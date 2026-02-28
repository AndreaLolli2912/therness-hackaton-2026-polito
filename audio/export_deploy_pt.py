"""Export a self-contained deployable TorchScript .pt for single-file binary prediction.

The exported module includes:
- waveform chunking
- mel preprocessing (inside WeldModel)
- per-window inference
- top-k MIL pooling
- thresholded final label (0=good_weld, 1=defect)

Usage:
    python -m audio.export_deploy_pt \
        --checkpoint checkpoints/audio/best_model.pt \
        --output checkpoints/audio/deploy_single_label.pt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from audio_model import AudioCNN
from audio.audio_processing import DEFAULT_AUDIO_CFG, WeldModel


def _load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def _infer_num_classes_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> int:
    key = "backbone.head.3.weight"
    if key not in state_dict:
        raise ValueError("Could not infer num_classes from checkpoint state_dict")
    return int(state_dict[key].shape[0])


def _load_sidecar_config(checkpoint_path: Path) -> Optional[Dict]:
    cfg_path = checkpoint_path.parent / "config.json"
    if not cfg_path.exists():
        return None
    return _load_json(cfg_path)


class DeploySingleLabelMIL(nn.Module):
    """Self-contained deployable binary predictor.

    Input:
      - waveform: Tensor shape (C, N) or (1, N)
    Output:
      - dict with:
          label: int64 tensor scalar (0=good_weld, 1=defect)
          p_defect: float32 tensor scalar
    """

    def __init__(
        self,
        base_model: nn.Module,
        chunk_samples: int,
        defect_idx: int,
        eval_pool_ratio: float,
        threshold: float,
    ):
        super().__init__()
        self.base_model = base_model
        self.chunk_samples = int(chunk_samples)
        self.defect_idx = int(defect_idx)
        self.eval_pool_ratio = float(eval_pool_ratio)
        self.threshold = float(threshold)

    def _aggregate_topk(self, prob_seq: torch.Tensor) -> torch.Tensor:
        k = max(1, int(math.ceil(self.eval_pool_ratio * float(prob_seq.numel()))))
        k = min(k, int(prob_seq.numel()))
        values, _ = torch.topk(prob_seq, k=k)
        return values.mean()

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        if waveform.dim() != 2:
            raise RuntimeError("waveform must have shape (C, N)")

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        n_samples = int(waveform.size(1))
        n_chunks = n_samples // self.chunk_samples
        if n_chunks < 1:
            raise RuntimeError("audio shorter than one chunk")

        used = waveform[:, : n_chunks * self.chunk_samples]
        windows = used.reshape(1, n_chunks, self.chunk_samples).permute(1, 0, 2)  # (T, 1, S)

        logits = self.base_model(windows)
        if logits.dim() == 2 and logits.size(1) == 1:
            probs = torch.sigmoid(logits).squeeze(1)
        else:
            probs = torch.softmax(logits, dim=1)[:, self.defect_idx]

        p_defect = self._aggregate_topk(probs)
        label = (p_defect >= self.threshold).to(torch.int64)

        return {
            "label": label,
            "p_defect": p_defect,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export deployable TorchScript .pt")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--eval_pool_ratio", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--defect_idx", type=int, default=None)
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sidecar = _load_sidecar_config(ckpt_path)

    audio_cfg = dict(DEFAULT_AUDIO_CFG)
    dropout = 0.15
    defect_idx = 0
    eval_pool_ratio = 0.05
    threshold = 0.5

    if sidecar is not None:
        audio_cfg = {**DEFAULT_AUDIO_CFG, **sidecar.get("audio", {}).get("feature_params", {})}
        dropout = float(sidecar.get("audio", {}).get("model", {}).get("dropout", 0.15))

        train_cfg = sidecar.get("audio", {}).get("training", {})
        mil_cfg = train_cfg.get("sequence_mil", {})
        eval_pool_ratio = float(mil_cfg.get("eval_pool_ratio", mil_cfg.get("topk_ratio_pos", 0.05)))
        threshold = float(mil_cfg.get("threshold", 0.5))

        label_to_idx = sidecar.get("label_to_idx", {})
        if isinstance(label_to_idx, dict) and "defect" in label_to_idx:
            defect_idx = int(label_to_idx["defect"])

    if args.defect_idx is not None:
        defect_idx = int(args.defect_idx)
    if args.eval_pool_ratio is not None:
        eval_pool_ratio = float(args.eval_pool_ratio)
    if args.threshold is not None:
        threshold = float(args.threshold)

    device = torch.device("cpu")
    try:
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location=device)

    model_state = checkpoint["model_state_dict"]
    num_classes = _infer_num_classes_from_state_dict(model_state)

    backbone = AudioCNN(num_classes=num_classes, dropout=dropout)
    base_model = WeldModel(backbone, cfg=audio_cfg)
    base_model.load_state_dict(model_state)
    base_model.eval()

    checkpoint_threshold = checkpoint.get("threshold", None)
    if args.threshold is None and checkpoint_threshold is not None:
        threshold = float(checkpoint_threshold)

    chunk_samples = int(float(audio_cfg["chunk_length_in_s"]) * int(audio_cfg["sampling_rate"]))

    deploy_model = DeploySingleLabelMIL(
        base_model=base_model,
        chunk_samples=chunk_samples,
        defect_idx=defect_idx,
        eval_pool_ratio=eval_pool_ratio,
        threshold=threshold,
    ).eval()

    scripted = torch.jit.script(deploy_model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    print(f"Exported deploy model: {out_path}")
    print(f"chunk_samples={chunk_samples} | defect_idx={defect_idx} | eval_pool_ratio={eval_pool_ratio} | threshold={threshold}")


if __name__ == "__main__":
    main()
