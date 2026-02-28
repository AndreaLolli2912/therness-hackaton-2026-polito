"""Export self-contained deployable TorchScript .pt models.

Two deploy classes are provided:

  DeploySingleLabelMIL  — binary classifier (good_weld vs defect).
    forward(waveform)           → full file prediction via top-k MIL pooling.
    predict_window(window)      → single 0.2 s chunk prediction.

  DeployMulticlassFile  — multiclass classifier (7 defect types).
    forward(waveform)           → full file prediction via mean pooling.
    predict_window(window)      → single 0.2 s chunk prediction.

Both are exported as TorchScript and work from a single .pt file.
The correct class is chosen automatically from the checkpoint:
  - 2 output classes  → DeploySingleLabelMIL
  - >2 output classes → DeployMulticlassFile

Usage
-----
Binary:
    python -m audio.export_deploy_pt \\
        --checkpoint checkpoints/audio_binary/best_model.pt \\
        --output     checkpoints/audio_binary/deploy_binary.pt

Multiclass:
    python -m audio.export_deploy_pt \\
        --checkpoint checkpoints/audio_multiclass/best_model.pt \\
        --output     checkpoints/audio_multiclass/deploy_multiclass.pt
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


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

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


def _resolve_device(device_arg: str) -> torch.device:
    dev = str(device_arg).strip().lower()
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested, but CUDA is not available")
        return torch.device("cuda")
    if dev == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_arg}. Use one of: auto, cuda, cpu")


# ──────────────────────────────────────────────────────────────────────────────
# Binary deploy model (good_weld vs defect)
# ──────────────────────────────────────────────────────────────────────────────

class DeploySingleLabelMIL(nn.Module):
    """Self-contained binary predictor (TorchScript-exportable).

    forward(waveform: Tensor[C, N]) → {"label": int64, "p_defect": float32}
      Full waveform → chunks → per-window defect prob → top-k pool → threshold

    predict_window(window: Tensor[1, S]) → {"label": int64, "p_defect": float32}
      Single pre-chunked window → direct prediction
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

    def _chunk_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
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
        return windows

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full waveform (C, N) → top-k pooled binary prediction."""
        windows = self._chunk_waveform(waveform)

        logits = self.base_model(windows)
        if logits.dim() == 2 and logits.size(1) == 1:
            probs = torch.sigmoid(logits).squeeze(1)
        else:
            probs = torch.softmax(logits, dim=1)[:, self.defect_idx]

        p_defect = self._aggregate_topk(probs)
        label = (p_defect >= self.threshold).to(torch.int64)

        return {"label": label, "p_defect": p_defect}

    @torch.jit.export
    def extract_window_activation(self, window: torch.Tensor) -> torch.Tensor:
        """Return penultimate embedding for a single window: (128,) tensor."""
        if window.dim() == 2:
            window = window.unsqueeze(0)  # (1, 1, S)
        feats = self.base_model.forward_features(window)  # (1, 128)
        return feats.squeeze(0)

    @torch.jit.export
    def extract_window_activations(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return full stage/head activations for a single window."""
        if window.dim() == 2:
            window = window.unsqueeze(0)  # (1, 1, S)
        acts = self.base_model.forward_activations(window)
        return {
            "stem": acts["stem"].squeeze(0),
            "stage1": acts["stage1"].squeeze(0),
            "stage2": acts["stage2"].squeeze(0),
            "stage3": acts["stage3"].squeeze(0),
            "head_pool": acts["head_pool"].squeeze(0),
            "head_flat": acts["head_flat"].squeeze(0),
            "head_dropout": acts["head_dropout"].squeeze(0),
            "logits": acts["logits"].squeeze(0),
        }

    @torch.jit.export
    def extract_file_activations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return per-window embeddings for full file: (T, 128)."""
        windows = self._chunk_waveform(waveform)
        feats = self.base_model.forward_features(windows)  # (T, 128)
        return feats

    @torch.jit.export
    def extract_file_activation_mean(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled file embedding: (128,)."""
        feats = self.extract_file_activations(waveform)
        return feats.mean(dim=0)

    @torch.jit.export
    def extract_file_activation_summary(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return mean-pooled stage/head activations for a full file."""
        windows = self._chunk_waveform(waveform)
        acts = self.base_model.forward_activations(windows)
        return {
            "stem": acts["stem"].mean(dim=0),
            "stage1": acts["stage1"].mean(dim=0),
            "stage2": acts["stage2"].mean(dim=0),
            "stage3": acts["stage3"].mean(dim=0),
            "head_pool": acts["head_pool"].mean(dim=0),
            "head_flat": acts["head_flat"].mean(dim=0),
            "head_dropout": acts["head_dropout"].mean(dim=0),
            "logits": acts["logits"].mean(dim=0),
        }

    @torch.jit.export
    def predict_window(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single pre-chunked window (1, S) → direct binary prediction.

        Useful for real-time / streaming inference where you already have
        a fixed-size chunk and want an immediate per-window prediction.
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)  # (1, 1, S)

        logits = self.base_model(window)  # (1, num_classes)

        if logits.size(1) == 1:
            p_defect = torch.sigmoid(logits).squeeze()
        else:
            p_defect = torch.softmax(logits, dim=1).squeeze()[self.defect_idx]

        label = (p_defect >= self.threshold).to(torch.int64)
        return {"label": label, "p_defect": p_defect}


# ──────────────────────────────────────────────────────────────────────────────
# Multiclass deploy model (7 defect types)
# ──────────────────────────────────────────────────────────────────────────────

class DeployMulticlassFile(nn.Module):
    """Self-contained multiclass predictor (TorchScript-exportable).

    forward(waveform: Tensor[C, N]) → {"label": int64, "probs": float32[num_classes]}
      Full waveform → chunks → per-window softmax → mean over all windows → argmax

    predict_window(window: Tensor[1, S]) → {"label": int64, "probs": float32[num_classes]}
      Single pre-chunked window → direct prediction
    """

    def __init__(
        self,
        base_model: nn.Module,
        chunk_samples: int,
        num_classes: int,
    ):
        super().__init__()
        self.base_model = base_model
        self.chunk_samples = int(chunk_samples)
        self.num_classes = int(num_classes)

    def _chunk_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
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
        return windows

    def forward(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Full waveform (C, N) → mean-pooled multiclass prediction."""
        windows = self._chunk_waveform(waveform)

        logits = self.base_model(windows)          # (T, num_classes)
        probs = torch.softmax(logits, dim=1)       # (T, num_classes)
        mean_probs = probs.mean(dim=0)             # (num_classes,)
        label = mean_probs.argmax().to(torch.int64)

        return {"label": label, "probs": mean_probs}

    @torch.jit.export
    def extract_window_activation(self, window: torch.Tensor) -> torch.Tensor:
        """Return penultimate embedding for a single window: (128,) tensor."""
        if window.dim() == 2:
            window = window.unsqueeze(0)  # (1, 1, S)
        feats = self.base_model.forward_features(window)  # (1, 128)
        return feats.squeeze(0)

    @torch.jit.export
    def extract_window_activations(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return full stage/head activations for a single window."""
        if window.dim() == 2:
            window = window.unsqueeze(0)  # (1, 1, S)
        acts = self.base_model.forward_activations(window)
        return {
            "stem": acts["stem"].squeeze(0),
            "stage1": acts["stage1"].squeeze(0),
            "stage2": acts["stage2"].squeeze(0),
            "stage3": acts["stage3"].squeeze(0),
            "head_pool": acts["head_pool"].squeeze(0),
            "head_flat": acts["head_flat"].squeeze(0),
            "head_dropout": acts["head_dropout"].squeeze(0),
            "logits": acts["logits"].squeeze(0),
        }

    @torch.jit.export
    def extract_file_activations(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return per-window embeddings for full file: (T, 128)."""
        windows = self._chunk_waveform(waveform)
        feats = self.base_model.forward_features(windows)  # (T, 128)
        return feats

    @torch.jit.export
    def extract_file_activation_mean(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return mean-pooled file embedding: (128,)."""
        feats = self.extract_file_activations(waveform)
        return feats.mean(dim=0)

    @torch.jit.export
    def extract_file_activation_summary(self, waveform: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return mean-pooled stage/head activations for a full file."""
        windows = self._chunk_waveform(waveform)
        acts = self.base_model.forward_activations(windows)
        return {
            "stem": acts["stem"].mean(dim=0),
            "stage1": acts["stage1"].mean(dim=0),
            "stage2": acts["stage2"].mean(dim=0),
            "stage3": acts["stage3"].mean(dim=0),
            "head_pool": acts["head_pool"].mean(dim=0),
            "head_flat": acts["head_flat"].mean(dim=0),
            "head_dropout": acts["head_dropout"].mean(dim=0),
            "logits": acts["logits"].mean(dim=0),
        }

    @torch.jit.export
    def predict_window(self, window: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single pre-chunked window (1, S) → direct multiclass prediction.

        Useful for real-time / streaming inference.
        """
        if window.dim() == 2:
            window = window.unsqueeze(0)  # (1, 1, S)

        logits = self.base_model(window)          # (1, num_classes)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        label = probs.argmax().to(torch.int64)

        return {"label": label, "probs": probs}


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Export deployable TorchScript .pt")
    parser.add_argument("--checkpoint",      type=str,   required=True)
    parser.add_argument("--output",          type=str,   required=True)
    # Binary-specific overrides (ignored for multiclass)
    parser.add_argument("--eval_pool_ratio", type=float, default=None)
    parser.add_argument("--threshold",       type=float, default=None)
    parser.add_argument("--defect_idx",      type=int,   default=None)
    parser.add_argument("--device",          type=str,   default="auto",
                        help="Export device: auto|cuda|cpu (default: auto)")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    sidecar = _load_sidecar_config(ckpt_path)

    # ── Defaults ──────────────────────────────────────────────────
    audio_cfg      = dict(DEFAULT_AUDIO_CFG)
    dropout        = 0.15
    defect_idx     = 0
    eval_pool_ratio = 0.05
    threshold      = 0.5
    task           = "binary"   # will be overridden from sidecar if available

    if sidecar is not None:
        audio_cfg  = {**DEFAULT_AUDIO_CFG, **sidecar.get("audio", {}).get("feature_params", {})}
        dropout    = float(sidecar.get("audio", {}).get("model", {}).get("dropout", 0.15))

        train_cfg  = sidecar.get("audio", {}).get("training", {})
        task       = train_cfg.get("task", "binary")
        mil_cfg    = train_cfg.get("sequence_mil", {})
        eval_pool_ratio = float(mil_cfg.get("eval_pool_ratio", mil_cfg.get("topk_ratio_pos", 0.05)))
        threshold  = float(mil_cfg.get("threshold", 0.5))

        label_to_idx = sidecar.get("label_to_idx", {})
        if isinstance(label_to_idx, dict) and "defect" in label_to_idx:
            defect_idx = int(label_to_idx["defect"])

    # ── CLI overrides ──────────────────────────────────────────────
    if args.defect_idx      is not None: defect_idx      = int(args.defect_idx)
    if args.eval_pool_ratio is not None: eval_pool_ratio = float(args.eval_pool_ratio)
    if args.threshold       is not None: threshold       = float(args.threshold)

    # ── Load checkpoint ────────────────────────────────────────────
    device = _resolve_device(args.device)
    try:
        checkpoint = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(str(ckpt_path), map_location=device)

    model_state = checkpoint["model_state_dict"]
    num_classes = _infer_num_classes_from_state_dict(model_state)

    # Use threshold stored in checkpoint (learned via auto_threshold) unless overridden
    if args.threshold is None:
        checkpoint_threshold = checkpoint.get("threshold", None)
        if checkpoint_threshold is not None:
            threshold = float(checkpoint_threshold)

    # ── Build base model ───────────────────────────────────────────
    backbone   = AudioCNN(num_classes=num_classes, dropout=dropout)
    base_model = WeldModel(backbone, cfg=audio_cfg)
    base_model.load_state_dict(model_state)
    base_model.to(device).eval()

    chunk_samples = int(float(audio_cfg["chunk_length_in_s"]) * int(audio_cfg["sampling_rate"]))

    # ── Wrap in deploy class ───────────────────────────────────────
    is_binary = (num_classes == 2) or (task == "binary")

    if is_binary:
        deploy_model = DeploySingleLabelMIL(
            base_model=base_model,
            chunk_samples=chunk_samples,
            defect_idx=defect_idx,
            eval_pool_ratio=eval_pool_ratio,
            threshold=threshold,
        ).to(device).eval()
        print(f"Export mode : binary (DeploySingleLabelMIL)")
        print(f"  defect_idx     = {defect_idx}")
        print(f"  eval_pool_ratio= {eval_pool_ratio}")
        print(f"  threshold      = {threshold}")
    else:
        deploy_model = DeployMulticlassFile(
            base_model=base_model,
            chunk_samples=chunk_samples,
            num_classes=num_classes,
        ).to(device).eval()
        print(f"Export mode : multiclass (DeployMulticlassFile)")
        print(f"  num_classes    = {num_classes}")

    print(f"  export_device  = {device}")
    print(f"  chunk_samples  = {chunk_samples}  ({audio_cfg['chunk_length_in_s']}s @ {audio_cfg['sampling_rate']} Hz)")

    # ── TorchScript export ─────────────────────────────────────────
    scripted = torch.jit.script(deploy_model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))

    print(f"\nSaved: {out_path.resolve()}")
    print("Methods available on loaded model:")
    print("  model(waveform)              → file-level prediction")
    print("  model.predict_window(window) → single-window prediction")
    print("  model.extract_window_activation(window)    → (128,) embedding")
    print("  model.extract_file_activations(waveform)   → (T, 128) embeddings")
    print("  model.extract_file_activation_mean(waveform) → (128,) embedding")
    print("  model.extract_window_activations(window)   → all stage/head activations")
    print("  model.extract_file_activation_summary(waveform) → mean stage/head activations")


if __name__ == "__main__":
    main()
