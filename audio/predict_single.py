"""Single-file inference for trained audio checkpoints.

Usage:
    python -m audio.predict_single \
        --checkpoint checkpoints/audio/best_model.pt \
        --audio_file /abs/path/sample.flac
"""

import argparse
import json
import math
import os

import torch
import torchaudio

from audio_model import AudioCNN
from audio.audio_processing import DEFAULT_AUDIO_CFG, WeldModel


def load_json(path):
    with open(path) as f:
        return json.load(f)


def load_label_mapping(checkpoint_path, task):
    ckpt_dir = os.path.dirname(checkpoint_path)
    saved_cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(saved_cfg_path):
        saved_cfg = load_json(saved_cfg_path)
        mapping = saved_cfg.get("label_to_idx")
        if mapping:
            return {k: int(v) for k, v in mapping.items()}

    if task == "binary":
        return {"defect": 0, "good_weld": 1}

    raise ValueError("Could not recover label mapping from checkpoint config.json")


def infer_num_classes_from_state_dict(state_dict):
    # AudioCNN head is: AdaptiveAvgPool2d, Flatten, Dropout, Linear
    linear_weight_key = "backbone.head.3.weight"
    if linear_weight_key in state_dict:
        return int(state_dict[linear_weight_key].shape[0])
    raise ValueError("Could not infer num_classes from checkpoint state_dict")


def load_runtime_cfg(checkpoint_path):
    """Load runtime settings from checkpoint sidecar config, with safe fallbacks."""
    ckpt_dir = os.path.dirname(checkpoint_path)
    sidecar_path = os.path.join(ckpt_dir, "config.json")

    runtime = {
        "audio_cfg": dict(DEFAULT_AUDIO_CFG),
        "dropout": 0.15,
        "task": "binary",
        "eval_pool_ratio": 0.05,
        "threshold": 0.5,
    }

    if not os.path.exists(sidecar_path):
        return runtime

    sidecar = load_json(sidecar_path)
    audio_cfg = sidecar.get("audio", {}).get("feature_params", {})
    runtime["audio_cfg"] = {**DEFAULT_AUDIO_CFG, **audio_cfg}
    runtime["dropout"] = float(sidecar.get("audio", {}).get("model", {}).get("dropout", 0.15))

    train_cfg = sidecar.get("audio", {}).get("training", {})
    mil_cfg = train_cfg.get("sequence_mil", {})
    runtime["task"] = train_cfg.get("task", "binary")
    runtime["eval_pool_ratio"] = float(mil_cfg.get("eval_pool_ratio", mil_cfg.get("topk_ratio_pos", 0.05)))
    runtime["threshold"] = float(mil_cfg.get("threshold", 0.5))
    return runtime


def aggregate_topk(prob_seq, ratio):
    k = max(1, int(math.ceil(ratio * prob_seq.numel())))
    k = min(k, int(prob_seq.numel()))
    vals, _ = torch.topk(prob_seq, k=k)
    return vals.mean()


def window_defect_probs(model, waveform, chunk_samples, defect_idx, device, batch_size):
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    n_samples = waveform.shape[-1]
    n_chunks = n_samples // chunk_samples
    if n_chunks < 1:
        raise ValueError(
            f"Audio is too short: got {n_samples} samples, requires at least {chunk_samples}."
        )

    used = waveform[:, : n_chunks * chunk_samples]
    windows = used.reshape(1, n_chunks, chunk_samples).permute(1, 0, 2)  # (T, 1, S)

    probs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, n_chunks, batch_size):
            batch = windows[i : i + batch_size].to(device)
            logits = model(batch)
            if logits.dim() == 2 and logits.size(1) == 1:
                p_def = torch.sigmoid(logits).squeeze(1)
            else:
                p_def = torch.softmax(logits, dim=1)[:, defect_idx]
            probs.append(p_def.cpu())

    return torch.cat(probs, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Predict one label for one .flac file")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--audio_file", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--eval_pool_ratio", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    runtime_cfg = load_runtime_cfg(args.checkpoint)
    audio_cfg = runtime_cfg["audio_cfg"]
    task = runtime_cfg["task"]

    if task != "binary":
        raise ValueError("predict_single currently supports binary task.")

    label_to_idx = load_label_mapping(args.checkpoint, task=task)
    if "defect" not in label_to_idx or "good_weld" not in label_to_idx:
        raise ValueError("Binary task requires 'defect' and 'good_weld' labels.")

    defect_idx = int(label_to_idx["defect"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.checkpoint, map_location=device)

    num_classes = infer_num_classes_from_state_dict(checkpoint["model_state_dict"])
    backbone = AudioCNN(
        num_classes=num_classes,
        dropout=runtime_cfg["dropout"],
    )
    model = WeldModel(backbone, cfg=audio_cfg)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    waveform, sr = torchaudio.load(args.audio_file)
    target_sr = int(audio_cfg["sampling_rate"])
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    chunk_samples = int(float(audio_cfg["chunk_length_in_s"]) * target_sr)
    probs = window_defect_probs(
        model=model,
        waveform=waveform,
        chunk_samples=chunk_samples,
        defect_idx=defect_idx,
        device=device,
        batch_size=args.batch_size,
    )

    eval_pool_ratio = (
        float(args.eval_pool_ratio)
        if args.eval_pool_ratio is not None
        else float(runtime_cfg["eval_pool_ratio"])
    )
    file_prob = float(aggregate_topk(probs, eval_pool_ratio).item())

    checkpoint_threshold = checkpoint.get("threshold", runtime_cfg["threshold"])
    threshold = float(args.threshold) if args.threshold is not None else float(checkpoint_threshold)

    pred_label = "defect" if file_prob >= threshold else "good_weld"

    if args.verbose:
        print(f"audio_file: {args.audio_file}")
        print(f"device: {device}")
        print(f"num_windows: {int(probs.numel())}")
        print(f"pool_ratio: {eval_pool_ratio:.4f}")
        print(f"threshold: {threshold:.4f}")
        print(f"p_defect: {file_prob:.6f}")
        print(f"prediction: {pred_label}")
    else:
        print(pred_label)


if __name__ == "__main__":
    main()
