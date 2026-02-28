"""Train and evaluate the VideoCNNBackbone on welding video data.

Usage:
    # Train with default config
    python -m video.run_video --config configs/master_config.json

    # Test only
    python -m video.run_video --config configs/master_config.json --test_only --checkpoint checkpoints/video/best_model.pt
"""

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from models.video_backbone import VideoCNNBackbone
from video.video_processing import (
    WeldingVideoDataset, WeldingFrameDataset, WeldVideoModel,
    get_video_files_and_labels,
    LABEL_CODE_MAP, CODE_TO_IDX,
)
from video.train import train_epoch, validate_epoch
from video.test import run_test


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return json.load(f)


def save_config(cfg: dict, path: str):
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to {path}")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels, num_classes=7, power=1.0):
    """Inverse-frequency weights with optional power scaling."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * count)
    if power != 1.0:
        weights = torch.pow(weights, power)
    mean_w = float(weights.mean()) if len(weights) > 0 else 1.0
    if mean_w > 0:
        weights = weights / mean_w
    return weights


def build_balanced_sampler(dataset, num_classes=7, power=0.35, max_samples=None):
    """Build a WeightedRandomSampler from window labels.

    Args:
        max_samples: if set, cap epoch size to this many windows (speeds up training
                     when the dataset has far more windows than needed per epoch).
    """
    if not getattr(dataset, "samples", None):
        return None

    labels = [int(s.get("label", 0)) for s in dataset.samples]
    counts = Counter(labels)
    if not counts:
        return None

    class_weights = {
        cls_idx: (1.0 / max(counts.get(cls_idx, 1), 1)) ** power
        for cls_idx in range(num_classes)
    }
    sample_weights = torch.tensor(
        [class_weights.get(lbl, 1.0) for lbl in labels],
        dtype=torch.double,
    )
    num_samples = len(sample_weights) if max_samples is None else min(max_samples, len(sample_weights))
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,
        replacement=True,
    )


def _to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_cpu(v) for v in obj)
    return obj


def _write_json(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Video welding defect classification")
    parser.add_argument("--config", type=str, default="configs/master_config.json")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    w_conf = cfg["video_window"]["training"]
    m_conf = cfg["video_window"]["model"]
    data_root = cfg["data_root"]
    num_classes = cfg.get("num_classes", 7)
    seed = w_conf.get("seed", 42)

    set_seed(seed)

    device_str = cfg.get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"Device: {device}")

    # ── Discover video files ─────────────────────────────────────
    print(f"\nDiscovering video files in {os.path.abspath(data_root)}...")
    video_data = get_video_files_and_labels(data_root)
    if not video_data:
        raise FileNotFoundError(f"No video data found in {data_root}")

    paths, labels, groups = zip(*video_data)
    paths = list(paths)
    labels = list(labels)
    groups = np.array(groups)
    label_indices = [CODE_TO_IDX.get(lbl, 0) for lbl in labels]

    print(f"Found {len(paths)} videos")
    label_counts = Counter(labels)
    for code in sorted(label_counts):
        idx = CODE_TO_IDX.get(code, '?')
        print(f"  Code {code} (class {idx}): {label_counts[code]} videos")

    # ── Train/val split ──────────────────────────────────────────
    split_strategy = w_conf.get("split_strategy", "group_shuffle")

    if split_strategy == "group_shuffle":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        train_idx, val_idx = next(gss.split(paths, label_indices, groups=groups))
    else:
        indices = list(range(len(paths)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=seed, stratify=label_indices
        )

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    train_label_indices = [label_indices[i] for i in train_idx]

    print(f"Train: {len(train_paths)} videos | Val: {len(val_paths)} videos")

    # ── Datasets ─────────────────────────────────────────────────
    img_size    = int(w_conf.get("img_size", 160))
    num_frames  = int(w_conf.get("num_frames", 8))
    clip_seconds = w_conf.get("clip_seconds", None)
    frames_dir  = w_conf.get("frames_dir", None)

    if frames_dir and os.path.isfile(os.path.join(frames_dir, "manifest.json")):
        print(f"Using pre-extracted frames from {frames_dir}")
        train_dataset = WeldingFrameDataset(
            train_paths, train_labels,
            frames_dir=frames_dir, num_frames=num_frames, img_size=img_size,
            data_root=data_root,
        )
        val_dataset = WeldingFrameDataset(
            val_paths, val_labels,
            frames_dir=frames_dir, num_frames=num_frames, img_size=img_size,
            data_root=data_root,
        )
        print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | num_frames={num_frames} [JPEG]")
    else:
        train_dataset = WeldingVideoDataset(
            train_paths, train_labels,
            num_frames=num_frames, img_size=img_size,
            clip_seconds=clip_seconds, data_root=data_root,
        )
        val_dataset = WeldingVideoDataset(
            val_paths, val_labels,
            num_frames=num_frames, img_size=img_size,
            clip_seconds=clip_seconds, data_root=data_root,
        )
        print(
            f"Train videos: {len(train_dataset)} | Val videos: {len(val_dataset)} "
            f"| num_frames={num_frames} | clip_seconds={clip_seconds} [AVI seek]"
        )

    # ── DataLoaders ──────────────────────────────────────────────
    num_workers = int(w_conf.get("num_workers", 4))
    use_balanced_sampler = bool(w_conf.get("use_balanced_sampler", False))
    sampler_power = float(w_conf.get("balanced_sampler_power", 0.35))

    train_sampler = None
    if use_balanced_sampler:
        # Build sampler from video-level labels (one weight per video)
        from collections import Counter as _Counter
        vid_labels = [s["label"] for s in train_dataset.samples]
        counts = _Counter(vid_labels)
        cls_w = {c: (1.0 / max(counts[c], 1)) ** sampler_power for c in range(num_classes)}
        sw = torch.tensor([cls_w[l] for l in vid_labels], dtype=torch.double)
        train_sampler = WeightedRandomSampler(weights=sw, num_samples=len(sw), replacement=True)
        print(f"Balanced sampler: enabled (power={sampler_power}, videos/epoch={len(sw)})")

    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
    }
    if num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": 4,
        })

    train_loader = DataLoader(
        train_dataset,
        batch_size=w_conf["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=w_conf["batch_size"],
        shuffle=False,
        **loader_kwargs,
    )

    # ── Model ────────────────────────────────────────────────────
    dropout = float(m_conf.get("dropout", 0.2))
    backbone = VideoCNNBackbone(num_classes=num_classes, dropout=dropout)
    model = WeldVideoModel(backbone)
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ─────────────────────────────────────────────────────
    class_weight_power = float(w_conf.get("class_weight_power", 1.0))
    use_weights = w_conf.get("class_weights", "inverse_frequency")
    if use_weights == "inverse_frequency":
        train_vid_labels = [s["label"] for s in train_dataset.samples]
        weights = compute_class_weights(
            train_vid_labels, num_classes=num_classes, power=class_weight_power,
        ).to(device)
        print(f"Class weights: {[f'{w:.3f}' for w in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    # ── Test-only mode ───────────────────────────────────────────
    if args.test_only:
        assert args.checkpoint, "--checkpoint required for --test_only"
        result = run_test(model, val_loader, criterion, device, args.checkpoint)
        print(f"Val loss:           {result['loss']:.4f}")
        print(f"Val Macro F1:       {result['macro_f1']:.4f}")
        print(f"Val Binary F1:      {result['binary_f1']:.4f}")
        print(f"Val Hackathon Score: {result['hackathon_score']:.4f}")
        return

    # ── Optimizer & scheduler ────────────────────────────────────
    weight_decay = float(w_conf.get("weight_decay", 0.0))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=w_conf["lr"], weight_decay=weight_decay,
    )

    lr_sched_cfg = w_conf.get("lr_schedule", {})
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_sched_cfg.get("plateau_factor", 0.5),
        patience=lr_sched_cfg.get("plateau_patience", 4),
        threshold=lr_sched_cfg.get("plateau_threshold", 1e-3),
        min_lr=lr_sched_cfg.get("plateau_min_lr", 1e-6),
    )

    total_steps = w_conf["epochs"] * len(train_loader)
    warmup_ratio = lr_sched_cfg.get("warmup_ratio", 0.1)
    warmup_steps = int(total_steps * warmup_ratio) if warmup_ratio else 0
    base_lrs = [w_conf["lr"]] * len(optimizer.param_groups)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── Save config ──────────────────────────────────────────────
    checkpoint_dir = w_conf.get("checkpoint_dir", "checkpoints/video")
    os.makedirs(checkpoint_dir, exist_ok=True)
    save_config(cfg, os.path.join(checkpoint_dir, "config.json"))

    # ── Training loop ────────────────────────────────────────────
    num_epochs = w_conf["epochs"]
    patience = w_conf.get("patience", None)
    best_score = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    global_step = 0
    train_losses = []
    val_losses = []

    print(f"\n{'='*60}")
    print(f"  TRAINING START — {num_epochs} epochs")
    print(f"  Checkpoint dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_result = train_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler,
            warmup_steps=warmup_steps,
            global_step=global_step,
            base_lrs=base_lrs,
        )
        global_step = train_result["global_step"]

        val_result = validate_epoch(model, val_loader, criterion, device)

        train_losses.append(train_result["loss"])
        val_losses.append(val_result["loss"])

        print(
            f"Train loss: {train_result['loss']:.4f} | "
            f"Train F1: {train_result['macro_f1']:.4f} | "
            f"Val loss: {val_result['loss']:.4f} | "
            f"Val Macro F1: {val_result['macro_f1']:.4f} | "
            f"Val Binary F1: {val_result['binary_f1']:.4f} | "
            f"Hackathon: {val_result['hackathon_score']:.4f} | "
            f"LR: {train_result['lr']:.6g}"
        )

        plateau_scheduler.step(val_result["loss"])

        score = val_result["hackathon_score"]

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": _to_cpu(model.state_dict()),
            "optimizer_state_dict": _to_cpu(optimizer.state_dict()),
            "val_loss": val_result["loss"],
            "val_f1": val_result["macro_f1"],
            "val_binary_f1": val_result["binary_f1"],
            "hackathon_score": score,
        }

        metrics_payload = {
            "epoch": epoch + 1,
            "val_loss": float(val_result["loss"]),
            "val_f1": float(val_result["macro_f1"]),
            "val_binary_f1": float(val_result["binary_f1"]),
            "hackathon_score": float(score),
        }

        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pt"))
        _write_json(os.path.join(checkpoint_dir, "last_metrics.json"), metrics_payload)

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            _write_json(os.path.join(checkpoint_dir, "best_metrics.json"), metrics_payload)
            print(f"New best model (hackathon_score={score:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if patience and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining complete. Best epoch: {best_epoch} (score={best_score:.4f})")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }


if __name__ == "__main__":
    main()
