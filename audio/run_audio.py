"""Train and evaluate the AudioCNN on welding audio data.

Usage:
    # Train with default config
    python -m audio.run_audio --config configs/master_config.json

    # Test only (evaluate checkpoint on val set)
    python -m audio.run_audio --config configs/master_config.json --test_only --checkpoint checkpoints/audio/best_model.pt

    # Generate submission CSV
    python -m audio.run_audio --config configs/master_config.json --submission --checkpoint checkpoints/audio/best_model.pt
"""

import argparse
import glob
import json
import os
import random
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from audio_model import AudioCNN, AudioCNNBackbone
from audio.audio_processing import AudioDataset, AudioFileDataset, WeldModel, WeldBackboneModel
from audio.run_train import run_training
from audio.run_train_mil import run_training_mil
from audio.test import run_test, run_test_mil, generate_submission


def load_config(config_path: str) -> dict:
    """Load the full JSON config."""
    with open(config_path) as f:
        return json.load(f)


def save_config(cfg: dict, path: str):
    """Save config dict as JSON alongside checkpoints for reproducibility."""
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Config saved to {path}")


# ── Collate functions ─────────────────────────────────────────────────
# AudioDataset.__getitem__ returns dicts with "waveform", "label", etc.
# train_epoch / validate_epoch expect (inputs, targets).

def train_collate_fn(batch):
    """Collate for training/validation: (waveform_tensor, label) pairs."""
    waveforms = torch.stack([item["waveform"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch])

    if "material_id" in batch[0]:
        material_ids = torch.tensor([item["material_id"] for item in batch])
        inputs = {"waveforms": waveforms, "material_id": material_ids}
        return inputs, labels

    return waveforms, labels


def submission_collate_fn(batch):
    """Collate for submission inference: (waveform_tensor, sample_ids) pairs."""
    waveforms = torch.stack([item["waveform"] for item in batch])
    sample_ids = [item["sample_id"] for item in batch]
    return waveforms, sample_ids


def mil_collate_fn(batch):
    """Collate file-level batches where each sample has a variable number of windows."""
    lengths = torch.tensor([item["num_windows"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    max_windows = int(lengths.max().item())
    sample_shape = batch[0]["windows"].shape[1:]  # (1, S)

    padded = torch.zeros((len(batch), max_windows, *sample_shape), dtype=batch[0]["windows"].dtype)
    for i, item in enumerate(batch):
        t = item["num_windows"]
        padded[i, :t] = item["windows"]
    return padded, lengths, labels


_DEFECT_RE = re.compile(r"^(?P<defect>.+?)(?:_weld)?_\d+_")


def infer_file_label(path: str, data_root: str, task: str) -> str:
    """Infer file-level label directly from path for stratified split."""
    rel = os.path.relpath(path, data_root)
    parts = rel.split(os.sep)

    top_folder = parts[0] if parts else ""
    if top_folder == "good_weld":
        defect_type = "good_weld"
    else:
        defect_folder = parts[1] if len(parts) > 1 else ""
        m = _DEFECT_RE.match(defect_folder)
        defect_type = m.group("defect") if m else defect_folder

    if task == "binary":
        return "good_weld" if defect_type == "good_weld" else "defect"
    return defect_type


def format_counts(counts: Counter) -> str:
    total = sum(counts.values())
    if total == 0:
        return "{}"
    parts = []
    for label in sorted(counts):
        n = counts[label]
        pct = (100.0 * n) / total
        parts.append(f"{label}: {n} ({pct:.1f}%)")
    return "{ " + ", ".join(parts) + " }"


def chunk_label_counts(dataset, data_root: str, task: str) -> Counter:
    """Compute chunk-level class counts without re-loading waveforms."""
    labels_by_file = [infer_file_label(p, data_root, task) for p in dataset.files]
    counts = Counter()
    for file_idx, _ in dataset._index:
        counts[labels_by_file[file_idx]] += 1
    return counts


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audio welding defect classification")
    parser.add_argument("--config", type=str, default="configs/master_config.json",
                        help="Path to JSON config file")
    parser.add_argument("--print_config", action="store_true",
                        help="Print full loaded config JSON")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for --test_only or --submission")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training, evaluate checkpoint on val set")
    parser.add_argument("--submission", action="store_true",
                        help="Generate submission CSV from unlabeled test set")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    cfg = load_config(args.config)

    audio_cfg = cfg["audio"]["feature_params"]
    model_cfg = cfg["audio"]["model"]
    train_cfg = cfg["audio"]["training"]
    num_classes = cfg["num_classes"]
    data_root = cfg.get("data_root", cfg.get("train_data_root"))
    test_root = cfg.get("test_root", cfg.get("test_data_root", ""))
    if not data_root:
        raise KeyError(
            "Missing dataset root in config. Provide either 'data_root' or 'train_data_root'."
        )

    optim_cfg = {"lr": train_cfg["lr"], "weight_decay": train_cfg.get("weight_decay", 1e-4)}
    lr_sched_cfg = train_cfg.get("lr_schedule", {})

    if args.print_config or bool(train_cfg.get("print_config", False)):
        print("=" * 50)
        print("Configuration:")
        print(json.dumps(cfg, indent=2))
        print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Submission mode (no labeled data needed) ──────────────────
    if args.submission:
        assert args.checkpoint, "--checkpoint required for --submission"

        test_dataset = AudioDataset(test_root, cfg=audio_cfg, labeled=False)
        submission_batch_size = int(train_cfg.get("batch_size", mil_cfg.get("batch_size", 8)))
        test_loader = DataLoader(
            test_dataset, batch_size=submission_batch_size, shuffle=False,
            num_workers=train_cfg["num_workers"], collate_fn=submission_collate_fn,
        )

        # Load label_map from saved config in checkpoint dir
        ckpt_dir = os.path.dirname(args.checkpoint)
        saved_cfg_path = os.path.join(ckpt_dir, "config.json")
        if os.path.exists(saved_cfg_path):
            with open(saved_cfg_path) as f:
                saved = json.load(f)
            label_map = {int(k): v for k, v in saved["label_map"].items()}
        else:
            print("WARNING: no config.json found in checkpoint dir, using default label_map")
            label_map = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}

        backbone = AudioCNNBackbone(num_classes=len(label_map), dropout=model_cfg["dropout"])
        model = WeldBackboneModel(backbone, cfg=audio_cfg)
        generate_submission(
            model, test_loader, device, args.checkpoint,
            label_map=label_map, output_path="submission.csv",
        )
        return

    # ── File-level train/val split (no data leakage) ──────────────
    seed = train_cfg["seed"]
    val_split = train_cfg["val_split"]

    all_files = sorted(
        glob.glob(os.path.join(data_root, "**", "*.flac"), recursive=True)
    )

    if len(all_files) == 0:
        raise FileNotFoundError(
            "No .flac files found under data_root. "
            f"Configured data_root='{data_root}'. "
            "Check mount/path permissions and re-run Cell 2 to refresh config."
        )

    random.seed(seed)
    random.shuffle(all_files)

    # Optional: use only a fraction of data for fast iteration
    train_fraction = train_cfg.get("train_fraction", 1.0)
    if train_fraction < 1.0:
        all_files = all_files[:int(len(all_files) * train_fraction)]

    if len(all_files) == 0:
        raise ValueError(
            "Dataset became empty after applying train_fraction. "
            f"train_fraction={train_fraction}. Increase it above 0.0."
        )

    task = train_cfg.get("task", "multiclass")
    use_material = train_cfg.get("use_material", False)
    mil_cfg = train_cfg.get("sequence_mil", {})
    mil_enabled = bool(mil_cfg.get("enabled", False))

    file_labels = [infer_file_label(p, data_root, task) for p in all_files]
    file_label_counts = Counter(file_labels)

    print(f"\nTotal weld files: {len(all_files)}")
    print(f"File label distribution ({task}): {dict(file_label_counts)}")

    # Stratified split keeps stable class proportions between train/val.
    can_stratify = len(file_label_counts) > 1 and min(file_label_counts.values()) >= 2
    if can_stratify:
        train_files, val_files = train_test_split(
            all_files,
            test_size=val_split,
            random_state=seed,
            stratify=file_labels,
        )
        print("Split strategy: stratified")
    else:
        val_size = int(len(all_files) * val_split)
        val_files = all_files[:val_size]
        train_files = all_files[val_size:]
        print("Split strategy: random (fallback, insufficient class counts for stratify)")

    if len(train_files) == 0 or len(val_files) == 0:
        raise ValueError(
            "Train/val split produced an empty set. "
            f"train={len(train_files)}, val={len(val_files)}, val_split={val_split}. "
            "Adjust val_split or train_fraction."
        )

    print(f"Train welds: {len(train_files)} | Val welds: {len(val_files)}")
    print(
        "File split stats | "
        f"train={format_counts(Counter(infer_file_label(p, data_root, task) for p in train_files))} | "
        f"val={format_counts(Counter(infer_file_label(p, data_root, task) for p in val_files))}"
    )

    if mil_enabled:
        train_dataset = AudioFileDataset(
            data_root=data_root,
            cfg=audio_cfg,
            task=task,
            files=train_files,
        )
        val_dataset = AudioFileDataset(
            data_root=data_root,
            cfg=audio_cfg,
            task=task,
            files=val_files,
            label_to_idx=train_dataset.label_to_idx,
        )
    else:
        train_dataset = AudioDataset(
            data_root=data_root,
            cfg=audio_cfg,
            task=task,
            use_material=use_material,
            files=train_files,
        )
        val_dataset = AudioDataset(
            data_root=data_root,
            cfg=audio_cfg,
            task=task,
            use_material=use_material,
            files=val_files,
            label_to_idx=train_dataset.label_to_idx,
            material_to_idx=getattr(train_dataset, "material_to_idx", None),
        )

    num_classes = len(train_dataset.label_to_idx)
    if mil_enabled:
        print(f"Train files: {len(train_dataset)} | Val files: {len(val_dataset)}")
    else:
        print(f"Train chunks: {len(train_dataset)} | Val chunks: {len(val_dataset)}")
    print(f"Classes ({num_classes}): {train_dataset.label_to_idx}")
    if not mil_enabled:
        print(
            "Chunk split stats | "
            f"train={format_counts(chunk_label_counts(train_dataset, data_root, task))} | "
            f"val={format_counts(chunk_label_counts(val_dataset, data_root, task))}"
        )

    # Store label info in config for checkpoint reproducibility
    cfg["label_to_idx"] = train_dataset.label_to_idx
    cfg["idx_to_label"] = train_dataset.idx_to_label

    train_sampler = None
    if mil_enabled and task == "multiclass":
        use_balanced_sampler = bool(mil_cfg.get("use_balanced_sampler", False))
        balanced_sampler_power = float(mil_cfg.get("balanced_sampler_power", 0.35))
        if use_balanced_sampler:
            class_counts_for_sampler = Counter(infer_file_label(p, data_root, task) for p in train_files)
            inv_freq = {
                label_name: (float(count) + 1e-6) ** (-balanced_sampler_power)
                for label_name, count in class_counts_for_sampler.items()
            }
            sample_weights = [
                inv_freq[infer_file_label(path, data_root, task)]
                for path in train_dataset.files
            ]
            train_sampler = WeightedRandomSampler(
                weights=torch.tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            print(
                "Multiclass balanced sampler enabled | "
                f"power={balanced_sampler_power}"
            )

    if mil_enabled:
        mil_batch_size = int(mil_cfg.get("batch_size", 8))
        train_loader = DataLoader(
            train_dataset,
            batch_size=mil_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=train_cfg["num_workers"], collate_fn=mil_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=mil_batch_size, shuffle=False,
            num_workers=train_cfg["num_workers"], collate_fn=mil_collate_fn,
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
            num_workers=train_cfg["num_workers"], collate_fn=train_collate_fn,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
            num_workers=train_cfg["num_workers"], collate_fn=train_collate_fn,
        )
    print(
        f"DataLoader stats | train_batches={len(train_loader)} | "
        f"val_batches={len(val_loader)} | batch_size="
        f"{(mil_cfg.get('batch_size', 8) if mil_enabled else train_cfg['batch_size'])}"
    )
    if mil_enabled:
        print(
            "MIL mode enabled | "
            f"topk_ratio_pos={mil_cfg.get('topk_ratio_pos', 0.05)} | "
            f"topk_ratio_neg={mil_cfg.get('topk_ratio_neg', 0.2)} | "
            f"eval_pool_ratio={mil_cfg.get('eval_pool_ratio', 0.05)} | "
            f"auto_threshold={mil_cfg.get('auto_threshold', True)} | "
            f"threshold={mil_cfg.get('threshold', 0.5)} | "
            f"good_window_weight={mil_cfg.get('good_window_weight', 0.0)} | "
            f"multiclass_eval_mode={mil_cfg.get('multiclass_eval_mode', 'topk_per_class')}"
        )

    multiclass_class_weights = None
    if mil_enabled and task == "multiclass":
        use_class_weights = bool(mil_cfg.get("use_class_weights", True))
        class_weight_power = float(mil_cfg.get("class_weight_power", 0.5))
        if use_class_weights:
            class_counts = Counter(infer_file_label(p, data_root, task) for p in train_files)
            class_weights = torch.ones(num_classes, dtype=torch.float32)
            for label_name, idx in train_dataset.label_to_idx.items():
                count = float(class_counts.get(label_name, 0.0))
                class_weights[idx] = (count + 1e-6) ** (-class_weight_power)
            class_weights = class_weights / class_weights.mean().clamp(min=1e-8)
            multiclass_class_weights = class_weights.to(device)
            printable = {
                label_name: float(class_weights[idx].item())
                for label_name, idx in train_dataset.label_to_idx.items()
            }
            print(
                "Multiclass class-weighting enabled | "
                f"power={class_weight_power} | weights={printable}"
            )

    # ── Model (WeldBackboneModel = AudioTransform + AudioCNNBackbone) ──
    backbone = AudioCNNBackbone(num_classes=num_classes, dropout=model_cfg["dropout"])
    model = WeldBackboneModel(backbone, cfg=audio_cfg)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
    )

    total_steps = train_cfg["num_epochs"] * len(train_loader)
    warmup_ratio = lr_sched_cfg.get("warmup_ratio", None)
    warmup_epochs = lr_sched_cfg.get("warmup_epochs", 0)
    if warmup_ratio is not None:
        warmup_steps = int(total_steps * float(warmup_ratio))
    else:
        warmup_steps = int(warmup_epochs * len(train_loader))

    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=lr_sched_cfg.get("plateau_factor", 0.5),
        patience=lr_sched_cfg.get("plateau_patience", 2),
        threshold=lr_sched_cfg.get("plateau_threshold", 1e-3),
        min_lr=lr_sched_cfg.get("plateau_min_lr", 1e-6),
    )

    print(
        "LR schedule: "
        f"base_lr={optim_cfg['lr']} | warmup_steps={warmup_steps}/{total_steps} "
        f"({100.0 * warmup_steps / max(total_steps, 1):.1f}%) | "
        f"plateau_factor={lr_sched_cfg.get('plateau_factor', 0.5)} "
        f"| plateau_patience={lr_sched_cfg.get('plateau_patience', 2)}"
    )

    # ── Loss ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Test-only mode ────────────────────────────────────────────
    if args.test_only:
        assert args.checkpoint, "--checkpoint required for --test_only"
        if mil_enabled:
            label_to_idx = train_dataset.label_to_idx
            defect_idx   = label_to_idx.get("defect", 0)
            good_idx     = label_to_idx.get("good_weld", 1)
            if task == "binary" and ("defect" not in label_to_idx or "good_weld" not in label_to_idx):
                raise ValueError("binary task must contain labels 'defect' and 'good_weld'.")

            result = run_test_mil(
                model=model,
                dataloader=val_loader,
                device=device,
                checkpoint_path=args.checkpoint,
                task=task,
                defect_idx=defect_idx,
                good_idx=good_idx,
                topk_ratio_pos=float(mil_cfg.get("topk_ratio_pos", 0.05)),
                topk_ratio_neg=float(mil_cfg.get("topk_ratio_neg", 0.2)),
                eval_pool_ratio=float(mil_cfg.get("eval_pool_ratio", 0.05)),
                threshold=float(mil_cfg.get("threshold", 0.5)),
                auto_threshold=bool(mil_cfg.get("auto_threshold", False)),
                multiclass_eval_mode=str(mil_cfg.get("multiclass_eval_mode", "topk_per_class")),
            )
            print(f"Val loss: {result['loss']:.4f}")
            print(f"Val macro F1: {result['macro_f1']:.4f}")
            print(f"Val accuracy: {result['accuracy']:.4f}")
            if task == "binary":
                print(f"Val AUC: {result['auc']:.4f}")
                print(f"Threshold used: {result['threshold']:.2f}")
        else:
            result = run_test(model, val_loader, criterion, device, args.checkpoint)
            print(f"Val loss: {result['loss']:.4f}")
            print(f"Val macro F1: {result['macro_f1']:.4f}")
            print(f"Val accuracy: {result['accuracy']:.4f}")
        return

    # ── Save config alongside checkpoints ─────────────────────────
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    save_config(cfg, os.path.join(train_cfg["checkpoint_dir"], "config.json"))

    # ── Training ──────────────────────────────────────────────────
    patience = train_cfg["patience"] if train_cfg["patience"] > 0 else None

    if mil_enabled:
        label_to_idx = train_dataset.label_to_idx
        # Binary requires the two standard labels; multiclass works with any labels.
        defect_idx = label_to_idx.get("defect", 0)
        good_idx   = label_to_idx.get("good_weld", 1)
        if task == "binary" and ("defect" not in label_to_idx or "good_weld" not in label_to_idx):
            raise ValueError("binary task must contain labels 'defect' and 'good_weld'.")

        history = run_training_mil(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=train_cfg["num_epochs"],
            task=task,
            defect_idx=defect_idx,
            good_idx=good_idx,
            topk_ratio_pos=float(mil_cfg.get("topk_ratio_pos", 0.05)),
            topk_ratio_neg=float(mil_cfg.get("topk_ratio_neg", 0.2)),
            eval_pool_ratio=float(mil_cfg.get("eval_pool_ratio", 0.05)),
            auto_threshold=bool(mil_cfg.get("auto_threshold", True)),
            good_window_weight=float(mil_cfg.get("good_window_weight", 0.0)),
            class_weights_multiclass=multiclass_class_weights,
            multiclass_pred_mode=str(mil_cfg.get("multiclass_eval_mode", "topk_per_class")),
            threshold=float(mil_cfg.get("threshold", 0.5)),
            checkpoint_dir=train_cfg["checkpoint_dir"],
            plateau_scheduler=plateau_scheduler,
            warmup_steps=warmup_steps,
            base_lrs=[optim_cfg["lr"]] * len(optimizer.param_groups),
            patience=patience,
            seed=seed,
            target_metric_threshold=train_cfg.get("target_metric_threshold", None),
        )
    else:
        history = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            num_epochs=train_cfg["num_epochs"],
            checkpoint_dir=train_cfg["checkpoint_dir"],
            plateau_scheduler=plateau_scheduler,
            warmup_steps=warmup_steps,
            base_lrs=[optim_cfg["lr"]] * len(optimizer.param_groups),
            patience=patience,
            seed=seed,
            target_metric_threshold=train_cfg.get("target_metric_threshold", None),
        )

    print(f"\nBest epoch: {history['best_epoch']}")
    print(f"Train losses: {[f'{l:.4f}' for l in history['train_losses']]}")
    print(f"Val losses:   {[f'{l:.4f}' for l in history['val_losses']]}")


if __name__ == "__main__":
    main()
