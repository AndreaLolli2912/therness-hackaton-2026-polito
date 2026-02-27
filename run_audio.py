"""Train and evaluate the AudioCNN on welding audio data.

Usage:
    # Train with default config
    python run_audio.py --config configs/audio_config.json

    # Test only (evaluate checkpoint on val set)
    python run_audio.py --config configs/audio_config.json --test_only --checkpoint checkpoints/audio/best_model.pt

    # Generate submission CSV
    python run_audio.py --config configs/audio_config.json --submission --checkpoint checkpoints/audio/best_model.pt
"""

import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from audio_model import AudioCNN
from audio_processing import AudioDataset
from run_train import run_training
from test import run_test, generate_submission


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
# AudioDataset returns dicts; train_epoch/validate_epoch expect (inputs, targets).

def train_collate_fn(batch):
    """Collate for training/validation: (audio_tensor, label) pairs."""
    audios = torch.stack([item["audio"] for item in batch])   # (B, 1, n_mels, T)
    labels = torch.tensor([item["label"] for item in batch])  # (B,)
    return audios, labels


def submission_collate_fn(batch):
    """Collate for submission inference: (audio_tensor, sample_ids) pairs."""
    audios = torch.stack([item["audio"] for item in batch])   # (B, 1, n_mels, T)
    sample_ids = [item["sample_id"] for item in batch]        # list of str
    return audios, sample_ids


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Audio welding defect classification")
    parser.add_argument("--config", type=str, default="configs/audio_config.json",
                        help="Path to JSON config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for --test_only or --submission")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training, evaluate checkpoint on val set")
    parser.add_argument("--submission", action="store_true",
                        help="Generate submission CSV from unlabeled test set")
    args = parser.parse_args()

    # ── Load config ───────────────────────────────────────────────
    cfg = load_config(args.config)
    audio_cfg = cfg["audio"]
    model_cfg = cfg["model"]
    optim_cfg = cfg["optimizer"]
    train_cfg = cfg["training"]
    data_cfg = cfg["data"]

    print("=" * 50)
    print("Configuration:")
    print(json.dumps(cfg, indent=2))
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Submission mode (no labeled data needed) ──────────────────
    if args.submission:
        assert args.checkpoint, "--checkpoint required for --submission"

        test_dataset = AudioDataset(data_cfg["test_root"], cfg=audio_cfg, labeled=False)
        test_loader = DataLoader(
            test_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
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

        model = AudioCNN(num_classes=len(label_map), dropout=model_cfg["dropout"])
        generate_submission(
            model, test_loader, device, args.checkpoint,
            label_map=label_map, output_path="submission.csv",
        )
        return

    # ── Load labeled dataset ──────────────────────────────────────
    full_dataset = AudioDataset(data_cfg["data_root"], cfg=audio_cfg, labeled=True)
    num_classes = len(full_dataset.label_to_idx)
    print(f"\nLoaded {len(full_dataset)} samples, {num_classes} classes")
    print(f"Label mapping: {full_dataset.label_to_idx}")

    # Store label_map in config so it gets saved with checkpoints
    cfg["label_map"] = full_dataset.idx_to_label

    # ── Train/val split ───────────────────────────────────────────
    val_size = int(len(full_dataset) * train_cfg["val_split"])
    train_size = len(full_dataset) - val_size

    generator = torch.Generator().manual_seed(train_cfg["seed"])
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator,
    )
    print(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,
        num_workers=train_cfg["num_workers"], collate_fn=train_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,
        num_workers=train_cfg["num_workers"], collate_fn=train_collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────
    model = AudioCNN(num_classes=num_classes, dropout=model_cfg["dropout"])
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Optimizer ─────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=optim_cfg["lr"], weight_decay=optim_cfg["weight_decay"],
    )

    # ── Loss ──────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss()

    # ── Test-only mode ────────────────────────────────────────────
    if args.test_only:
        assert args.checkpoint, "--checkpoint required for --test_only"
        result = run_test(model, val_loader, criterion, device, args.checkpoint)
        print(f"Val loss: {result['loss']:.4f}")
        return

    # ── Save config alongside checkpoints ─────────────────────────
    os.makedirs(train_cfg["checkpoint_dir"], exist_ok=True)
    save_config(cfg, os.path.join(train_cfg["checkpoint_dir"], "config.json"))

    # ── Training ──────────────────────────────────────────────────
    patience = train_cfg["patience"] if train_cfg["patience"] > 0 else None

    history = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=train_cfg["num_epochs"],
        checkpoint_dir=train_cfg["checkpoint_dir"],
        patience=patience,
        seed=train_cfg["seed"],
    )

    print(f"\nBest epoch: {history['best_epoch']}")
    print(f"Train losses: {[f'{l:.4f}' for l in history['train_losses']]}")
    print(f"Val losses:   {[f'{l:.4f}' for l in history['val_losses']]}")


if __name__ == "__main__":
    main()
