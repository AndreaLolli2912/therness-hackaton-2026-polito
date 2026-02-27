"""Full training loop with checkpointing, early stopping, and logging."""

import os
import random

import numpy as np
import torch

from audio.train import train_epoch, validate_epoch


def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_training(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    num_epochs,
    checkpoint_dir="checkpoints",
    scheduler=None,
    patience=None,
    seed=42,
):
    """Run the full training loop.

    Args:
        model: nn.Module to train.
        train_loader: training dataloader yielding (inputs, targets).
        val_loader: validation dataloader yielding (inputs, targets).
        criterion: loss function.
        optimizer: optimizer.
        device: torch device.
        num_epochs: number of epochs to train.
        checkpoint_dir: directory to save checkpoints.
        scheduler: optional LR scheduler (stepped per batch inside train_epoch).
        patience: early stopping patience (None to disable).
        seed: random seed for reproducibility.

    Returns:
        dict with "train_losses", "val_losses", "best_epoch".
    """
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_result = train_epoch(model, train_loader, criterion, optimizer, device, scheduler=scheduler)
        val_result = validate_epoch(model, val_loader, criterion, device)

        train_loss = train_result["loss"]
        val_loss = val_result["loss"]
        val_f1 = val_result["macro_f1"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val Macro F1: {val_f1:.4f}")

        # Save last checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_f1": val_f1,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pt"))

        # Save best checkpoint based on Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            print(f"New best model saved (val_f1={val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining complete. Best epoch: {best_epoch} (val_loss={best_val_loss:.4f})")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
    }
