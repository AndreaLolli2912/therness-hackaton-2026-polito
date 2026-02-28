"""Full training loop with checkpointing, early stopping, and logging."""

import os
import random
from datetime import datetime

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
    plateau_scheduler=None,
    warmup_steps=0,
    base_lrs=None,
    patience=None,
    seed=42,
):
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)

    # AMP scaler (created once)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    best_val_f1 = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    train_losses = []
    val_losses = []
    global_step = 0
    base_lrs = base_lrs or [pg["lr"] for pg in optimizer.param_groups]
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_result = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            scheduler=scheduler,
            scaler=scaler,
            warmup_steps=warmup_steps,
            global_step=global_step,
            base_lrs=base_lrs,
        )
        global_step = train_result["global_step"]

        val_result = validate_epoch(model, val_loader, criterion, device)

        train_loss = train_result["loss"]
        val_loss = val_result["loss"]
        val_f1 = val_result["macro_f1"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f} | "
            f"LR: {train_result['lr']:.6g}"
        )

        if plateau_scheduler is not None:
            plateau_scheduler.step(val_loss)

        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                plateau_scheduler.state_dict() if plateau_scheduler is not None else None
            ),
            "val_loss": val_loss,
            "val_f1": val_f1,
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pt"))

        # Save best checkpoint (based on F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            best_snapshot_dir = os.path.join(
                checkpoint_dir,
                "best_models",
                f"{run_stamp}_ep{epoch + 1:03d}_f1_{val_f1:.4f}_vl_{val_loss:.4f}",
            )
            os.makedirs(best_snapshot_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(best_snapshot_dir, "model.pt"))
            print(f"New best model saved (val_f1={val_f1:.4f})")
            print(f"Best snapshot dir: {best_snapshot_dir}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        # Early stopping
        if patience is not None and epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(
        f"\nTraining complete. "
        f"Best epoch: {best_epoch} "
        f"(Best Val Macro F1={best_val_f1:.4f})"
    )
    print(f"\nTraining complete. Best epoch: {best_epoch} (val_f1={best_val_f1:.4f})")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
