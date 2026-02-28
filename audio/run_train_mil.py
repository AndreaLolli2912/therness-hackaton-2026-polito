"""MIL-style training loop for sequence-level weak supervision.

Each sample is a weld file represented by a sequence of fixed-size windows.
The model predicts per-window logits, which are aggregated to a file-level defect score.
"""

import math
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score
from tqdm import tqdm


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_topk(prob_seq, topk_ratio):
    k = max(1, int(math.ceil(topk_ratio * prob_seq.numel())))
    k = min(k, int(prob_seq.numel()))
    values, _ = torch.topk(prob_seq, k=k)
    return values.mean()


def _linear_warmup(optimizer, base_lrs, warmup_steps, global_step):
    if warmup_steps <= 0 or global_step >= warmup_steps:
        return
    scale = float(global_step + 1) / float(warmup_steps)
    for pg, base_lr in zip(optimizer.param_groups, base_lrs):
        pg["lr"] = base_lr * scale


def _window_probs_per_file(model, windows, lengths, defect_idx):
    # windows: (B, T_max, 1, S)
    bsz, _ = windows.shape[:2]

    flat = []
    owner = []
    for b in range(bsz):
        t = int(lengths[b].item())
        flat.append(windows[b, :t])
        owner.extend([b] * t)
    flat = torch.cat(flat, dim=0)
    owner = torch.tensor(owner, device=flat.device)

    logits = model(flat)
    probs = torch.softmax(logits, dim=1)[:, defect_idx]  # (sum_t,)

    seq_probs = []
    for b in range(bsz):
        seq_probs.append(probs[owner == b])
    return seq_probs


def _bag_probs_from_sequences(seq_probs, target_defect=None, topk_ratio_pos=0.05, topk_ratio_neg=0.2, eval_pool_ratio=None):
    """Aggregate per-window probabilities into file-level probs.

    If target_defect is provided, use asymmetric pooling:
      - positive bags: topk_ratio_pos
      - negative bags: topk_ratio_neg
    Otherwise use eval_pool_ratio for all bags.
    """
    bag_probs = []
    for i, seq in enumerate(seq_probs):
        if target_defect is None:
            ratio = eval_pool_ratio if eval_pool_ratio is not None else topk_ratio_pos
        else:
            ratio = topk_ratio_pos if float(target_defect[i].item()) > 0.5 else topk_ratio_neg
        bag_probs.append(_aggregate_topk(seq, ratio))
    return torch.stack(bag_probs, dim=0)


def train_epoch_mil(
    model,
    dataloader,
    optimizer,
    device,
    defect_idx,
    good_idx,
    topk_ratio_pos=0.05,
    topk_ratio_neg=0.2,
    good_window_weight=0.0,
    warmup_steps=0,
    global_step=0,
    base_lrs=None,
    scaler=None,
):
    model.train()
    running_loss = 0.0
    n_samples = 0

    pbar = tqdm(dataloader, desc="Training (MIL)", leave=False)
    for windows, lengths, labels in pbar:
        _linear_warmup(optimizer, base_lrs, warmup_steps, global_step)

        windows = windows.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        target_defect = (labels == defect_idx).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seq_probs = _window_probs_per_file(
                model=model,
                windows=windows,
                lengths=lengths,
                defect_idx=defect_idx,
            )
            file_probs = _bag_probs_from_sequences(
                seq_probs=seq_probs,
                target_defect=target_defect,
                topk_ratio_pos=topk_ratio_pos,
                topk_ratio_neg=topk_ratio_neg,
            )
        # BCE on probabilities is not autocast-safe: force fp32 here.
        with torch.amp.autocast("cuda", enabled=False):
            bag_loss = F.binary_cross_entropy(
                file_probs.float().clamp(1e-6, 1.0 - 1e-6),
                target_defect.float(),
            )
            if good_window_weight > 0.0:
                good_windows = [
                    seq_probs[i]
                    for i in range(len(seq_probs))
                    if float(target_defect[i].item()) < 0.5
                ]
                if good_windows:
                    good_probs = torch.cat(good_windows, dim=0).float().clamp(1e-6, 1.0 - 1e-6)
                    good_loss = F.binary_cross_entropy(good_probs, torch.zeros_like(good_probs))
                else:
                    good_loss = torch.tensor(0.0, device=file_probs.device)
                loss = bag_loss + good_window_weight * good_loss
            else:
                loss = bag_loss

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        global_step += 1
        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size
        pbar.set_postfix(loss=running_loss / max(n_samples, 1))

    return {
        "loss": running_loss / max(n_samples, 1),
        "global_step": global_step,
        "lr": optimizer.param_groups[0]["lr"],
    }


def validate_epoch_mil(
    model,
    dataloader,
    device,
    defect_idx,
    good_idx,
    topk_ratio_pos=0.05,
    topk_ratio_neg=0.2,
    eval_pool_ratio=0.05,
    threshold=0.5,
    auto_threshold=True,
):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_targets = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Validation (MIL)", leave=False)
    with torch.no_grad():
        for windows, lengths, labels in pbar:
            windows = windows.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            target_defect = (labels == defect_idx).float()

            seq_probs = _window_probs_per_file(
                model=model,
                windows=windows,
                lengths=lengths,
                defect_idx=defect_idx,
            )
            file_probs = _bag_probs_from_sequences(
                seq_probs=seq_probs,
                target_defect=target_defect,
                topk_ratio_pos=topk_ratio_pos,
                topk_ratio_neg=topk_ratio_neg,
            )
            loss = F.binary_cross_entropy(
                file_probs.float().clamp(1e-6, 1.0 - 1e-6),
                target_defect.float(),
            )
            all_targets.append(labels.cpu())
            all_probs.append(
                _bag_probs_from_sequences(
                    seq_probs=seq_probs,
                    target_defect=None,
                    eval_pool_ratio=eval_pool_ratio,
                ).cpu()
            )

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size
            pbar.set_postfix(loss=running_loss / max(n_samples, 1))

    targets = torch.cat(all_targets).numpy()
    probs = torch.cat(all_probs).numpy()

    if auto_threshold:
        best_thr = threshold
        best_f1 = -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            pred_def = (probs >= thr).astype(np.int64)
            preds = np.where(pred_def == 1, defect_idx, good_idx)
            f1 = f1_score(targets, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = float(thr)
        threshold = best_thr

    pred_defect = (probs >= threshold).astype(np.int64)
    preds = np.where(pred_defect == 1, defect_idx, good_idx)
    accuracy = float((preds == targets).sum()) / max(len(targets), 1)
    binary_targets = (targets == defect_idx).astype(np.int64)
    auc = roc_auc_score(binary_targets, probs) if len(np.unique(binary_targets)) > 1 else float("nan")

    return {
        "loss": running_loss / max(n_samples, 1),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "accuracy": accuracy,
        "threshold": float(threshold),
        "auc": float(auc),
    }


def run_training_mil(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    defect_idx,
    good_idx,
    topk_ratio_pos=0.05,
    topk_ratio_neg=0.2,
    eval_pool_ratio=0.05,
    auto_threshold=True,
    good_window_weight=0.0,
    threshold=0.5,
    checkpoint_dir="checkpoints",
    plateau_scheduler=None,
    warmup_steps=0,
    base_lrs=None,
    patience=None,
    seed=42,
):
    set_seed(seed)
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    base_lrs = base_lrs or [pg["lr"] for pg in optimizer.param_groups]
    global_step = 0

    best_val_f1 = -1.0
    best_epoch = -1
    no_improve = 0
    train_losses = []
    val_losses = []
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        train_result = train_epoch_mil(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            defect_idx=defect_idx,
            good_idx=good_idx,
            topk_ratio_pos=topk_ratio_pos,
            topk_ratio_neg=topk_ratio_neg,
            good_window_weight=good_window_weight,
            warmup_steps=warmup_steps,
            global_step=global_step,
            base_lrs=base_lrs,
            scaler=scaler,
        )
        global_step = train_result["global_step"]

        val_result = validate_epoch_mil(
            model=model,
            dataloader=val_loader,
            device=device,
            defect_idx=defect_idx,
            good_idx=good_idx,
            topk_ratio_pos=topk_ratio_pos,
            topk_ratio_neg=topk_ratio_neg,
            eval_pool_ratio=eval_pool_ratio,
            threshold=threshold,
            auto_threshold=auto_threshold,
        )
        threshold = val_result["threshold"]

        train_loss = train_result["loss"]
        val_loss = val_result["loss"]
        val_f1 = val_result["macro_f1"]
        val_auc = val_result["auc"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"Thr: {threshold:.2f} | "
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
            "val_auc": val_auc,
            "threshold": threshold,
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch + 1
            no_improve = 0
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
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s)")

        if patience is not None and no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(
        f"\nTraining complete. "
        f"Best epoch: {best_epoch} "
        f"(Best Val Macro F1={best_val_f1:.4f})"
    )
    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
    }
