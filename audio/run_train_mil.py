"""MIL-style training loop for sequence-level weak supervision.

Each sample is a weld file represented by a sequence of fixed-size windows.
The model predicts per-window logits, which are aggregated to a file-level
prediction.  Works for both binary and multiclass tasks.

BINARY
------
  - Per-window: extract probability of the defect class.
  - Aggregation: asymmetric top-k pooling (topk_ratio_pos for defect files,
    topk_ratio_neg for good files).
  - Loss: binary cross-entropy on the pooled defect probability.
  - Threshold learned automatically on the validation set.

MULTICLASS
----------
  - Per-window: extract full softmax probability vector (C classes).
  - Aggregation: for each file labelled class c, select the top-k windows
    by P(class=c) and average their full probability vector.
  - Loss: NLL loss on the averaged probability vector vs the file label.
  - Prediction: argmax of mean-pooled probabilities over all windows.
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


# ──────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _aggregate_topk(prob_seq, topk_ratio):
    """Mean of the top-k values in a 1-D probability sequence."""
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


# ──────────────────────────────────────────────────────────────────────────────
# Binary MIL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _window_probs_per_file(model, windows, lengths, defect_idx):
    """Return list of per-file 1-D defect-probability tensors."""
    bsz = windows.shape[0]

    flat, owner = [], []
    for b in range(bsz):
        t = int(lengths[b].item())
        flat.append(windows[b, :t])
        owner.extend([b] * t)
    flat  = torch.cat(flat, dim=0)
    owner = torch.tensor(owner, device=flat.device)

    logits = model(flat)
    probs  = torch.softmax(logits, dim=1)[:, defect_idx]  # (sum_t,)

    return [probs[owner == b] for b in range(bsz)]


def _bag_probs_from_sequences(
    seq_probs,
    target_defect=None,
    topk_ratio_pos=0.05,
    topk_ratio_neg=0.2,
    eval_pool_ratio=None,
):
    """Aggregate per-window defect probabilities → file-level scalar per sample."""
    bag_probs = []
    for i, seq in enumerate(seq_probs):
        if target_defect is None:
            ratio = eval_pool_ratio if eval_pool_ratio is not None else topk_ratio_pos
        else:
            ratio = topk_ratio_pos if float(target_defect[i].item()) > 0.5 else topk_ratio_neg
        bag_probs.append(_aggregate_topk(seq, ratio))
    return torch.stack(bag_probs, dim=0)


# ──────────────────────────────────────────────────────────────────────────────
# Multiclass MIL helpers
# ──────────────────────────────────────────────────────────────────────────────

def _window_full_probs_per_file(model, windows, lengths):
    """Return list of per-file (T_i, C) full softmax probability tensors."""
    bsz = windows.shape[0]

    flat, owner = [], []
    for b in range(bsz):
        t = int(lengths[b].item())
        flat.append(windows[b, :t])
        owner.extend([b] * t)
    flat  = torch.cat(flat, dim=0)
    owner = torch.tensor(owner, device=flat.device)

    logits = model(flat)                         # (sum_t, C)
    probs  = torch.softmax(logits, dim=1)        # (sum_t, C)

    return [probs[owner == b] for b in range(bsz)]


def _aggregate_topk_multiclass(probs_per_file, labels, topk_ratio):
    """For each file, select top-k windows by correct-class probability,
    then average their full probability vector.

    Args:
        probs_per_file: list of (T_i, C) tensors.
        labels:         (B,) integer class labels.
        topk_ratio:     fraction of windows to select.

    Returns:
        (B, C) tensor of bag probability vectors (NOT normalised).
    """
    bag_probs = []
    for probs, label in zip(probs_per_file, labels):
        label = int(label.item())
        p_correct = probs[:, label]                    # (T,)
        k = max(1, int(math.ceil(topk_ratio * float(p_correct.numel()))))
        k = min(k, int(p_correct.numel()))
        topk_idx  = torch.topk(p_correct, k=k).indices
        bag_probs.append(probs[topk_idx].mean(dim=0))  # (C,)
    return torch.stack(bag_probs, dim=0)               # (B, C)


def _predict_multiclass_file(probs_per_file, eval_pool_ratio):
    """Mean-pool over all windows → return (B,) predicted class indices and (B, C) probs."""
    preds, mean_probs_list = [], []
    for probs in probs_per_file:
        # Mean over ALL windows (robust for inference)
        mean_p = probs.mean(dim=0)        # (C,)
        mean_probs_list.append(mean_p)
        preds.append(mean_p.argmax())
    return torch.stack(preds), torch.stack(mean_probs_list)


# ──────────────────────────────────────────────────────────────────────────────
# Binary MIL epoch functions
# ──────────────────────────────────────────────────────────────────────────────

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

    pbar = tqdm(dataloader, desc="Training MIL binary", leave=False)
    for windows, lengths, labels in pbar:
        _linear_warmup(optimizer, base_lrs, warmup_steps, global_step)

        windows = windows.to(device)
        lengths = lengths.to(device)
        labels  = labels.to(device)
        target_defect = (labels == defect_idx).float()

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seq_probs  = _window_probs_per_file(model, windows, lengths, defect_idx)
            file_probs = _bag_probs_from_sequences(
                seq_probs,
                target_defect=target_defect,
                topk_ratio_pos=topk_ratio_pos,
                topk_ratio_neg=topk_ratio_neg,
            )

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
                    good_probs = torch.cat(good_windows).float().clamp(1e-6, 1.0 - 1e-6)
                    good_loss  = F.binary_cross_entropy(good_probs, torch.zeros_like(good_probs))
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

        global_step  += 1
        batch_size    = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples    += batch_size
        pbar.set_postfix(loss=running_loss / max(n_samples, 1))

    return {
        "loss":        running_loss / max(n_samples, 1),
        "global_step": global_step,
        "lr":          optimizer.param_groups[0]["lr"],
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
    n_samples    = 0
    all_targets  = []
    all_probs    = []

    pbar = tqdm(dataloader, desc="Validation MIL binary", leave=False)
    with torch.no_grad():
        for windows, lengths, labels in pbar:
            windows = windows.to(device)
            lengths = lengths.to(device)
            labels  = labels.to(device)
            target_defect = (labels == defect_idx).float()

            seq_probs  = _window_probs_per_file(model, windows, lengths, defect_idx)
            file_probs = _bag_probs_from_sequences(
                seq_probs,
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
                    seq_probs, target_defect=None, eval_pool_ratio=eval_pool_ratio,
                ).cpu()
            )
            batch_size    = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples    += batch_size
            pbar.set_postfix(loss=running_loss / max(n_samples, 1))

    targets = torch.cat(all_targets).numpy()
    probs   = torch.cat(all_probs).numpy()

    if auto_threshold:
        best_thr = threshold
        best_f1  = -1.0
        for thr in np.linspace(0.05, 0.95, 19):
            pred_def = (probs >= thr).astype(np.int64)
            preds    = np.where(pred_def == 1, defect_idx, good_idx)
            f1 = f1_score(targets, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1  = f1
                best_thr = float(thr)
        threshold = best_thr

    pred_defect   = (probs >= threshold).astype(np.int64)
    preds         = np.where(pred_defect == 1, defect_idx, good_idx)
    accuracy      = float((preds == targets).sum()) / max(len(targets), 1)
    binary_targets = (targets == defect_idx).astype(np.int64)
    auc = (
        roc_auc_score(binary_targets, probs)
        if len(np.unique(binary_targets)) > 1
        else float("nan")
    )

    return {
        "loss":      running_loss / max(n_samples, 1),
        "macro_f1":  f1_score(targets, preds, average="macro", zero_division=0),
        "accuracy":  accuracy,
        "threshold": float(threshold),
        "auc":       float(auc),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Multiclass MIL epoch functions
# ──────────────────────────────────────────────────────────────────────────────

def train_epoch_mil_multiclass(
    model,
    dataloader,
    optimizer,
    device,
    topk_ratio=0.05,
    warmup_steps=0,
    global_step=0,
    base_lrs=None,
    scaler=None,
):
    """MIL training for multiclass task.

    For each file labelled class c:
      1. Compute per-window softmax probabilities (T, C).
      2. Select the top-k windows by P(class=c).
      3. Average the full probability vector over those windows.
      4. NLL loss against the file label c.
    """
    model.train()
    running_loss = 0.0
    n_samples    = 0

    pbar = tqdm(dataloader, desc="Training MIL multiclass", leave=False)
    for windows, lengths, labels in pbar:
        _linear_warmup(optimizer, base_lrs, warmup_steps, global_step)

        windows = windows.to(device)
        lengths = lengths.to(device)
        labels  = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            seq_probs = _window_full_probs_per_file(model, windows, lengths)
            bag_probs = _aggregate_topk_multiclass(seq_probs, labels, topk_ratio)  # (B, C)

        with torch.amp.autocast("cuda", enabled=False):
            log_bag = bag_probs.float().clamp(min=1e-8).log()
            loss    = F.nll_loss(log_bag, labels)

        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        global_step  += 1
        batch_size    = labels.size(0)
        running_loss += loss.item() * batch_size
        n_samples    += batch_size
        pbar.set_postfix(loss=running_loss / max(n_samples, 1))

    return {
        "loss":        running_loss / max(n_samples, 1),
        "global_step": global_step,
        "lr":          optimizer.param_groups[0]["lr"],
    }


def validate_epoch_mil_multiclass(
    model,
    dataloader,
    device,
    eval_pool_ratio=0.05,
):
    """MIL validation for multiclass: mean-pool all windows → argmax."""
    model.eval()
    running_loss = 0.0
    n_samples    = 0
    all_targets  = []
    all_preds    = []

    pbar = tqdm(dataloader, desc="Validation MIL multiclass", leave=False)
    with torch.no_grad():
        for windows, lengths, labels in pbar:
            windows = windows.to(device)
            lengths = lengths.to(device)
            labels  = labels.to(device)

            seq_probs = _window_full_probs_per_file(model, windows, lengths)

            # Loss: top-k aggregation with true label (teacher-forced at val time)
            bag_probs = _aggregate_topk_multiclass(seq_probs, labels, eval_pool_ratio)
            log_bag   = bag_probs.float().clamp(min=1e-8).log()
            loss      = F.nll_loss(log_bag, labels)

            # Prediction: mean pool (no label info)
            preds, _ = _predict_multiclass_file(seq_probs, eval_pool_ratio)

            all_targets.append(labels.cpu())
            all_preds.append(preds.cpu())

            batch_size    = labels.size(0)
            running_loss += loss.item() * batch_size
            n_samples    += batch_size
            pbar.set_postfix(loss=running_loss / max(n_samples, 1))

    targets = torch.cat(all_targets).numpy()
    preds   = torch.cat(all_preds).numpy()
    accuracy = float((preds == targets).sum()) / max(len(targets), 1)

    return {
        "loss":     running_loss / max(n_samples, 1),
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "accuracy": accuracy,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Unified training loop
# ──────────────────────────────────────────────────────────────────────────────

def run_training_mil(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    task="binary",
    # binary-specific
    defect_idx=0,
    good_idx=1,
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

    scaler     = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    base_lrs   = base_lrs or [pg["lr"] for pg in optimizer.param_groups]
    global_step = 0

    best_val_f1 = -1.0
    best_epoch  = -1
    no_improve  = 0
    train_losses, val_losses = [], []
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    is_binary = (task == "binary")

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 40)

        # ── Train ──────────────────────────────────────────────────
        if is_binary:
            train_result = train_epoch_mil(
                model=model, dataloader=train_loader, optimizer=optimizer,
                device=device, defect_idx=defect_idx, good_idx=good_idx,
                topk_ratio_pos=topk_ratio_pos, topk_ratio_neg=topk_ratio_neg,
                good_window_weight=good_window_weight,
                warmup_steps=warmup_steps, global_step=global_step,
                base_lrs=base_lrs, scaler=scaler,
            )
        else:
            train_result = train_epoch_mil_multiclass(
                model=model, dataloader=train_loader, optimizer=optimizer,
                device=device, topk_ratio=topk_ratio_pos,
                warmup_steps=warmup_steps, global_step=global_step,
                base_lrs=base_lrs, scaler=scaler,
            )
        global_step = train_result["global_step"]

        # ── Validate ───────────────────────────────────────────────
        if is_binary:
            val_result = validate_epoch_mil(
                model=model, dataloader=val_loader, device=device,
                defect_idx=defect_idx, good_idx=good_idx,
                topk_ratio_pos=topk_ratio_pos, topk_ratio_neg=topk_ratio_neg,
                eval_pool_ratio=eval_pool_ratio,
                threshold=threshold, auto_threshold=auto_threshold,
            )
            threshold = val_result["threshold"]
        else:
            val_result = validate_epoch_mil_multiclass(
                model=model, dataloader=val_loader, device=device,
                eval_pool_ratio=eval_pool_ratio,
            )

        train_loss = train_result["loss"]
        val_loss   = val_result["loss"]
        val_f1     = val_result["macro_f1"]
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log_line = (
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )
        if is_binary:
            log_line += f" | Val AUC: {val_result.get('auc', float('nan')):.4f} | Thr: {threshold:.2f}"
        log_line += f" | LR: {train_result['lr']:.6g}"
        print(log_line)

        if plateau_scheduler is not None:
            plateau_scheduler.step(-val_f1)  # track F1 improvement, not val_loss

        # ── Checkpoint ─────────────────────────────────────────────
        checkpoint = {
            "epoch":              epoch + 1,
            "model_state_dict":   model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": (
                plateau_scheduler.state_dict() if plateau_scheduler is not None else None
            ),
            "val_loss": val_loss,
            "val_f1":   val_f1,
        }
        if is_binary:
            checkpoint["val_auc"]   = val_result.get("auc", float("nan"))
            checkpoint["threshold"] = threshold

        torch.save(checkpoint, os.path.join(checkpoint_dir, "last_model.pt"))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch  = epoch + 1
            no_improve  = 0
            torch.save(checkpoint, os.path.join(checkpoint_dir, "best_model.pt"))
            snap_dir = os.path.join(
                checkpoint_dir, "best_models",
                f"{run_stamp}_ep{epoch+1:03d}_f1_{val_f1:.4f}_vl_{val_loss:.4f}",
            )
            os.makedirs(snap_dir, exist_ok=True)
            torch.save(checkpoint, os.path.join(snap_dir, "model.pt"))
            print(f"New best model saved (val_f1={val_f1:.4f})")
        else:
            no_improve += 1
            print(f"No improvement for {no_improve} epoch(s)")

        if patience is not None and no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}")
            break

    print(f"\nTraining complete. Best epoch: {best_epoch} (val_f1={best_val_f1:.4f})")
    return {
        "train_losses": train_losses,
        "val_losses":   val_losses,
        "best_epoch":   best_epoch,
        "best_val_f1":  best_val_f1,
    }
