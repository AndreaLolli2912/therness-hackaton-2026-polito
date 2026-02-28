"""Training and validation epoch functions for video models."""

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scaler=None,
    warmup_steps=0,
    global_step=0,
    base_lrs=None,
):
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Training", leave=False)

    for frames, targets in pbar:
        # Linear warmup
        if warmup_steps > 0 and global_step < warmup_steps and base_lrs is not None:
            warmup_scale = float(global_step + 1) / float(warmup_steps)
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * warmup_scale

        frames = frames.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(frames)
            loss = criterion(outputs, targets)

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

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        _, predicted = outputs.max(1)
        all_preds.append(predicted.detach().cpu())
        all_targets.append(targets.detach().cpu())

        pbar.set_postfix(loss=running_loss / n_samples)

    preds = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()

    return {
        "loss": running_loss / n_samples,
        "macro_f1": f1_score(targets_np, preds, average="macro", zero_division=0),
        "global_step": global_step,
        "lr": optimizer.param_groups[0]["lr"],
    }


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for frames, targets in pbar:
            frames = frames.to(device)
            targets = targets.to(device)

            outputs = model(frames)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())
            all_probs.append(probs.cpu())

            pbar.set_postfix(loss=running_loss / n_samples)

    preds = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()

    correct = (preds == targets_np).sum()

    # Binary F1: class 0 = good_weld, classes 1-6 = defect
    binary_true = [0 if y == 0 else 1 for y in targets_np]
    binary_pred = [0 if p == 0 else 1 for p in preds]
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, zero_division=0)
    macro_f1 = f1_score(targets_np, preds, average="macro", zero_division=0)
    hackathon_score = 0.6 * binary_f1 + 0.4 * macro_f1

    return {
        "loss": running_loss / n_samples,
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "hackathon_score": hackathon_score,
        "accuracy": correct / len(targets_np),
    }
