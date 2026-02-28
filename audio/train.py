"""Functions for training and validation epochs."""

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    scheduler=None,
    scaler=None,
    collect_preds=False,
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

    for inputs, targets in pbar:

        # Linear warmup for the first N optimizer steps.
        if warmup_steps > 0 and global_step < warmup_steps and base_lrs is not None:
            warmup_scale = float(global_step + 1) / float(warmup_steps)
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * warmup_scale

        targets = targets.to(device)

        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(device)

        optimizer.zero_grad(set_to_none=True)

        # ---- AMP forward ----
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            if isinstance(inputs, dict):
                outputs = model(**inputs)
            else:
                outputs = model(inputs)

            loss = criterion(outputs, targets)

        # ---- AMP backward ----
        if scaler is not None and device.type == "cuda":
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        global_step += 1

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        if collect_preds:
            _, predicted = outputs.max(1)
            all_preds.append(predicted.detach().cpu())
            all_targets.append(targets.detach().cpu())

        pbar.set_postfix(loss=running_loss / n_samples)

    epoch_loss = running_loss / n_samples

    result = {"loss": epoch_loss}
    result["global_step"] = global_step
    result["lr"] = optimizer.param_groups[0]["lr"]

    if collect_preds:
        result["predictions"] = torch.cat(all_preds, dim=0)
        result["targets"] = torch.cat(all_targets, dim=0)

    return result


def validate_epoch(model, dataloader, criterion, device):
    model.eval()

    running_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for inputs, targets in pbar:

            targets = targets.to(device)

            if isinstance(inputs, dict):
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
            else:
                inputs = inputs.to(device)
                outputs = model(inputs)

            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            n_samples += batch_size

            _, predicted = outputs.max(1)
            all_preds.append(predicted.cpu())
            all_targets.append(targets.cpu())

            pbar.set_postfix(loss=running_loss / n_samples)

    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    correct = (preds == targets).sum()
    accuracy = correct / len(targets)

    return {
        "loss": running_loss / n_samples,
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
        "accuracy": accuracy,
    }
