"""Functions for training and validation epochs."""

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


def train_epoch(model, dataloader, criterion, optimizer, device, scheduler=None, collect_preds=True):
    """Run one training epoch.

    Args:
        model: nn.Module to train.
        dataloader: yields (inputs, targets) tuples. inputs can be a Tensor
            or a dict (unpacked as **kwargs to the model).
        criterion: loss function.
        optimizer: optimizer.
        device: torch device.
        scheduler: optional LR scheduler (stepped per batch).
        collect_preds: if True, return predictions and targets.

    Returns:
        dict with "loss" (float) and optionally "predictions" / "targets" (Tensors).
    """
    model.train()
    running_loss = 0.0
    n_samples = 0
    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in pbar:
        targets = targets.to(device)
        if isinstance(inputs, dict):
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
        else:
            inputs = inputs.to(device)
            outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        n_samples += batch_size

        if collect_preds:
            _, predicted = outputs.max(1)
            all_preds.append(predicted.detach().cpu())
            all_targets.append(targets.detach().cpu())

        pbar.set_postfix(loss=running_loss / n_samples)

    result = {"loss": running_loss / n_samples}
    if collect_preds:
        preds = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        result["macro_f1"] = f1_score(targets, preds, average='macro', zero_division=0)
    return result


def validate_epoch(model, dataloader, criterion, device):
    """Run one validation epoch.

    Args:
        model: nn.Module to evaluate.
        dataloader: yields (inputs, targets) tuples.
        criterion: loss function.
        device: torch device.

    Returns:
        dict with "loss" (float), "predictions" and "targets" (Tensors).
    """
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
    
    return {
        "loss": running_loss / n_samples,
        "macro_f1": f1_score(targets, preds, average='macro', zero_division=0)
    }
