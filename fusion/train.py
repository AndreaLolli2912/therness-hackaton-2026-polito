"""Training and validation epoch functions for the fusion model."""

import torch
from tqdm import tqdm
from sklearn.metrics import f1_score


def fusion_collate_fn(batch):
    """Collate fusion dataset items into batched tensors."""
    audio_embs = torch.stack([item["audio_emb"] for item in batch])
    video_embs = torch.stack([item["video_emb"] for item in batch])
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return audio_embs, video_embs, labels


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

    for audio_embs, video_embs, targets in pbar:
        # Linear warmup
        if warmup_steps > 0 and global_step < warmup_steps and base_lrs is not None:
            warmup_scale = float(global_step + 1) / float(warmup_steps)
            for pg, base_lr in zip(optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * warmup_scale

        audio_embs = audio_embs.to(device)
        video_embs = video_embs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(audio_embs, video_embs)
            loss = criterion(logits, targets)

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

        _, predicted = logits.max(1)
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


def validate_epoch(model, dataloader, device, good_weld_idx=0):
    """Validate fusion model with both multiclass and binary metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    pbar = tqdm(dataloader, desc="Validation", leave=False)

    with torch.no_grad():
        for audio_embs, video_embs, targets in pbar:
            audio_embs = audio_embs.to(device)
            video_embs = video_embs.to(device)

            result = model.predict(audio_embs, video_embs, good_weld_idx=good_weld_idx)

            all_preds.append(result["pred_class"].cpu())
            all_targets.append(targets)
            all_probs.append(result["probs"].cpu())

    preds = torch.cat(all_preds).numpy()
    targets_np = torch.cat(all_targets).numpy()

    # Multiclass metrics
    macro_f1 = f1_score(targets_np, preds, average="macro", zero_division=0)
    correct = (preds == targets_np).sum()
    accuracy = correct / len(targets_np)

    # Binary metrics: good_weld vs everything else
    binary_true = [0 if y == good_weld_idx else 1 for y in targets_np]
    binary_pred = [0 if p == good_weld_idx else 1 for p in preds]
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, zero_division=0)

    hackathon_score = 0.6 * binary_f1 + 0.4 * macro_f1

    return {
        "macro_f1": macro_f1,
        "binary_f1": binary_f1,
        "hackathon_score": hackathon_score,
        "accuracy": accuracy,
    }
