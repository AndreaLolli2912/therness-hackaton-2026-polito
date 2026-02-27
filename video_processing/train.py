"""
Video classifier training with hackathon-optimized scoring.

Key improvements over baseline:
  - Inverse-frequency class weights for imbalanced defect types
  - GroupShuffleSplit on configuration folders (prevents data leakage)
  - Binary + multi-class metrics logged each epoch
  - Combined hackathon score as the checkpoint metric:
      FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1
"""
import os
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, roc_auc_score
from collections import Counter

from src.models.video_model import StreamingVideoClassifier
from src.data.dataset import (
    WeldingSequenceDataset, get_video_transforms, get_video_files_and_labels,
)


# ── Label mapping (index → code) ─────────────────────────────────
LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}
CODE_TO_IDX = {v: k for k, v in LABEL_CODE_MAP.items()}


def compute_class_weights(labels, num_classes=7):
    """
    Inverse-frequency weights: w_c = N / (num_classes * count_c).
    Ensures rare classes (overlap, crater_cracks) get higher loss weight.
    """
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 1)  # avoid division by zero
        weights[cls_idx] = total / (num_classes * count)
    return weights


def compute_binary_metrics(all_labels, all_preds, all_probs=None):
    """
    Derive binary metrics from 7-class predictions.
    Binary: good_weld (class 0) vs any defect (classes 1-6).
    """
    binary_true = [0 if y == 0 else 1 for y in all_labels]
    binary_pred = [0 if p == 0 else 1 for p in all_preds]
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, zero_division=0)

    roc = None
    if all_probs is not None:
        # p_defect = 1 - P(good_weld)
        p_defect = [1.0 - p[0] for p in all_probs]
        try:
            roc = roc_auc_score(binary_true, p_defect)
        except ValueError:
            roc = None

    return binary_f1, roc


def compute_hackathon_score(binary_f1, macro_f1):
    """FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1"""
    return 0.6 * binary_f1 + 0.4 * macro_f1


def train_video(config):
    # Extract configs
    v_conf = config['video']['training']
    m_conf = config['video']['model']
    data_root = config['data_root']
    device = config.get('device', 'auto')
    if device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device(device)

    num_classes = config.get('num_classes', 7)

    # ── 1. Discover files, labels, and groups ────────────────────
    video_data = get_video_files_and_labels(data_root)
    if not video_data:
        print(f"No video data found in {data_root}")
        return

    paths, labels, groups = zip(*video_data)
    paths = list(paths)
    labels = list(labels)
    groups = np.array(groups)

    # Convert label codes to indices
    label_indices = [CODE_TO_IDX.get(lbl, 0) for lbl in labels]

    # ── 2. Group-aware split (prevents data leakage) ─────────────
    split_strategy = v_conf.get('split_strategy', 'group_shuffle')

    if split_strategy == 'group_shuffle':
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(paths, label_indices, groups=groups))
    else:
        # Fallback to stratified split (no group awareness)
        from sklearn.model_selection import train_test_split
        indices = list(range(len(paths)))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.2, random_state=42, stratify=label_indices
        )

    train_paths = [paths[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    train_label_indices = [label_indices[i] for i in train_idx]

    n_train_groups = len(set(groups[train_idx]))
    n_val_groups = len(set(groups[val_idx]))
    overlap = set(groups[train_idx]) & set(groups[val_idx])
    print(f"Split: {len(train_paths)} train ({n_train_groups} groups), "
          f"{len(val_paths)} val ({n_val_groups} groups), "
          f"group overlap: {len(overlap)}")

    if overlap:
        print(f"WARNING: {len(overlap)} configuration folder(s) appear in both "
              f"train and val sets. Consider adjusting split strategy.")

    # ── 3. Prepare datasets ──────────────────────────────────────
    train_dataset = WeldingSequenceDataset(
        train_paths, train_labels,
        transform=get_video_transforms(),
        seq_len=v_conf['seq_len'],
        frame_skip=v_conf['frame_skip']
    )
    val_dataset = WeldingSequenceDataset(
        val_paths, val_labels,
        transform=get_video_transforms(),
        seq_len=v_conf['seq_len'],
        frame_skip=v_conf['frame_skip']
    )

    train_loader = DataLoader(
        train_dataset, batch_size=v_conf['batch_size'],
        shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=v_conf['batch_size'],
        shuffle=False, num_workers=4, pin_memory=True
    )

    # ── 4. Initialize model ──────────────────────────────────────
    model = StreamingVideoClassifier(
        num_classes=num_classes,
        hidden_size=m_conf['hidden_size'],
        pretrained=m_conf['pretrained']
    ).to(device)

    # Inverse-frequency class weights
    use_weights = v_conf.get('class_weights', 'inverse_frequency')
    if use_weights == 'inverse_frequency':
        weights = compute_class_weights(train_label_indices, num_classes).to(device)
        print(f"Class weights: {weights.tolist()}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=v_conf['lr'])

    # Mixed precision scaler
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_score = 0.0
    epochs = v_conf['epochs']
    checkpoint_path = v_conf['checkpoint_path']

    print(f"\nStarting video training on {device}.")
    print(f"Checkpoint metric: hackathon combined score "
          f"(0.6 * Binary_F1 + 0.4 * Macro_F1)")

    for epoch in range(epochs):
        # ── Training loop ────────────────────────────────────────
        model.train()
        train_loss = 0
        all_preds = []
        all_labels_epoch = []

        for i, (sequences, batch_labels) in enumerate(train_loader):
            sequences, batch_labels = sequences.to(device), batch_labels.to(device)

            with autocast('cuda'):
                logits, _ = model(sequences)
                loss = criterion(logits, batch_labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels_epoch.extend(batch_labels.cpu().numpy())

            if i % 10 == 0:
                iter_f1 = f1_score(
                    batch_labels.cpu().numpy(), predicted.cpu().numpy(),
                    average='macro', zero_division=0
                )
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Macro F1: {iter_f1:.4f}")

        train_macro_f1 = f1_score(all_labels_epoch, all_preds, average='macro', zero_division=0)
        train_binary_f1, _ = compute_binary_metrics(all_labels_epoch, all_preds)
        train_hackathon = compute_hackathon_score(train_binary_f1, train_macro_f1)

        print(f"\n--- Epoch {epoch+1} Train Summary ---")
        print(f"  Macro F1: {train_macro_f1:.4f} | Binary F1: {train_binary_f1:.4f} | "
              f"Hackathon Score: {train_hackathon:.4f}")

        # ── Validation loop ──────────────────────────────────────
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        val_probs = []

        with torch.no_grad():
            for sequences, batch_labels in val_loader:
                sequences, batch_labels = sequences.to(device), batch_labels.to(device)
                logits, _ = model(sequences)
                loss = criterion(logits, batch_labels)
                val_loss += loss.item()

                probs = torch.softmax(logits, dim=1)
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(batch_labels.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)
        val_macro_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        val_binary_f1, val_roc_auc = compute_binary_metrics(
            val_labels_list, val_preds, val_probs
        )
        val_hackathon = compute_hackathon_score(val_binary_f1, val_macro_f1)

        print(f"\n--- Epoch {epoch+1} Validation ---")
        print(f"  Val Loss: {avg_val_loss:.4f}")
        print(f"  Macro F1: {val_macro_f1:.4f} | Binary F1: {val_binary_f1:.4f} | "
              f"ROC-AUC: {val_roc_auc if val_roc_auc else 'N/A'}")
        print(f"  Hackathon Score: {val_hackathon:.4f}")
        print(classification_report(val_labels_list, val_preds, digits=4, zero_division=0))

        # ── Checkpoint on best hackathon score ───────────────────
        if val_hackathon > best_score:
            best_score = val_hackathon
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Saved new best model (Hackathon Score: {best_score:.4f})")

    print(f"\nTraining complete. Best Val Hackathon Score: {best_score:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train video classifier with JSON config")
    parser.add_argument("--config", type=str, default="../configs/master_config.json",
                        help="Path to master config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    train_video(config)
