"""
Video classifier training with hackathon-optimized scoring.

Key improvements over baseline:
  - Inverse-frequency class weights for imbalanced defect types
  - GroupShuffleSplit on configuration folders (prevents data leakage)
  - Binary + multi-class metrics logged each epoch
  - Combined hackathon score as the checkpoint metric:
      FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1
"""
import time
_t0 = time.time()

print("[1/8] Importing standard libraries...")
import os
import json
import numpy as np
from collections import Counter

print("[2/8] Importing PyTorch...")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
print(f"       PyTorch {torch.__version__} loaded.")

print("[3/8] Importing scikit-learn...")
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, roc_auc_score
print(f"       scikit-learn loaded.")

print("[4/8] Importing project modules...")
from src.models.video_model import StreamingVideoClassifier
from src.data.dataset import (
    WeldingSequenceDataset, get_video_transforms, get_video_files_and_labels,
)
print(f"       All imports done in {time.time()-_t0:.1f}s.\n")


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


def train_video(config, full_train=False, checkpoint=None):
    # Extract configs
    v_conf = config['video']['training']
    m_conf = config['video']['model']
    data_root = config['train_data_root']
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
    print(f"[5/8] Configuration loaded.")
    print(f"       data_root:   {os.path.abspath(data_root)}")
    print(f"       device:      {device}")
    print(f"       num_classes: {num_classes}")
    print(f"       epochs:      {v_conf['epochs']}")
    print(f"       batch_size:  {v_conf['batch_size']}")
    print(f"       lr:          {v_conf['lr']}")
    print(f"       seq_len:     {v_conf['seq_len']}")
    print(f"       frame_skip:  {v_conf['frame_skip']}")
    print(f"       full_train:  {full_train}")
    if full_train:
        print(f"       ⚠ FULL TRAINING MODE: using 100% of data, no validation")
    if checkpoint:
        print(f"       ⚠ RESUMING from checkpoint: {checkpoint}")

    # ── 1. Discover files, labels, and groups ────────────────────
    print(f"\n[6/8] Discovering video files in {os.path.abspath(data_root)}...")
    t_discover = time.time()
    video_data = get_video_files_and_labels(data_root)
    if not video_data:
        print(f"  ERROR: No video data found in {data_root}")
        print(f"  Make sure the dataset folder contains good_weld/ and defect_data_weld/ subdirectories.")
        return
    print(f"       Found {len(video_data)} videos in {time.time()-t_discover:.1f}s.")

    paths, labels, groups = zip(*video_data)
    paths = list(paths)
    labels = list(labels)
    groups = np.array(groups)

    # Convert label codes to indices
    label_indices = [CODE_TO_IDX.get(lbl, 0) for lbl in labels]

    # Print class distribution
    label_counts = Counter(labels)
    print(f"       Class distribution:")
    for code in sorted(label_counts.keys()):
        idx = CODE_TO_IDX.get(code, '?')
        print(f"         Code {code} (class {idx}): {label_counts[code]} videos")

    # ── 2. Split or use full data ─────────────────────────────────
    if full_train:
        print(f"\n       Using ALL {len(paths)} videos for training (no validation split)")
        train_paths = paths
        train_labels = labels
        val_paths = []
        val_labels = []
        train_label_indices = label_indices
    else:
        print(f"\n       Splitting data with GroupShuffleSplit (no leakage)...")
        split_strategy = v_conf.get('split_strategy', 'group_shuffle')

        if split_strategy == 'group_shuffle':
            gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
            train_idx, val_idx = next(gss.split(paths, label_indices, groups=groups))
        else:
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
        print(f"       Split: {len(train_paths)} train ({n_train_groups} groups), "
              f"{len(val_paths)} val ({n_val_groups} groups)")
        print(f"       Group overlap: {len(overlap)} (should be 0)")

        if overlap:
            print(f"  WARNING: {len(overlap)} configuration folder(s) appear in both "
                  f"train and val sets. Consider adjusting split strategy.")

    # ── 3. Prepare datasets ──────────────────────────────────────
    print(f"\n[7/8] Building datasets and dataloaders...")
    t_dataset = time.time()
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
    print(f"       Train dataset: {len(train_dataset)} sequences")
    if not full_train:
        print(f"       Val dataset:   {len(val_dataset)} sequences")
    print(f"       Datasets built in {time.time()-t_dataset:.1f}s.")

    train_loader = DataLoader(
        train_dataset, batch_size=v_conf['batch_size'],
        shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=v_conf['batch_size'],
        shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True
    )
    print(f"       Train batches: {len(train_loader)}")
    if not full_train:
        print(f"       Val batches:   {len(val_loader)}")

    # ── 4. Initialize model ──────────────────────────────────────
    print(f"\n[8/8] Initializing model...")
    model = StreamingVideoClassifier(
        num_classes=num_classes,
        hidden_size=m_conf['hidden_size'],
        pretrained=m_conf['pretrained']
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"       StreamingVideoClassifier loaded on {device}")
    print(f"       Parameters: {n_params:,}")
    print(f"       Pretrained backbone: {m_conf['pretrained']}")

    if "resume_from" in config.get('video', {}).get('training', {}):
        checkpoint_path_to_load = config['video']['training']['resume_from']
    else:
        checkpoint_path_to_load = args.checkpoint if 'args' in locals() and hasattr(args, 'checkpoint') and args.checkpoint else None

    if checkpoint_path_to_load and os.path.exists(checkpoint_path_to_load):
        print(f"       Loading checkpoint from {checkpoint_path_to_load}...")
        model.load_state_dict(torch.load(checkpoint_path_to_load, map_location=device))
        print(f"       Checkpoint loaded successfully.")


    # Inverse-frequency class weights
    use_weights = v_conf.get('class_weights', 'inverse_frequency')
    if use_weights == 'inverse_frequency':
        weights = compute_class_weights(train_label_indices, num_classes).to(device)
        print(f"       Class weights (inverse-freq): {[f'{w:.3f}' for w in weights.tolist()]}")
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        print(f"       Class weights: none (uniform)")
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=v_conf['lr'])

    # Mixed precision scaler
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    print(f"       Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    best_score = 0.0
    epochs = v_conf['epochs']
    checkpoint_path = v_conf['checkpoint_path']

    # In full mode, save to a separate file
    if full_train:
        base, ext = os.path.splitext(checkpoint_path)
        checkpoint_path = f"{base}_full{ext}"

    total_setup_time = time.time() - _t0
    print(f"\n{'='*65}")
    print(f"  TRAINING START — {epochs} epochs, setup took {total_setup_time:.1f}s")
    if full_train:
        print(f"  Mode: FULL TRAINING (100% data, no validation)")
    else:
        print(f"  Checkpoint metric: 0.6 * Binary_F1 + 0.4 * Macro_F1")
    print(f"  Model saved to: {checkpoint_path}")
    print(f"{'='*65}\n")

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Training loop ────────────────────────────────────────
        model.train()
        train_loss = 0
        all_preds = []
        all_labels_epoch = []

        print(f"━━━ Epoch {epoch+1}/{epochs} ━━━ TRAINING ━━━")
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

            if i % 5 == 0:
                avg_loss = train_loss / (i + 1)
                iter_f1 = f1_score(
                    batch_labels.cpu().numpy(), predicted.cpu().numpy(),
                    average='macro', zero_division=0
                )
                elapsed = time.time() - epoch_start
                eta = elapsed / (i + 1) * (len(train_loader) - i - 1) if i > 0 else 0
                print(f"  Step [{i+1:>4}/{len(train_loader)}] "
                      f"Loss: {loss.item():.4f} (avg: {avg_loss:.4f}) "
                      f"F1: {iter_f1:.3f} "
                      f"[{elapsed:.0f}s elapsed, ~{eta:.0f}s remaining]")

        train_macro_f1 = f1_score(all_labels_epoch, all_preds, average='macro', zero_division=0)
        train_binary_f1, _ = compute_binary_metrics(all_labels_epoch, all_preds)
        train_hackathon = compute_hackathon_score(train_binary_f1, train_macro_f1)
        train_time = time.time() - epoch_start

        print(f"\n  ── Epoch {epoch+1} Train Summary ({train_time:.1f}s) ──")
        print(f"     Avg Loss:       {train_loss/len(train_loader):.4f}")
        print(f"     Macro F1:       {train_macro_f1:.4f}")
        print(f"     Binary F1:      {train_binary_f1:.4f}")
        print(f"     Hackathon Score: {train_hackathon:.4f}")

        # ── Validation or save (full mode) ────────────────────────
        if full_train:
            # No validation — save after every epoch
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            epoch_total = time.time() - epoch_start
            print(f"\n  ✓ Model saved (epoch {epoch+1}) → {os.path.abspath(checkpoint_path)}")
            print(f"  Epoch {epoch+1} total time: {epoch_total:.1f}s")
            print(f"{'─'*65}\n")
        else:
            # ── Validation loop ──────────────────────────────────
            print(f"\n  ━━━ Epoch {epoch+1}/{epochs} ━━━ VALIDATION ━━━")
            val_start = time.time()
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels_list = []
            val_probs = []

            with torch.no_grad():
                for j, (sequences, batch_labels) in enumerate(val_loader):
                    sequences, batch_labels = sequences.to(device), batch_labels.to(device)
                    logits, _ = model(sequences)
                    loss = criterion(logits, batch_labels)
                    val_loss += loss.item()

                    probs = torch.softmax(logits, dim=1)
                    _, predicted = logits.max(1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_labels_list.extend(batch_labels.cpu().numpy())
                    val_probs.extend(probs.cpu().numpy())

                    if j % 5 == 0:
                        print(f"  Val Step [{j+1:>4}/{len(val_loader)}]")

            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_macro_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
            val_binary_f1, val_roc_auc = compute_binary_metrics(
                val_labels_list, val_preds, val_probs
            )
            val_hackathon = compute_hackathon_score(val_binary_f1, val_macro_f1)
            val_time = time.time() - val_start

            print(f"\n  ── Epoch {epoch+1} Validation Summary ({val_time:.1f}s) ──")
            print(f"     Val Loss:       {avg_val_loss:.4f}")
            print(f"     Macro F1:       {val_macro_f1:.4f}")
            print(f"     Binary F1:      {val_binary_f1:.4f}")
            print(f"     ROC-AUC:        {val_roc_auc if val_roc_auc else 'N/A'}")
            print(f"     Hackathon Score: {val_hackathon:.4f}")
            print()
            print(classification_report(val_labels_list, val_preds, digits=4, zero_division=0))

            # ── Checkpoint on best hackathon score ───────────────
            if val_hackathon > best_score:
                best_score = val_hackathon
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ NEW BEST MODEL SAVED! (Hackathon Score: {best_score:.4f})")
                print(f"    → {os.path.abspath(checkpoint_path)}")
            else:
                print(f"  ✗ No improvement (best={best_score:.4f}, current={val_hackathon:.4f})")

            epoch_total = time.time() - epoch_start
            print(f"\n  Epoch {epoch+1} total time: {epoch_total:.1f}s")
            print(f"{'─'*65}\n")

    total_time = time.time() - _t0
    print(f"{'='*65}")
    print(f"  TRAINING COMPLETE")
    if not full_train:
        print(f"  Best Val Hackathon Score: {best_score:.4f}")
    print(f"  Model saved to: {os.path.abspath(checkpoint_path)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train video classifier with JSON config")
    parser.add_argument("--config", type=str, default="../configs/master_config.json",
                        help="Path to master config")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint to resume training from")
    parser.add_argument("--full", action="store_true",
                        help="Train on 100%% of data (no validation split). "
                             "Use for final submission model.")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Config loaded: project={config.get('project_name', '?')}\n")

    train_video(config, full_train=args.full, checkpoint=args.checkpoint)
