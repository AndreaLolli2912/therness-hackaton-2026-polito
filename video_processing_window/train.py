"""
Sliding-window video classifier training with hackathon-optimized scoring.

Same training loop structure as the GRU pipeline, but each window of N frames
is independently classified (no hidden state / temporal model).

Key features:
  - Inverse-frequency class weights for imbalanced defect types
  - GroupShuffleSplit on config folders (prevents data leakage)
  - Binary + multi-class metrics logged each epoch
  - Combined hackathon score: FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler
from torch.amp import autocast
print(f"       PyTorch {torch.__version__} loaded.")

print("[3/8] Importing scikit-learn...")
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score, classification_report, roc_auc_score
print(f"       scikit-learn loaded.")

print("[4/8] Importing project modules...")
from src.models.video_model import WindowVideoClassifier
from src.data.dataset import (
    WeldingWindowDataset, get_video_transforms, get_video_files_and_labels,
)
print(f"       All imports done in {time.time()-_t0:.1f}s.\n")


# ── Label mapping ────────────────────────────────────────────────
LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}
CODE_TO_IDX = {v: k for k, v in LABEL_CODE_MAP.items()}


def compute_class_weights(labels, num_classes=7, power=1.0):
    """Inverse-frequency weights with optional power scaling."""
    counts = Counter(labels)
    total = len(labels)
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = counts.get(cls_idx, 1)
        weights[cls_idx] = total / (num_classes * count)
    if power != 1.0:
        weights = torch.pow(weights, power)
    mean_w = float(weights.mean()) if len(weights) > 0 else 1.0
    if mean_w > 0:
        weights = weights / mean_w
    return weights


def build_balanced_sampler(dataset, num_classes=7, power=0.35):
    """Build a WeightedRandomSampler from window labels in the dataset."""
    if not getattr(dataset, "samples", None):
        return None

    labels = [int(s.get("label", 0)) for s in dataset.samples]
    counts = Counter(labels)
    if not counts:
        return None

    class_weights = {
        cls_idx: (1.0 / max(counts.get(cls_idx, 1), 1)) ** power
        for cls_idx in range(num_classes)
    }
    sample_weights = torch.tensor(
        [class_weights.get(lbl, 1.0) for lbl in labels],
        dtype=torch.double,
    )
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def compute_binary_metrics(all_labels, all_preds, all_probs=None):
    """Binary: good_weld (class 0) vs any defect (classes 1-6)."""
    binary_true = [0 if y == 0 else 1 for y in all_labels]
    binary_pred = [0 if p == 0 else 1 for p in all_preds]
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, zero_division=0)

    roc = None
    if all_probs is not None:
        p_defect = [1.0 - p[0] for p in all_probs]
        try:
            roc = roc_auc_score(binary_true, p_defect)
        except ValueError:
            roc = None

    return binary_f1, roc


def compute_hackathon_score(binary_f1, macro_f1):
    """FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1"""
    return 0.6 * binary_f1 + 0.4 * macro_f1


def train_video_window(config, full_train=False):
    # Extract configs — use video_window section
    w_conf = config['video_window']['training']
    m_conf = config['video_window']['model']
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
    print(f"[5/8] Configuration loaded.")
    print(f"       data_root:     {os.path.abspath(data_root)}")
    print(f"       device:        {device}")
    print(f"       num_classes:   {num_classes}")
    print(f"       epochs:        {w_conf['epochs']}")
    print(f"       batch_size:    {w_conf['batch_size']}")
    print(f"       lr:            {w_conf['lr']}")
    print(f"       window_size:   {w_conf['window_size']}")
    print(f"       window_stride: {w_conf['window_stride']}")
    print(f"       full_train:    {full_train}")
    if full_train:
        print(f"       ⚠ FULL TRAINING MODE: using 100% of data, no validation")

    # ── 1. Discover files ────────────────────────────────────────
    print(f"\n[6/8] Discovering video files in {os.path.abspath(data_root)}...")
    t_discover = time.time()
    video_data = get_video_files_and_labels(data_root)
    if not video_data:
        print(f"  ERROR: No video data found in {data_root}")
        return
    print(f"       Found {len(video_data)} videos in {time.time()-t_discover:.1f}s.")

    paths, labels, groups = zip(*video_data)
    paths = list(paths)
    labels = list(labels)
    groups = np.array(groups)

    label_indices = [CODE_TO_IDX.get(lbl, 0) for lbl in labels]

    label_counts = Counter(labels)
    print(f"       Class distribution:")
    for code in sorted(label_counts.keys()):
        idx = CODE_TO_IDX.get(code, '?')
        print(f"         Code {code} (class {idx}): {label_counts[code]} videos")

    # ── 2. Split ─────────────────────────────────────────────────
    if full_train:
        print(f"\n       Using ALL {len(paths)} videos for training")
        train_paths, train_labels = paths, labels
        val_paths, val_labels = [], []
        train_label_indices = label_indices
    else:
        print(f"\n       Splitting data with GroupShuffleSplit...")
        split_strategy = w_conf.get('split_strategy', 'group_shuffle')

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

    # ── 3. Build datasets ────────────────────────────────────────
    print(f"\n[7/8] Building datasets and dataloaders...")
    t_dataset = time.time()
    train_dataset = WeldingWindowDataset(
        train_paths, train_labels,
        transform=get_video_transforms(),
        window_size=w_conf['window_size'],
        window_stride=w_conf['window_stride'],
        data_root=data_root,
    )
    val_dataset = WeldingWindowDataset(
        val_paths, val_labels,
        transform=get_video_transforms(),
        window_size=w_conf['window_size'],
        window_stride=w_conf['window_stride'],
        data_root=data_root,
    )
    print(f"       Train dataset: {len(train_dataset)} windows")
    if not full_train:
        print(f"       Val dataset:   {len(val_dataset)} windows")
    print(f"       Datasets built in {time.time()-t_dataset:.1f}s.")

    num_workers = int(w_conf.get('num_workers', 4))
    use_balanced_sampler = bool(w_conf.get('use_balanced_sampler', False))
    sampler_power = float(w_conf.get('balanced_sampler_power', 0.35))

    train_sampler = None
    if use_balanced_sampler:
        train_sampler = build_balanced_sampler(
            train_dataset,
            num_classes=num_classes,
            power=sampler_power,
        )
        if train_sampler is not None:
            print(f"       Balanced sampler: enabled (power={sampler_power})")
        else:
            print("       Balanced sampler: requested but unavailable, falling back to shuffle")

    train_loader = DataLoader(
        train_dataset,
        batch_size=w_conf['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=w_conf['batch_size'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    print(f"       Train batches: {len(train_loader)}")
    if not full_train:
        print(f"       Val batches:   {len(val_loader)}")

    # ── 4. Model ─────────────────────────────────────────────────
    print(f"\n[8/8] Initializing model...")
    model = WindowVideoClassifier(
        num_classes=num_classes,
        pretrained=m_conf['pretrained'],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"       WindowVideoClassifier loaded on {device}")
    print(f"       Parameters: {n_params:,}")
    print(f"       Pretrained backbone: {m_conf['pretrained']}")

    # Class weights
    use_weights = w_conf.get('class_weights', 'inverse_frequency')
    class_weight_power = float(w_conf.get('class_weight_power', 1.0))
    if use_weights == 'inverse_frequency':
        weights = compute_class_weights(
            train_label_indices,
            num_classes=num_classes,
            power=class_weight_power,
        ).to(device)
        print(
            f"       Class weights: {[f'{w:.3f}' for w in weights.tolist()]} "
            f"(power={class_weight_power})"
        )
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    weight_decay = float(w_conf.get('weight_decay', 0.0))
    optimizer = optim.Adam(model.parameters(), lr=w_conf['lr'], weight_decay=weight_decay)
    if weight_decay > 0:
        print(f"       Weight decay: {weight_decay}")

    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    print(f"       Mixed precision (AMP): {'enabled' if use_amp else 'disabled'}")

    best_score = 0.0
    epochs = w_conf['epochs']
    checkpoint_path = w_conf['checkpoint_path']

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

        # ── Training ─────────────────────────────────────────────
        model.train()
        train_loss = 0
        all_preds = []
        all_labels_epoch = []

        print(f"━━━ Epoch {epoch+1}/{epochs} ━━━ TRAINING ━━━")
        for i, (windows, batch_labels) in enumerate(train_loader):
            windows, batch_labels = windows.to(device), batch_labels.to(device)

            with autocast('cuda'):
                logits = model(windows)
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

        # ── Validation / save ────────────────────────────────────
        if full_train:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            epoch_total = time.time() - epoch_start
            print(f"\n  ✓ Model saved (epoch {epoch+1}) → {os.path.abspath(checkpoint_path)}")
            print(f"  Epoch {epoch+1} total time: {epoch_total:.1f}s")
            print(f"{'─'*65}\n")
        else:
            print(f"\n  ━━━ Epoch {epoch+1}/{epochs} ━━━ VALIDATION ━━━")
            val_start = time.time()
            model.eval()
            val_loss = 0
            val_preds = []
            val_labels_list = []
            val_probs = []

            with torch.no_grad():
                for j, (windows, batch_labels) in enumerate(val_loader):
                    windows, batch_labels = windows.to(device), batch_labels.to(device)
                    logits = model(windows)
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
    parser = argparse.ArgumentParser(description="Train sliding-window video classifier")
    parser.add_argument("--config", type=str, default="../configs/master_config.json",
                        help="Path to master config")
    parser.add_argument("--full", action="store_true",
                        help="Train on 100%% of data (no validation split).")
    args = parser.parse_args()

    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    print(f"Config loaded: project={config.get('project_name', '?')}\n")

    train_video_window(config, full_train=args.full)
