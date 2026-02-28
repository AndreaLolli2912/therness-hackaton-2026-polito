"""
Temperature scaling calibration for the sliding-window video classifier.

Learns a single scalar temperature T on validation logits such that
softmax(logits / T) produces well-calibrated probabilities.

Usage:
    python calibrate.py \
        --checkpoint checkpoints/video_window_classifier.pth \
        --config ../configs/master_config.json \
        --output checkpoints/temperature.json
"""
import os
import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.optimize import minimize_scalar

from sklearn.model_selection import GroupShuffleSplit

from src.models.video_model import WindowVideoClassifier
from src.data.dataset import (
    WeldingWindowDataset, get_video_transforms, get_video_files_and_labels,
)


LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}
CODE_TO_IDX = {v: k for k, v in LABEL_CODE_MAP.items()}


def expected_calibration_error(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / len(labels)


def collect_val_logits(model, val_loader, device):
    all_logits = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for windows, labels in val_loader:
            windows = windows.to(device)
            logits = model(windows)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

    return torch.cat(all_logits), torch.cat(all_labels)


def find_temperature(logits, labels, t_range=(0.1, 10.0)):
    def nll_at_temp(T):
        scaled = logits / T
        log_probs = torch.log_softmax(scaled, dim=1)
        nll = nn.NLLLoss()(log_probs, labels)
        return nll.item()

    result = minimize_scalar(nll_at_temp, bounds=t_range, method='bounded')
    return result.x


def main():
    parser = argparse.ArgumentParser(description="Temperature scaling calibration")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="../configs/master_config.json")
    parser.add_argument("--output", default="checkpoints/temperature.json")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    w_conf = config['video_window']['training']
    m_conf = config['video_window']['model']
    data_root = config['data_root']
    num_classes = config.get('num_classes', 7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WindowVideoClassifier(num_classes=num_classes, pretrained=False)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    # Validation set (same split as training)
    video_data = get_video_files_and_labels(data_root)
    paths, labels, groups = zip(*video_data)
    paths = list(paths)
    labels = list(labels)
    groups = np.array(groups)
    label_indices = [CODE_TO_IDX.get(lbl, 0) for lbl in labels]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, val_idx = next(gss.split(paths, label_indices, groups=groups))

    val_paths = [paths[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    val_dataset = WeldingWindowDataset(
        val_paths, val_labels,
        transform=get_video_transforms(),
        window_size=w_conf['window_size'],
        window_stride=w_conf['window_stride'],
    )
    val_loader = DataLoader(val_dataset, batch_size=w_conf['batch_size'],
                            shuffle=False, num_workers=4)

    print("Collecting validation logits...")
    logits, true_labels = collect_val_logits(model, val_loader, device)
    print(f"  Collected {len(logits)} samples")

    probs_before = torch.softmax(logits, dim=1).numpy()
    ece_before = expected_calibration_error(probs_before, true_labels.numpy())
    print(f"\nBefore calibration:")
    print(f"  ECE = {ece_before:.4f}")

    print("\nFinding optimal temperature...")
    T = find_temperature(logits, true_labels)
    print(f"  Optimal T = {T:.4f}")

    probs_after = torch.softmax(logits / T, dim=1).numpy()
    ece_after = expected_calibration_error(probs_after, true_labels.numpy())
    print(f"\nAfter calibration:")
    print(f"  ECE = {ece_after:.4f}")
    print(f"  ECE improvement: {ece_before - ece_after:.4f}")

    result = {
        "temperature": round(T, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
