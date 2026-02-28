"""
Full hackathon evaluation suite.

Computes:
  - Binary metrics: F1, Precision, Recall, ROC-AUC, PR-AUC
  - Multi-class metrics: Macro F1, Weighted F1, per-class F1
  - Calibration: ECE
  - Combined: FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1 - 0.05 * ECE

Usage:
    python evaluate.py \
        --predictions predictions.csv \
        --ground_truth /path/to/test_data_ground_truth.csv
"""
import argparse
import csv
import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)


LABEL_NAMES = {
    "00": "good_weld",
    "01": "excessive_penetration",
    "02": "burn_through",
    "06": "overlap",
    "07": "lack_of_fusion",
    "08": "excessive_convexity",
    "11": "crater_cracks",
}


def load_csv(path):
    """Load CSV and return list of dicts."""
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def compute_binary_ece(p_defect, binary_true, n_bins=15):
    """ECE for binary defect probability."""
    p_defect = np.array(p_defect, dtype=float)
    binary_true = np.array(binary_true, dtype=float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (p_defect > lo) & (p_defect <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = binary_true[mask].mean()
        bin_conf = p_defect[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)

    return ece / len(binary_true)


def main():
    parser = argparse.ArgumentParser(description="Hackathon evaluation suite")
    parser.add_argument("--predictions", required=True, help="Predictions CSV path")
    parser.add_argument("--ground_truth", required=True, help="Ground truth CSV path")
    args = parser.parse_args()

    preds = load_csv(args.predictions)
    truth = load_csv(args.ground_truth)

    # Build lookup: sample_id → ground truth label code
    gt_map = {}
    for row in truth:
        sid = row.get("sample_id", "")
        label = row.get("label_code", row.get("label", ""))
        gt_map[sid] = label

    # Align predictions with ground truth
    pred_codes = []
    true_codes = []
    p_defects = []

    for row in preds:
        sid = row["sample_id"]
        if sid not in gt_map:
            print(f"WARNING: {sid} not in ground truth, skipping")
            continue
        pred_codes.append(row["pred_label_code"])
        true_codes.append(gt_map[sid])
        p_defects.append(float(row["p_defect"]))

    if not pred_codes:
        print("ERROR: No matching samples between predictions and ground truth")
        return

    # Binary labels
    binary_true = [0 if c == "00" else 1 for c in true_codes]
    binary_pred = [0 if c == "00" else 1 for c in pred_codes]

    # ── Binary Metrics ──────────────────────────────────────────
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1)
    binary_prec = precision_score(binary_true, binary_pred, pos_label=1, zero_division=0)
    binary_rec = recall_score(binary_true, binary_pred, pos_label=1, zero_division=0)
    try:
        roc_auc = roc_auc_score(binary_true, p_defects)
    except ValueError:
        roc_auc = None
    try:
        pr_auc = average_precision_score(binary_true, p_defects)
    except ValueError:
        pr_auc = None

    # ── Multi-class Metrics ─────────────────────────────────────
    all_codes = sorted(set(true_codes) | set(pred_codes))
    macro_f1 = f1_score(true_codes, pred_codes, average='macro', zero_division=0)
    weighted_f1 = f1_score(true_codes, pred_codes, average='weighted', zero_division=0)

    # ── Calibration ─────────────────────────────────────────────
    ece = compute_binary_ece(p_defects, binary_true)

    # ── Combined Score ──────────────────────────────────────────
    final_score = 0.6 * binary_f1 + 0.4 * macro_f1 - 0.05 * ece

    # ── Print Report ────────────────────────────────────────────
    print("=" * 65)
    print("         HACKATHON EVALUATION REPORT")
    print("=" * 65)

    print("\n── Binary Detection (good vs defect) ────────────────")
    print(f"  F1 (defect):    {binary_f1:.4f}")
    print(f"  Precision:      {binary_prec:.4f}")
    print(f"  Recall:         {binary_rec:.4f}")
    print(f"  ROC-AUC:        {roc_auc:.4f}" if roc_auc else "  ROC-AUC:        N/A")
    print(f"  PR-AUC:         {pr_auc:.4f}" if pr_auc else "  PR-AUC:         N/A")

    cm = confusion_matrix(binary_true, binary_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion Matrix:")
    print(f"              Predicted Good  Predicted Defect")
    print(f"  True Good       {tn:>5}           {fp:>5}")
    print(f"  True Defect     {fn:>5}           {tp:>5}")

    print("\n── Multi-class Defect Type ──────────────────────────")
    print(f"  Macro F1:       {macro_f1:.4f}")
    print(f"  Weighted F1:    {weighted_f1:.4f}")
    print()
    print(classification_report(
        true_codes, pred_codes,
        target_names=[f"{c} ({LABEL_NAMES.get(c, '?')})" for c in all_codes],
        digits=4, zero_division=0,
    ))

    print("── Calibration ─────────────────────────────────────")
    print(f"  ECE (binary):   {ece:.4f}")

    print("\n── Combined Score ─────────────────────────")
    print(f"  0.6 × Binary F1 ({binary_f1:.4f}) + "
          f"0.4 × Macro F1 ({macro_f1:.4f}) - "
          f"0.05 × ECE ({ece:.4f})")
    print(f"  FinalScore =    {final_score:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
