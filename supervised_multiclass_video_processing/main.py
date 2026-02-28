"""
Supervised Multiclass Video Processing — Multimodal Pipeline.
Fuses Sensor (CSV) aggregated stats with offline GPU Video Embeddings.

7-class output:
  0=good_weld, 1=excessive_penetration, 2=burn_through,
  3=overlap, 4=lack_of_fusion, 5=excessive_convexity, 6=crater_cracks

Key differences from binary version:
  - Labels are 0–6 (not 0/1)
  - ROC-AUC uses multiclass one-vs-rest
  - Reports per-class + macro F1
  - Reports hackathon combined score: 0.6 * Binary_F1 + 0.4 * Macro_F1
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
import joblib
from src.data.dataset import build_video_index, IDX_TO_CODE


# ── Label constants ──────────────────────────────────────────────
NUM_CLASSES = 7
LABEL_NAMES = [
    "good_weld", "excessive_penetration", "burn_through",
    "overlap", "lack_of_fusion", "excessive_convexity", "crater_cracks"
]


def extract_sensor_features(csv_path: str):
    """
    Extracts aggregated physics-based stats from welding sensor signals.
    Prevents temporal smearing and handles dropped packets (NaNs).
    Returns a 1x42 dimensional vector (6 channels * 7 statistics).
    """
    try:
        df = pd.read_csv(csv_path)
        # Columns 3 to 8: Pressure, CO2 Flow, Feed, Current, Wire, Voltage
        sensors = df.iloc[:, 3:9].values.astype(np.float32)

        # Failsafe for empty or fully corrupt files
        if len(sensors) == 0:
            return np.zeros(42, dtype=np.float32)

        feats = []
        for j in range(sensors.shape[1]):
            col = sensors[:, j]

            # Use nan-safe NumPy functions for dropped-packet resilience
            all_nan = np.all(np.isnan(col))
            c_mean = 0.0 if all_nan else float(np.nanmean(col))
            c_std  = 0.0 if all_nan else float(np.nanstd(col))
            c_max  = 0.0 if all_nan else float(np.nanmax(col))
            c_min  = 0.0 if all_nan else float(np.nanmin(col))

            # Percentiles and max absolute diff (ignore NaNs)
            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                p10, p90 = np.percentile(valid_col, [10, 90])
                max_diff = float(np.max(np.abs(np.diff(valid_col)))) if len(valid_col) > 1 else 0.0
            else:
                p10, p90, max_diff = 0.0, 0.0, 0.0

            feats.extend([c_mean, c_std, c_max, c_min, p10, p90, max_diff])

        return np.array(feats, dtype=np.float32)
    except Exception as e:
        # Silently fail to zero vector so pipeline doesn't crash on one broken weld
        print(f"WARNING: Failed to extract sensor features from {csv_path}: {e}")
        return np.zeros(42, dtype=np.float32)


def load_multimodal_dataset(dataset_root: str):
    """
    Traverses the dataset, loads cached GPU embeddings, and extracts sensor stats.
    Returns X (features), y (multiclass labels 0–6), groups, and kept samples.
    """
    print(f"Indexing dataset at: {dataset_root}...")
    samples = build_video_index(dataset_root)
    cache_dir = os.path.join(dataset_root, "features_cache")

    X_list = []
    y_list = []
    groups = []
    kept = []

    print("\nStarting Multimodal Feature Fusion (GPU Embeddings + Sensor Stats)...")
    for i, s in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Processing: {s.weld_id}", end="\r")

        # 1. Load Pre-computed GPU Video Embedding (1x1152)
        v_feat = np.zeros(1152, dtype=np.float32)
        cache_path = os.path.join(cache_dir, f"{s.weld_id.replace('/', '_')}_video.npy")
        if os.path.exists(cache_path):
            try:
                v_feat = np.load(cache_path)
            except Exception:
                pass  # Fallback to zeros is already set

        # 2. Extract Robust Sensor Features (1x42)
        csv_files = [f for f in os.listdir(s.weld_root) if f.endswith(".csv")]
        s_feat = np.zeros(42, dtype=np.float32)
        if csv_files:
            s_feat = extract_sensor_features(os.path.join(s.weld_root, csv_files[0]))

        # 3. Fuse Modalities: 1152 (Video) + 42 (Sensor) = 1194 features
        combined = np.concatenate([v_feat, s_feat])

        X_list.append(combined)
        y_list.append(s.label)  # multiclass label (0–6)
        groups.append(s.weld_id)
        kept.append(s)

    print(f"\nCompleted loading {len(y_list)} multimodal samples.")
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups)

    # Print class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Final Feature Dimension = {X.shape[1]} (1152 Video + 42 Sensor)")
    print(f"Class distribution:")
    for cls_idx, cnt in zip(unique, counts):
        code = IDX_TO_CODE.get(cls_idx, "??")
        name = LABEL_NAMES[cls_idx] if cls_idx < len(LABEL_NAMES) else "unknown"
        print(f"  Class {cls_idx} (code {code}, {name}): {cnt} samples")

    return X, y, groups, kept


def compute_binary_metrics(y_true, y_pred, y_proba=None):
    """
    Derive binary metrics from 7-class predictions.
    Binary: good_weld (class 0) vs any defect (classes 1–6).
    """
    binary_true = [0 if y == 0 else 1 for y in y_true]
    binary_pred = [0 if p == 0 else 1 for p in y_pred]
    binary_f1 = f1_score(binary_true, binary_pred, pos_label=1, zero_division=0)

    roc = None
    if y_proba is not None:
        # p_defect = 1 - P(good_weld)
        p_defect = [1.0 - p[0] for p in y_proba]
        try:
            roc = roc_auc_score(binary_true, p_defect)
        except ValueError:
            roc = None

    return binary_f1, roc


def train_baseline_classifier(X, y, groups, full_train=False):
    """
    Trains a robust Random Forest baseline using group-aware splits (or 100% of data),
    balanced multi-class weighting, and NaN imputation.
    Reports full 7-class metrics + hackathon combined score (if not full_train).
    """
    from sklearn.ensemble import RandomForestClassifier
    print(f"\nTraining Multimodal Random Forest classifier (7-class) - Full Train: {full_train}...")

    if full_train:
        X_train, y_train = X, y
        # In full_train mode, validation metrics aren't possible as there is no validation set
        X_val, y_val = None, None
    else:
        # STRICT LEAKAGE PREVENTION: GroupShuffleSplit on weld_id
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(gss.split(X, y, groups=groups))
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

    # NaN SAFETY NET: Impute any residual NaN to 0 before sklearn
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train = imputer.fit_transform(X_train)
    if not full_train:
        X_val = imputer.transform(X_val)

    # Print class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training Class Distribution: {dict(zip(unique, counts))}")

    # MULTI-CLASS WEIGHTING: compute_sample_weight works for any number of classes
    weights = compute_sample_weight('balanced', y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train, sample_weight=weights)

    if not full_train:
        val_pred = clf.predict(X_val)
        val_proba = clf.predict_proba(X_val)  # Shape: [n_samples, n_classes]

        # ── Multiclass Metrics ─────────────────────────────────────
        print("\n" + "=" * 65)
        print("  MULTIMODAL VALIDATION RESULTS (Sensor + GPU Video Embeddings)")
        print("=" * 65)
        print(classification_report(
            y_val, val_pred,
            target_names=LABEL_NAMES[:len(np.unique(np.concatenate([y_train, y_val])))],
            digits=4, zero_division=0
        ))

        # Macro F1 (treats each class equally)
        macro_f1 = f1_score(y_val, val_pred, average='macro', zero_division=0)
        print(f"Macro F1: {macro_f1:.4f}")

        # ── Binary Metrics ─────────────────────────────────────────
        binary_f1, roc = compute_binary_metrics(y_val, val_pred, val_proba)
        print(f"Binary F1 (defect vs good): {binary_f1:.4f}")
        if roc is not None:
            print(f"Binary ROC-AUC: {roc:.4f}")

        # ── Multiclass ROC-AUC (One-vs-Rest) ──────────────────────
        try:
            mc_roc = roc_auc_score(y_val, val_proba, multi_class='ovr')
            print(f"Multiclass ROC-AUC (OVR): {mc_roc:.4f}")
        except ValueError as e:
            print(f"Multiclass ROC-AUC: N/A ({e})")

        # ── Hackathon Combined Score ──────────────────────────────
        hackathon_score = 0.6 * binary_f1 + 0.4 * macro_f1
        print(f"\nHackathon Combined Score: {hackathon_score:.4f}")
        print(f"  (0.6 × Binary_F1={binary_f1:.4f} + 0.4 × Macro_F1={macro_f1:.4f})")
        print("=" * 65)
    else:
        print("\n" + "=" * 65)
        print("  MODEL TRAINED ON 100% OF DATA (No validation metrics available)")
        print("=" * 65)

    return clf, imputer


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train multiclass random forest on GPU embeddings and sensor stats.")
    parser.add_argument("--full", action="store_true", help="Train on 100% of the dataset without a validation split. Useful for generating final submission models.")
    args = parser.parse_args()

    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    # Load Fused Dataset
    X, y, groups, kept = load_multimodal_dataset(dataset_root=dataset_root)

    if len(y) > 0:
        model, imputer = train_baseline_classifier(X, y, groups, full_train=args.full)
        
        # Save model and imputer together
        save_dir = os.path.join(os.path.dirname(__file__), "weights")
        os.makedirs(save_dir, exist_ok=True)
        filename = "multimodal_rf_model_full.pkl" if args.full else "multimodal_rf_model.pkl"
        save_path = os.path.join(save_dir, filename)
        
        print(f"\nSaving model and imputer to {save_path}...")
        joblib.dump({'model': model, 'imputer': imputer}, save_path)
        print("Done.")
    else:
        print("No valid data samples found. Check dataset path and contents.")
