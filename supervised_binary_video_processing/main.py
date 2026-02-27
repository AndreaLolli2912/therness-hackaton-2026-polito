"""
Supervised Video Processing Multimodal Pipeline.
Fuses Sensor (CSV) aggregated stats with offline GPU Video Embeddings.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.impute import SimpleImputer
from src.data.dataset import build_video_index


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
        cache_path = os.path.join(cache_dir, f"{s.weld_id}_video.npy")
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
        y_list.append(s.label)
        groups.append(s.weld_id)
        kept.append(s)

    print(f"\nCompleted loading {len(y_list)} multimodal samples.")
    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(groups)

    print(f"Final Feature Dimension = {X.shape[1]} (1152 Video + 42 Sensor)")
    return X, y, groups, kept


def train_baseline_classifier(X, y, groups):
    """
    Trains a robust Random Forest baseline using group-aware splits,
    balanced multi-class weighting, and NaN imputation.
    """
    from sklearn.ensemble import RandomForestClassifier
    print("\nTraining Multimodal Random Forest classifier...")

    # STRICT LEAKAGE PREVENTION: GroupShuffleSplit on weld_id
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    # NaN SAFETY NET: Impute any residual NaN to 0 before sklearn
    imputer = SimpleImputer(strategy='constant', fill_value=0)
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)

    # Print class distribution to be aware of imbalance
    unique, counts = np.unique(y_train, return_counts=True)
    print(f"Training Class Distribution: {dict(zip(unique, counts))}")

    # MULTI-CLASS WEIGHTING: compute_sample_weight works for any number of classes
    weights = compute_sample_weight('balanced', y_train)

    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train, sample_weight=weights)

    val_proba = clf.predict_proba(X_val)[:, 1]
    val_pred = clf.predict(X_val)

    print("\nMultimodal Validation Results (Sensor + GPU Video Embeddings):")
    print(classification_report(y_val, val_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_val, val_proba))

    return clf


if __name__ == "__main__":
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset"))

    # Load Fused Dataset
    X, y, groups, kept = load_multimodal_dataset(dataset_root=dataset_root)

    if len(y) > 0:
        model = train_baseline_classifier(X, y, groups)
    else:
        print("No valid data samples found. Check dataset path and contents.")
