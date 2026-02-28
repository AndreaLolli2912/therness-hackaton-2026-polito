import os
import argparse
import numpy as np
import pandas as pd
import joblib

def extract_sensor_features(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        sensors = df.iloc[:, 3:9].values.astype(np.float32)
        if len(sensors) == 0:
            return np.zeros(42, dtype=np.float32)

        feats = []
        for j in range(sensors.shape[1]):
            col = sensors[:, j]
            all_nan = np.all(np.isnan(col))
            c_mean = 0.0 if all_nan else float(np.nanmean(col))
            c_std  = 0.0 if all_nan else float(np.nanstd(col))
            c_max  = 0.0 if all_nan else float(np.nanmax(col))
            c_min  = 0.0 if all_nan else float(np.nanmin(col))

            valid_col = col[~np.isnan(col)]
            if len(valid_col) > 0:
                p10, p90 = np.percentile(valid_col, [10, 90])
                max_diff = float(np.max(np.abs(np.diff(valid_col)))) if len(valid_col) > 1 else 0.0
            else:
                p10, p90, max_diff = 0.0, 0.0, 0.0

            feats.extend([c_mean, c_std, c_max, c_min, p10, p90, max_diff])
        return np.array(feats, dtype=np.float32)
    except Exception as e:
        return np.zeros(42, dtype=np.float32)

def generate_multimodal_submission(test_dir, model_pkl_path, output_csv):
    if not os.path.exists(model_pkl_path):
        print(f"ERROR: Model file {model_pkl_path} not found.")
        return

    print(f"Loading MultiModal RF Model from {model_pkl_path}...")
    pipeline = joblib.load(model_pkl_path)
    model = pipeline['model']
    imputer = pipeline['imputer']

    # Also need the cache dir to get GPU embeddings during inference
    dataset_root = os.path.abspath(os.path.join(test_dir, ".."))
    cache_dir = os.path.join(dataset_root, "features_cache")

    sample_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and d.startswith("sample_")
    ])
    
    IDX_TO_CODE = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}

    rows = []
    print(f"Found {len(sample_dirs)} samples to process.")
    for i, sample_name in enumerate(sample_dirs):
        print(f"[{i+1}/{len(sample_dirs)}] Processing {sample_name}...", end="\r")
        sample_path = os.path.join(test_dir, sample_name)
        
        # 1. Load Pre-computed GPU Video Embedding (1x1152) if available
        # Note: the cache key for test set is usually just the sample_name
        v_feat = np.zeros(1152, dtype=np.float32)
        cache_path = os.path.join(cache_dir, f"{sample_name}_video.npy")
        if os.path.exists(cache_path):
            try:
                v_feat = np.load(cache_path)
            except Exception:
                pass
                
        # 2. Extract Sensor Features (1x42)
        csv_files = [f for f in os.listdir(sample_path) if f.endswith(".csv")]
        s_feat = np.zeros(42, dtype=np.float32)
        if csv_files:
            s_feat = extract_sensor_features(os.path.join(sample_path, csv_files[0]))

        # 3. Fuse Modalities
        x_sample = np.concatenate([v_feat, s_feat]).reshape(1, -1)
        
        # 4. Predict
        x_sample = imputer.transform(x_sample)
        probs = model.predict_proba(x_sample)[0]
        
        pred_idx = int(np.argmax(probs))
        pred_code = IDX_TO_CODE.get(pred_idx, "00")
        p_defect = float(1.0 - probs[0])
        
        rows.append({
            "sample_id": sample_name,
            "pred_label_code": pred_code,
            "p_defect": round(p_defect, 4)
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n\nSubmission saved to {output_csv}")
    print(df['pred_label_code'].value_counts())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--rf_model", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    args = parser.parse_args()
    
    generate_multimodal_submission(args.test_dir, args.rf_model, args.output_csv)
