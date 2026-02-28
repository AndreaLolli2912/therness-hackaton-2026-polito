import os
import pandas as pd
import numpy as np
import glob
import json
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from collections import Counter

def extract_features(csv_path):
    """
    Extracts statistical features from a sensor CSV file.
    Columns are assumed to be: Date, Time, Part No, Pressure, CO2 Weld Flow, Feed, 
    Primary Weld Current, Wire Consumed, Secondary Weld Voltage
    """
    try:
        # Load CSV
        df = pd.read_csv(csv_path)
        
        # Clean column names (strip whitespace)
        df.columns = [c.strip() for c in df.columns]
        
        # Numeric columns to use for features
        feature_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Secondary Weld Voltage']
        
        # Ensure columns exist
        existing_cols = [c for c in feature_cols if c in df.columns]
        if not existing_cols:
            return None
            
        # Convert to numeric, handle errors
        for col in existing_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in critical columns
        df = df.dropna(subset=existing_cols)
        
        if len(df) < 5:
            return None
            
        # Compute stats (Mean, Std, Max, Min) as features
        features = {}
        for col in existing_cols:
            features[f"{col}_mean"] = df[col].mean()
            features[f"{col}_std"] = df[col].std()
            features[f"{col}_max"] = df[col].max()
            features[f"{col}_min"] = df[col].min()
            
        return features
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return None

def main():
    data_root = "dataset"
    print(f"Scanning {data_root} for sensor CSV files...")
    
    # Recursively find all CSV files
    csv_files = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} potential sensor files.")
    
    data = []
    labels = []
    
    for i, f in enumerate(csv_files):
        # Determine label from filename or path: "00" is good, others are defects
        run_id = os.path.splitext(os.path.basename(f))[0]
        parts = run_id.split('-')
        label_code = parts[-1] if len(parts) > 1 else "00"
        
        # Binary label: 0 for good (00), 1 for defect (everything else)
        binary_label = 0 if label_code == "00" else 1
        
        feat = extract_features(f)
        if feat:
            data.append(feat)
            labels.append(binary_label)
            
    if not data:
        print("No valid data extracted. Please check the dataset structure.")
        return
    
    # Create DataFrame and prepare features/labels
    df_features = pd.DataFrame(data)
    X = df_features.values
    y = np.array(labels)
    
    print(f"Summary: Total {len(data)} valid samples found.")
    print(f"Class distribution: {Counter(y)}")
    
    # Split into Train and Test
    # LOF in Novelty mode should be trained on normal ("good") data only
    X_good = X[y == 0]
    X_defect = X[y == 1]
    
    if len(X_good) < 10:
        print("Insufficient 'good' samples for training. LOF requires enough normal data.")
        return

    # Use 80% of good data for training, 20% for testing (plus all defects)
    X_train, X_test_good = train_test_split(X_good, test_size=0.2, random_state=42)
    X_test = np.vstack([X_test_good, X_defect])
    y_test = np.array([0] * len(X_test_good) + [1] * len(X_defect))
    
    print(f"Training on {len(X_train)} 'good' samples.")
    print(f"Testing on {len(X_test)} samples ({len(X_test_good)} good, {len(X_defect)} defect).")
    
    # 1. Feature Scaling (LOF is distance-based, so scaling is critical)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Local Outlier Factor Model
    # novelty=True allows us to fit on training data and predict on new data
    # n_neighbors: Number of neighbors to use for LOF calculation (default 20)
    # contamination: Expected proportion of outliers (used to set the threshold)
    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    clf.fit(X_train_scaled)
    
    # 3. Predict
    # LOF returns 1 for inliers (normal) and -1 for outliers (anomalies)
    y_pred_raw = clf.predict(X_test_scaled)
    # Convert to our binary labels: 1 -> 0 (good), -1 -> 1 (defect)
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    # 4. Evaluation Metrics
    print("\n" + "="*50)
    print("LOCAL OUTLIER FACTOR (LOF) ANOMALY DETECTION RESULTS")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["Good", "Defect"]))
    
    f1 = f1_score(y_test, y_pred)
    print(f"Anomaly Detection F1 Score: {f1:.4f}")
    
    # 5. Extract Anomaly Scores
    # LOF negative_outlier_factor_: lower (more negative) means more anomalous
    # For novelty=True, we use decision_function or score_samples
    scores = clf.decision_function(X_test_scaled)
    
    results = {
        "model": "Local Outlier Factor",
        "parameters": {"n_neighbors": 20, "contamination": 0.1},
        "metrics": {
            "f1_score": float(f1),
            "good_count": int(len(X_test_good)),
            "defect_count": int(len(X_defect)),
            "avg_score_good": float(scores[y_test == 0].mean()),
            "avg_score_defect": float(scores[y_test == 1].mean()),
        },
        "classification_report": classification_report(y_test, y_pred, target_names=["Good", "Defect"], output_dict=True)
    }
    
    # Save the results to JSON
    with open("lof_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
    print("\nResults saved to lof_results.json")
    print("\n[TIP] LOF is effective at detecting samples in low-density regions of the feature space.")

if __name__ == "__main__":
    main()
