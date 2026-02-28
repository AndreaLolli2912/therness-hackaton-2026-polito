import os
import pandas as pd
import numpy as np
import glob
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, f1_score
from collections import Counter

def extract_features(csv_path):
    """
    Extracts statistical features from a sensor CSV file.
    Columns: Date, Time, Part No, Pressure, CO2 Weld Flow, Feed, Primary Weld Current, Wire Consumed, Secondary Weld Voltage
    """
    try:
        # Load CSV, skip headers if necessary. The file we saw has a header.
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
        
        df = df.dropna(subset=existing_cols)
        
        if len(df) < 5:
            return None
            
        # Compute stats
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
    
    csv_files = glob.glob(os.path.join(data_root, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} potential sensor files.")
    
    data = []
    labels = []
    
    for i, f in enumerate(csv_files):
        # Determine label from filename or path
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
        print("No valid data extracted. Check dataset structure.")
        return
    
    print(f"Summary: Total {len(data)} valid samples found.")
        
    df_features = pd.DataFrame(data)
    X = df_features.values
    y = np.array(labels)
    
    print(f"Extracted features for {len(X)} runs.")
    print(f"Class distribution: {Counter(y)}")
    
    # Split into train (only good samples for One-Class SVM) and test
    X_good = X[y == 0]
    X_defect = X[y == 1]
    
    # We use some good samples for training, and remaining for testing
    from sklearn.model_selection import train_test_split
    X_train, X_test_good = train_test_split(X_good, test_size=0.2, random_state=42)
    
    # Test set contains remaining good samples and all defect samples
    X_test = np.vstack([X_test_good, X_defect])
    y_test = np.array([0] * len(X_test_good) + [1] * len(X_defect))
    
    print(f"Training on {len(X_train)} 'good' samples...")
    print(f"Testing on {len(X_test)} samples ({len(X_test_good)} good, {len(X_defect)} defect)")
    
    # 1. Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Train One-Class SVM
    # nu: An upper bound on the fraction of training errors and a lower bound of the fraction of support vectors.
    # kernel: 'rbf' is standard for non-linear boundaries
    clf = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(X_train_scaled)
    
    # 3. Predict
    # OneClassSVM returns 1 for inliers (normal) and -1 for outliers (anomalies)
    y_pred_raw = clf.predict(X_test_scaled)
    # Convert to our binary labels: 1 -> 0 (good), -1 -> 1 (defect)
    y_pred = np.where(y_pred_raw == 1, 0, 1)
    
    # 4. Evaluate
    print("\n" + "="*40)
    print("ONE-CLASS SVM ANOMALY DETECTION RESULTS")
    print("="*40)
    print(classification_report(y_test, y_pred, target_names=["Good", "Defect"]))
    
    f1 = f1_score(y_test, y_pred)
    print(f"Anomaly Detection F1 Score: {f1:.4f}")
    
    # 5. Example of getting anomaly scores (decision function)
    # Lower values = more anomalous
    scores = clf.decision_function(X_test_scaled)
    
    # 6. Save results to a file for robustness
    results = {
        "class_distribution": {str(k): int(v) for k, v in Counter(y).items()},
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "f1_score": f1,
        "avg_score_good": float(scores[y_test == 0].mean()),
        "avg_score_defect": float(scores[y_test == 1].mean()),
        "report": classification_report(y_test, y_pred, target_names=["Good", "Defect"], output_dict=True)
    }
    
    import json
    with open("svm_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print("\nResults saved to svm_results.json")

if __name__ == "__main__":
    main()
