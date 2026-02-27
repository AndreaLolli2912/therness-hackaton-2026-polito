import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.sensor_model import SensorClassifier
from src.data.sensor_dataset import WeldingSensorDataset, get_sensor_files_and_labels
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import joblib
import os
import numpy as np
import pandas as pd
import json

def train_sensor(config):
    s_conf = config['sensor']['training']
    m_conf = config['sensor']['model']
    data_root = config['data_root']
    device = config['device']
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Discover data and labels
    sensor_data = get_sensor_files_and_labels(data_root)
    if not sensor_data:
        print(f"No sensor data found in {data_root}")
        return

    # Shuffle and split files (80% train, 20% val)
    np.random.seed(42)
    np.random.shuffle(sensor_data)
    split_idx = int(len(sensor_data) * 0.8)
    train_data = sensor_data[:split_idx]
    val_data = sensor_data[split_idx:]

    print(f"Using {len(train_data)} files for training and {len(val_data)} for validation.")

    # 2. Fit scaler ONLY on training files
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
    
    for path, _ in train_data:
        df = pd.read_csv(path)
        scaler.partial_fit(df[numerical_cols].values)
    
    # Save the scaler
    os.makedirs(os.path.dirname(s_conf['scaler_path']), exist_ok=True)
    joblib.dump(scaler, s_conf['scaler_path'])

    # 3. Create Datasets with the fitted scaler
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    
    train_dataset = WeldingSensorDataset(train_paths, labels=train_labels, scaler=scaler, window_size=s_conf['window_size'])
    val_dataset = WeldingSensorDataset(val_paths, labels=val_labels, scaler=scaler, window_size=s_conf['window_size'])
    
    train_loader = DataLoader(train_dataset, batch_size=s_conf['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=s_conf['batch_size'], shuffle=False)

    # 4. Initialize supervised classifier
    model = SensorClassifier(
        input_size=m_conf['input_size'], 
        hidden_size=m_conf['hidden_size'], 
        num_classes=config['num_classes']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=s_conf['lr'])
    
    best_metric = 0.0
    epochs = s_conf['epochs']

    print(f"Starting sensor classifier training on {device}. Metric: {s_conf['metric']}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []

        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            
            logits = model(windows)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for windows, labels in val_loader:
                windows, labels = windows.to(device), labels.to(device)
                logits = model(windows)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        
        print(f"Epoch [{epoch+1}/{epochs}] Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_metric:
            best_metric = val_f1
            torch.save(model.state_dict(), s_conf['checkpoint_path'])
            print(f"Saved new best model with F1: {best_metric:.4f}")

    print(f"Training complete. Best Val Macro F1: {best_metric:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train supervised sensor classifier with JSON config")
    parser.add_argument("--config", type=str, default="../configs/master_config.json", help="Path to master config")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_sensor(config)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train supervised sensor classifier")
    parser.add_argument("--data_root", type=str, default=os.getenv("DATA_ROOT", "/data1/malto/therness/data/Hackathon"),
                        help="Path to the root data directory")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    train_sensor(data_root=args.data_root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, device=device)
