import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.sensor_model import SensorClassifier
from src.data.sensor_dataset import WeldingSensorDataset, get_sensor_files_and_labels
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np
import pandas as pd

def train_sensor(data_root, epochs=30, batch_size=32, lr=1e-3, device='cpu'):
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
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(scaler, 'checkpoints/sensor_scaler.pkl')

    # 3. Create Datasets with the fitted scaler
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)
    
    train_dataset = WeldingSensorDataset(train_paths, labels=train_labels, scaler=scaler, window_size=100)
    val_dataset = WeldingSensorDataset(val_paths, labels=val_labels, scaler=scaler, window_size=100)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize supervised classifier
    model = SensorClassifier(input_size=6, hidden_size=64, num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0

    print(f"Starting sensor classifier training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        for windows, labels in train_loader:
            windows, labels = windows.to(device), labels.to(device)
            
            logits = model(windows)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for windows, labels in val_loader:
                windows, labels = windows.to(device), labels.to(device)
                logits = model(windows)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_acc = 100. * val_correct / val_total if val_total > 0 else 0
        
        print(f"Epoch [{epoch+1}/{epochs}] Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'checkpoints/sensor_classifier.pth')

    print(f"Training complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    data_path = os.path.expanduser("~/Desktop/Hackathon")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_sensor(data_root=data_path, device=device)
