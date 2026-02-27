import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.models.sensor_model import SensorAutoencoder
from src.data.sensor_dataset import WeldingSensorDataset, get_sensor_files
from sklearn.preprocessing import StandardScaler
import joblib
import os
import numpy as np

def train_sensor(data_root, epochs=50, batch_size=64, lr=1e-3, device='cpu'):
    # 1. Discover all files and split by session (file) to avoid leakage
    all_files = get_sensor_files(data_root)
    if not all_files:
        print(f"No sensor data found in {data_root}")
        return

    # Shuffle and split files (80% train, 20% val)
    np.random.seed(42)
    np.random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.8)
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Using {len(train_files)} files for training and {len(val_files)} for validation.")

    # 2. Fit scaler ONLY on training files
    print("Fitting scaler on training data...")
    scaler = StandardScaler()
    numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
    
    for f in train_files:
        df = pd.read_csv(f)
        scaler.partial_fit(df[numerical_cols].values)
    
    # Save the scaler immediately
    os.makedirs('checkpoints', exist_ok=True)
    joblib.dump(scaler, 'checkpoints/sensor_scaler.pkl')

    # 3. Create Datasets with the fitted scaler
    train_dataset = WeldingSensorDataset(train_files, scaler=scaler)
    val_dataset = WeldingSensorDataset(val_files, scaler=scaler)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 4. Initialize model
    model = SensorAutoencoder(input_size=6, hidden_size=64, latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    print(f"Starting sensor model training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for windows in train_loader:
            windows = windows.to(device)
            reconstructed = model(windows)
            loss = criterion(reconstructed, windows)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for windows in val_loader:
                windows = windows.to(device)
                reconstructed = model(windows)
                loss = criterion(reconstructed, windows)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'checkpoints/sensor_autoencoder.pth')

    print(f"Training complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    import pandas as pd
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_sensor(data_root='../sampleData', device=device)
