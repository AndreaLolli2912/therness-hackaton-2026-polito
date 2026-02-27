import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.sensor_model import SensorAutoencoder
from src.data.sensor_dataset import WeldingSensorDataset
import os

def train_sensor(data_root, epochs=20, batch_size=32, lr=1e-3, device='cpu'):
    # Prepare dataset and loader
    dataset = WeldingSensorDataset(data_root, window_size=50, step_size=10)
    if len(dataset) == 0:
        print("No sensor data found.")
        return
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    # input_size=6 based on selected columns
    model = SensorAutoencoder(input_size=6, hidden_size=64, latent_dim=16).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting sensor model training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, windows in enumerate(dataloader):
            windows = windows.to(device)
            
            # Forward pass
            reconstructed = model(windows)
            loss = criterion(reconstructed, windows)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    # Save the model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/sensor_autoencoder.pth')
    print("Training complete. Model saved to checkpoints/sensor_autoencoder.pth")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_sensor(data_root='../sampleData', device=device)
