import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.video_model import VideoAutoencoder
from src.data.dataset import WeldingVideoDataset, get_video_transforms, get_video_files
import os
import numpy as np

def train_video(data_root, epochs=20, batch_size=16, lr=1e-4, device='cpu'):
    # 1. Split by video file to prevent temporal leakage
    all_video_files = get_video_files(data_root)
    if not all_video_files:
        print(f"No video data found in {data_root}")
        return

    np.random.seed(42)
    np.random.shuffle(all_video_files)
    split_idx = int(len(all_video_files) * 0.8)
    train_files = all_video_files[:split_idx]
    val_files = all_video_files[split_idx:]

    print(f"Using {len(train_files)} videos for training and {len(val_files)} for validation.")

    # 2. Prepare datasets
    train_dataset = WeldingVideoDataset(train_files, transform=get_video_transforms(), frame_skip=15)
    val_dataset = WeldingVideoDataset(val_files, transform=get_video_transforms(), frame_skip=15)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 3. Initialize model
    model = VideoAutoencoder(latent_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    print(f"Starting video model training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for i, frames in enumerate(train_loader):
            frames = frames.to(device)
            
            reconstructed = model(frames)
            loss = criterion(reconstructed, frames)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.6f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for frames in val_loader:
                frames = frames.to(device)
                reconstructed = model(frames)
                loss = criterion(reconstructed, frames)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        print(f"Epoch [{epoch+1}/{epochs}] Avg Train Loss: {avg_train:.6f} | Avg Val Loss: {avg_val:.6f}")
        
        # Save best model
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/video_autoencoder.pth')

    print(f"Video training complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_video(data_root='../sampleData', device=device)
