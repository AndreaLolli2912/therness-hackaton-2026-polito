import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.video_model import VideoAutoencoder
from src.data.dataset import WeldingVideoDataset, get_transforms
import os

def train(data_root, epochs=10, batch_size=8, lr=1e-4, device='cpu'):
    # Prepare dataset and loader
    dataset = WeldingVideoDataset(data_root, transform=get_transforms(), frame_skip=10)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = VideoAutoencoder(latent_dim=256).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training on {device}...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for i, frames in enumerate(dataloader):
            frames = frames.to(device)
            
            # Forward pass
            reconstructed = model(frames)
            loss = criterion(reconstructed, frames)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
        
    # Save the model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save(model.state_dict(), 'checkpoints/video_autoencoder.pth')
    print("Training complete. Model saved to checkpoints/video_autoencoder.pth")

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train(data_root='../sampleData', device=device)
