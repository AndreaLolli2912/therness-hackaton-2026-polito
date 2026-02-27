import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.video_model import StreamingVideoClassifier
from src.data.dataset import WeldingSequenceDataset, get_video_transforms, get_video_files_and_labels
import os
import numpy as np

def train_video(data_root, epochs=20, batch_size=8, lr=1e-4, device='cpu'):
    # 1. Discover files and labels
    video_data = get_video_files_and_labels(data_root)
    if not video_data:
        print(f"No video data found in {data_root}")
        return

    # Split by video file to prevent temporal leakage
    np.random.seed(42)
    np.random.shuffle(video_data)
    split_idx = int(len(video_data) * 0.8)
    train_data = video_data[:split_idx]
    val_data = video_data[split_idx:]

    print(f"Using {len(train_data)} videos for training and {len(val_data)} for validation.")

    # 2. Prepare datasets
    train_paths, train_labels = zip(*train_data)
    val_paths, val_labels = zip(*val_data)

    train_dataset = WeldingSequenceDataset(train_paths, train_labels, transform=get_video_transforms(), seq_len=15, frame_skip=5)
    val_dataset = WeldingSequenceDataset(val_paths, val_labels, transform=get_video_transforms(), seq_len=15, frame_skip=5)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 3. Initialize model (7 classes for welding defects)
    model = StreamingVideoClassifier(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_acc = 0.0

    print(f"Starting supervised video classification training on {device}...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            # StreamingVideoClassifier returns (logits, hidden_state)
            logits, _ = model(sequences)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, Acc: {100.*correct/total:.2f}%")

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                logits, _ = model(sequences)
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
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/video_classifier.pth')

    print(f"Video training complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    # Point to the user's actual data location
    data_path = os.path.expanduser("~/Desktop/Hackathon")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    train_video(data_root=data_path, device=device)
