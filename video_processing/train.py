import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.video_model import StreamingVideoClassifier
from src.data.dataset import WeldingSequenceDataset, get_video_transforms, get_video_files_and_labels
import os
import numpy as np
import json
from sklearn.metrics import f1_score, classification_report

def train_video(config):
    # Extract configs
    v_conf = config['video']['training']
    m_conf = config['video']['model']
    data_root = config['data_root']
    device = config['device']
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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

    train_dataset = WeldingSequenceDataset(
        train_paths, train_labels, 
        transform=get_video_transforms(), 
        seq_len=v_conf['seq_len'], 
        frame_skip=v_conf['frame_skip']
    )
    val_dataset = WeldingSequenceDataset(
        val_paths, val_labels, 
        transform=get_video_transforms(), 
        seq_len=v_conf['seq_len'], 
        frame_skip=v_conf['frame_skip']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=v_conf['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=v_conf['batch_size'], shuffle=False, num_workers=2)
    
    # 3. Initialize model
    model = StreamingVideoClassifier(
        num_classes=config['num_classes'],
        hidden_size=m_conf['hidden_size'],
        pretrained=m_conf['pretrained']
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=v_conf['lr'])
    
    best_metric = 0.0
    epochs = v_conf['epochs']

    print(f"Starting video training on {device}. Metric: {v_conf['metric']}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for i, (sequences, labels) in enumerate(train_loader):
            sequences, labels = sequences.to(device), labels.to(device)
            
            logits, _ = model(sequences)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if i % 10 == 0:
                iter_f1 = f1_score(labels.cpu().numpy(), predicted.cpu().numpy(), average='macro', zero_division=0)
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i}/{len(train_loader)}], Loss: {loss.item():.4f}, Macro F1: {iter_f1:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                logits, _ = model(sequences)
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        
        print(f"\n--- Epoch {epoch+1} Validation ---")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        # Save best model based on configured metric
        if val_f1 > best_metric:
            best_metric = val_f1
            os.makedirs(os.path.dirname(v_conf['checkpoint_path']), exist_ok=True)
            torch.save(model.state_dict(), v_conf['checkpoint_path'])
            print(f"Saved new best model with F1: {best_metric:.4f}")

    print(f"\nTraining complete. Best Val Macro F1: {best_metric:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train video classifier with JSON config")
    parser.add_argument("--config", type=str, default="../configs/master_config.json", help="Path to master config")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    train_video(config)
