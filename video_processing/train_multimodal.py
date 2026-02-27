import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.models.fusion_model import MultimodalFusionClassifier
from src.data.multimodal_dataset import MultimodalWeldingDataset, get_multimodal_data
import os
import numpy as np
import json
import joblib
from sklearn.metrics import f1_score

def train_multimodal(config):
    v_conf = config['video']['training']
    s_conf = config['sensor']['training']
    m_conf = config['video']['model'] # Using some video defaults for fusion
    data_root = config['data_root']
    device = config['device']
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Discover Pairs
    data_pairs = get_multimodal_data(data_root)
    if not data_pairs:
        print(f"No multimodal data pairs found in {data_root}")
        return

    np.random.seed(42)
    np.random.shuffle(data_pairs)
    split_idx = int(len(data_pairs) * 0.8)
    train_pairs = data_pairs[:split_idx]
    val_pairs = data_pairs[split_idx:]

    print(f"Found {len(data_pairs)} pairs. Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    # 2. Load Scaler
    scaler = None
    if os.path.exists(s_conf['scaler_path']):
        scaler = joblib.load(s_conf['scaler_path'])
        print(f"Loaded sensor scaler from {s_conf['scaler_path']}")
    else:
        print("Warning: Sensor scaler not found. Training without scaling.")

    # 3. Datasets
    train_dataset = MultimodalWeldingDataset(
        train_pairs, 
        v_seq_len=v_conf['seq_len'], v_frame_skip=v_conf['frame_skip'],
        s_window_size=s_conf['window_size'],
        scaler=scaler
    )
    val_dataset = MultimodalWeldingDataset(
        val_pairs, 
        v_seq_len=v_conf['seq_len'], v_frame_skip=v_conf['frame_skip'],
        s_window_size=s_conf['window_size'],
        scaler=scaler
    )
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2) # Small batch for fusion
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)
    
    # 4. Model
    model = MultimodalFusionClassifier(
        num_classes=config['num_classes'],
        video_hidden=config['video']['model']['hidden_size'],
        sensor_hidden=config['sensor']['model']['hidden_size'],
        fusion_hidden=64
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    best_f1 = 0.0
    epochs = 20

    print(f"Starting Multimodal Fusion Training on {device}")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        all_preds = []
        all_labels = []
        
        for v_batch, s_batch, labels in train_loader:
            v_batch, s_batch, labels = v_batch.to(device), s_batch.to(device), labels.to(device)
            
            logits = model(v_batch, s_batch)
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
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for v_batch, s_batch, labels in val_loader:
                v_batch, s_batch, labels = v_batch.to(device), s_batch.to(device), labels.to(device)
                logits = model(v_batch, s_batch)
                _, predicted = logits.max(1)
                val_preds.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        val_f1 = f1_score(val_labels_list, val_preds, average='macro', zero_division=0)
        print(f"Epoch [{epoch+1}/{epochs}] Val Macro F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), 'checkpoints/multimodal_fusion.pth')
            print(f"Saved best fusion model (F1: {best_f1:.4f})")

    print(f"Fusion Training Complete. Best Val F1: {best_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="../configs/master_config.json")
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = json.load(f)
    train_multimodal(config)
