"""
extract_window_features.py

Extracts MobileNetV3 spatial features for every frame of every video in the dataset.
Saves a single `.npy` file per video with shape [Total_Frames, 576].
This completely bypasses the slow Google Drive IO bottleneck during DataLoader training,
as the DataLoader can just load the `.npy` file and slice the sliding windows instantly.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import sys

# Add parent to path for dataset utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from video_processing_window.src.data.dataset import get_video_files_and_labels, get_video_transforms

BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    print(f"Loading MobileNetV3-Small on {DEVICE}...")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    class EmbeddingModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1) # [Batch, 576]
            return x

    extractor = EmbeddingModel(model).to(DEVICE)
    extractor.eval()
    return extractor

def process_video_to_tensor(video_path, model, transform):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((1, 576), dtype=np.float32)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    
    if not frames:
        return np.zeros((1, 576), dtype=np.float32)

    from PIL import Image
    tensor_list = [transform(Image.fromarray(f)) for f in frames]
    video_tensor = torch.stack(tensor_list)

    embeddings = []
    with torch.no_grad():
        for i in range(0, len(video_tensor), BATCH_SIZE):
            batch = video_tensor[i:i+BATCH_SIZE].to(DEVICE)
            out = model(batch)
            embeddings.append(out.cpu().numpy())

    return np.vstack(embeddings) # [Total_Frames, 576]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_root = os.path.abspath(os.path.join(script_dir, "..", "dataset"))
    out_dir = os.path.join(data_root, "window_features")
    os.makedirs(out_dir, exist_ok=True)

    print("--- Phase 1: Scanning Videos ---")
    video_data = get_video_files_and_labels(data_root)
    print(f"Found {len(video_data)} videos.")

    print("\n--- Phase 2: Extracting Features ---")
    model = get_feature_extractor()
    transform = get_video_transforms()

    for i, (path, code, group) in enumerate(video_data):
        rel_path = os.path.relpath(path, data_root)
        save_name = rel_path.replace(os.sep, "_").replace(".avi", ".npy")
        out_path = os.path.join(out_dir, save_name)

        print(f"[{i+1}/{len(video_data)}] {rel_path}", end="\r")

        if os.path.exists(out_path):
            continue

        try:
            feat = process_video_to_tensor(path, model, transform)
            np.save(out_path, feat)
        except Exception as e:
            print(f"\nError processing {rel_path}: {e}")
            np.save(out_path, np.zeros((1, 576), dtype=np.float32))

    print("\nFeature extraction complete!")

if __name__ == "__main__":
    main()
