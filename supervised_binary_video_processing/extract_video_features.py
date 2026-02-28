"""
extract_video_features.py

Standalone script to extract deep learning features from welding videos.
Uses a pre-trained frozen CNN (MobileNetV3) to extract frame embeddings.
Aggregates temporal dynamics using Mean and Standard Deviation to prevent
'temporal smearing' of anomalies like burn-through or lack of fusion.
Saves features to disk (.npy) for ultra-fast downstream tabular training.
"""

import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from src.data.dataset import build_video_index

# Configuration
N_FRAMES = 40
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_feature_extractor():
    """Loads a pre-trained MobileNetV3 and removes the classification head."""
    print(f"Loading MobileNetV3 on {DEVICE}...")
    # Load pretrained model
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    
    # Remove the final classification layer.
    # The 'classifier' block in mobilenet_v3_small looks like:
    # (0): Linear(in_features=576, out_features=1024, bias=True)
    # (1): Hardswish()
    # (2): Dropout(p=0.2, inplace=True)
    # (3): Linear(in_features=1024, out_features=1000, bias=True)
    
    # We want the rich 576-dimensional features BEFORE the final MLP layers.
    # We can just use the 'features' and 'avgpool' modules.
    
    class EmbeddingModel(torch.nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.features = base_model.features
            self.avgpool = base_model.avgpool
            
        def forward(self, x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1) # Shape: [Batch, 576]
            return x

    extractor = EmbeddingModel(model)
    extractor = extractor.to(DEVICE)
    extractor.eval() # CRITICAL: Set to eval mode
    return extractor

def get_transforms():
    """Standard ImageNet preprocessing."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def extract_frames(video_path, n_frames=40):
    """Uniformly extracts frames from a video. Returns list of numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            # Convert BGR (OpenCV) to RGB (PyTorch)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)

    cap.release()
    return frames

def process_video_to_embedding(video_path, model, transform):
    """
    Extracts frames, passes them through the CNN in mini-batches,
    and aggregates the temporal embeddings (Mean + Std).
    """
    frames = extract_frames(video_path, N_FRAMES)
    
    # Failsafe: Corrupt or unreadable video
    # Return a zero vector of the expected concatenated size (576 * 2 = 1152)
    if len(frames) == 0:
        return np.zeros(576 * 2, dtype=np.float32)

    # Preprocess all frames
    tensor_list = [transform(f) for f in frames]
    video_tensor = torch.stack(tensor_list) # Shape: [N_FRAMES, 3, 224, 224]

    embeddings = []
    
    # Memory Safe Mini-Batching
    with torch.no_grad(): # CRITICAL: Prevent OOM by not building gradients
        for i in range(0, len(video_tensor), BATCH_SIZE):
            batch = video_tensor[i:i+BATCH_SIZE].to(DEVICE)
            out = model(batch)
            embeddings.append(out.cpu().numpy())

    # Shape: [N_FRAMES, 576]
    embeddings = np.vstack(embeddings)

    # Temporal Aggregation (Avoids Smearing)
    # Calculate Mean and Std across the temporal axis (axis=0)
    mu = np.mean(embeddings, axis=0)
    sigma = np.std(embeddings, axis=0)

    # Final feature vector Shape: [1152]
    final_representation = np.concatenate([mu, sigma])
    
    return final_representation

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract GPU video features")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Path to test_data/ directory with sample_XXXX/weld.avi structure")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_root = os.path.abspath(os.path.join(script_dir, "..", "dataset"))
    cache_dir = os.path.join(dataset_root, "features_cache")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Initialize PyTorch components
    model = get_feature_extractor()
    transform = get_transforms()

    # --- Phase 1: Process Training Data ---
    print(f"Indexing dataset at: {dataset_root}...")
    samples = build_video_index(dataset_root)
    print(f"Found {len(samples)} potential weld samples.")

    print(f"\nStarting GPU Feature Extraction. Saving to {cache_dir}...")
    
    for i, s in enumerate(samples):
        print(f"[{i+1}/{len(samples)}] Processing: {s.weld_id}", end="\r")
        
        cache_path = os.path.join(cache_dir, f"{s.weld_id}_video.npy")
        
        # Skip if already cached
        if os.path.exists(cache_path):
            continue
            
        try:
            if s.video_path:
                feat = process_video_to_embedding(s.video_path, model, transform)
            else:
                # Fallback for missing video
                feat = np.zeros(1152, dtype=np.float32)
                
            # Save to disk
            np.save(cache_path, feat)
        except Exception as e:
            print(f"\nError processing {s.weld_id}: {e}")
            # Save zero vector so we don't retry a broken file
            np.save(cache_path, np.zeros(1152, dtype=np.float32))

    print("\nTraining data extraction complete!")

    # --- Phase 2: Process Test Data (if provided) ---
    if args.test_dir:
        test_dir = os.path.abspath(args.test_dir)
        print(f"\nProcessing test data from: {test_dir}")
        
        if not os.path.isdir(test_dir):
            print(f"ERROR: Test directory not found: {test_dir}")
            return
        
        # Discover test samples: test_data/sample_XXXX/weld.avi
        test_samples = sorted([d for d in os.listdir(test_dir) 
                               if os.path.isdir(os.path.join(test_dir, d))])
        print(f"Found {len(test_samples)} test samples.")
        
        for i, sample_name in enumerate(test_samples):
            print(f"[{i+1}/{len(test_samples)}] Test: {sample_name}", end="\r")
            
            cache_path = os.path.join(cache_dir, f"{sample_name}_video.npy")
            if os.path.exists(cache_path):
                continue
            
            # Look for any video file in the sample directory
            sample_dir = os.path.join(test_dir, sample_name)
            video_files = [f for f in os.listdir(sample_dir)
                           if f.lower().endswith(('.avi', '.mp4', '.mov'))]
            
            try:
                if video_files:
                    video_path = os.path.join(sample_dir, video_files[0])
                    feat = process_video_to_embedding(video_path, model, transform)
                else:
                    feat = np.zeros(1152, dtype=np.float32)
                np.save(cache_path, feat)
            except Exception as e:
                print(f"\nError processing test {sample_name}: {e}")
                np.save(cache_path, np.zeros(1152, dtype=np.float32))
        
        print("\nTest data extraction complete!")

    print("\nAll extractions finished. Features cached successfully.")

if __name__ == "__main__":
    main()
