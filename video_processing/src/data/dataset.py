import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class WeldingSequenceDataset(Dataset):
    def __init__(self, video_paths, labels, seq_len=15, frame_skip=2, transform=None):
        """
        Args:
            video_paths (list): List of paths to AVI files.
            labels (list): List of label codes (e.g., '00', '01') corresponding to videos.
            seq_len (int): Number of frames per sequence.
            frame_skip (int): Step between frames in a sequence (temporal resolution).
            transform (callable): Image transforms.
        """
        self.seq_len = seq_len
        self.frame_skip = frame_skip
        self.transform = transform
        self.samples = []
        
        # Map label codes to 0-6 indices
        self.label_map = {
            "00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6
        }

        for path, code in zip(video_paths, labels):
            if not os.path.exists(path):
                continue
                
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # Sequence duration in frames
            window_size = seq_len * frame_skip
            
            # Create overlapping windows across the video
            # Using a stride of window_size // 2 for data augmentation
            for start_f in range(0, total_frames - window_size, window_size // 2):
                self.samples.append({
                    "video_path": path,
                    "start_frame": start_f,
                    "label": self.label_map.get(code, 0)
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample["video_path"])
        
        # Seek once to the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["start_frame"])
        
        frames = []
        for i in range(self.seq_len):
            ret, frame = cap.read()
            
            if not ret:
                # Padding with zeros if frame read fails
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                frame = self.transform(Image.fromarray(frame))
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            frames.append(frame)

            # Skip frames efficiently using grab() instead of set()
            if i < self.seq_len - 1:
                for _ in range(self.frame_skip - 1):
                    cap.grab()
        
        cap.release()
        
        # Stack frames into [seq_len, 3, 224, 224]
        return torch.stack(frames), sample["label"]

def get_video_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_video_files_and_labels(data_root):
    """
    Scans directory and returns pairs of (video_path, label_code).
    """
    video_data = []
    # This logic matches the DatasetExplorer structure
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.avi'):
                path = os.path.join(root, file)
                # Label is suffix of parent folder or filename
                # Standard pattern: .../run_id/run_id.avi where run_id is xx-xx-xx-xxxx-LL
                run_id = os.path.splitext(file)[0]
                parts = run_id.split('-')
                label = parts[-1] if len(parts) > 1 else "00"
                video_data.append((path, label))
    return video_data
