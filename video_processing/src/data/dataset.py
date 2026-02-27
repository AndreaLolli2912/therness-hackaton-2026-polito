import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WeldingVideoDataset(Dataset):
    def __init__(self, video_files, transform=None, frame_skip=10):
        """
        Args:
            video_files (list): List of paths to .avi files.
            transform (callable, optional): Transform to be applied on a frame.
            frame_skip (int): Number of frames to skip.
        """
        self.transform = transform
        self.frame_skip = frame_skip
        self.frame_paths = []
        
        for video_path in video_files:
            if not os.path.exists(video_path):
                print(f"Warning: Video file {video_path} not found.")
                continue
            self._extract_frame_info(video_path)

    def _extract_frame_info(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in range(0, frame_count, self.frame_skip):
            self.frame_paths.append((video_path, i))
        cap.release()

    def __len__(self):
        return len(self.frame_paths)

    def __getitem__(self, idx):
        video_path, frame_idx = self.frame_paths[idx]
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback for failed frame read
            return torch.zeros((3, 224, 224))
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)
        else:
            # Default minimal transform
            frame = transforms.ToTensor()(frame)
            frame = transforms.Resize((224, 224))(frame)
            
        return frame

def get_video_transforms():
    """
    Standard transforms for MobileNetV3 encoder.
    Includes ImageNet normalization to match pretrained weights.
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

def get_video_files(data_root):
    """Utility to find all AVI files."""
    video_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.avi'):
                video_files.append(os.path.join(root, file))
    return sorted(video_files)
