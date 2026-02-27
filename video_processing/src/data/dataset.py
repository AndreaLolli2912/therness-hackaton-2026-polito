import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WeldingVideoDataset(Dataset):
    def __init__(self, data_root, transform=None, frame_skip=5):
        self.data_root = data_root
        self.transform = transform
        self.frame_skip = frame_skip
        self.frame_paths = []
        
        # Discover all .avi files
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.avi'):
                    video_path = os.path.join(root, file)
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
            # Return a blank frame or handle error
            frame = torch.zeros((3, 224, 224))
            return frame
        
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            frame = self.transform(frame)
        else:
            # Default transform if none provided
            frame = transforms.ToTensor()(frame)
            frame = transforms.Resize((224, 224))(frame)
            
        return frame

def get_transforms():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Removed Normalize to match Sigmoid output [0, 1] in decoder
    ])
