"""
Sliding-window video datasets for welding defect classification.

Mirrors the audio pipeline's chunking strategy:
  - WeldingWindowDataset: each item = one fixed-size window of frames (like AudioDataset chunks)
  - WeldingFileDataset:   each item = all windows from one video (like AudioFileDataset for MIL)
"""

import os
import cv2
import torch
import numpy as np
import concurrent.futures
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class WeldingWindowDataset(Dataset):
    """
    Slides a fixed-size window across every video and yields individual
    windows as training samples.  Analogous to AudioDataset which chunks
    audio into fixed-length waveforms.

    Each item: (frames_tensor, label)
        frames_tensor: (window_size, 3, 224, 224)
        label:         int  (0-6)
    """

    def __init__(self, video_paths, labels, window_size=8, window_stride=4,
                 transform=None):
        """
        Args:
            video_paths: list of paths to .avi files
            labels:      list of label codes (e.g. '00', '01')
            window_size: number of frames per window
            window_stride: step between consecutive window starts
            transform:   torchvision transform for individual frames
        """
        self.window_size = window_size
        self.transform = transform
        self.samples = []  # list of {video_path, start_frame, label}

        self.label_map = {
            "00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6
        }

        # ── Parallel Frame Counting & Sample Discovery ──
        print(f"       Checking {len(video_paths)} videos for windows (parallel)...")
        
        def process_one_video(item):
            path, code = item
            if not os.path.exists(path):
                return []
            
            cap = cv2.VideoCapture(path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            video_samples = []
            if total_frames < window_size:
                video_samples.append({
                    "video_path": path,
                    "start_frame": 0,
                    "total_frames": total_frames,
                    "label": self.label_map.get(code, 0),
                })
            else:
                for start_f in range(0, total_frames - window_size + 1, window_stride):
                    video_samples.append({
                        "video_path": path,
                        "start_frame": start_f,
                        "total_frames": total_frames,
                        "label": self.label_map.get(code, 0),
                    })
            return video_samples

        # Use ThreadPoolExecutor for I/O bound tasks (opening/closing files)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            video_items = list(zip(video_paths, labels))
            results = list(tqdm(executor.map(process_one_video, video_items), 
                               total=len(video_items), desc="       Windexing", 
                               leave=False, disable=len(video_items) < 100))
        
        for res in results:
            self.samples.extend(res)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["start_frame"])

        frames = []
        for _ in range(self.window_size):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.transform:
                frame = self.transform(Image.fromarray(frame))
            else:
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            frames.append(frame)

        cap.release()
        return torch.stack(frames), sample["label"]


class WeldingFileDataset(Dataset):
    """
    File-level dataset for MIL-style training/inference.
    Each item returns ALL windows from a single video file.

    Mirrors AudioFileDataset which returns all chunks from one FLAC file.
    """

    def __init__(self, video_paths, labels, window_size=8, window_stride=4,
                 transform=None):
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform

        self.label_map = {
            "00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6
        }

        self.files = []
        self.labels = []
        self._frame_counts = []

        for path, code in zip(video_paths, labels):
            if not os.path.exists(path):
                continue
            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if total < window_size:
                continue
            self.files.append(path)
            self.labels.append(self.label_map.get(code, 0))
            self._frame_counts.append(total)

    def __len__(self):
        return len(self.files)

    def _num_windows(self, total_frames):
        if total_frames < self.window_size:
            return 1
        return (total_frames - self.window_size) // self.window_stride + 1

    def __getitem__(self, idx):
        path = self.files[idx]
        total = self._frame_counts[idx]
        n_win = self._num_windows(total)

        cap = cv2.VideoCapture(path)
        all_windows = []

        for w in range(n_win):
            start = w * self.window_stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames = []
            for _ in range(self.window_size):
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if self.transform:
                    frame = self.transform(Image.fromarray(frame))
                else:
                    frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

                frames.append(frame)

            all_windows.append(torch.stack(frames))  # (window_size, 3, H, W)

        cap.release()

        windows_tensor = torch.stack(all_windows)  # (n_win, window_size, 3, H, W)
        return {
            "windows": windows_tensor,
            "num_windows": n_win,
            "label": self.labels[idx],
        }


# ── Shared utilities ─────────────────────────────────────────────

def get_video_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_group_from_path(video_path: str) -> str:
    """
    Extract config folder name as group key for GroupShuffleSplit.

    Structure: .../good_weld/<config_folder>/<run_id>/<run_id>.avi
           or: .../defect_data_weld/<config_folder>/<run_id>/<run_id>.avi
    """
    run_dir = os.path.dirname(video_path)
    config_dir = os.path.dirname(run_dir)
    return os.path.basename(config_dir)


def get_video_files_and_labels(data_root):
    """
    Scan directory and return list of (video_path, label_code, group).
    Optimized for Google Drive/Network mounts by scanning only labeled roots.
    """
    video_data = []
    
    # Priority folders to scan
    target_dirs = ["good_weld", "defect_data_weld"]
    scan_roots = []
    
    for d in target_dirs:
        potential_path = os.path.join(data_root, d)
        if os.path.isdir(potential_path):
            scan_roots.append(potential_path)
    
    # If target dirs don't exist, fallback to scanning entire data_root
    if not scan_roots:
        scan_roots = [data_root]

    for root_dir in scan_roots:
        print(f"       Scanning {os.path.basename(root_dir)}...")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith('.avi'):
                    path = os.path.join(root, file)
                    run_id = os.path.splitext(file)[0]
                    parts = run_id.split('-')
                    label = parts[-1] if len(parts) > 1 else "00"
                    group = get_group_from_path(path)
                    video_data.append((path, label, group))
                    
    return video_data
