"""
Sliding-window video datasets for welding defect classification.

Mirrors the audio pipeline's chunking strategy:
  - WeldingWindowDataset: each item = one fixed-size window of frames (like AudioDataset chunks)
  - WeldingFileDataset:   each item = all windows from one video (like AudioFileDataset for MIL)
"""

import os
import cv2
import json
import torch
import numpy as np
import concurrent.futures
import subprocess
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
                 transform=None, data_root=None, load_features=False):
        """
        Args:
            video_paths: list of paths to .avi files
            labels:      list of label codes (e.g. '00', '01')
            window_size: number of frames per window
            window_stride: step between consecutive window starts
            transform:   torchvision transform for individual frames
            data_root:   optional root to save/load .video_metadata.json cache
            load_features: if True, load pre-computed .npy instead of video
        """
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform
        self.data_root = data_root
        self.load_features = load_features
        self.samples = []  # list of {video_path, start_frame, label}

        self.label_map = {
            "00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6
        }

        # ── Persistent Metadata Caching ──
        cache_path = os.path.join(data_root, ".video_metadata.json") if data_root else None
        cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        # ── Parallel Frame Counting & Sample Discovery ──
        to_scan = []
        for p, c in zip(video_paths, labels):
            rel_path = os.path.relpath(p, data_root) if data_root else p
            if rel_path in cache:
                # Add from cache immediately
                total_frames = cache[rel_path]
                if total_frames < window_size:
                    self.samples.append({
                        "video_path": p, "start_frame": 0,
                        "total_frames": total_frames, "label": self.label_map.get(c, 0),
                    })
                else:
                    for start_f in range(0, total_frames - window_size + 1, window_stride):
                        self.samples.append({
                            "video_path": p, "start_frame": start_f,
                            "total_frames": total_frames, "label": self.label_map.get(c, 0),
                        })
            else:
                to_scan.append((p, c))

        if to_scan:
            print(f"       Scanning {len(to_scan)} new videos (parallel)...")
            def process_one_video(item):
                path, code = item
                if not os.path.exists(path): return None, []
                
                # Fast frame count via ffprobe (skips slow cv2 header reading over network)
                total_frames = -1
                try:
                    cmd = [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_packets", "-show_entries", "stream=nb_read_packets",
                        "-of", "csv=p=0", path
                    ]
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
                    total_frames = int(output.decode('utf-8').strip())
                except Exception:
                    pass
                
                if total_frames <= 0:
                    # Fallback to OpenCV
                    cap = cv2.VideoCapture(path)
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                
                rel_p = os.path.relpath(path, data_root) if data_root else path
                
                video_samples = []
                if total_frames < window_size:
                    video_samples.append({
                        "video_path": path, "start_frame": 0,
                        "total_frames": total_frames, "label": self.label_map.get(code, 0),
                    })
                else:
                    for start_f in range(0, total_frames - window_size + 1, window_stride):
                        video_samples.append({
                            "video_path": path, "start_frame": start_f,
                            "total_frames": total_frames, "label": self.label_map.get(code, 0),
                        })
                return (rel_p, total_frames), video_samples

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(process_one_video, to_scan), 
                                   total=len(to_scan), desc="       Windexing", 
                                   leave=False, disable=len(to_scan) < 50))
            
            new_metadata = {}
            for meta, samples in results:
                if meta:
                    new_metadata[meta[0]] = meta[1]
                self.samples.extend(samples)
            
            # Update cache file
            if cache_path:
                cache.update(new_metadata)
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
                except Exception:
                    pass
        else:
            print(f"       Loaded {len(video_paths)} videos from cache.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Fast path: load pre-extracted 1D features
        if self.load_features and self.data_root:
            rel_path = os.path.relpath(sample["video_path"], self.data_root)
            npy_name = rel_path.replace(os.sep, "_").replace(".avi", ".npy")
            npy_path = os.path.join(self.data_root, "window_features", npy_name)
            
            try:
                # Shape: [Total_Frames, 576]
                feats = np.load(npy_path)
            except Exception:
                # Fallback to zeros (e.g. if extraction missing/corrupt)
                feats = np.zeros((sample["total_frames"] or self.window_size, 576), dtype=np.float32)
                
            start = sample["start_frame"]
            end = start + self.window_size
            
            if feats.shape[0] < self.window_size:
                pad_len = self.window_size - feats.shape[0]
                pad = np.zeros((pad_len, 576), dtype=np.float32)
                feats = np.vstack([feats, pad])
                
            window_feat = feats[start:end] # [window_size, 576]
            return torch.from_numpy(window_feat), sample["label"]

        # Slow path: extract CNN frames dynamically (slow over Google Drive)
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
                 transform=None, data_root=None, load_features=False):
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = transform
        self.data_root = data_root
        self.load_features = load_features

        self.label_map = {
            "00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6
        }

        self.files = []
        self.labels = []
        self._frame_counts = []

        # ── Persistent Metadata Caching ──
        cache_path = os.path.join(data_root, ".video_metadata.json") if data_root else None
        cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        to_scan = []
        for path, code in zip(video_paths, labels):
            if not os.path.exists(path): continue
            rel_path = os.path.relpath(path, data_root) if data_root else path
            
            if rel_path in cache:
                total = cache[rel_path]
                if total >= window_size:
                    self.files.append(path)
                    self.labels.append(self.label_map.get(code, 0))
                    self._frame_counts.append(total)
            else:
                to_scan.append((path, code))

        if to_scan:
            print(f"       Scanning {len(to_scan)} for file indices (parallel)...")
            def process_one(item):
                p, c = item
                
                total = -1
                try:
                    cmd = [
                        "ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-count_packets", "-show_entries", "stream=nb_read_packets",
                        "-of", "csv=p=0", p
                    ]
                    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
                    total = int(output.decode('utf-8').strip())
                except Exception:
                    pass
                    
                if total <= 0:
                    cap = cv2.VideoCapture(p)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                rel_p = os.path.relpath(p, data_root) if data_root else p
                return rel_p, total, c

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(executor.map(process_one, to_scan), 
                                   total=len(to_scan), desc="       Windexing", 
                                   leave=False, disable=len(to_scan) < 100))
            
            for rel_p, total, code in results:
                cache[rel_p] = total
                if total >= window_size:
                    self.files.append(os.path.join(data_root, rel_p) if data_root else rel_p)
                    self.labels.append(self.label_map.get(code, 0))
                    self._frame_counts.append(total)
            
            if cache_path:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
                except Exception: pass
        
        if not to_scan:
            print(f"       Loaded {len(self.files)} files from cache index.")

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
        
        if self.load_features and self.data_root:
            rel_path = os.path.relpath(path, self.data_root)
            npy_name = rel_path.replace(os.sep, "_").replace(".avi", ".npy")
            npy_path = os.path.join(self.data_root, "window_features", npy_name)
            
            try:
                feats = np.load(npy_path)
            except Exception:
                feats = np.zeros((total or self.window_size, 576), dtype=np.float32)
                
            all_windows = []
            for w in range(n_win):
                start = w * self.window_stride
                end = start + self.window_size
                
                # We do the padding inside the loop instead of modifying `feats` array to avoid accumulating pad in memory
                f_slice = feats[start:end]
                if f_slice.shape[0] < self.window_size:
                    pad_len = self.window_size - f_slice.shape[0]
                    pad = np.zeros((pad_len, 576), dtype=np.float32)
                    f_slice = np.vstack([f_slice, pad])
                    
                all_windows.append(torch.from_numpy(f_slice))
                
            windows_tensor = torch.stack(all_windows) # [n_win, window_size, 576]
            return {
                "windows": windows_tensor,
                "num_windows": n_win,
                "label": self.labels[idx],
            }

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
    target_dirs = ["good_weld", "defect-weld", "defect_data_weld"]
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
