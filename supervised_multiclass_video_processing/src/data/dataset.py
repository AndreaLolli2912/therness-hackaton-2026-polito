"""
Multiclass dataset module for 7-class welding defect detection.

Provides two interfaces:
  1. build_video_index()       — returns VideoSample namedtuples for the
                                  frozen-embedding + sensor-fusion pipeline (main.py).
  2. WeldingSequenceDataset    — PyTorch Dataset for end-to-end MobileNetV3+GRU
     + get_video_files_and_labels()   training (train.py).

Label mapping (7 classes):
  Code  Index  Description
  00    0      good_weld
  01    1      excessive_penetration
  02    2      burn_through
  06    3      overlap
  07    4      lack_of_fusion
  08    5      excessive_convexity
  11    6      crater_cracks
"""

import os
import re
import cv2
import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ═══════════════════════════════════════════════════════════════════
# Shared constants
# ═══════════════════════════════════════════════════════════════════

VIDEO_EXTS = (".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg")
WELD_ID_RE = re.compile(r"^\d{2}-\d{2}-\d{2}-\d{4}-\d{2}$")  # e.g. 08-17-22-0011-00

# Multiclass label mapping: defect-code suffix → class index
LABEL_CODE_MAP = {"00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6}
CODE_TO_IDX = LABEL_CODE_MAP  # alias
IDX_TO_CODE = {v: k for k, v in LABEL_CODE_MAP.items()}


# ═══════════════════════════════════════════════════════════════════
# Part 1: VideoSample index builder (for main.py embedding-fusion pipeline)
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VideoSample:
    weld_id: str
    label: int            # 0–6 multiclass index
    label_code: str       # original code string ("00", "01", …)
    defect_type: str      # human-readable type
    weld_root: str
    video_path: Optional[str]


# Human-readable type names keyed by code suffix
_CODE_TO_TYPE = {
    "00": "good_weld",
    "01": "excessive_penetration",
    "02": "burn_through",
    "06": "overlap",
    "07": "lack_of_fusion",
    "08": "excessive_convexity",
    "11": "crater_cracks",
}


def _is_weld_id_folder(name: str) -> bool:
    return bool(WELD_ID_RE.match(name))


def _pick_video(videos: List[str], weld_id: str) -> Optional[str]:
    if not videos:
        return None
    w = weld_id.lower()
    hits = [p for p in videos if w in os.path.basename(p).lower()]
    if hits:
        hits.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
        return hits[0]
    videos.sort(key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))
    return videos[0]


def _find_videos_under(root: str) -> List[str]:
    vids = []
    for r, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(VIDEO_EXTS):
                vids.append(os.path.join(r, fn))
    return vids


def _extract_label_code(weld_id: str) -> str:
    """Extract the 2-digit defect code from a weld run ID like '04-03-23-0010-11'."""
    parts = weld_id.split("-")
    if len(parts) >= 5:
        return parts[-1]
    return "00"


def build_video_index(dataset_root: str) -> List[VideoSample]:
    """
    Traverse good_weld/ and defect_data_weld/ folders and return a list
    of VideoSample with 7-class multiclass labels (0–6).
    """
    dataset_root = os.path.abspath(dataset_root)
    good_root = os.path.join(dataset_root, "good_weld")

    # Try both naming conventions for the defect folder
    defect_root = os.path.join(dataset_root, "defect_data_weld")
    if not os.path.isdir(defect_root):
        defect_root = os.path.join(dataset_root, "defect-weld")
    if not os.path.isdir(defect_root):
        defect_root = os.path.join(dataset_root, "defect_weld")

    samples: List[VideoSample] = []

    # ── GOOD WELDS (label code "00" → index 0) ──────────────────
    if os.path.isdir(good_root):
        for group in sorted(os.listdir(good_root)):
            group_path = os.path.join(good_root, group)
            if not os.path.isdir(group_path):
                continue
            for weld_id in sorted(os.listdir(group_path)):
                weld_path = os.path.join(group_path, weld_id)
                if not os.path.isdir(weld_path) or not _is_weld_id_folder(weld_id):
                    continue
                videos = _find_videos_under(weld_path)
                unique_id = f"{group}/{weld_id}"
                samples.append(VideoSample(
                    weld_id=unique_id,
                    label=0,
                    label_code="00",
                    defect_type="good_weld",
                    weld_root=weld_path,
                    video_path=_pick_video(videos, weld_id),
                ))

    # ── DEFECT WELDS (label derived from run-ID suffix) ──────────
    if os.path.isdir(defect_root):
        for defect_group in sorted(os.listdir(defect_root)):
            defect_group_path = os.path.join(defect_root, defect_group)
            if not os.path.isdir(defect_group_path):
                continue
            for weld_id in sorted(os.listdir(defect_group_path)):
                weld_path = os.path.join(defect_group_path, weld_id)
                if not os.path.isdir(weld_path) or not _is_weld_id_folder(weld_id):
                    continue

                code = _extract_label_code(weld_id)
                label_idx = LABEL_CODE_MAP.get(code, None)
                if label_idx is None:
                    # Unknown defect code — skip to avoid silent mislabelling
                    print(f"WARNING: Unknown defect code '{code}' in {weld_id}, skipping.")
                    continue

                videos = _find_videos_under(weld_path)
                unique_id = f"{defect_group}/{weld_id}"
                samples.append(VideoSample(
                    weld_id=unique_id,
                    label=label_idx,
                    label_code=code,
                    defect_type=_CODE_TO_TYPE.get(code, "unknown"),
                    weld_root=weld_path,
                    video_path=_pick_video(videos, weld_id),
                ))

    return samples


# ═══════════════════════════════════════════════════════════════════
# Part 2: PyTorch Dataset for end-to-end MobileNetV3+GRU training
# ═══════════════════════════════════════════════════════════════════

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
                    "label": LABEL_CODE_MAP.get(code, 0)
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


def get_group_from_path(video_path: str) -> str:
    """
    Extract the configuration folder name as a group key for GroupShuffleSplit.
    
    Directory structure: .../good_weld/<config_folder>/<run_id>/<run_id>.avi
                     or: .../defect_data_weld/<config_folder>/<run_id>/<run_id>.avi
    
    The config folder (e.g., 'butt_Fe410_04-03-23') groups runs from the same
    welding setup. Splitting on this prevents data leakage.
    """
    # video_path → run_id folder → config_folder
    run_dir = os.path.dirname(video_path)
    config_dir = os.path.dirname(run_dir)
    return os.path.basename(config_dir)


def get_video_files_and_labels(data_root):
    """
    Scans directory and returns triples of (video_path, label_code, group).
    
    group is the configuration folder name, used for leakage-free splits
    via GroupShuffleSplit.
    """
    video_data = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.avi'):
                path = os.path.join(root, file)
                # Label is suffix of parent folder or filename
                # Standard pattern: .../run_id/run_id.avi where run_id is xx-xx-xx-xxxx-LL
                run_id = os.path.splitext(file)[0]
                parts = run_id.split('-')
                label = parts[-1] if len(parts) > 1 else "00"
                group = get_group_from_path(path)
                video_data.append((path, label, group))
    return video_data
