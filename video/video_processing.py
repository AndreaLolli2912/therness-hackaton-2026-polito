"""Video datasets and preprocessing for welding defect classification.

Datasets are moved from video_processing_window/src/data/dataset.py and
adapted to work with the new VideoCNNBackbone.
"""

import os
import cv2
import json
import re
import subprocess
import concurrent.futures

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# Pre-computed normalization constants (ImageNet stats) as (3,1,1) tensors
# used by the fast decode path to avoid PIL overhead
_NORM_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_NORM_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

from models.video_backbone import VideoCNNBackbone


# ── Label mapping (hackathon standard) ────────────────────────────
LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}
CODE_TO_IDX = {v: k for k, v in LABEL_CODE_MAP.items()}


# ── Video transforms ─────────────────────────────────────────────

def get_video_transforms(img_size: int = 224):
    """PIL-based transform pipeline (kept for external callers / legacy use)."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def _decode_frame_fast(frame_bgr: np.ndarray, img_size: int) -> torch.Tensor:
    """Resize + BGR→RGB + ToTensor + Normalize without PIL.

    ~2-3× faster than the PIL path for small target sizes.
    Returns: (3, img_size, img_size) float32 tensor.
    """
    frame = cv2.resize(frame_bgr, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
    t.sub_(_NORM_MEAN).div_(_NORM_STD)
    return t


# ── File discovery ────────────────────────────────────────────────

def get_group_from_path(video_path: str) -> str:
    """Extract config folder name as group key for GroupShuffleSplit."""
    run_dir = os.path.dirname(video_path)
    config_dir = os.path.dirname(run_dir)
    return os.path.basename(config_dir)


def get_video_files_and_labels(data_root):
    """Scan directory and return list of (video_path, label_code, group)."""
    video_data = []

    target_dirs = ["good_weld", "defect-weld", "defect_data_weld"]
    scan_roots = []
    for d in target_dirs:
        potential_path = os.path.join(data_root, d)
        if os.path.isdir(potential_path):
            scan_roots.append(potential_path)

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


def _count_frames_fast(path):
    """Count frames via ffprobe (fast), fallback to OpenCV."""
    total = -1
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-count_packets", "-show_entries", "stream=nb_read_packets",
            "-of", "csv=p=0", path
        ]
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5)
        total = int(output.decode('utf-8').strip())
    except Exception:
        pass

    if total <= 0:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    return total


# ── Datasets ──────────────────────────────────────────────────────

class WeldingWindowDataset(Dataset):
    """Sliding-window video dataset.

    Each item: (frames_tensor, label)
        frames_tensor: (window_size, 3, 224, 224)
        label:         int  (0-6)
    """

    def __init__(self, video_paths, labels, window_size=8, window_stride=4,
                 img_size=160, transform=None, data_root=None):
        self.window_size = window_size
        self.window_stride = window_stride
        self.img_size = img_size
        self.transform = transform  # kept for backward compat; fast path is preferred
        self.data_root = data_root
        self.samples = []

        self.label_map = CODE_TO_IDX

        # ── Persistent metadata cache ──
        cache_path = os.path.join(data_root, ".video_metadata.json") if data_root else None
        cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        to_scan = []
        for p, c in zip(video_paths, labels):
            rel_path = os.path.relpath(p, data_root) if data_root else p
            if rel_path in cache:
                total_frames = cache[rel_path]
                if total_frames < window_size:
                    self.samples.append({
                        "video_path": p, "start_frame": 0,
                        "total_frames": total_frames,
                        "label": self.label_map.get(c, 0),
                    })
                else:
                    for start_f in range(0, total_frames - window_size + 1, window_stride):
                        self.samples.append({
                            "video_path": p, "start_frame": start_f,
                            "total_frames": total_frames,
                            "label": self.label_map.get(c, 0),
                        })
            else:
                to_scan.append((p, c))

        if to_scan:
            print(f"       Scanning {len(to_scan)} new videos (parallel)...")

            def process_one(item):
                path, code = item
                if not os.path.exists(path):
                    return None, []
                total_frames = _count_frames_fast(path)
                rel_p = os.path.relpath(path, data_root) if data_root else path
                samples = []
                if total_frames < window_size:
                    samples.append({
                        "video_path": path, "start_frame": 0,
                        "total_frames": total_frames,
                        "label": self.label_map.get(code, 0),
                    })
                else:
                    for start_f in range(0, total_frames - window_size + 1, window_stride):
                        samples.append({
                            "video_path": path, "start_frame": start_f,
                            "total_frames": total_frames,
                            "label": self.label_map.get(code, 0),
                        })
                return (rel_p, total_frames), samples

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(
                    executor.map(process_one, to_scan),
                    total=len(to_scan), desc="       Windexing",
                    leave=False, disable=len(to_scan) < 50,
                ))

            new_metadata = {}
            for meta, samples in results:
                if meta:
                    new_metadata[meta[0]] = meta[1]
                self.samples.extend(samples)

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
        cap = cv2.VideoCapture(sample["video_path"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample["start_frame"])

        frames = []
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        for _ in range(self.window_size):
            ret, frame = cap.read()
            if not ret:
                frame = blank
            if self.transform:
                # Legacy PIL path (used if caller passes a custom transform)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else frame
                t = self.transform(Image.fromarray(frame_rgb))
            else:
                t = _decode_frame_fast(frame, self.img_size)
            frames.append(t)

        cap.release()
        return torch.stack(frames), sample["label"]


class WeldingFileDataset(Dataset):
    """File-level dataset for MIL-style video training.

    Each item returns ALL windows from a single video file.
    """

    def __init__(self, video_paths, labels, window_size=8, window_stride=4,
                 img_size=160, transform=None, data_root=None):
        self.window_size = window_size
        self.window_stride = window_stride
        self.img_size = img_size
        self.transform = transform  # kept for backward compat
        self.data_root = data_root
        self.label_map = CODE_TO_IDX

        self.files = []
        self.labels = []
        self._frame_counts = []

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
            if not os.path.exists(path):
                continue
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
                total = _count_frames_fast(p)
                rel_p = os.path.relpath(p, data_root) if data_root else p
                return rel_p, total, c

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                results = list(tqdm(
                    executor.map(process_one, to_scan),
                    total=len(to_scan), desc="       Windexing",
                    leave=False, disable=len(to_scan) < 100,
                ))

            for rel_p, total, code in results:
                cache[rel_p] = total
                if total >= self.window_size:
                    self.files.append(
                        os.path.join(data_root, rel_p) if data_root else rel_p
                    )
                    self.labels.append(self.label_map.get(code, 0))
                    self._frame_counts.append(total)

            if cache_path:
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
                except Exception:
                    pass
        elif not to_scan:
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

        cap = cv2.VideoCapture(path)
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        all_windows = []

        for w in range(n_win):
            start = w * self.window_stride
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames = []
            for _ in range(self.window_size):
                ret, frame = cap.read()
                if not ret:
                    frame = blank
                if self.transform:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else frame
                    t = self.transform(Image.fromarray(frame_rgb))
                else:
                    t = _decode_frame_fast(frame, self.img_size)
                frames.append(t)
            all_windows.append(torch.stack(frames))

        cap.release()
        windows_tensor = torch.stack(all_windows)
        return {
            "windows": windows_tensor,
            "num_windows": n_win,
            "label": self.labels[idx],
        }


# ── File-level dataset (uniform frame sampling) ───────────────────

class WeldingVideoDataset(Dataset):
    """One item per video file — no sliding windows.

    Samples `num_frames` frames from either:
      - the full video duration (default), or
      - the first `clip_seconds` seconds (fast path, audio-aligned).

    Each item: (frames_tensor, label)
        frames_tensor: (num_frames, 3, img_size, img_size)
        label:         int (0-6)
    """

    def __init__(self, video_paths, labels, num_frames=16, img_size=160,
                 clip_seconds=None, data_root=None):
        self.num_frames = num_frames
        self.img_size   = img_size
        self.clip_seconds = clip_seconds
        self.label_map  = CODE_TO_IDX

        # Reuse the frame-count metadata cache
        cache_path = os.path.join(data_root, ".video_metadata.json") if data_root else None
        cache = {}
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    cache = json.load(f)
            except Exception:
                cache = {}

        to_scan = []
        self.samples = []   # list of {video_path, total_frames, label}
        for p, c in zip(video_paths, labels):
            if not os.path.exists(p):
                continue
            rel = os.path.relpath(p, data_root) if data_root else p
            if rel in cache:
                self.samples.append({
                    "video_path": p,
                    "total_frames": cache[rel],
                    "label": self.label_map.get(c, 0),
                })
            else:
                to_scan.append((p, c))

        if to_scan:
            print(f"       Scanning {len(to_scan)} new videos (parallel)...")

            def _scan(item):
                p, c = item
                total = _count_frames_fast(p)
                rel = os.path.relpath(p, data_root) if data_root else p
                return rel, total, c

            with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
                results = list(tqdm(
                    ex.map(_scan, to_scan), total=len(to_scan),
                    desc="       Scanning", leave=False, disable=len(to_scan) < 50,
                ))

            new_meta = {}
            for rel, total, c in results:
                new_meta[rel] = total
                p = os.path.join(data_root, rel) if data_root else rel
                self.samples.append({
                    "video_path": p,
                    "total_frames": total,
                    "label": self.label_map.get(c, 0),
                })

            if cache_path:
                cache.update(new_meta)
                try:
                    with open(cache_path, 'w') as f:
                        json.dump(cache, f)
                except Exception:
                    pass
        else:
            print(f"       Loaded {len(self.samples)} videos from cache.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        total = max(s["total_frames"], 1)
        cap = cv2.VideoCapture(s["video_path"])
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)

        end_frame = total
        if self.clip_seconds is not None:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0
            clip_frames = max(1, int(round(float(self.clip_seconds) * float(fps))))
            end_frame = min(total, clip_frames)

        # Uniform frame indices inside [0, end_frame)
        indices = np.linspace(0, end_frame - 1, self.num_frames, dtype=int)

        # Seek directly to each sampled frame — faster than sequential decode
        # when frames are widely spaced (e.g. 8 frames across 1000-frame video)
        frames = []
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ret, frame = cap.read()
            frames.append(_decode_frame_fast(frame if ret else blank, self.img_size))

        while len(frames) < self.num_frames:
            frames.append(_decode_frame_fast(blank, self.img_size))

        cap.release()

        return torch.stack(frames), s["label"]


# ── Pre-extracted JPEG dataset ────────────────────────────────────

class WeldingFrameDataset(Dataset):
    """Fast dataset that loads pre-extracted JPEG frames from disk.

    Run video/extract_frames.py once to populate frames_dir, then use this
    dataset for all training/validation — no video decoding, just JPEG reads.

    Each item: (frames_tensor, label)
        frames_tensor: (num_frames, 3, img_size, img_size)  float32, ImageNet-normalized
        label:         int (0-6)
    """

    def __init__(self, video_paths, labels, frames_dir, num_frames=8, img_size=160,
                 data_root=None):
        self.num_frames = num_frames
        self.img_size = img_size
        self.label_map = CODE_TO_IDX

        manifest_path = os.path.join(frames_dir, "manifest.json")
        with open(manifest_path) as f:
            manifest = json.load(f)

        def _norm(p: str) -> str:
            return os.path.normpath(p).replace('\\', '/')

        def _run_id(path: str) -> str:
            return os.path.splitext(os.path.basename(path))[0]

        schema_v2 = isinstance(manifest, dict) and ("schema_version" in manifest or "entries" in manifest)

        by_abs = {}
        by_rel = {}
        by_run_id = {}
        if schema_v2:
            by_abs = manifest.get("by_video_path", {}) or {}
            by_rel = manifest.get("by_rel_video_path", {}) or {}
            by_run_id = manifest.get("by_run_id", {}) or {}

        self.samples = []
        missing = 0
        for p, c in zip(video_paths, labels):
            entry = None
            if schema_v2:
                abs_key = _norm(os.path.abspath(p))
                rel_key = _norm(os.path.relpath(p, data_root)) if data_root else _norm(p)
                rid = _run_id(p)
                entry = by_abs.get(abs_key) or by_rel.get(rel_key) or by_run_id.get(rid)
            else:
                entry = manifest.get(p)

            if entry is None:
                missing += 1
                continue
            frame_paths = entry["frames"]
            # Subsample or pad to exactly num_frames
            if len(frame_paths) >= num_frames:
                indices = np.linspace(0, len(frame_paths) - 1, num_frames, dtype=int)
                frame_paths = [frame_paths[i] for i in indices]
            self.samples.append({
                "frames": frame_paths,
                "label": self.label_map.get(c, 0),
            })
        if missing:
            print(f"       WeldingFrameDataset: {missing} videos not in manifest (skipped)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        blank = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        frames = []
        for fp in s["frames"]:
            img = cv2.imread(fp)
            if img is None:
                img = blank
            # JPEGs are already img_size×img_size; skip resize in _decode_frame_fast
            # by converting BGR→RGB + normalizing directly
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
            t.sub_(_NORM_MEAN).div_(_NORM_STD)
            frames.append(t)
        while len(frames) < self.num_frames:
            frames.append(_decode_frame_fast(blank, self.img_size))
        return torch.stack(frames), s["label"]


# ── Model wrapper ────────────────────────────────────────────────

class WeldVideoModel(nn.Module):
    """End-to-end video model wrapper (mirrors WeldBackboneModel for audio).

    forward(frames)                        → (B, num_classes) logits
    forward(frames, return_features=True)  → dict with all layer activations
    forward_features(frames)               → (B, 128) embedding
    """

    def __init__(self, backbone, transform=None):
        super().__init__()
        self.backbone = backbone
        # Note: transforms are applied in the dataset, not here
        # (video transforms are CPU-side PIL/numpy ops, not differentiable)

    def forward(self, frames, return_features=False):
        return self.backbone(frames, return_features=return_features)

    def forward_features(self, frames):
        return self.backbone.extract_features(frames)

    def forward_per_frame_features(self, frames):
        """Return (B, N, 128) per-frame embeddings before mean-pooling."""
        return self.backbone.extract_per_frame_features(frames)
