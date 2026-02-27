"""
Stream processors for video, audio, and sensor data.

Each processor reads from a source (file or hardware) and produces
inference-ready tensors. All preprocessing is self-contained.
"""

import time
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torchaudio
from torchvision import transforms

from .models import (
    StreamingVideoClassifier, AudioCNN, AudioTransform,
    load_video_model, load_audio_model,
)
from .config import InferenceConfig


# ─────────────────────────────────────────────────────────────────
# Video Stream
# ─────────────────────────────────────────────────────────────────

def _get_video_transforms():
    """ImageNet-standard preprocessing (must match training)."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class VideoStreamProcessor:
    """
    Captures frames from an AVI file or camera and runs them through
    the MobileNetV3 + GRU model, maintaining GRU hidden state across
    successive frames for temporal awareness.
    """

    def __init__(self, model: StreamingVideoClassifier,
                 source: str, device: torch.device):
        self.model = model
        self.device = device
        self.transform = _get_video_transforms()
        self.hidden_state: Optional[torch.Tensor] = None

        # Source can be a camera index ("0") or file path
        src = int(source) if source.isdigit() else source
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_idx = 0
        print(f"[video] Opened source: {source}  "
              f"({self.frame_count} frames, {self.fps:.1f} fps)")

    def get_next(self) -> Optional[np.ndarray]:
        """
        Read next frame, run inference, return 7-class probabilities.
        Returns None when the stream is exhausted.
        """
        ret, frame_bgr = self.cap.read()
        if not ret:
            return None

        self._frame_idx += 1
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, self.hidden_state = self.model.predict_stream(
                tensor.squeeze(0), self.hidden_state
            )
        return probs.squeeze(0).cpu().numpy()

    @property
    def frame_idx(self) -> int:
        return self._frame_idx

    def reset(self):
        """Reset stream to beginning (for benchmark loops)."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._frame_idx = 0
        self.hidden_state = None

    def close(self):
        self.cap.release()


# ─────────────────────────────────────────────────────────────────
# Audio Stream
# ─────────────────────────────────────────────────────────────────

class AudioStreamProcessor:
    """
    Reads 0.5s audio chunks from a FLAC file and runs them through
    AudioTransform → AudioCNN, returning per-chunk class probabilities.
    """

    def __init__(self, model: AudioCNN, audio_transform: AudioTransform,
                 source: str, device: torch.device, cfg: InferenceConfig):
        self.model = model
        self.audio_transform = audio_transform.to(device)
        self.device = device
        self.cfg = cfg

        # Load full waveform once (industrial FLAC files are ~38s)
        waveform, sr = torchaudio.load(source)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != cfg.audio_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, cfg.audio_sample_rate)
            waveform = resampler(waveform)

        self.waveform = waveform  # (1, total_samples)
        self.chunk_samples = int(cfg.audio_chunk_length_s * cfg.audio_sample_rate)
        self.total_chunks = self.waveform.shape[1] // self.chunk_samples
        self._chunk_idx = 0

        print(f"[audio] Loaded source: {source}  "
              f"({self.total_chunks} chunks of {cfg.audio_chunk_length_s}s)")

    def get_next(self) -> Optional[np.ndarray]:
        """
        Get next 0.5s chunk, run through AudioTransform + AudioCNN.
        Returns 7-class probabilities or None when exhausted.
        """
        if self._chunk_idx >= self.total_chunks:
            return None

        start = self._chunk_idx * self.chunk_samples
        end = start + self.chunk_samples
        chunk = self.waveform[:, start:end].unsqueeze(0).to(self.device)
        self._chunk_idx += 1

        with torch.no_grad():
            mel = self.audio_transform(chunk)
            logits = self.model(mel)
            probs = torch.softmax(logits, dim=1)

        return probs.squeeze(0).cpu().numpy()

    @property
    def chunk_idx(self) -> int:
        return self._chunk_idx

    def reset(self):
        self._chunk_idx = 0

    def close(self):
        pass  # no resources to release


# ─────────────────────────────────────────────────────────────────
# Sensor Stream
# ─────────────────────────────────────────────────────────────────

class SensorStreamProcessor:
    """
    Reads sensor CSV rows and computes the same 42-dim feature vector
    used by the supervised_binary_video_processing pipeline.
    Simulates streaming by yielding one row-window at a time.
    """

    SENSOR_COLS = slice(3, 9)  # Pressure, CO2, Feed, Current, Wire, Voltage
    FEATURES_PER_CHANNEL = 7   # mean, std, max, min, p10, p90, max_diff
    TOTAL_FEATURES = 6 * 7     # 42

    def __init__(self, source: str):
        import pandas as pd
        self.df = pd.read_csv(source)
        self.sensors = self.df.iloc[:, self.SENSOR_COLS].values.astype(np.float32)
        self._row_idx = 0
        self._window = 50  # rows per inference window

        print(f"[sensor] Loaded source: {source}  "
              f"({len(self.df)} rows, {self.sensors.shape[1]} channels)")

    def get_next(self) -> Optional[np.ndarray]:
        """
        Returns 42-dim feature vector from the next window of sensor rows.
        Returns None when exhausted.
        """
        if self._row_idx >= len(self.sensors):
            return None

        end = min(self._row_idx + self._window, len(self.sensors))
        window = self.sensors[self._row_idx:end]
        self._row_idx = end

        return self._extract_features(window)

    def _extract_features(self, sensors: np.ndarray) -> np.ndarray:
        """Compute 42-dim aggregated stats (matches offline pipeline)."""
        if len(sensors) == 0:
            return np.zeros(self.TOTAL_FEATURES, dtype=np.float32)

        feats = []
        for j in range(sensors.shape[1]):
            col = sensors[:, j]
            all_nan = np.all(np.isnan(col))

            c_mean = 0.0 if all_nan else float(np.nanmean(col))
            c_std  = 0.0 if all_nan else float(np.nanstd(col))
            c_max  = 0.0 if all_nan else float(np.nanmax(col))
            c_min  = 0.0 if all_nan else float(np.nanmin(col))

            valid = col[~np.isnan(col)]
            if len(valid) > 0:
                p10, p90 = np.percentile(valid, [10, 90])
                max_diff = (float(np.max(np.abs(np.diff(valid))))
                            if len(valid) > 1 else 0.0)
            else:
                p10, p90, max_diff = 0.0, 0.0, 0.0

            feats.extend([c_mean, c_std, c_max, c_min, p10, p90, max_diff])

        return np.array(feats, dtype=np.float32)

    def reset(self):
        self._row_idx = 0

    def close(self):
        pass
