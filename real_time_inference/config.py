"""
Runtime configuration for the real-time welding defect inference pipeline.
All settings needed to run inference without touching training code.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


# 7-class welding defect label mapping (matches training code)
LABEL_MAP: Dict[int, str] = {
    0: "good_weld",
    1: "excessive_penetration",
    2: "burn_through",
    3: "overlap",
    4: "lack_of_fusion",
    5: "excessive_convexity",
    6: "crater_cracks",
}

LABEL_CODE_MAP: Dict[int, str] = {
    0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11",
}


@dataclass
class InferenceConfig:
    """All runtime parameters for the real-time pipeline."""

    # ── Model checkpoints ────────────────────────────────────────
    video_checkpoint: Optional[str] = None
    audio_checkpoint: Optional[str] = None
    fusion_checkpoint: Optional[str] = None  # optional attention weights

    # ── Device ───────────────────────────────────────────────────
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    # ── Input sources ────────────────────────────────────────────
    video_input: Optional[str] = None   # path to .avi or camera index ("0")
    audio_input: Optional[str] = None   # path to .flac file
    sensor_input: Optional[str] = None  # path to .csv file

    # ── Model hyperparameters (must match training) ──────────────
    num_classes: int = 7
    video_hidden_size: int = 128
    video_window_size: int = 8  # Frames per window for non-GRU models
    audio_dropout: float = 0.3

    # ── Audio preprocessing (must match training) ────────────────
    audio_sample_rate: int = 16000
    audio_chunk_length_s: float = 0.5  # 0.5s chunks → 8000 samples
    audio_n_fft: int = 1024
    audio_n_mels: int = 40
    audio_f_min: int = 0
    audio_f_max: int = 8000

    # ── Fusion defaults (used when no attention checkpoint) ──────
    fusion_weights: Dict[str, float] = field(
        default_factory=lambda: {"video": 0.6, "audio": 0.3, "sensor": 0.1}
    )

    # ── Performance ──────────────────────────────────────────────
    latency_target_ms: float = 50.0      # Strict target per paper (<50ms)
    latency_target_relaxed_ms: float = 200.0  # Relaxed target for full pipeline
    benchmark_frames: int = 100

    # ── Edge deployment ──────────────────────────────────────────
    quantized: bool = False              # Use INT8 quantized model
    onnx_path: Optional[str] = None      # Path to ONNX model (if using ORT)

    # ── Label mappings ───────────────────────────────────────────
    label_map: Dict[int, str] = field(default_factory=lambda: dict(LABEL_MAP))
    label_code_map: Dict[int, str] = field(default_factory=lambda: dict(LABEL_CODE_MAP))
