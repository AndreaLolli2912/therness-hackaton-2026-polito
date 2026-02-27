"""
Real-time multimodal welding defect inference.

Entry point for the self-contained real_time_inference package.
Runs video, audio, and sensor streams through MobileNetV3+GRU,
AudioCNN, and sensor feature extraction, then fuses predictions
via an attention-based late-fusion engine.

Usage
-----
    # Full multimodal inference
    python -m real_time_inference.main \
        --video_checkpoint checkpoints/video_classifier.pth \
        --audio_checkpoint checkpoints/audio/best_model.pt  \
        --video_input  sampleData/sample_0001/weld.avi      \
        --audio_input  sampleData/sample_0001/weld.flac     \
        --sensor_input sampleData/sample_0001/sensor.csv

    # Video only
    python -m real_time_inference.main \
        --video_checkpoint checkpoints/video_classifier.pth \
        --video_input sampleData/sample_0001/weld.avi

    # Benchmark mode (latency profiling)
    python -m real_time_inference.main \
        --video_checkpoint checkpoints/video_classifier.pth \
        --video_input sampleData/sample_0001/weld.avi       \
        --benchmark
"""

import argparse
import sys
import time
from typing import Optional

import numpy as np
import torch

from .config import InferenceConfig
from .models import (
    AudioTransform, load_audio_model, load_video_model,
)
from .streams import (
    AudioStreamProcessor, SensorStreamProcessor, VideoStreamProcessor,
)
from .fusion import AttentionFusionEngine


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _print_header():
    print("=" * 64)
    print("  Real-Time Welding Defect Inference Pipeline")
    print("  MobileNetV3-Small + GRU (video) | AudioCNN (audio)")
    print("  Attention-based late fusion (Zhang et al. 2026)")
    print("=" * 64)


def _print_result(cycle: int, result, latency_ms: float, target_ms: float):
    warn = " ⚠ SLOW" if latency_ms > target_ms else ""
    attn_str = "  ".join(f"{k}={v:.2f}" for k, v in result.attention_weights.items())

    print(f"[{cycle:04d}]  "
          f"{result.predicted_label:<24s}  "
          f"(code={result.predicted_code})  "
          f"conf={result.confidence:.3f}  "
          f"attn=[{attn_str}]  "
          f"{latency_ms:6.1f}ms{warn}")


# ─────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────

def run(cfg: InferenceConfig):
    """Main inference loop."""
    device = _resolve_device(cfg.device)
    print(f"\n[main] Device: {device}")

    # ── Load models ──────────────────────────────────────────────
    video_model = None
    if cfg.video_checkpoint and cfg.video_input:
        video_model = load_video_model(
            cfg.video_checkpoint, device,
            num_classes=cfg.num_classes,
            hidden_size=cfg.video_hidden_size,
        )

    audio_model = None
    audio_transform = None
    if cfg.audio_checkpoint and cfg.audio_input:
        audio_model = load_audio_model(
            cfg.audio_checkpoint, device,
            num_classes=cfg.num_classes,
            dropout=cfg.audio_dropout,
        )
        audio_transform = AudioTransform(
            sample_rate=cfg.audio_sample_rate,
            chunk_length_s=cfg.audio_chunk_length_s,
            n_fft=cfg.audio_n_fft,
            n_mels=cfg.audio_n_mels,
            f_min=cfg.audio_f_min,
            f_max=cfg.audio_f_max,
        )

    if video_model is None and audio_model is None:
        print("[main] ERROR: No models loaded. Provide at least one "
              "checkpoint + input pair.")
        sys.exit(1)

    # ── Open streams ─────────────────────────────────────────────
    video_stream: Optional[VideoStreamProcessor] = None
    audio_stream: Optional[AudioStreamProcessor] = None
    sensor_stream: Optional[SensorStreamProcessor] = None

    if video_model and cfg.video_input:
        video_stream = VideoStreamProcessor(video_model, cfg.video_input, device)

    if audio_model and audio_transform and cfg.audio_input:
        audio_stream = AudioStreamProcessor(
            audio_model, audio_transform, cfg.audio_input, device, cfg
        )

    if cfg.sensor_input:
        sensor_stream = SensorStreamProcessor(cfg.sensor_input)

    # ── Fusion engine ────────────────────────────────────────────
    fusion = AttentionFusionEngine(
        num_classes=cfg.num_classes,
        label_map=cfg.label_map,
        label_code_map=cfg.label_code_map,
        fallback_weights=cfg.fusion_weights,
        fusion_checkpoint=cfg.fusion_checkpoint,
        device=device,
    )

    # ── Inference loop ───────────────────────────────────────────
    print("\n[main] Starting inference loop (Ctrl+C to stop)\n")
    latencies = []
    cycle = 0

    try:
        while True:
            t0 = time.perf_counter()

            # Get next data from each stream
            video_probs = video_stream.get_next() if video_stream else None
            audio_probs = audio_stream.get_next() if audio_stream else None
            sensor_feats = sensor_stream.get_next() if sensor_stream else None

            # All streams exhausted?
            if video_probs is None and audio_probs is None and sensor_feats is None:
                print("\n[main] All input streams exhausted.")
                break

            # Fuse predictions
            result = fusion.fuse(video_probs, audio_probs, sensor_feats)

            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            cycle += 1

            _print_result(cycle, result, latency_ms, cfg.latency_target_ms)

    except KeyboardInterrupt:
        print("\n\n[main] Interrupted by user.")

    finally:
        # Cleanup
        if video_stream:
            video_stream.close()
        if audio_stream:
            audio_stream.close()
        if sensor_stream:
            sensor_stream.close()

    # ── Summary ──────────────────────────────────────────────────
    if latencies:
        arr = np.array(latencies)
        print("\n" + "=" * 64)
        print("  Latency Summary")
        print("=" * 64)
        print(f"  Cycles:  {len(arr)}")
        print(f"  Mean:    {arr.mean():.1f} ms")
        print(f"  Median:  {np.median(arr):.1f} ms")
        print(f"  P95:     {np.percentile(arr, 95):.1f} ms")
        print(f"  P99:     {np.percentile(arr, 99):.1f} ms")
        print(f"  Max:     {arr.max():.1f} ms")
        target_met = (arr <= cfg.latency_target_ms).sum()
        print(f"  Within {cfg.latency_target_ms}ms target: "
              f"{target_met}/{len(arr)} ({100*target_met/len(arr):.1f}%)")
        print("=" * 64)


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def parse_args() -> InferenceConfig:
    p = argparse.ArgumentParser(
        description="Real-time welding defect inference pipeline"
    )
    # Checkpoints
    p.add_argument("--video_checkpoint", type=str, default=None)
    p.add_argument("--audio_checkpoint", type=str, default=None)
    p.add_argument("--fusion_checkpoint", type=str, default=None)

    # Inputs
    p.add_argument("--video_input", type=str, default=None,
                   help="Path to .avi file or camera index (e.g. '0')")
    p.add_argument("--audio_input", type=str, default=None,
                   help="Path to .flac audio file")
    p.add_argument("--sensor_input", type=str, default=None,
                   help="Path to sensor .csv file")

    # Device
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda", "mps"])

    # Benchmark mode
    p.add_argument("--benchmark", action="store_true",
                   help="Profile latency on sample data then exit")

    args = p.parse_args()

    cfg = InferenceConfig(
        video_checkpoint=args.video_checkpoint,
        audio_checkpoint=args.audio_checkpoint,
        fusion_checkpoint=args.fusion_checkpoint,
        video_input=args.video_input,
        audio_input=args.audio_input,
        sensor_input=args.sensor_input,
        device=args.device,
    )
    return cfg


def main():
    _print_header()
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
