"""
Latency benchmarking for edge deployment compliance.

Measures inference latency for video model, audio model, and fusion pipeline.
Reports mean, p50, p95, p99 latency and PASS/FAIL against target thresholds.

Usage:
    # Benchmark video model against 50ms target
    python benchmark_latency.py \
        --video_checkpoint checkpoints/video_classifier.pth \
        --target_ms 50

    # Full pipeline benchmark
    python benchmark_latency.py \
        --video_checkpoint checkpoints/video_classifier.pth \
        --audio_checkpoint checkpoints/audio/audio_model.pth \
        --target_ms 50 \
        --output benchmark_results.json
"""
import os
import sys
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.models.video_model import StreamingVideoClassifier


class AudioCNN(nn.Module):
    """Inline AudioCNN for self-contained benchmarking."""
    def __init__(self, num_classes=7, dropout=0.3):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(), nn.MaxPool2d(2), nn.Dropout2d(dropout),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(dropout), nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.head(self.block3(self.block2(self.block1(x))))


def get_model_size_mb(model):
    """Get model size in MB (parameters only)."""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / 1024 / 1024


def benchmark_model(model, dummy_input, n_warmup=20, n_runs=100, device='cpu'):
    """
    Benchmark inference latency.
    Returns dict with timing statistics in milliseconds.
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy_input)

    # Timed runs
    if device == 'cuda':
        torch.cuda.synchronize()

    timings = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == 'cuda':
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(dummy_input)

            if device == 'cuda':
                torch.cuda.synchronize()

            end = time.perf_counter()
            timings.append((end - start) * 1000)  # ms

    timings = np.array(timings)
    return {
        "mean_ms": float(np.mean(timings)),
        "std_ms": float(np.std(timings)),
        "p50_ms": float(np.percentile(timings, 50)),
        "p95_ms": float(np.percentile(timings, 95)),
        "p99_ms": float(np.percentile(timings, 99)),
        "min_ms": float(np.min(timings)),
        "max_ms": float(np.max(timings)),
        "n_runs": n_runs,
    }


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def main():
    parser = argparse.ArgumentParser(description="Latency benchmarking")
    parser.add_argument("--video_checkpoint", help="Video model checkpoint")
    parser.add_argument("--audio_checkpoint", help="Audio model checkpoint")
    parser.add_argument("--target_ms", type=float, default=50.0,
                        help="Latency target in ms")
    parser.add_argument("--n_warmup", type=int, default=20)
    parser.add_argument("--n_runs", type=int, default=100)
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional output JSON path")
    parser.add_argument("--cpu_only", action="store_true",
                        help="Force CPU benchmarking")
    args = parser.parse_args()

    if args.cpu_only:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Benchmarking on: {device}")
    print(f"Target latency: {args.target_ms} ms")
    print(f"Warmup: {args.n_warmup}, Runs: {args.n_runs}")
    print("=" * 60)

    results = {"device": str(device), "target_ms": args.target_ms}
    total_mean = 0.0

    # ── Video Model ──────────────────────────────────────────────
    if args.video_checkpoint:
        print("\n── Video Model (StreamingVideoClassifier) ──")
        model = StreamingVideoClassifier(
            num_classes=args.num_classes,
            hidden_size=args.hidden_size,
            pretrained=False,
        )
        if os.path.exists(args.video_checkpoint):
            state = torch.load(args.video_checkpoint, map_location=device,
                               weights_only=True)
            model.load_state_dict(state)
        model.to(device).eval()

        n_params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        print(f"  Parameters: {n_params:,}")
        print(f"  Model size: {size_mb:.2f} MB")

        # Single frame input: [1, 1, 3, 224, 224]
        dummy = torch.randn(1, 1, 3, 224, 224).to(device)

        # Wrap forward to handle tuple return
        class VideoWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                logits, _ = self.m(x)
                return logits

        wrapper = VideoWrapper(model).to(device)
        timing = benchmark_model(wrapper, dummy, args.n_warmup, args.n_runs, str(device))

        passed = timing["p95_ms"] <= args.target_ms
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"  Mean:  {timing['mean_ms']:.2f} ms")
        print(f"  P50:   {timing['p50_ms']:.2f} ms")
        print(f"  P95:   {timing['p95_ms']:.2f} ms")
        print(f"  P99:   {timing['p99_ms']:.2f} ms")
        print(f"  Result: {status} (p95={timing['p95_ms']:.2f} vs target={args.target_ms})")

        results["video"] = {
            "parameters": n_params,
            "size_mb": round(size_mb, 2),
            "latency": timing,
            "passed": passed,
        }
        total_mean += timing["mean_ms"]

    # ── Audio Model ──────────────────────────────────────────────
    if args.audio_checkpoint:
        print("\n── Audio Model (AudioCNN) ──")
        model = AudioCNN(num_classes=args.num_classes)
        if os.path.exists(args.audio_checkpoint):
            state = torch.load(args.audio_checkpoint, map_location=device,
                               weights_only=True)
            model.load_state_dict(state)
        model.to(device).eval()

        n_params = count_parameters(model)
        size_mb = get_model_size_mb(model)
        print(f"  Parameters: {n_params:,}")
        print(f"  Model size: {size_mb:.2f} MB")

        # Mel spectrogram input: [1, 1, 40, 24]
        dummy = torch.randn(1, 1, 40, 24).to(device)
        timing = benchmark_model(model, dummy, args.n_warmup, args.n_runs, str(device))

        passed = timing["p95_ms"] <= args.target_ms
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"  Mean:  {timing['mean_ms']:.2f} ms")
        print(f"  P50:   {timing['p50_ms']:.2f} ms")
        print(f"  P95:   {timing['p95_ms']:.2f} ms")
        print(f"  P99:   {timing['p99_ms']:.2f} ms")
        print(f"  Result: {status} (p95={timing['p95_ms']:.2f} vs target={args.target_ms})")

        results["audio"] = {
            "parameters": n_params,
            "size_mb": round(size_mb, 2),
            "latency": timing,
            "passed": passed,
        }
        total_mean += timing["mean_ms"]

    # ── Combined Summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Total estimated pipeline latency (mean): {total_mean:.2f} ms")
    combined_pass = total_mean <= args.target_ms
    print(f"Combined: {'✅ PASS' if combined_pass else '❌ FAIL'} "
          f"(mean={total_mean:.2f} vs target={args.target_ms})")

    results["combined_mean_ms"] = round(total_mean, 2)
    results["combined_passed"] = combined_pass

    # ── Memory Summary ───────────────────────────────────────────
    if device.type == 'cuda':
        mem_alloc = torch.cuda.max_memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.max_memory_reserved() / 1024 / 1024
        print(f"\nGPU Memory:")
        print(f"  Allocated: {mem_alloc:.1f} MB")
        print(f"  Reserved:  {mem_reserved:.1f} MB")
        results["gpu_memory_allocated_mb"] = round(mem_alloc, 1)
        results["gpu_memory_reserved_mb"] = round(mem_reserved, 1)

    # ── Save results ─────────────────────────────────────────────
    if args.output:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
