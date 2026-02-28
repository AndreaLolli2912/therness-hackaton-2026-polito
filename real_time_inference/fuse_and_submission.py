"""
Multimodal submission generator — fuses video (.pth) and audio (.pt) models.

Iterates over test_data/ samples, runs both models, fuses their probability
vectors, and writes a hackathon-format submission CSV:
    sample_id, pred_label_code, p_defect

Two fusion strategies are supported via --fusion_mode:
  • "weighted"   — fixed-weight average (configurable via --video_weight / --audio_weight)
  • "attention"  — learned ModalityAttention network (requires --fusion_checkpoint)

Usage
-----
    # Weighted average (default 0.6 video / 0.4 audio)
    python -m real_time_inference.fuse_and_submission `
        --video_checkpoint checkpoints/video/video_classifier_gru_full_heron_f1binary93_f1multi81_h88.pth `
        --audio_checkpoint checkpoints/audio/deploy_multiclass.pt `
        --test_dir test_data `
        --output submission_fused.csv

    # Custom weights
    python -m real_time_inference.fuse_and_submission `
        --video_checkpoint checkpoints/video/video_classifier_gru_full_heron_f1binary93_f1multi81_h88.pth `
        --audio_checkpoint checkpoints/audio/deploy_multiclass.pt `
        --test_dir test_data `
        --output submission_fused.csv `
        --video_weight 0.7 --audio_weight 0.3

    # Attention fusion (requires a trained fusion checkpoint)
    python -m real_time_inference.fuse_and_submission `
        --video_checkpoint checkpoints/video/video_classifier_gru_full_heron_f1binary93_f1multi81_h88.pth `
        --audio_checkpoint checkpoints/audio/deploy_multiclass.pt `
        --test_dir test_data `
        --output submission_fused.csv `
        --fusion_mode attention `
        --fusion_checkpoint checkpoints/fusion_attention.pth
"""

import argparse
import csv
import os
import sys
import time

import cv2
import numpy as np
import torch
from torchvision import transforms

from .config import LABEL_CODE_MAP
from .models import load_video_model, load_audio_model
from .fusion import AttentionFusionEngine


# ─────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────

NUM_CLASSES = 7

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────
# Video inference helpers
# ─────────────────────────────────────────────────────────────────

def extract_frames(video_path: str, n_frames: int = 40):
    """Uniformly sample n_frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()
    return frames


def infer_video(model, video_path: str, device: torch.device,
                seq_len: int = 15, frame_skip: int = 5) -> np.ndarray:
    """Run video model on a sample, return 7-class softmax probabilities."""
    frames = extract_frames(video_path, n_frames=seq_len * frame_skip)

    if not frames:
        return np.ones(NUM_CLASSES) / NUM_CLASSES

    selected = frames[::frame_skip][:seq_len]

    # Pad if shorter than expected
    while len(selected) < seq_len:
        selected.append(np.zeros((224, 224, 3), dtype=np.uint8))

    tensors = [TRANSFORM(f) for f in selected]
    sequence = torch.stack(tensors).unsqueeze(0).to(device)  # [1, T, 3, 224, 224]

    use_amp = device.type == 'cuda'
    from torch.amp import autocast
    with torch.no_grad():
        if use_amp:
            with autocast('cuda'):
                logits, _ = model(sequence)
        else:
            logits, _ = model(sequence)
            
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return probs


# ─────────────────────────────────────────────────────────────────
# Audio inference helpers
# ─────────────────────────────────────────────────────────────────

def load_audio_waveform(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load a FLAC file and return waveform tensor (1, N)."""
    import soundfile as sf

    data, sr = sf.read(audio_path)
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono

    waveform = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

    if sr != target_sr:
        import torchaudio
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)

    return waveform


def infer_audio(model, audio_path: str, device: torch.device,
                is_torchscript: bool) -> np.ndarray:
    """Run audio model on a sample, return 7-class probabilities."""
    waveform = load_audio_waveform(audio_path)  # always on CPU

    with torch.no_grad():
        if is_torchscript:
            # TorchScript deploy models have internal params on CPU,
            # so keep waveform on CPU regardless of device arg
            result = model(waveform.cpu())

            if isinstance(result, dict):
                if "probs" in result:
                    # DeployMulticlassFile: {"label": int, "probs": float[7]}
                    probs = result["probs"].cpu().numpy()
                    return probs.astype(np.float32)
                elif "p_defect" in result:
                    # DeploySingleLabelMIL: {"label": int, "p_defect": float}
                    # Convert binary output to 7-class distribution
                    p_defect = float(result["p_defect"])
                    p_good = 1.0 - p_defect
                    probs = np.zeros(NUM_CLASSES, dtype=np.float32)
                    probs[0] = p_good
                    probs[1:] = p_defect / (NUM_CLASSES - 1)
                    return probs
            else:
                # Tensor output
                probs = torch.softmax(result, dim=-1)
                if probs.dim() > 1 and probs.shape[0] > 1:
                    probs = probs.mean(dim=0, keepdim=True)
                return probs.squeeze(0).cpu().numpy()
        else:
            # Raw AudioCNN — would need AudioTransform preprocessing
            # For submission, TorchScript deploy models are preferred
            raise NotImplementedError(
                "Raw AudioCNN inference not supported in submission mode. "
                "Use a TorchScript deploy .pt model instead."
            )

    return np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES


# ─────────────────────────────────────────────────────────────────
# Sample discovery
# ─────────────────────────────────────────────────────────────────

def find_test_samples(test_dir: str):
    """
    Find test samples in test_data/ directory.
    Returns sorted list of (sample_id, video_path, audio_path).
    audio_path may be None if no FLAC file is found.
    """
    samples = []
    for entry in sorted(os.listdir(test_dir)):
        sample_dir = os.path.join(test_dir, entry)
        if not os.path.isdir(sample_dir):
            continue

        video_path = None
        audio_path = None
        for f in os.listdir(sample_dir):
            fl = f.lower()
            if fl.endswith(('.avi', '.mp4')) and video_path is None:
                video_path = os.path.join(sample_dir, f)
            elif fl.endswith('.flac') and audio_path is None:
                audio_path = os.path.join(sample_dir, f)

        if video_path or audio_path:
            samples.append((entry, video_path, audio_path))
        else:
            print(f"  WARNING: No video or audio found in {sample_dir}")

    return samples


# ─────────────────────────────────────────────────────────────────
# Fusion helpers
# ─────────────────────────────────────────────────────────────────

def fuse_weighted(video_probs, audio_probs, video_weight, audio_weight):
    """Fixed-weight average fusion."""
    available = {}
    weights = {}

    if video_probs is not None:
        available["video"] = video_probs
        weights["video"] = video_weight
    if audio_probs is not None:
        available["audio"] = audio_probs
        weights["audio"] = audio_weight

    if not available:
        return np.ones(NUM_CLASSES) / NUM_CLASSES, weights

    # Renormalize weights to sum to 1
    total_w = sum(weights.values())
    weights = {k: v / total_w for k, v in weights.items()}

    fused = np.zeros(NUM_CLASSES, dtype=np.float64)
    for modality, probs in available.items():
        fused += weights[modality] * probs

    # Normalize to valid distribution
    total_p = fused.sum()
    if total_p > 0:
        fused /= total_p

    return fused.astype(np.float32), weights


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate multimodal submission CSV (video + audio fusion)"
    )

    # Models
    parser.add_argument("--video_checkpoint", type=str, required=True,
                        help="Path to video model .pth")
    parser.add_argument("--audio_checkpoint", type=str, required=True,
                        help="Path to audio deploy .pt")

    # Inputs
    parser.add_argument("--test_dir", type=str, default="test_data",
                        help="Path to test_data/ directory")
    parser.add_argument("--output", type=str, default="submission_fused.csv",
                        help="Output CSV path")

    # Fusion
    parser.add_argument("--fusion_mode", type=str, default="weighted",
                        choices=["weighted", "attention"],
                        help="Fusion strategy: 'weighted' or 'attention'")
    parser.add_argument("--video_weight", type=float, default=0.6,
                        help="Video weight for weighted fusion (default: 0.6)")
    parser.add_argument("--audio_weight", type=float, default=0.4,
                        help="Audio weight for weighted fusion (default: 0.4)")
    parser.add_argument("--fusion_checkpoint", type=str, default=None,
                        help="Path to trained attention fusion checkpoint "
                             "(required for --fusion_mode attention)")

    # Video model params
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--frame_skip", type=int, default=5)

    # Temperature scaling
    parser.add_argument("--video_temperature", type=float, default=1.0,
                        help="Temperature scaling for video model probabilities")
    parser.add_argument("--audio_temperature", type=float, default=1.0,
                        help="Temperature scaling for audio model probabilities")

    # Device (default cpu — TorchScript audio deploy models are saved on CPU)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["auto", "cpu", "cuda", "mps"])

    args = parser.parse_args()

    # ── Resolve device ───────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # ── Validate ─────────────────────────────────────────────────
    if args.fusion_mode == "attention" and args.fusion_checkpoint is None:
        print("ERROR: --fusion_checkpoint is required when using "
              "--fusion_mode attention")
        sys.exit(1)

    if not os.path.exists(args.test_dir):
        print(f"ERROR: Test directory not found: {args.test_dir}")
        sys.exit(1)

    for ckpt_name, ckpt_path in [("Video", args.video_checkpoint),
                                   ("Audio", args.audio_checkpoint)]:
        if not os.path.exists(ckpt_path):
            print(f"ERROR: {ckpt_name} checkpoint not found: {ckpt_path}")
            sys.exit(1)

    # ── Print header ─────────────────────────────────────────────
    print("=" * 64)
    print("  Multimodal Submission Generator")
    print("  Video + Audio Fusion Pipeline")
    print("=" * 64)
    print(f"  Device:            {device}")
    print(f"  Fusion mode:       {args.fusion_mode}")
    if args.fusion_mode == "weighted":
        print(f"  Video weight:      {args.video_weight}")
        print(f"  Audio weight:      {args.audio_weight}")
    else:
        print(f"  Fusion checkpoint: {args.fusion_checkpoint}")
    print(f"  Video checkpoint:  {args.video_checkpoint}")
    print(f"  Audio checkpoint:  {args.audio_checkpoint}")
    print(f"  Test directory:    {args.test_dir}")
    print(f"  Output:            {args.output}")
    print("=" * 64)

    # ── Load models ──────────────────────────────────────────────
    print("\n[1/4] Loading models...")

    video_model = load_video_model(
        args.video_checkpoint, device,
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
    )

    audio_model = load_audio_model(
        args.audio_checkpoint, device,
        num_classes=args.num_classes,
    )
    audio_is_torchscript = isinstance(audio_model, torch.jit.ScriptModule)
    print(f"  Audio model type: {'TorchScript' if audio_is_torchscript else 'state_dict'}")

    # ── Setup fusion engine (for attention mode) ─────────────────
    fusion_engine = None
    if args.fusion_mode == "attention":
        fusion_engine = AttentionFusionEngine(
            num_classes=args.num_classes,
            fusion_checkpoint=args.fusion_checkpoint,
            device=device,
        )

    # ── Discover test samples ────────────────────────────────────
    print("\n[2/4] Scanning test directory...")
    samples = find_test_samples(args.test_dir)
    print(f"  Found {len(samples)} test samples")

    samples_with_video = sum(1 for _, v, _ in samples if v is not None)
    samples_with_audio = sum(1 for _, _, a in samples if a is not None)
    samples_both = sum(1 for _, v, a in samples if v is not None and a is not None)
    print(f"  With video: {samples_with_video}  |  "
          f"With audio: {samples_with_audio}  |  "
          f"Both: {samples_both}")

    # ── Run inference ────────────────────────────────────────────
    print("\n[3/4] Running multimodal inference...")
    rows = []
    t_start = time.perf_counter()

    for i, (sample_id, video_path, audio_path) in enumerate(samples):
        # Video inference
        video_probs = None
        if video_path:
            video_probs = infer_video(
                video_model, video_path, device,
                seq_len=args.seq_len, frame_skip=args.frame_skip,
            )
            # Apply temperature scaling
            if args.video_temperature != 1.0:
                logits = np.log(video_probs + 1e-10)
                scaled = logits / args.video_temperature
                exp_scaled = np.exp(scaled - np.max(scaled))
                video_probs = exp_scaled / exp_scaled.sum()

        # Audio inference
        audio_probs = None
        if audio_path:
            try:
                audio_probs = infer_audio(
                    audio_model, audio_path, device, audio_is_torchscript
                )
                # Apply temperature scaling
                if args.audio_temperature != 1.0:
                    logits = np.log(audio_probs + 1e-10)
                    scaled = logits / args.audio_temperature
                    exp_scaled = np.exp(scaled - np.max(scaled))
                    audio_probs = exp_scaled / exp_scaled.sum()
            except Exception as e:
                print(f"  WARNING [{sample_id}]: Audio inference failed: {e}")

        # Fuse
        if args.fusion_mode == "attention" and fusion_engine is not None:
            result = fusion_engine.fuse(video_probs, audio_probs)
            fused_probs = np.zeros(NUM_CLASSES, dtype=np.float32)
            for modality_probs in result.per_modality_probs.values():
                w = result.attention_weights.get("video", 0.5)
                # Reconstruct fused from result
            # Simpler: get predicted_idx and confidence from result
            pred_idx = result.predicted_idx
            pred_code = result.predicted_code
            # Derive p_defect from per-modality probs
            # Reconstruct the full fused distribution
            available_probs = {}
            if video_probs is not None:
                available_probs["video"] = video_probs
            if audio_probs is not None:
                available_probs["audio"] = audio_probs
            fused = np.zeros(NUM_CLASSES, dtype=np.float64)
            for mod, probs in available_probs.items():
                fused += result.attention_weights.get(mod, 0.5) * probs
            total_p = fused.sum()
            if total_p > 0:
                fused /= total_p
            p_defect = 1.0 - float(fused[0])
            attn_str = " | ".join(f"{k}={v:.3f}"
                                   for k, v in result.attention_weights.items())
        else:
            fused, weights = fuse_weighted(
                video_probs, audio_probs,
                args.video_weight, args.audio_weight,
            )
            pred_idx = int(np.argmax(fused))
            pred_code = LABEL_CODE_MAP[pred_idx]
            p_defect = 1.0 - float(fused[0])
            attn_str = " | ".join(f"{k}={v:.3f}" for k, v in weights.items())

        rows.append({
            "sample_id": sample_id,
            "pred_label_code": pred_code,
            "p_defect": f"{p_defect:.4f}",
        })

        # Per-sample status
        modalities = []
        if video_probs is not None:
            modalities.append("V")
        if audio_probs is not None:
            modalities.append("A")
        mod_str = "+".join(modalities) if modalities else "?"

        print(f"  [{i+1:3d}/{len(samples)}] {sample_id}: "
              f"pred={pred_code}  p_defect={p_defect:.4f}  "
              f"[{mod_str}]  ({attn_str})")

    elapsed = time.perf_counter() - t_start

    # ── Write CSV ────────────────────────────────────────────────
    print(f"\n[4/4] Writing submission CSV...")
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sample_id", "pred_label_code", "p_defect"]
        )
        writer.writeheader()
        writer.writerows(rows)

    # ── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  Submission Summary")
    print("=" * 64)
    print(f"  Total samples:     {len(rows)}")
    print(f"  Output file:       {args.output}")
    print(f"  Elapsed time:      {elapsed:.1f}s "
          f"({elapsed/max(len(rows),1):.2f}s/sample)")

    # Distribution summary
    from collections import Counter
    code_counts = Counter(r["pred_label_code"] for r in rows)
    print(f"\n  Label distribution:")
    for code in sorted(code_counts.keys()):
        count = code_counts[code]
        pct = 100 * count / len(rows)
        # Reverse-lookup label name
        label_name = "?"
        for idx, c in LABEL_CODE_MAP.items():
            if c == code:
                from .config import LABEL_MAP
                label_name = LABEL_MAP.get(idx, "?")
                break
        print(f"    {code} ({label_name:24s}): {count:4d} ({pct:5.1f}%)")

    defect_probs = [float(r["p_defect"]) for r in rows]
    print(f"\n  p_defect stats:")
    print(f"    Mean:   {np.mean(defect_probs):.4f}")
    print(f"    Median: {np.median(defect_probs):.4f}")
    print(f"    Min:    {np.min(defect_probs):.4f}")
    print(f"    Max:    {np.max(defect_probs):.4f}")
    print("=" * 64)


if __name__ == "__main__":
    main()
