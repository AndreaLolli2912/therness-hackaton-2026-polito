"""
Generate hackathon submission CSV from a trained 7-class video model.

Reads test_data/ samples, runs inference, and outputs:
  sample_id, pred_label_code, p_defect

Usage:
    python generate_submission.py \
        --checkpoint checkpoints/video_classifier.pth \
        --test_dir /path/to/test_data \
        --output predictions.csv
"""
import os
import csv
import argparse
import numpy as np

import torch
import cv2
from torchvision import transforms

from src.models.video_model import StreamingVideoClassifier


# ── Constants matching training ──────────────────────────────────
LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}

TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def derive_binary(probs):
    """p_defect = 1 - P(good_weld).  probs is a list/array of 7 class probs."""
    return 1.0 - float(probs[0])


def extract_frames(video_path, n_frames=40):
    """Uniformly sample n_frames from a video."""
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


def infer_sample(model, video_path, device, seq_len=15, frame_skip=5):
    """Run inference on a single video and return softmax probabilities."""
    frames = extract_frames(video_path, n_frames=seq_len * frame_skip)

    if not frames:
        # Return uniform distribution if video can't be read
        return np.ones(7) / 7.0

    # Take frames at the skip interval
    selected = frames[::frame_skip][:seq_len]

    # Pad if needed
    while len(selected) < seq_len:
        selected.append(np.zeros((224, 224, 3), dtype=np.uint8))

    # Transform and stack: [seq_len, 3, 224, 224]
    tensors = [TRANSFORM(f) for f in selected]
    sequence = torch.stack(tensors).unsqueeze(0).to(device)  # [1, T, 3, 224, 224]

    with torch.no_grad():
        logits, _ = model(sequence)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    return probs


def find_test_samples(test_dir):
    """
    Find test samples in test_data/ directory.
    Returns sorted list of (sample_id, video_path).
    """
    samples = []
    for entry in sorted(os.listdir(test_dir)):
        sample_dir = os.path.join(test_dir, entry)
        if not os.path.isdir(sample_dir):
            continue

        # Look for video file
        video_path = None
        for f in os.listdir(sample_dir):
            if f.endswith(('.avi', '.mp4')):
                video_path = os.path.join(sample_dir, f)
                break

        if video_path:
            samples.append((entry, video_path))
        else:
            print(f"WARNING: No video found in {sample_dir}")

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate hackathon submission CSV")
    parser.add_argument("--checkpoint", required=True, help="Path to model .pth")
    parser.add_argument("--test_dir", required=True, help="Path to test_data/")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--seq_len", type=int, default=15)
    parser.add_argument("--frame_skip", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature scaling factor for calibration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = StreamingVideoClassifier(
        num_classes=args.num_classes,
        hidden_size=args.hidden_size,
        pretrained=False,
    )
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {args.checkpoint}")

    # Find test samples
    samples = find_test_samples(args.test_dir)
    print(f"Found {len(samples)} test samples")

    # Run inference
    rows = []
    for i, (sample_id, video_path) in enumerate(samples):
        probs = infer_sample(model, video_path, device,
                             seq_len=args.seq_len, frame_skip=args.frame_skip)

        # Apply temperature scaling if specified
        if args.temperature != 1.0:
            logits = np.log(probs + 1e-10)
            scaled = logits / args.temperature
            exp_scaled = np.exp(scaled - np.max(scaled))
            probs = exp_scaled / exp_scaled.sum()

        pred_idx = int(np.argmax(probs))
        pred_code = LABEL_CODE_MAP[pred_idx]
        p_defect = derive_binary(probs)

        rows.append({
            "sample_id": sample_id,
            "pred_label_code": pred_code,
            "p_defect": f"{p_defect:.4f}",
        })

        print(f"  [{i+1}/{len(samples)}] {sample_id}: "
              f"pred={pred_code}, p_defect={p_defect:.4f}")

    # Write CSV
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_label_code", "p_defect"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSubmission saved to {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
