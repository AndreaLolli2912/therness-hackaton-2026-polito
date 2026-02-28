"""
Generate hackathon submission CSV from the sliding-window video classifier.

For each test sample, slides windows across the video, classifies each
independently, and soft-averages to produce a final prediction.

Output:  sample_id, pred_label_code, p_defect

Usage:
    python generate_submission.py \
        --checkpoint checkpoints/video_window_classifier.pth \
        --test_dir /path/to/test_data \
        --output predictions.csv
"""
import os
import csv
import argparse
import numpy as np

import torch
import cv2
from PIL import Image
from torchvision import transforms

from src.models.video_model import WindowVideoClassifier


LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def infer_video_windows(model, video_path, device, window_size=8, window_stride=4):
    """Slide windows, classify each, return soft-averaged probs."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total < window_size:
        cap.release()
        return np.ones(7) / 7.0

    all_probs = []
    for start in range(0, total - window_size + 1, window_stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames = []
        for _ in range(window_size):
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(TRANSFORM(Image.fromarray(frame)))

        window = torch.stack(frames).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(window)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        all_probs.append(probs)

    cap.release()
    return np.mean(all_probs, axis=0) if all_probs else np.ones(7) / 7.0


def find_test_samples(test_dir):
    samples = []
    for entry in sorted(os.listdir(test_dir)):
        sample_dir = os.path.join(test_dir, entry)
        if not os.path.isdir(sample_dir):
            continue
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
    parser = argparse.ArgumentParser(description="Generate hackathon submission (window model)")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--output", default="predictions.csv")
    parser.add_argument("--num_classes", type=int, default=7)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WindowVideoClassifier(num_classes=args.num_classes, pretrained=False)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded model from {args.checkpoint}")

    samples = find_test_samples(args.test_dir)
    print(f"Found {len(samples)} test samples")

    rows = []
    for i, (sample_id, video_path) in enumerate(samples):
        probs = infer_video_windows(
            model, video_path, device,
            window_size=args.window_size, window_stride=args.window_stride,
        )

        if args.temperature != 1.0:
            logits = np.log(probs + 1e-10)
            scaled = logits / args.temperature
            exp_scaled = np.exp(scaled - np.max(scaled))
            probs = exp_scaled / exp_scaled.sum()

        pred_idx = int(np.argmax(probs))
        pred_code = LABEL_CODE_MAP[pred_idx]
        p_defect = 1.0 - float(probs[0])

        rows.append({
            "sample_id": sample_id,
            "pred_label_code": pred_code,
            "p_defect": f"{p_defect:.4f}",
        })
        print(f"  [{i+1}/{len(samples)}] {sample_id}: pred={pred_code}, p_defect={p_defect:.4f}")

    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "pred_label_code", "p_defect"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSubmission saved to {args.output} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
