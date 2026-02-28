"""
Multiclass inference pipeline for welding defect detection.

Supports two modes:
  1. Single sample analysis (interactive debugging)
  2. Batch test-set inference → submission CSV generation

Submission CSV format:
  sample_id,pred_label_code,p_defect
  sample_0001,11,0.94
  sample_0002,00,0.08
"""
import torch
import cv2
import pandas as pd
import numpy as np
import os
from src.models.video_model import StreamingVideoClassifier
from src.data.dataset import get_video_transforms, LABEL_CODE_MAP, IDX_TO_CODE


# ── Constants ────────────────────────────────────────────────────
NUM_CLASSES = 7
LABEL_NAMES = [
    "good_weld", "excessive_penetration", "burn_through",
    "overlap", "lack_of_fusion", "excessive_convexity", "crater_cracks"
]


class WeldingInference:
    def __init__(self, video_model_path, device='cpu', num_classes=7, hidden_size=128):
        self.device = device
        self.num_classes = num_classes

        # Load Video Model
        self.video_model = StreamingVideoClassifier(
            num_classes=num_classes, hidden_size=hidden_size
        ).to(device)
        self.video_model.load_state_dict(
            torch.load(video_model_path, map_location=device, weights_only=True)
        )
        self.video_model.eval()
        self.video_transform = get_video_transforms()

    def predict_video_frame(self, frame):
        """
        Predict defect probabilities for a single BGR frame.
        Returns numpy array of shape [num_classes] with class probabilities.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.video_transform(frame_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits, _ = self.video_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.squeeze().cpu().numpy()

    def predict_video_sequence(self, video_path, n_frames=15, frame_skip=5):
        """
        Predict across a sequence of frames from a video file.
        Returns average probability distribution over the sampled sequence.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return np.zeros(self.num_classes, dtype=np.float32)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return np.zeros(self.num_classes, dtype=np.float32)

        # Uniformly sample frame indices
        window = n_frames * frame_skip
        start = max(0, (total_frames - window) // 2)  # center the window

        all_probs = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        for i in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            probs = self.predict_video_frame(frame)
            all_probs.append(probs)
            # Skip frames
            for _ in range(frame_skip - 1):
                cap.grab()

        cap.release()

        if not all_probs:
            return np.zeros(self.num_classes, dtype=np.float32)

        # Average probabilities across frames
        return np.mean(all_probs, axis=0)


def run_inference_sample(sample_dir, video_model_pth):
    """Run inference on a single sample directory and print results."""
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not os.path.exists(video_model_pth):
        print("Video model file not found. Please train the model first.")
        return

    inference = WeldingInference(video_model_pth, device=device)

    # Find video file in sample_dir
    video_file = None
    for f in os.listdir(sample_dir):
        if f.endswith('.avi'):
            video_file = os.path.join(sample_dir, f)
            break

    if video_file:
        print(f"\nProcessing video: {video_file}")
        probs = inference.predict_video_sequence(video_file)
        pred_idx = int(np.argmax(probs))
        pred_code = IDX_TO_CODE[pred_idx]
        pred_name = LABEL_NAMES[pred_idx]
        p_defect = 1.0 - probs[0]  # P(defect) = 1 - P(good_weld)

        print(f"\n  ── Prediction ──")
        print(f"  Predicted class: {pred_idx} (code {pred_code}, {pred_name})")
        print(f"  P(defect):       {p_defect:.4f}")
        print(f"  Confidence:      {probs[pred_idx]:.4f}")
        print(f"\n  ── Full probability distribution ──")
        for i, (name, prob) in enumerate(zip(LABEL_NAMES, probs)):
            code = IDX_TO_CODE[i]
            bar = "█" * int(prob * 40)
            print(f"    {code} {name:<26s} {prob:.4f} {bar}")
    else:
        print(f"No .avi video found in {sample_dir}")


def generate_submission_csv(test_dir, video_model_pth, output_csv="submission.csv"):
    """
    Generate hackathon submission CSV for all test samples.

    Expected directory structure:
      test_data/
        sample_0001/
          weld.avi
        sample_0002/
          weld.avi
        ...
        sample_0090/
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not os.path.exists(video_model_pth):
        print("ERROR: Video model file not found. Please train the model first.")
        return

    print(f"Loading model from {video_model_pth}...")
    inference = WeldingInference(video_model_pth, device=device)

    test_dir = os.path.abspath(test_dir)
    if not os.path.isdir(test_dir):
        print(f"ERROR: Test directory not found: {test_dir}")
        return

    # Discover test samples
    sample_dirs = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d)) and d.startswith("sample_")
    ])
    print(f"Found {len(sample_dirs)} test samples in {test_dir}")

    rows = []
    for i, sample_name in enumerate(sample_dirs):
        sample_path = os.path.join(test_dir, sample_name)
        print(f"[{i+1}/{len(sample_dirs)}] {sample_name}", end="\r")

        # Find video file
        video_files = [
            f for f in os.listdir(sample_path)
            if f.lower().endswith(('.avi', '.mp4', '.mov'))
        ]

        if video_files:
            video_path = os.path.join(sample_path, video_files[0])
            probs = inference.predict_video_sequence(video_path)
        else:
            probs = np.zeros(NUM_CLASSES, dtype=np.float32)
            probs[0] = 1.0  # default to good_weld if no video

        pred_idx = int(np.argmax(probs))
        pred_code = IDX_TO_CODE[pred_idx]
        p_defect = float(1.0 - probs[0])  # P(defect) = 1 - P(good_weld)

        rows.append({
            "sample_id": sample_name,
            "pred_label_code": pred_code,
            "p_defect": round(p_defect, 4)
        })

    # Write submission CSV
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"\n\nSubmission CSV saved to: {os.path.abspath(output_csv)}")
    print(f"Rows: {len(df)}")
    print(f"\nPrediction distribution:")
    for code, count in df['pred_label_code'].value_counts().sort_index().items():
        idx = LABEL_CODE_MAP.get(code, -1)
        name = LABEL_NAMES[idx] if 0 <= idx < len(LABEL_NAMES) else "unknown"
        print(f"  Code {code} ({name}): {count}")

    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multiclass welding defect inference")
    parser.add_argument("--sample_dir", type=str, default=None,
                        help="Path to a single sample directory to analyze")
    parser.add_argument("--video_model", type=str, default="checkpoints/video_classifier.pth",
                        help="Path to trained video model checkpoint")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Path to test_data/ directory for batch submission generation")
    parser.add_argument("--output_csv", type=str, default="submission.csv",
                        help="Output path for submission CSV")
    args = parser.parse_args()

    if args.test_dir:
        # Batch mode: generate submission CSV
        generate_submission_csv(args.test_dir, args.video_model, args.output_csv)
    elif args.sample_dir:
        # Single sample mode
        run_inference_sample(args.sample_dir, args.video_model)
    else:
        print("Usage:")
        print("  Single sample: python inference.py --sample_dir /path/to/sample --video_model checkpoint.pth")
        print("  Batch submit:  python inference.py --test_dir /path/to/test_data --video_model checkpoint.pth")
