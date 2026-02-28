"""
Inference for the sliding-window video classifier.

Slides windows across a video, classifies each independently, then
aggregates via soft-averaging (like audio chunk aggregation).
"""
import torch
import cv2
import numpy as np
import os
from src.models.video_model import WindowVideoClassifier
from src.data.dataset import get_video_transforms
from PIL import Image


LABEL_CODE_MAP = {0: "00", 1: "01", 2: "02", 3: "06", 4: "07", 5: "08", 6: "11"}
LABEL_NAMES = {
    "00": "Good Weld", "01": "Excessive Penetration", "02": "Burn Through",
    "06": "Overlap", "07": "Lack of Fusion", "08": "Excessive Convexity",
    "11": "Crater Cracks",
}


class WindowInference:
    """Run sliding-window inference on a single video."""

    def __init__(self, model_path, device='cpu', window_size=8, window_stride=4):
        self.device = torch.device(device)
        self.window_size = window_size
        self.window_stride = window_stride
        self.transform = get_video_transforms()

        self.model = WindowVideoClassifier(num_classes=7, pretrained=False)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device).eval()

    def predict_video(self, video_path):
        """
        Classify each window and soft-average across the video.

        Returns:
            avg_probs: (7,) numpy array of class probabilities
            per_window_probs: list of (7,) arrays for each window
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if total_frames < self.window_size:
            cap.release()
            return np.ones(7) / 7.0, []

        per_window_probs = []

        for start in range(0, total_frames - self.window_size + 1, self.window_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames = []
            for _ in range(self.window_size):
                ret, frame = cap.read()
                if not ret:
                    frame = np.zeros((224, 224, 3), dtype=np.uint8)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(self.transform(Image.fromarray(frame)))

            window = torch.stack(frames).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = self.model(window)
                probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            per_window_probs.append(probs)

        cap.release()

        if not per_window_probs:
            return np.ones(7) / 7.0, []

        avg_probs = np.mean(per_window_probs, axis=0)
        return avg_probs, per_window_probs


def run_inference_sample(sample_dir, model_path, window_size=8, window_stride=4):
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    if not os.path.exists(model_path):
        print("Model file not found. Please train the model first.")
        return

    engine = WindowInference(model_path, device, window_size, window_stride)

    video_file = None
    for f in os.listdir(sample_dir):
        if f.endswith('.avi'):
            video_file = os.path.join(sample_dir, f)

    if video_file:
        print(f"\nProcessing video: {video_file}")
        avg_probs, per_window = engine.predict_video(video_file)

        pred_idx = int(np.argmax(avg_probs))
        pred_code = LABEL_CODE_MAP[pred_idx]
        print(f"\nAggregated prediction over {len(per_window)} windows:")
        print(f"  Predicted: {pred_code} ({LABEL_NAMES.get(pred_code, '?')})")
        print(f"  Confidence: {avg_probs[pred_idx]:.4f}")
        print(f"  p_defect: {1.0 - avg_probs[0]:.4f}")
        print(f"  Full probs: {[f'{p:.4f}' for p in avg_probs]}")
    else:
        print(f"No .avi file found in {sample_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Window-based video inference")
    parser.add_argument("--sample_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="checkpoints/video_window_classifier.pth")
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--window_stride", type=int, default=4)
    args = parser.parse_args()

    run_inference_sample(args.sample_dir, args.model, args.window_size, args.window_stride)
