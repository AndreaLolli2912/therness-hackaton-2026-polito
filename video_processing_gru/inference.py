import torch
import cv2
import pandas as pd
import numpy as np
import os
from src.models.video_model import StreamingVideoClassifier
from src.data.dataset import get_video_transforms

class WeldingInference:
    def __init__(self, video_model_path, device='cpu'):
        self.device = device
        
        # 1. Load Video Model
        self.video_model = StreamingVideoClassifier(num_classes=7).to(device)
        self.video_model.load_state_dict(torch.load(video_model_path, map_location=device))
        self.video_model.eval()
        
        self.video_transform = get_video_transforms()

    def predict_video_frame(self, frame):
        """
        Predict defect probabilities for a single BGR frame.
        """
        # frame: BGR from OpenCV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.video_transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, _ = self.video_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.squeeze().cpu().numpy()

def run_inference_sample(sample_dir, video_model_pth):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if not os.path.exists(video_model_pth):
        print("Video model file not found. Please train the model first.")
        return

    inference = WeldingInference(video_model_pth, device=device)
    
    # Find files in sample_dir
    video_file = None
    for f in os.listdir(sample_dir):
        if f.endswith('.avi'): video_file = os.path.join(sample_dir, f)
        
    if video_file:
        print(f"\nProcessing video: {video_file}")
        cap = cv2.VideoCapture(video_file)
        # Check first 5 frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret: break
            probs = inference.predict_video_frame(frame)
            # Find the most likely class
            class_idx = np.argmax(probs)
            score = probs[class_idx]
            print(f"Frame {i} - Predicted Class: {class_idx}, Score: {score:.6f}")
        cap.release()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a welding sample")
    parser.add_argument("--sample_dir", type=str, default=os.path.expanduser("~/Desktop/Hackathon/good_weld/config_1/run_1"),
                        help="Path to the sample directory to analyze")
    parser.add_argument("--video_model", type=str, default="checkpoints/video_classifier.pth")
    args = parser.parse_args()

    run_inference_sample(args.sample_dir, args.video_model)
