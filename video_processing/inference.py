import torch
import cv2
import pandas as pd
import numpy as np
import joblib
import os
from src.models.video_model import StreamingVideoClassifier
from src.models.sensor_model import SensorClassifier
from src.data.dataset import get_video_transforms

class WeldingInference:
    def __init__(self, video_model_path, sensor_model_path, scaler_path, device='cpu'):
        self.device = device
        
        # 1. Load Video Model
        self.video_model = StreamingVideoClassifier(num_classes=7).to(device)
        self.video_model.load_state_dict(torch.load(video_model_path, map_location=device))
        self.video_model.eval()
        
        # 2. Load Sensor Model
        self.sensor_model = SensorClassifier(input_size=6, hidden_size=64, num_classes=7).to(device)
        self.sensor_model.load_state_dict(torch.load(sensor_model_path, map_location=device))
        self.sensor_model.eval()
        
        # 3. Load the Scaler fitted during training
        if os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            print(f"Loaded scaler from {scaler_path}")
        else:
            print(f"Warning: Scaler not found at {scaler_path}. Inference might be inaccurate.")
            self.scaler = None
        
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

    def predict_sensor_window(self, window_data):
        """
        Predict defect probabilities for a window of sensor data.
        """
        if self.scaler is not None:
            window_data = self.scaler.transform(window_data)
            
        input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.sensor_model(input_tensor)
            probs = torch.softmax(logits, dim=1)
        return probs.squeeze().cpu().numpy()

def run_inference_sample(sample_dir, video_model_pth, sensor_model_pth, scaler_pth):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    if not os.path.exists(video_model_pth) or not os.path.exists(sensor_model_pth):
        print("Model files not found. Please train the models first.")
        return

    inference = WeldingInference(video_model_pth, sensor_model_pth, scaler_pth, device=device)
    
    # Find files in sample_dir
    video_file = None
    csv_file = None
    for f in os.listdir(sample_dir):
        if f.endswith('.avi'): video_file = os.path.join(sample_dir, f)
        if f.endswith('.csv'): csv_file = os.path.join(sample_dir, f)
        
    if video_file:
        print(f"\nProcessing video: {video_file}")
        cap = cv2.VideoCapture(video_file)
        # Check first 5 frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret: break
            score = inference.predict_video_frame(frame)
            print(f"Frame {i} Anomaly Score: {score:.6f}")
        cap.release()
        
    if csv_file:
        print(f"\nProcessing CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
        data = df[numerical_cols].values
        
        if len(data) >= 50:
            window = data[0:50] # Use first 50 samples
            score = inference.predict_sensor_window(window)
            print(f"First Sensor Window Anomaly Score: {score:.6f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run inference on a welding sample")
    parser.add_argument("--sample_dir", type=str, default=os.path.expanduser("~/Desktop/Hackathon/good_weld/config_1/run_1"),
                        help="Path to the sample directory to analyze")
    parser.add_argument("--video_model", type=str, default="checkpoints/video_classifier.pth")
    parser.add_argument("--sensor_model", type=str, default="checkpoints/sensor_classifier.pth")
    parser.add_argument("--scaler", type=str, default="checkpoints/sensor_scaler.pkl")
    args = parser.parse_args()

    run_inference_sample(args.sample_dir, args.video_model, args.sensor_model, args.scaler)
