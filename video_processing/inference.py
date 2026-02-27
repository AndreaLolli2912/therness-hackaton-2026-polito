import torch
import cv2
import pandas as pd
import numpy as np
from src.models.video_model import VideoAutoencoder
from src.models.sensor_model import SensorAutoencoder
from src.data.dataset import get_transforms
from sklearn.preprocessing import StandardScaler
import os

class WeldingInference:
    def __init__(self, video_model_path, sensor_model_path, device='cpu'):
        self.device = device
        
        # Load Video Model
        self.video_model = VideoAutoencoder(latent_dim=256).to(device)
        self.video_model.load_state_dict(torch.load(video_model_path, map_location=device))
        self.video_model.eval()
        
        # Load Sensor Model
        self.sensor_model = SensorAutoencoder(input_size=6, hidden_size=64, latent_dim=16).to(device)
        self.sensor_model.load_state_dict(torch.load(sensor_model_path, map_location=device))
        self.sensor_model.eval()
        
        self.video_transform = get_transforms()
        # Note: In a real scenario, you'd load the scaler used during training
        self.scaler = StandardScaler() 

    def predict_video_frame(self, frame):
        # frame: BGR from OpenCV
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = self.video_transform(frame_rgb).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.video_model.get_anomaly_score(input_tensor)
        return score.item()

    def predict_sensor_window(self, window_data):
        # window_data: [window_size, 6]
        input_tensor = torch.tensor(window_data, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            score = self.sensor_model.get_anomaly_score(input_tensor)
        return score.item()

def run_inference_sample(sample_dir, video_model_pth, sensor_model_pth):
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    inference = WeldingInference(video_model_pth, sensor_model_pth, device=device)
    
    # Find files in sample_dir
    video_file = None
    csv_file = None
    for f in os.listdir(sample_dir):
        if f.endswith('.avi'): video_file = os.path.join(sample_dir, f)
        if f.endswith('.csv'): csv_file = os.path.join(sample_dir, f)
        
    if video_file:
        print(f"Processing video: {video_file}")
        cap = cv2.VideoCapture(video_file)
        # Check first 5 frames
        for i in range(5):
            ret, frame = cap.read()
            if not ret: break
            score = inference.predict_video_frame(frame)
            print(f"Frame {i} Anomaly Score: {score:.6f}")
        cap.release()
        
    if csv_file:
        print(f"Processing CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
        data = df[numerical_cols].values
        # Simple scaling for demonstration (should use training scaler)
        data_scaled = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)
        
        if len(data_scaled) >= 50:
            window = data_scaled[0:50]
            score = inference.predict_sensor_window(window)
            print(f"First Sensor Window Anomaly Score: {score:.6f}")

if __name__ == "__main__":
    run_inference_sample('../sampleData/08-17-22-0011-00', 
                         'checkpoints/video_autoencoder.pth', 
                         'checkpoints/sensor_autoencoder.pth')
