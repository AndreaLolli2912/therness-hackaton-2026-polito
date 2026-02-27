import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from .dataset import get_video_transforms

class MultimodalWeldingDataset(Dataset):
    def __init__(self, data_pairs, v_seq_len=15, v_frame_skip=5, s_window_size=100, scaler=None, transform=None):
        """
        data_pairs: list of tuples (video_path, csv_path, label_code)
        """
        self.v_seq_len = v_seq_len
        self.v_frame_skip = v_frame_skip
        self.s_window_size = s_window_size
        self.scaler = scaler
        self.transform = transform or get_video_transforms()
        
        self.label_map = {"00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6}
        self.samples = []
        
        for v_path, s_path, label_code in data_pairs:
            if not os.path.exists(v_path) or not os.path.exists(s_path):
                continue
            
            # Load sensor data to find window count
            df = pd.read_csv(s_path)
            s_len = len(df)
            
            # Check video frames
            cap = cv2.VideoCapture(v_path)
            v_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            
            # We create segments based on video (coarser modality)
            v_window = v_seq_len * v_frame_skip
            
            # Rough alignment: we assume video and sensor cover same duration
            # Stride of half a window
            for v_start in range(0, v_len - v_window, v_window // 2):
                # Map v_start to sensor_start
                # Ratio: s_len / v_len
                s_start = int(v_start * (s_len / v_len))
                
                if s_start + s_window_size < s_len:
                    self.samples.append({
                        "v_path": v_path,
                        "v_start": v_start,
                        "s_path": s_path,
                        "s_start": s_start,
                        "label": self.label_map.get(label_code, 0)
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        
        # 1. Video
        cap = cv2.VideoCapture(s["v_path"])
        v_frames = []
        for i in range(self.v_seq_len):
            idx = s["v_start"] + (i * self.v_frame_skip)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            v_frames.append(self.transform(Image.fromarray(frame)))
        cap.release()
        v_tensor = torch.stack(v_frames)
        
        # 2. Sensor
        df = pd.read_csv(s["s_path"])
        numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
        s_data = df.iloc[s["s_start"] : s["s_start"] + self.s_window_size][numerical_cols].values
        if self.scaler:
            s_data = self.scaler.transform(s_data)
        s_tensor = torch.tensor(s_data, dtype=torch.float32)
        
        return v_tensor, s_tensor, s["label"]

def get_multimodal_data(data_root):
    pairs = []
    # Assumes same run_id for .avi and .csv in the same subfolder
    for root, dirs, files in os.walk(data_root):
        v_files = {f.replace('.avi', ''): os.path.join(root, f) for f in files if f.endswith('.avi')}
        s_files = {f.replace('.csv', ''): os.path.join(root, f) for f in files if f.endswith('.csv')}
        
        for run_id in v_files:
            if run_id in s_files:
                label = run_id.split('-')[-1] if '-' in run_id else "00"
                pairs.append((v_files[run_id], s_files[run_id], label))
    return pairs
