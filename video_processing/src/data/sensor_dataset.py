import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class WeldingSensorDataset(Dataset):
    def __init__(self, csv_files, labels=None, window_size=100, step_size=50, scaler=None):
        """
        Args:
            csv_files (list): List of paths to .csv files.
            labels (list, optional): List of label codes corresponding to files. 
                                    If None, it tries to extract from filenames.
            window_size (int): Number of timesteps per window.
            step_size (int): Overlap between windows.
            scaler (StandardScaler, optional): Pre-fitted scaler.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        self.window_labels = []
        self.scaler = scaler
        
        # Map label codes to 0-6 indices
        self.label_map = {
            "00": 0, "01": 1, "02": 2, "06": 3, "07": 4, "08": 5, "11": 6
        }
        
        numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
        
        for i, csv_path in enumerate(csv_files):
            df = pd.read_csv(csv_path)
            data = df[numerical_cols].values
            
            # Determine label for this file
            if labels is not None:
                label_code = labels[i]
            else:
                # Extract label from filename (run_id pattern)
                run_id = os.path.splitext(os.path.basename(csv_path))[0]
                label_code = run_id.split('-')[-1] if '-' in run_id else "00"
            
            label_idx = self.label_map.get(label_code, 0)
            
            # Apply scaling if available
            if self.scaler is not None:
                data = self.scaler.transform(data)
                
            # Create windows
            for start_idx in range(0, len(data) - window_size, step_size):
                window = data[start_idx : start_idx + window_size]
                self.windows.append(window)
                self.window_labels.append(label_idx)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32), self.window_labels[idx]

def get_sensor_files_and_labels(data_root):
    """Utility to find all CSV files and their labels."""
    sensor_data = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.csv'):
                path = os.path.join(root, file)
                run_id = os.path.splitext(file)[0]
                label = run_id.split('-')[-1] if '-' in run_id else "00"
                sensor_data.append((path, label))
    return sensor_data
