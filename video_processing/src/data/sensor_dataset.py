import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class WeldingSensorDataset(Dataset):
    def __init__(self, csv_files, window_size=50, step_size=10, scaler=None):
        """
        Args:
            csv_files (list): List of paths to .csv files.
            window_size (int): Number of timesteps per window.
            step_size (int): Overlap between windows.
            scaler (StandardScaler, optional): Pre-fitted scaler. If None, it will NOT fit one.
        """
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        self.scaler = scaler
        
        numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
        
        all_data = []
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            # Ensure columns exist
            data = df[numerical_cols].values
            all_data.append(data)
        
        if not all_data:
            return

        # If a scaler is provided, use it. Otherwise, data remains raw until handled externally.
        if self.scaler is not None:
            for data in all_data:
                scaled_data = self.scaler.transform(data)
                for i in range(0, len(scaled_data) - window_size, step_size):
                    window = scaled_data[i : i + window_size]
                    self.windows.append(window)
        else:
            # Fallback: Just create windows from raw data (useful for fitting the scaler)
            for data in all_data:
                for i in range(0, len(data) - window_size, step_size):
                    window = data[i : i + window_size]
                    self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)

def get_sensor_files(data_root):
    """Utility to find all CSV files."""
    csv_files = []
    for root, dirs, files in os.walk(data_root):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return sorted(csv_files)
