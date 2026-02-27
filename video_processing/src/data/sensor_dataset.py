import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class WeldingSensorDataset(Dataset):
    def __init__(self, data_root, window_size=50, step_size=10):
        self.data_root = data_root
        self.window_size = window_size
        self.step_size = step_size
        self.windows = []
        
        all_data = []
        # Discover all .csv files
        for root, dirs, files in os.walk(data_root):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    df = pd.read_csv(csv_path)
                    # Selecting numerical columns: Pressure, CO2 Weld Flow, Feed, Primary Weld Current, Wire Consumed, Secondary Weld Voltage
                    # Based on the sample: indices 3 to 8
                    numerical_cols = ['Pressure', 'CO2 Weld Flow', 'Feed', 'Primary Weld Current', 'Wire Consumed', 'Secondary Weld Voltage']
                    data = df[numerical_cols].values
                    all_data.append(data)
        
        if not all_data:
            return

        # Fit scaler on all data
        import numpy as np
        concatenated_data = np.concatenate(all_data, axis=0)
        self.scaler = StandardScaler()
        self.scaler.fit(concatenated_data)
        
        # Create windows
        for data in all_data:
            scaled_data = self.scaler.transform(data)
            for i in range(0, len(scaled_data) - window_size, step_size):
                window = scaled_data[i : i + window_size]
                self.windows.append(window)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return torch.tensor(self.windows[idx], dtype=torch.float32)
