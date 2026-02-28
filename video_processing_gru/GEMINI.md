# Supervised Welding Defect Detection

This project implements supervised models for welding defect detection using Video and Audio data, optimized for high-accuracy classification and edge/ARM deployment.

## Architecture

We currently train independent models for different modalities to maintain modularity and robustness.

### 1. Video Branch (Spatial-Temporal Features)
- **Backbone:** MobileNetV3-Small + GRU.
- **Role:** Extracts spatial features from frames and aggregates them temporally to detect defect patterns.
- **Input:** 224x224 welding frames (sequences).
- **Output:** 7-class probability distribution.

### 2. Audio Branch (Spectral Features)
- **Backbone:** 2D CNN (ResNet-style).
- **Input:** Log-Mel Spectrograms generated from FLAC audio.
- **Output:** 7-class probability distribution.

### 3. Sensor Data
- **Status:** Sensor data (CSV) is utilized for dashboard visualization and real-time monitoring but is not currently used for training the primary defect detection models.

## Setup

1. Navigate to the video processing folder:
   ```bash
   cd video_processing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure sample data is available:
   ```bash
   ../download_sample_data.sh
   ```

## Workflow

### 1. Video Training
Train the video classifier:
```bash
python3 train.py --config ../configs/master_config.json
```

### 2. Audio Training
Train the audio classifier (from the project root):
```bash
python run_audio.py --config configs/master_config.json
```

### 3. Inference
Run the video inference script:
```bash
python3 inference.py --video_model checkpoints/video_classifier.pth
```

## Edge Deployment Considerations
- **Efficiency:** MobileNetV3 and small LSTMs/GRUs are used to maintain low latency on ARM/Jetson devices.
- **Quantization:** Compatible with INT8 quantization for deployment on constrained hardware.
