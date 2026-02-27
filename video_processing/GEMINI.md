# Unsupervised Welding Defect Detection

This project implements a robust, lightweight architecture for unsupervised welding defect detection using Video and Sensor (CSV) data, optimized for edge/ARM deployment.

All code and models are located in the `video_processing` directory.

## Architecture

### Video Branch
- **Model:** Convolutional Autoencoder (CAE).
- **Encoder:** MobileNetV3-Small (Pretrained on ImageNet for robust feature extraction).
- **Decoder:** Transposed convolutions to reconstruct the 224x224 input frame.
- **Anomaly Detection:** Reconstruction error (MSE) is used as the anomaly score. Higher scores indicate deviations from the "normal" welding process.

### Sensor Branch
- **Model:** LSTM Autoencoder.
- **Input:** 6-dimensional time-series data (Pressure, Flow, Feed, Current, Wire, Voltage).
- **Mechanism:** Learns to compress and reconstruct normal sensor patterns.
- **Anomaly Detection:** MSE loss between the input window and reconstructed window.

## Setup

1. Navigate to the video processing folder:
   ```bash
   cd video_processing
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure sample data is downloaded in the root:
   ```bash
   ../download_sample_data.sh
   ```

## Training

### Video Branch
Train the video autoencoder on normal welding videos:
```bash
cd video_processing
python3 train.py
```

### Sensor Branch
Train the sensor autoencoder on CSV data:
```bash
cd video_processing
python3 train_sensor.py
```

## Inference

Run the unified inference script:
```bash
cd video_processing
python3 inference.py
```

## Edge Deployment Considerations
- **Backbone:** MobileNetV3-Small is used to minimize latency and power consumption on ARM devices.
- **Quantization:** The models are compatible with PyTorch quantization (e.g., INT8) for further optimization on Jetson or Raspberry Pi.
- **Fusion:** For final deployment, anomaly scores from both branches can be fused (e.g., via weighted sum or max) to improve robustness.
