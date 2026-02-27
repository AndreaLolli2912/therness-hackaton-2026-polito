# Supervised Multimodal Welding Defect Detection

This project implements a supervised, multimodal architecture for welding defect detection using Video, Sensor (CSV), and Audio (FLAC) data, optimized for high-accuracy classification and edge/ARM deployment.

All code and models are located in the `video_processing` directory.

## Architecture: Multimodal Fusion Classifier

Instead of independent anomaly scores, we use a unified **Late Fusion** architecture to classify defect types directly.

### 1. Video Branch (Spatial Features)
- **Backbone:** MobileNetV3-Small (Pretrained on ImageNet).
- **Role:** Extracts high-level spatial features from 224x224 welding frames.
- **Output:** 576-dimensional feature vector per frame.

### 2. Sensor Branch (Temporal Features)
- **Backbone:** LSTM (Long Short-Term Memory).
- **Input:** 6-dimensional time-series data (Pressure, Flow, Feed, Current, Wire, Voltage).
- **Output:** 64-dimensional temporal context vector.

### 3. Audio Branch (Spectral Features)
- **Backbone:** 2D CNN (ResNet-style).
- **Input:** Log-Mel Spectrograms generated from FLAC audio.
- **Output:** 128-dimensional spectral feature vector.

### 4. Fusion & Classification Head
- **Mechanism:** Concatenates features from all three branches.
- **Head:** Multi-layer Perceptron (MLP) with Dropout.
- **Outputs:**
  - **Primary:** Softmax probabilities for all 7 classes (00, 01, 02, 06, 07, 08, 11).
  - **Secondary:** Binary `p_defect` (derived from the sum of non-00 probabilities).

## Setup

1. Navigate to the video processing folder:
   ```bash
   cd video_processing
   ```

2. Install dependencies (added torchaudio for the audio branch):
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure sample data is available:
   ```bash
   ../download_sample_data.sh
   ```

## Workflow

### 1. Data Preparation
The dataset is split by **Run ID** to prevent temporal leakage. Labels are extracted from folder names in `good_weld` and `defect_data_weld`.

### 2. Training
Train the unified multimodal classifier:
```bash
python3 train_multimodal.py
```
*Note: Supports training on Video-only, Sensor-only, or Full Fusion by toggling flags in the script.*

### 3. Inference & Submission
Run the inference script to generate the `submission.csv` for the hackathon:
```bash
python3 generate_submission.py --data_path ../test_data
```

## Edge Deployment Considerations
- **Efficiency:** MobileNetV3 and small LSTMs are used to maintain low latency on ARM/Jetson devices.
- **Modularity:** The architecture allows "Graceful Degradation"—if a sensor or camera fails, the model can still provide a prediction based on the remaining modalities.
- **Quantization:** Compatible with INT8 quantization for deployment on constrained hardware.

