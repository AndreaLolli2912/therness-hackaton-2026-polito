# Welding Defect Detection - Video-GRU Approach

This project implements a video-based welding defect classification system. It uses a MobileNetV3-Small backbone for spatial feature extraction and a GRU for temporal aggregation.

## Project Structure

- `video_processing_gru/`: Core logic for training and inference.
- `configs/`: JSON configuration for model and training parameters.
- `dashboard.py`: Streamlit application for dataset visualization and run inspection.

## Getting Started

### 1. Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Training
```bash
cd video_processing_gru
python train.py --config ../configs/master_config.json
```

### 3. Submission
Generate the hackathon submission CSV:
```bash
python video_processing_gru/generate_submission.py \
    --checkpoint checkpoints/video/gru_classifier.pth \
    --test_dir /path/to/test_data \
    --output predictions_video_gru.csv
```

### 4. Dashboard
```bash
streamlit run dashboard.py
```

## Labels

| Code | Label |
|---|---|
| `00` | `good_weld` |
| `01` | `excessive_penetration` |
| `02` | `burn_through` |
| `06` | `overlap` |
| `07` | `lack_of_fusion` |
| `08` | `excessive_convexity` |
| `11` | `crater_cracks` |
