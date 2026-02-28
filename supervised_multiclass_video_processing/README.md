# Supervised Multiclass Video Processing

7-class welding defect detection using video and sensor data.

## Label Mapping

| Code | Index | Defect Type              | Train Count |
|------|-------|--------------------------|-------------|
| `00` | 0     | good_weld                | 750         |
| `01` | 1     | excessive_penetration    | 479         |
| `02` | 2     | burn_through             | 317         |
| `06` | 3     | overlap                  | 155         |
| `07` | 4     | lack_of_fusion           | 320         |
| `08` | 5     | excessive_convexity      | 159         |
| `11` | 6     | crater_cracks            | 150         |

## Pipelines

### Pipeline A: Random Forest on Frozen Embeddings + Sensor Stats

Fast tabular baseline. Uses pre-computed MobileNetV3 embeddings (1152-dim) fused with aggregated sensor statistics (42-dim).

```bash
# Step 1: Extract GPU video features → .npy cache
python extract_video_features.py

# Step 2: Train Random Forest classifier
python main.py
```

### Pipeline B: End-to-End MobileNetV3 + GRU Training

Trainable CNN-RNN that learns spatial-temporal features directly from video frames.

```bash
# Train with config
python train.py --config ../configs/master_config.json

# Full training (no validation split) for final submission
python train.py --config ../configs/master_config.json --full
```

## Inference

```bash
# Single sample analysis
python inference.py --sample_dir ../dataset/test_data/sample_0001 \
                    --video_model checkpoints/video_classifier.pth

# Generate submission CSV for all test samples
python inference.py --test_dir ../dataset/test_data \
                    --video_model checkpoints/video_classifier.pth \
                    --output_csv submission.csv
```

### Submission Format

```csv
sample_id,pred_label_code,p_defect
sample_0001,11,0.94
sample_0002,00,0.08
```

## Hackathon Scoring

```
FinalScore = 0.6 * Binary_F1 + 0.4 * Type_MacroF1
```

Where:
- **Binary_F1**: F1 score for defect vs non-defect (class 0 vs classes 1–6)
- **Type_MacroF1**: Macro-averaged F1 across all 7 classes

## Directory Structure

```
supervised_multiclass_video_processing/
├── README.md
├── extract_video_features.py      # Frozen MobileNetV3 → .npy cache
├── main.py                        # RF on fused embeddings + sensor stats
├── train.py                       # End-to-end MobileNetV3+GRU training
├── inference.py                   # Inference + submission CSV
└── src/
    ├── data/
    │   └── dataset.py             # 7-class dataset (VideoSample + PyTorch Dataset)
    └── models/
        └── video_model.py         # StreamingVideoClassifier (MobileNetV3+GRU)
```
