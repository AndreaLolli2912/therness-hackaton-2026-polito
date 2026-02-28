#!/usr/bin/env bash

# 1. Install System Dependencies (OpenCV needs these)
echo "📦 Installing system dependencies..."
apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 unzip wget curl

# 2. Install Python Requirements
echo "📦 Installing Python requirements..."
pip install -r video_processing_gru/requirements.txt
pip install gdown  # For Google Drive downloads

# 3. Create Data Directories
mkdir -p /root/data/train_data
mkdir -p /root/data/test_data

echo "--------------------------------------------------------"
echo "✅ Environment Ready!"
echo "--------------------------------------------------------"
echo "Next steps:"
echo "1. Download your data from Google Drive."
echo "   Example: gdown --folder <FOLDER_ID> -O /root/data/train_data"
echo "   Example: gdown --folder <FOLDER_ID> -O /root/data/test_data"
echo ""
echo "2. Run training (Video GRU):"
echo "   python3 -m video_processing_gru.train --config configs/master_config.json"
echo ""
echo "3. Run inference (after training):"
echo "   python3 video_processing_gru/generate_submission.py "
echo "       --checkpoint checkpoints/video/gru_classifier.pth "
echo "       --test_dir /root/data/test_data "
echo "       --output predictions_vast.csv"
echo "--------------------------------------------------------"
