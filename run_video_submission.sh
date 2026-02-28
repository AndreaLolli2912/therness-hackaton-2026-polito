#!/usr/bin/env bash

# Paths
MODEL_PATH="/Users/timon/Desktop/Hackathon/gru_classifier_4_epochs.pth"
OUTPUT_CSV="predictions_video_gru.csv"

# CHECK THIS: Set the path to the test_data folder (sample_XXXX subfolders inside)
TEST_DATA_DIR="${1:-TODO_SET_PATH_TO_TEST_DATA}"

if [ "$TEST_DATA_DIR" == "TODO_SET_PATH_TO_TEST_DATA" ]; then
    echo "❌ Error: Please provide the path to test_data as the first argument, or edit this script."
    echo "Usage: ./run_video_submission.sh /path/to/test_data"
    exit 1
fi

echo "📦 Installing/Checking requirements..."
python3 -m pip install -r video_processing_gru/requirements.txt --quiet

echo "🚀 Running video GRU inference..."
echo "📍 Model: $MODEL_PATH"
echo "📍 Test Data: $TEST_DATA_DIR"

# Ensure script is run from the project root
export PYTHONPATH=$PYTHONPATH:.

python3 video_processing_gru/generate_submission.py \
    --checkpoint "$MODEL_PATH" \
    --test_dir "$TEST_DATA_DIR" \
    --output "$OUTPUT_CSV" \
    --hidden_size 128 \
    --num_classes 7 \
    --seq_len 15 \
    --frame_skip 5

echo "✅ Done! Predictions saved to $OUTPUT_CSV"
