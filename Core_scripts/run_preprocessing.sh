#!/bin/bash

# Example Usage Script for 3D Gaussian Splatting Pipeline
# This script demonstrates how to preprocess a video for Gaussian Splatting

set -e  # Exit on error

echo "=================================================="
echo "3D Gaussian Splatting - Preprocessing Example"
echo "=================================================="

# Configuration
VIDEO_PATH="${1:-example_video.mp4}"
OUTPUT_DIR="${2:-./data/example_scene}"
FPS="${3:-2}"
MAX_FRAMES="${4:-300}"

# Validate input
if [ ! -f "$VIDEO_PATH" ]; then
    echo "Error: Video file not found: $VIDEO_PATH"
    echo ""
    echo "Usage: $0 <video_path> [output_dir] [fps] [max_frames]"
    echo ""
    echo "Examples:"
    echo "  $0 my_video.mp4"
    echo "  $0 my_video.mp4 ./data/my_scene 2 300"
    echo "  $0 my_video.mp4 ./data/my_scene 5 500"
    exit 1
fi

echo "Configuration:"
echo "  Video: $VIDEO_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  FPS: $FPS"
echo "  Max Frames: $MAX_FRAMES"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run preprocessing
echo "Step 1: Extracting frames..."
python scripts/preprocessing_pipeline.py \
    --video "$VIDEO_PATH" \
    --output "$OUTPUT_DIR" \
    --fps "$FPS" \
    --max-frames "$MAX_FRAMES" \
    --quality 95

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "Preprocessing Complete!"
    echo "=================================================="
    echo ""
    echo "Next steps:"
    echo "1. Review the extracted frames in: $OUTPUT_DIR/extracted_frames/"
    echo "2. Compress the training data:"
    echo "   cd $OUTPUT_DIR && zip -r training_data.zip training_data/"
    echo "3. Upload to Google Colab and run the training notebook"
    echo ""
    echo "Files generated:"
    ls -lh "$OUTPUT_DIR"/extracted_frames/ | head -10
    echo "..."
    echo ""
    echo "Ready for training!"
else
    echo "Error: Preprocessing failed"
    exit 1
fi
