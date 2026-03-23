#!/bin/bash
# Script to run shape analysis tests for Video Kamba

PROJECT_ROOT="/home/WUR/stiva001/WUR/video_kamba"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: venv not found at $PROJECT_ROOT/venv"
    exit 1
fi

export PYTHONPATH=.

echo "Running shape analysis tests..."
python -m pytest tests/test_analysis_shapes.py -v

echo "Testing complete."
