#!/bin/bash
# Quick script to download vLLM models for development
# Run this before starting docker-compose

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"

echo "========================================"
echo "vLLM Model Download Script"
echo "========================================"
echo ""

# Create models directory
mkdir -p "${MODELS_DIR}"

# Check if Python and pip are available
if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found. Please install Python 3.10+"
    exit 1
fi

echo "Installing huggingface-hub..."
pip install --quiet huggingface-hub

echo ""
echo "Downloading gpt-oss-20b model..."
echo "This may take 10-20 minutes and ~40GB disk space..."
python3 << 'PYTHON_EOF'
from huggingface_hub import snapshot_download
import os

model_dir = os.path.expanduser("${MODELS_DIR}")
print(f"Saving to: {model_dir}")

snapshot_download(
    repo_id="gpt-oss/gpt-oss-20b",
    local_dir=f"{model_dir}/gpt-oss-20b",
    local_dir_use_symlinks=False,
    cache_dir=None
)
print("✓ gpt-oss-20b downloaded successfully")
PYTHON_EOF

echo ""
echo "Downloading mistral-7b model..."
echo "This may take 5-10 minutes and ~15GB disk space..."
python3 << 'PYTHON_EOF'
from huggingface_hub import snapshot_download
import os

model_dir = os.path.expanduser("${MODELS_DIR}")

snapshot_download(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    local_dir=f"{model_dir}/mistral-7b",
    local_dir_use_symlinks=False,
    cache_dir=None
)
print("✓ mistral-7b downloaded successfully")
PYTHON_EOF

echo ""
echo "========================================"
echo "✓ Models downloaded successfully!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run: docker compose -f containers/docker-compose.dev.yml up -d"
echo "2. Models will be mounted from ./models/ directory"
echo ""
