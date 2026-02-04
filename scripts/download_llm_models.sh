#!/bin/bash
# Download LLM models - full precision or quantized
# Usage: ./scripts/download_models.sh [full|quantized|both]
# Default: full

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"

MODE="${1:-full}"

if [ "$MODE" != "full" ] && [ "$MODE" != "quantized" ] && [ "$MODE" != "both" ]; then
    echo "Usage: $0 [full|quantized|both]"
    echo ""
    echo "  full      - Download gpt-oss-20b + mistral-7b-AWQ (~45GB)"
    echo "  quantized - Download gpt-oss-20b + mistral-7b-AWQ (~45GB)"
    echo "  both      - Download all model variants"
    echo ""
    echo "For A40 (48GB VRAM): Use 'full'"
    echo "For RTX 3090/4090 (24GB VRAM): Use 'quantized'"
    exit 1
fi

mkdir -p "${MODELS_DIR}"

if ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found. Please install Python 3.12+"
    exit 1
fi

echo "Installing huggingface-hub..."
pip install --quiet huggingface-hub

# Download full precision models
download_full() {
    echo ""
    echo "========================================"
    echo "Downloading Full Precision Models"
    echo "========================================"

    echo "Downloading gpt-oss-20b (22B MoE, ~41GB BF16, uses ~16GB VRAM with MXFP4)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b', local_dir='${MODELS_DIR}/gpt-oss-20b', local_dir_use_symlinks=False)"
    echo "✓ gpt-oss-20b"

    echo "Downloading mistral-7b-awq (AWQ 4-bit, ~4GB)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('TheBloke/Mistral-7B-Instruct-v0.2-AWQ', local_dir='${MODELS_DIR}/mistral-7b-awq', local_dir_use_symlinks=False)"
    echo "✓ mistral-7b-awq"
}

# Download quantized models
# gpt-oss-20b has built-in MXFP4 quantization (fits 16GB VRAM), no separate AWQ needed
download_quantized() {
    echo ""
    echo "========================================"
    echo "Downloading Quantized Models"
    echo "========================================"

    echo "Downloading gpt-oss-20b (22B MoE, ~41GB BF16, uses ~16GB VRAM with MXFP4)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('openai/gpt-oss-20b', local_dir='${MODELS_DIR}/gpt-oss-20b', local_dir_use_symlinks=False)"
    echo "✓ gpt-oss-20b"

    echo "Downloading mistral-7b-awq (AWQ 4-bit, ~4GB)..."
    python3 -c "from huggingface_hub import snapshot_download; snapshot_download('TheBloke/Mistral-7B-Instruct-v0.2-AWQ', local_dir='${MODELS_DIR}/mistral-7b-awq', local_dir_use_symlinks=False)"
    echo "✓ mistral-7b-awq"
}

# Main
if [ "$MODE" = "full" ] || [ "$MODE" = "both" ]; then
    download_full
fi

if [ "$MODE" = "quantized" ] || [ "$MODE" = "both" ]; then
    download_quantized
fi

echo ""
echo "========================================"
echo "✓ Download completed!"
echo "========================================"
echo ""
echo "Disk usage: $(du -sh "${MODELS_DIR}" | cut -f1)"
echo ""
echo "Next: docker compose -f containers/docker-compose.dev.yml up -d"
