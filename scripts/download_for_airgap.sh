#!/bin/bash
# Download all dependencies for air-gapped deployment.
# Run this on an internet-connected machine before export_images.sh.
#
# Downloads:
#   1. LLM models (gpt-oss-20b + mistral-7b-awq)
#   2. Embedding model (nomic-ai/nomic-embed-text-v1.5)
#   3. Python wheels (all pip dependencies for offline install)
#   4. Container images (vllm, pgvector, app build)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${PROJECT_DIR}/models"
WHEELS_DIR="${PROJECT_DIR}/wheels"

echo "========================================"
echo "RAG MVP - Air-Gap Preparation"
echo "========================================"
echo ""
echo "Project directory: ${PROJECT_DIR}"
echo ""

# ------------------------------------------------------------------
# 1. LLM Models
# ------------------------------------------------------------------
echo "========================================"
echo "Step 1/5: Downloading LLM models"
echo "========================================"

if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
    echo "ERROR: pip not found. Please install Python 3.12+"
    exit 1
fi

PIP="${PIP:-pip3}"

# Ensure huggingface-hub is available
${PIP} install --quiet huggingface-hub

bash "${SCRIPT_DIR}/download_llm_models.sh"

echo ""

# ------------------------------------------------------------------
# 2. Embedding Model
# ------------------------------------------------------------------
echo "========================================"
echo "Step 2/5: Downloading embedding model"
echo "========================================"

EMBEDDING_DIR="${MODELS_DIR}/embedding/nomic-embed-text-v1.5"
if [ -d "${EMBEDDING_DIR}" ] && [ "$(ls -A "${EMBEDDING_DIR}" 2>/dev/null)" ]; then
    echo "Embedding model already exists at ${EMBEDDING_DIR}, skipping."
else
    mkdir -p "${EMBEDDING_DIR}"
    echo "Downloading nomic-ai/nomic-embed-text-v1.5..."
    ${PIP} install --quiet sentence-transformers einops
    python3 -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
model.save('${EMBEDDING_DIR}')
print('Saved to ${EMBEDDING_DIR}')
"
    echo "Embedding model downloaded."
fi
