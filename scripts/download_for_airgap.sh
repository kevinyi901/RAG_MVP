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

echo ""

# ------------------------------------------------------------------
# 3. Python Wheels
# ------------------------------------------------------------------
echo "========================================"
echo "Step 3/5: Downloading Python wheels"
echo "========================================"

bash "${SCRIPT_DIR}/download_wheels.sh"

# Ensure einops wheel is present (required by nomic embedding model)
if ! ls "${WHEELS_DIR}"/einops-*.whl &> /dev/null; then
    echo "Downloading einops wheel..."
    ${PIP} download einops --no-deps -d "${WHEELS_DIR}"
fi

echo ""

# ------------------------------------------------------------------
# 4. Container Images
# ------------------------------------------------------------------
echo "========================================"
echo "Step 4/5: Pulling and building container images"
echo "========================================"

# Detect container runtime
if command -v docker &> /dev/null; then
    CTR="docker"
elif command -v podman &> /dev/null; then
    CTR="podman"
else
    echo "ERROR: Neither docker nor podman found."
    exit 1
fi

echo "Using container runtime: ${CTR}"
echo ""

echo "Pulling vllm/vllm-openai:latest..."
${CTR} pull vllm/vllm-openai:latest

echo "Pulling pgvector/pgvector:pg16..."
${CTR} pull pgvector/pgvector:pg16

echo "Building application image..."
${CTR} build -f "${PROJECT_DIR}/containers/Containerfile.app" -t localhost/rag-app:latest "${PROJECT_DIR}"

# Tag images for air-gap (localhost/ prefix)
${CTR} tag vllm/vllm-openai:latest localhost/rag-vllm:latest 2>/dev/null || true
${CTR} tag pgvector/pgvector:pg16 localhost/rag-postgres:latest 2>/dev/null || true

echo ""

# ------------------------------------------------------------------
# 5. Summary
# ------------------------------------------------------------------
echo "========================================"
echo "Step 5/5: Verification"
echo "========================================"

echo ""
echo "LLM Models:"
du -sh "${MODELS_DIR}/gpt-oss-20b" 2>/dev/null || echo "  gpt-oss-20b: NOT FOUND"
du -sh "${MODELS_DIR}/mistral-7b-awq" 2>/dev/null || echo "  mistral-7b-awq: NOT FOUND"

echo ""
echo "Embedding Model:"
du -sh "${MODELS_DIR}/embedding/nomic-embed-text-v1.5" 2>/dev/null || echo "  nomic-embed-text-v1.5: NOT FOUND"

echo ""
echo "Wheels:"
echo "  $(ls "${WHEELS_DIR}"/*.whl 2>/dev/null | wc -l | tr -d ' ') wheel files in ${WHEELS_DIR}/"

echo ""
echo "Container Images:"
${CTR} images | grep -E "(rag-app|rag-vllm|rag-postgres)" || true

echo ""
echo "========================================"
echo "Preparation complete!"
echo "========================================"
echo ""
echo "Next step: bash scripts/export_images.sh"
echo "This will package everything into a single tar.gz for transfer."
