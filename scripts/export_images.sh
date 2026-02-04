#!/bin/bash
# Create a complete self-contained offline package for air-gapped deployment.
# Run this AFTER download_for_airgap.sh on an internet-connected machine.
#
# Produces: rag-offline-package.tar.gz containing:
#   - Container images (rag-app, rag-vllm, rag-postgres)
#   - LLM models (gpt-oss-20b, mistral-7b-awq)
#   - Embedding model (nomic-embed-text-v1.5)
#   - Python wheels (for offline pip install)
#   - Compose files, init.sql, .env.example
#   - Import script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STAGING_DIR="${PROJECT_DIR}/rag-offline-staging"
OUTPUT_FILE="${PROJECT_DIR}/rag-offline-package.tar.gz"

echo "========================================"
echo "RAG MVP - Offline Package Builder"
echo "========================================"
echo ""
echo "Project directory: ${PROJECT_DIR}"
echo ""

# ------------------------------------------------------------------
# Detect container runtime
# ------------------------------------------------------------------
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

# ------------------------------------------------------------------
# Verify prerequisites
# ------------------------------------------------------------------
echo "Checking prerequisites..."

MISSING=0
if ! ${CTR} image inspect localhost/rag-app:latest &>/dev/null && \
   ! ${CTR} image inspect rag-app:latest &>/dev/null; then
    echo "  MISSING: rag-app:latest image (run download_for_airgap.sh first)"
    MISSING=1
fi

if ! ${CTR} image inspect localhost/rag-vllm:latest &>/dev/null && \
   ! ${CTR} image inspect rag-vllm:latest &>/dev/null && \
   ! ${CTR} image inspect vllm/vllm-openai:latest &>/dev/null; then
    echo "  MISSING: vllm image (run download_for_airgap.sh first)"
    MISSING=1
fi

if ! ${CTR} image inspect localhost/rag-postgres:latest &>/dev/null && \
   ! ${CTR} image inspect rag-postgres:latest &>/dev/null && \
   ! ${CTR} image inspect pgvector/pgvector:pg16 &>/dev/null; then
    echo "  MISSING: postgres image (run download_for_airgap.sh first)"
    MISSING=1
fi

if [ ! -d "${PROJECT_DIR}/models/gpt-oss-20b" ]; then
    echo "  MISSING: models/gpt-oss-20b (run download_for_airgap.sh first)"
    MISSING=1
fi

if [ ! -d "${PROJECT_DIR}/models/mistral-7b-awq" ]; then
    echo "  MISSING: models/mistral-7b-awq (run download_for_airgap.sh first)"
    MISSING=1
fi

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "Some prerequisites are missing. Run download_for_airgap.sh first."
    echo "Continuing anyway — missing items will be skipped."
    echo ""
fi

echo "Prerequisites check done."
echo ""

# ------------------------------------------------------------------
# Clean up any previous staging directory
# ------------------------------------------------------------------
if [ -d "${STAGING_DIR}" ]; then
    echo "Removing previous staging directory..."
    rm -rf "${STAGING_DIR}"
fi

mkdir -p "${STAGING_DIR}"

# ------------------------------------------------------------------
# 1. Export container images
# ------------------------------------------------------------------
echo "========================================"
echo "Step 1/4: Exporting container images"
echo "========================================"

mkdir -p "${STAGING_DIR}/images"

# Ensure images are tagged with localhost/ prefix
${CTR} tag vllm/vllm-openai:latest localhost/rag-vllm:latest 2>/dev/null || true
${CTR} tag pgvector/pgvector:pg16 localhost/rag-postgres:latest 2>/dev/null || true

echo "Saving all container images to a single tar..."

if [ "${CTR}" == "podman" ]; then
    podman save \
        localhost/rag-app:latest \
        localhost/rag-vllm:latest \
        localhost/rag-postgres:latest \
        -o "${STAGING_DIR}/images/rag_stack.tar"
else
    docker save \
        localhost/rag-app:latest \
        localhost/rag-vllm:latest \
        localhost/rag-postgres:latest \
        -o "${STAGING_DIR}/images/rag_stack.tar"
fi

echo "Compressing container images..."
gzip -f "${STAGING_DIR}/images/rag_stack.tar"

IMAGE_SIZE=$(du -sh "${STAGING_DIR}/images/rag_stack.tar.gz" | cut -f1)
echo "Container images: ${IMAGE_SIZE}"
echo ""

# ------------------------------------------------------------------
# 2. Copy models
# ------------------------------------------------------------------
echo "========================================"
echo "Step 2/4: Copying models"
echo "========================================"

mkdir -p "${STAGING_DIR}/models"

if [ -d "${PROJECT_DIR}/models/gpt-oss-20b" ]; then
    echo "Copying gpt-oss-20b..."
    cp -r "${PROJECT_DIR}/models/gpt-oss-20b" "${STAGING_DIR}/models/"
    echo "  $(du -sh "${STAGING_DIR}/models/gpt-oss-20b" | cut -f1)"
else
    echo "  SKIPPED: gpt-oss-20b not found"
fi

if [ -d "${PROJECT_DIR}/models/mistral-7b-awq" ]; then
    echo "Copying mistral-7b-awq..."
    cp -r "${PROJECT_DIR}/models/mistral-7b-awq" "${STAGING_DIR}/models/"
    echo "  $(du -sh "${STAGING_DIR}/models/mistral-7b-awq" | cut -f1)"
else
    echo "  SKIPPED: mistral-7b-awq not found"
fi

if [ -d "${PROJECT_DIR}/models/embedding" ]; then
    echo "Copying embedding model..."
    cp -r "${PROJECT_DIR}/models/embedding" "${STAGING_DIR}/models/"
    echo "  $(du -sh "${STAGING_DIR}/models/embedding" | cut -f1)"
else
    echo "  SKIPPED: embedding model not found"
fi

echo ""

# ------------------------------------------------------------------
# 3. Copy wheels
# ------------------------------------------------------------------
echo "========================================"
echo "Step 3/4: Copying wheels and config"
echo "========================================"

if [ -d "${PROJECT_DIR}/wheels" ] && [ "$(ls -A "${PROJECT_DIR}/wheels" 2>/dev/null)" ]; then
    echo "Copying Python wheels..."
    cp -r "${PROJECT_DIR}/wheels" "${STAGING_DIR}/"
    WHEEL_COUNT=$(ls "${STAGING_DIR}/wheels/"*.whl 2>/dev/null | wc -l | tr -d ' ')
    WHEEL_SIZE=$(du -sh "${STAGING_DIR}/wheels" | cut -f1)
    echo "  ${WHEEL_COUNT} wheels (${WHEEL_SIZE})"
else
    echo "  SKIPPED: No wheels found in ${PROJECT_DIR}/wheels/"
fi

# Copy compose files and config
echo "Copying compose files and configuration..."
mkdir -p "${STAGING_DIR}/containers"
cp "${PROJECT_DIR}/containers/podman-compose.yml" "${STAGING_DIR}/containers/" 2>/dev/null || true
cp "${PROJECT_DIR}/containers/docker-compose.dev.yml" "${STAGING_DIR}/containers/" 2>/dev/null || true
cp "${PROJECT_DIR}/containers/podman-compose.dev.yml" "${STAGING_DIR}/containers/" 2>/dev/null || true
cp "${PROJECT_DIR}/containers/init.sql" "${STAGING_DIR}/containers/" 2>/dev/null || true
cp "${PROJECT_DIR}/containers/Containerfile.app" "${STAGING_DIR}/containers/" 2>/dev/null || true

# Copy import script and other scripts
mkdir -p "${STAGING_DIR}/scripts"
cp "${PROJECT_DIR}/scripts/import_images.sh" "${STAGING_DIR}/scripts/" 2>/dev/null || true
cp "${PROJECT_DIR}/scripts/setup_rhel.sh" "${STAGING_DIR}/scripts/" 2>/dev/null || true

# Copy env and requirements
cp "${PROJECT_DIR}/.env.example" "${STAGING_DIR}/" 2>/dev/null || true
cp "${PROJECT_DIR}/requirements.txt" "${STAGING_DIR}/" 2>/dev/null || true

echo ""

# ------------------------------------------------------------------
# 4. Create the tar.gz package
# ------------------------------------------------------------------
echo "========================================"
echo "Step 4/4: Creating offline package"
echo "========================================"

echo "Creating ${OUTPUT_FILE}..."
echo "This may take a while for large model files..."

tar czf "${OUTPUT_FILE}" -C "${STAGING_DIR}" .

# Get final size
FINAL_SIZE=$(du -sh "${OUTPUT_FILE}" | cut -f1)

# Clean up staging directory
echo "Cleaning up staging directory..."
rm -rf "${STAGING_DIR}"

echo ""
echo "========================================"
echo "Offline package created!"
echo "========================================"
echo ""
echo "Output: ${OUTPUT_FILE}"
echo "Size:   ${FINAL_SIZE}"
echo ""
echo "Package contents:"
echo "  - Container images (rag-app, rag-vllm, rag-postgres)"
echo "  - LLM models (gpt-oss-20b, mistral-7b-awq)"
echo "  - Embedding model (nomic-embed-text-v1.5)"
echo "  - Python wheels"
echo "  - Compose files, init.sql, .env.example"
echo "  - Import script"
echo ""
echo "========================================"
echo "Transfer Instructions"
echo "========================================"
echo ""
echo "1. Copy rag-offline-package.tar.gz to the air-gapped machine"
echo ""
echo "2. On the air-gapped machine:"
echo "   mkdir -p RAG_MVP && cd RAG_MVP"
echo "   tar xzf rag-offline-package.tar.gz"
echo ""
echo "3. Import container images:"
echo "   bash scripts/import_images.sh images/rag_stack.tar.gz"
echo ""
echo "4. Set up environment:"
echo "   cp .env.example .env"
echo "   # Edit .env as needed (e.g., change default passwords)"
echo ""
echo "5. Start the stack:"
echo "   podman-compose -f containers/podman-compose.yml up -d"
echo ""
