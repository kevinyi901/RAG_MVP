#!/bin/bash
# Export container images for air-gapped deployment
# Run this after download_for_airgap.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="${PROJECT_DIR}/offline_packages"
OUTPUT_FILE="${OFFLINE_DIR}/images/rag_stack.tar"

echo "========================================"
echo "RAG System Image Export Script"
echo "========================================"
echo ""

# Detect container runtime
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "ERROR: Neither podman nor docker found."
    exit 1
fi

echo "Using container runtime: ${CONTAINER_CMD}"
echo ""

# Create output directory
mkdir -p "${OFFLINE_DIR}/images"

# Images to export
IMAGES=(
    "rag-app:latest"
    "rag-vllm:latest"
    "rag-postgres:latest"
)

echo "Exporting images..."
echo "-------------------"

# Tag images for localhost registry (for Podman compatibility)
for img in "${IMAGES[@]}"; do
    ${CONTAINER_CMD} tag "${img}" "localhost/${img}" 2>/dev/null || true
done

# Export all images to a single tar file
echo "Saving images to ${OUTPUT_FILE}..."

if [ "${CONTAINER_CMD}" == "podman" ]; then
    podman save \
        localhost/rag-app:latest \
        localhost/rag-vllm:latest \
        localhost/rag-postgres:latest \
        -o "${OUTPUT_FILE}"
else
    docker save \
        rag-app:latest \
        rag-vllm:latest \
        rag-postgres:latest \
        -o "${OUTPUT_FILE}"
fi

# Compress the tar file
echo "Compressing..."
gzip -f "${OUTPUT_FILE}"

# Get file size
SIZE=$(du -h "${OUTPUT_FILE}.gz" | cut -f1)

echo ""
echo "========================================"
echo "Export complete!"
echo "========================================"
echo ""
echo "Output file: ${OUTPUT_FILE}.gz"
echo "Size: ${SIZE}"
echo ""
echo "Transfer this file to your air-gapped machine along with:"
echo "- The scripts/ directory"
echo "- The containers/ directory"
echo "- The .env.example file"
echo ""
