#!/bin/bash
# Import container images on air-gapped machine
# Run this after copying rag_stack.tar.gz to the target machine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default location for the tar file
TAR_FILE="${1:-${PROJECT_DIR}/images/rag_stack.tar.gz}"

echo "========================================"
echo "RAG System Image Import Script"
echo "========================================"
echo ""

# Check if tar file exists
if [ ! -f "${TAR_FILE}" ]; then
    echo "ERROR: Image archive not found at ${TAR_FILE}"
    echo ""
    echo "Usage: $0 [path/to/rag_stack.tar.gz]"
    exit 1
fi

# Detect container runtime
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "ERROR: Neither podman nor docker found."
    echo "Please install podman: sudo dnf install podman podman-compose"
    exit 1
fi

echo "Using container runtime: ${CONTAINER_CMD}"
echo ""

# Import images
echo "Importing images from ${TAR_FILE}..."
echo "This may take several minutes..."
echo ""

if [ "${TAR_FILE: -3}" == ".gz" ]; then
    gunzip -c "${TAR_FILE}" | ${CONTAINER_CMD} load
else
    ${CONTAINER_CMD} load -i "${TAR_FILE}"
fi

echo ""

# List imported images
echo "Imported images:"
echo "----------------"
${CONTAINER_CMD} images | grep -E "(rag-app|rag-vllm|rag-postgres)"

echo ""
echo "========================================"
echo "Import complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Ensure NVIDIA drivers and nvidia-container-toolkit are installed"
echo "2. Generate CDI spec: sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml"
echo "3. Copy .env.example to .env and adjust if needed"
echo "4. Start the stack: podman-compose -f containers/podman-compose.yml up -d"
echo ""
