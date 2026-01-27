#!/bin/bash
# Download all dependencies for air-gapped deployment
# Run this script on an internet-connected machine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OFFLINE_DIR="${PROJECT_DIR}/offline_packages"

echo "========================================"
echo "RAG System Air-Gap Preparation Script"
echo "========================================"
echo ""

# Create offline directory
mkdir -p "${OFFLINE_DIR}"
mkdir -p "${OFFLINE_DIR}/images"
mkdir -p "${OFFLINE_DIR}/models"
mkdir -p "${OFFLINE_DIR}/wheels"
mkdir -p "${OFFLINE_DIR}/rpms"

# Detect container runtime
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "ERROR: Neither podman nor docker found. Please install one."
    exit 1
fi

echo "Using container runtime: ${CONTAINER_CMD}"
echo ""

# Step 1: Pull base images
echo "Step 1: Pulling container images..."
echo "-----------------------------------"

IMAGES=(
    "python:3.10.14-slim"
    "ollama/ollama:0.1.23"
    "pgvector/pgvector:pg15"
)

for img in "${IMAGES[@]}"; do
    echo "Pulling ${img}..."
    ${CONTAINER_CMD} pull "${img}"
done

echo ""

# Step 2: Build application image
echo "Step 2: Building application image..."
echo "--------------------------------------"

${CONTAINER_CMD} build \
    -t rag-app:latest \
    -f "${PROJECT_DIR}/containers/Containerfile.app" \
    "${PROJECT_DIR}"

echo ""

# Step 3: Start Ollama and pull models
echo "Step 3: Pulling Ollama models..."
echo "---------------------------------"

# Start Ollama container temporarily to pull models
${CONTAINER_CMD} run -d \
    --name ollama-temp \
    -v ollama-models:/root/.ollama \
    ollama/ollama:0.1.23

# Wait for Ollama to start
echo "Waiting for Ollama to start..."
sleep 10

# Pull models
echo "Pulling mxbai-embed-large..."
${CONTAINER_CMD} exec ollama-temp ollama pull mxbai-embed-large

echo "Pulling gpt-oss:20b (this may take a while)..."
${CONTAINER_CMD} exec ollama-temp ollama pull gpt-oss:20b

# Copy models to offline directory
echo "Copying models to offline directory..."
${CONTAINER_CMD} cp ollama-temp:/root/.ollama/models "${OFFLINE_DIR}/models/"

# Stop and remove temp container
${CONTAINER_CMD} stop ollama-temp
${CONTAINER_CMD} rm ollama-temp

# Build Ollama image with pre-loaded models
echo "Building Ollama image with models..."
${CONTAINER_CMD} build \
    -t rag-ollama:latest \
    -f "${PROJECT_DIR}/containers/Containerfile.ollama" \
    "${PROJECT_DIR}/containers"

# Copy models into the Ollama image
# Create a temporary container to copy models
${CONTAINER_CMD} create --name ollama-build rag-ollama:latest
${CONTAINER_CMD} cp "${OFFLINE_DIR}/models/." ollama-build:/root/.ollama/models/
${CONTAINER_CMD} commit ollama-build rag-ollama:latest
${CONTAINER_CMD} rm ollama-build

echo ""

# Step 4: Build PostgreSQL image with init script
echo "Step 4: Building PostgreSQL image..."
echo "-------------------------------------"

# Create a simple Containerfile for postgres
cat > "${PROJECT_DIR}/containers/Containerfile.postgres" << 'EOF'
FROM pgvector/pgvector:pg15
COPY init.sql /docker-entrypoint-initdb.d/
EOF

${CONTAINER_CMD} build \
    -t rag-postgres:latest \
    -f "${PROJECT_DIR}/containers/Containerfile.postgres" \
    "${PROJECT_DIR}/containers"

echo ""

# Step 5: Download Python wheels
echo "Step 5: Downloading Python wheels..."
echo "-------------------------------------"

${CONTAINER_CMD} run --rm \
    -v "${OFFLINE_DIR}/wheels:/wheels" \
    -v "${PROJECT_DIR}/requirements.txt:/requirements.txt:ro" \
    python:3.10.14-slim \
    pip download -d /wheels -r /requirements.txt

echo ""

# Step 6: Download NVIDIA container toolkit RPMs (for RHEL 8.10)
echo "Step 6: Preparing NVIDIA toolkit info..."
echo "-----------------------------------------"

cat > "${OFFLINE_DIR}/rpms/README.txt" << 'EOF'
NVIDIA Container Toolkit for RHEL 8.10

On a RHEL 8.10 system with internet access, run:

# Add NVIDIA repo
distribution=rhel8.10
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
    sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Download RPMs
sudo yum install --downloadonly --downloaddir=/path/to/rpms \
    nvidia-container-toolkit

Then copy the downloaded RPMs to this directory.

On the air-gapped machine, install with:
sudo rpm -ivh *.rpm
EOF

echo ""
echo "========================================"
echo "Download complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Run ./export_images.sh to save images to a tar file"
echo "2. Copy the following to your air-gapped machine:"
echo "   - ${OFFLINE_DIR}/images/rag_stack.tar.gz (after running export_images.sh)"
echo "   - ${OFFLINE_DIR}/wheels/ (Python packages)"
echo "   - ${OFFLINE_DIR}/rpms/ (NVIDIA toolkit, if needed)"
echo "3. On the air-gapped machine, run ./import_images.sh"
echo ""
