#!/bin/bash
# Setup script for RHEL 8.10 air-gapped deployment
# Run this script with sudo

set -e

echo "========================================"
echo "RAG System RHEL 8.10 Setup Script"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "Please run with sudo: sudo $0"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Step 1: Check NVIDIA driver
echo "Step 1: Checking NVIDIA driver..."
echo "----------------------------------"

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA driver detected:"
    nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv
    echo ""
else
    echo "WARNING: NVIDIA driver not detected!"
    echo "Please install NVIDIA driver 535+ before continuing."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Step 2: Install Podman and podman-compose
echo "Step 2: Checking Podman installation..."
echo "---------------------------------------"

if ! command -v podman &> /dev/null; then
    echo "Installing Podman..."
    dnf install -y podman
else
    echo "Podman already installed: $(podman --version)"
fi

if ! command -v podman-compose &> /dev/null; then
    echo "Installing podman-compose..."
    # podman-compose may need to be installed via pip if not in repos
    if command -v pip3 &> /dev/null; then
        pip3 install podman-compose
    else
        dnf install -y python3-pip
        pip3 install podman-compose
    fi
else
    echo "podman-compose already installed"
fi

echo ""

# Step 3: Install nvidia-container-toolkit
echo "Step 3: Checking NVIDIA Container Toolkit..."
echo "---------------------------------------------"

RPM_DIR="${PROJECT_DIR}/offline_packages/rpms"

if ! command -v nvidia-ctk &> /dev/null; then
    if [ -d "${RPM_DIR}" ] && ls "${RPM_DIR}"/*.rpm 1> /dev/null 2>&1; then
        echo "Installing NVIDIA Container Toolkit from offline RPMs..."
        rpm -ivh "${RPM_DIR}"/*.rpm || dnf install -y "${RPM_DIR}"/*.rpm
    else
        echo "WARNING: nvidia-container-toolkit not found and no offline RPMs available."
        echo ""
        echo "If you have internet access, install with:"
        echo "  distribution=rhel8.10"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.repo | \\"
        echo "      sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo"
        echo "  sudo dnf install -y nvidia-container-toolkit"
        echo ""
    fi
else
    echo "NVIDIA Container Toolkit already installed"
fi

echo ""

# Step 4: Generate CDI specification
echo "Step 4: Generating CDI specification..."
echo "---------------------------------------"

if command -v nvidia-ctk &> /dev/null; then
    mkdir -p /etc/cdi
    nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
    echo "CDI spec generated at /etc/cdi/nvidia.yaml"

    # Verify CDI
    echo ""
    echo "Available CDI devices:"
    nvidia-ctk cdi list
else
    echo "WARNING: nvidia-ctk not available, skipping CDI generation"
fi

echo ""

# Step 5: Configure SELinux (if enabled)
echo "Step 5: Checking SELinux..."
echo "---------------------------"

if command -v getenforce &> /dev/null; then
    SELINUX_STATUS=$(getenforce)
    echo "SELinux status: ${SELINUX_STATUS}"

    if [ "${SELINUX_STATUS}" == "Enforcing" ]; then
        echo "Setting SELinux context for volumes..."
        # Allow containers to use GPU devices
        setsebool -P container_use_devices on 2>/dev/null || true
    fi
else
    echo "SELinux not detected"
fi

echo ""

# Step 6: Create systemd service (optional)
echo "Step 6: Creating systemd service..."
echo "------------------------------------"

cat > /etc/systemd/system/rag-stack.service << EOF
[Unit]
Description=RAG System Stack
After=network-online.target nvidia-persistenced.service
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=${PROJECT_DIR}
ExecStart=/usr/bin/podman-compose -f containers/podman-compose.yml up -d
ExecStop=/usr/bin/podman-compose -f containers/podman-compose.yml down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
echo "Systemd service created: rag-stack.service"
echo "  Enable with: sudo systemctl enable rag-stack"
echo "  Start with:  sudo systemctl start rag-stack"

echo ""

# Step 7: Setup environment file
echo "Step 7: Setting up environment..."
echo "----------------------------------"

if [ ! -f "${PROJECT_DIR}/.env" ]; then
    if [ -f "${PROJECT_DIR}/.env.example" ]; then
        cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
        echo "Created .env from .env.example"
    fi
else
    echo ".env file already exists"
fi

echo ""
echo "========================================"
echo "Setup complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Import container images: ./scripts/import_images.sh"
echo "2. Start the stack:"
echo "   cd ${PROJECT_DIR}"
echo "   podman-compose -f containers/podman-compose.yml up -d"
echo ""
echo "Or use systemd:"
echo "   sudo systemctl enable --now rag-stack"
echo ""
echo "Access the application:"
echo "   Streamlit UI: http://localhost:8501"
echo "   API:          http://localhost:8000"
echo "   API Docs:     http://localhost:8000/docs"
echo ""
