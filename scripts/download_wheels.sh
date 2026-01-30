#!/bin/bash
# Download Python wheels for all requirements (air-gapped deployment)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHEELS_DIR="${PROJECT_DIR}/wheels"

mkdir -p "${WHEELS_DIR}"

echo "========================================"
echo "Downloading Python Wheels"
echo "========================================"
echo "Source: ${PROJECT_DIR}/requirements.txt"
echo "Target: ${WHEELS_DIR}"
echo ""

# Download all wheels matching exact versions in requirements.txt
/opt/anaconda3/bin/pip download \
  -r "${PROJECT_DIR}/requirements.txt" \
  -d "${WHEELS_DIR}" \
  --no-deps \
  --no-binary :all: || true  # Allow some pre-compiled wheels

echo ""
echo "========================================="
WHEEL_COUNT=$(ls "${WHEELS_DIR}" | wc -l)
WHEEL_SIZE=$(du -sh "${WHEELS_DIR}" | cut -f1)
echo "✓ Downloaded ${WHEEL_COUNT} packages"
echo "✓ Total size: ${WHEEL_SIZE}"
echo "✓ Saved to: ${WHEELS_DIR}"
echo ""
echo "For air-gapped deployment on EC2:"
echo "  1. Copy wheels/ to EC2"
echo "  2. Run: pip install --no-index --find-links ./wheels/ -r requirements.txt"
