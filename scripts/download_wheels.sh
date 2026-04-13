#!/bin/bash
# Build offline Python bundle for RHEL 8.10

set -euo pipefail

# Use system Python/pip or custom path if set
PYTHON="${PYTHON:-python3}"
PIP="${PIP:-pip3}"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WHEELS_DIR="${PROJECT_DIR}/wheels"

REQ_IN="${PROJECT_DIR}/requirements.txt"
REQ_LOCK="${PROJECT_DIR}/requirements.lock"

PY_VER="312"
PY_ABI="cp312"

PLATFORM="manylinux2014_x86_64"

mkdir -p "${WHEELS_DIR}"

echo "========================================"
echo "Locking Dependencies"
echo "========================================"

"${PIP}" install -r "${REQ_IN}"
"${PIP}" freeze > "${REQ_LOCK}"

echo "========================================"
echo "Downloading Linux Wheels"
echo "========================================"

"${PIP}" download \
  -r "${REQ_LOCK}" \
  -d "${WHEELS_DIR}" \
  --platform "${PLATFORM}" \
  --python-version "${PY_VER}" \
  --implementation cp \
  --abi "${PY_ABI}" \
  --only-binary=:all: \
  --progress-bar off

echo "========================================"
echo "✓ Bundle Ready"
echo "========================================"

echo "Install with:"
echo "${PIP} install --no-index --find-links ./wheels -r requirements.lock"
