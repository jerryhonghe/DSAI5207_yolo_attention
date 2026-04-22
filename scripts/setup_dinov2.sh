#!/bin/bash
# Clone DINOv2 source code for offline model definition loading.
# Run this on a machine with internet access.

set -e

THIRD_PARTY_DIR="third_party"
DINOV2_DIR="${THIRD_PARTY_DIR}/dinov2"

mkdir -p "${THIRD_PARTY_DIR}"

if [ -d "${DINOV2_DIR}" ]; then
    echo "DINOv2 source already exists at ${DINOV2_DIR}"
else
    echo "Cloning DINOv2 source code..."
    git clone https://github.com/facebookresearch/dinov2.git "${DINOV2_DIR}"
    echo "Done!"
fi

echo ""
echo "Next steps:"
echo "1. Run: python scripts/download_weights.py"
echo "2. Upload the entire project directory to your server"
