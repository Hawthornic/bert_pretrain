#!/bin/bash
# =============================================================
# Environment Setup
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

echo "============================================"
echo "  Environment Setup for BERT Pre-training"
echo "============================================"

echo "[1/2] Installing Python dependencies..."
pip install -q -r requirements.txt

echo "[2/2] Checking GPU..."
python -c "
import torch
if torch.cuda.is_available():
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  GPU: {name}')
    print(f'  VRAM: {vram:.1f} GB')
    print(f'  CUDA: {torch.version.cuda}')
else:
    print('  WARNING: No GPU detected! Training will be very slow on CPU.')
"

echo ""
echo "Setup complete!"
echo "============================================"
