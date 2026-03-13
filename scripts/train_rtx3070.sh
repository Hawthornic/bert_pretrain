#!/bin/bash
# =============================================================
# BERT Pre-training — RTX 3070 (8GB VRAM) Optimized
# =============================================================
# One-click training with optimized parameters:
#   Model:     BERT-Medium (512h, 8L, 8H) ~40M params
#   Seq len:   128
#   Batch:     32 x 4 grad_accum = 128 effective
#   Precision: FP16 (automatic)
#   Epochs:    10 (default)
#   VRAM:      ~4-5GB (safe margin on 8GB)
#
# Usage:
#   bash scripts/train_rtx3070.sh                # defaults
#   bash scripts/train_rtx3070.sh --epochs 15    # custom epochs
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults optimized for RTX 3070 8GB
EPOCHS=10
MAX_ARTICLES=100000
RESUME_FROM=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max_articles)
            MAX_ARTICLES="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

EXTRA_ARGS=""
if [ -n "$RESUME_FROM" ]; then
    EXTRA_ARGS="--resume_from $RESUME_FROM"
fi

# Step 1: Download data if not present
if [ ! -d "data/processed/wikipedia" ] || [ -z "$(ls data/processed/wikipedia/*.pkl 2>/dev/null)" ]; then
    echo "============================================"
    echo "  Step 1/2: Downloading data (${MAX_ARTICLES} articles)..."
    echo "============================================"
    bash scripts/download_data.sh --max_articles "$MAX_ARTICLES"
    echo ""
fi

# Step 2: Train
echo "============================================"
echo "  Step 2/2: BERT Pre-training (RTX 3070)"
echo "============================================"
echo "  Config:     BERT-Medium (~40M params)"
echo "  Epochs:     ${EPOCHS}"
echo "  Batch:      32 x 4 accum = 128 effective"
echo "  Seq len:    128"
echo "  FP16:       enabled"
echo "============================================"

python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

python src/pretrain.py \
    --data_dirs data/processed/wikipedia \
    --config_file configs/bert_medium.json \
    --output_dir output/bert_medium \
    --max_seq_length 128 \
    --train_batch_size 32 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs "$EPOCHS" \
    --learning_rate 5e-4 \
    --logging_steps 50 \
    --save_steps 2000 \
    --num_workers 4 \
    --fp16 \
    $EXTRA_ARGS

echo ""
echo "============================================"
echo "  Training complete!"
echo "  Model: output/bert_medium/"
echo "  Run app: bash scripts/run_app.sh"
echo "============================================"
