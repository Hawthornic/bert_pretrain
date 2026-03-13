#!/bin/bash
# =============================================================
# BERT Pre-training: Multi-GPU Distributed Training Script
# Uses HuggingFace Accelerate for easy distributed training.
# =============================================================
# Usage:
#   # First time: configure accelerate
#   accelerate config
#
#   # Then run:
#   bash scripts/train_distributed.sh [options]
#
#   # Or directly specify GPU count:
#   bash scripts/train_distributed.sh --num_gpus 4
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Defaults
CONFIG="bert_base"
SEQ_LEN=128
BATCH_SIZE=32
EPOCHS=3
MAX_STEPS=-1
LR=1e-4
FP16=""
GRAD_ACCUM=1
NUM_GPUS=0  # 0 = use all available / accelerate config
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --seq_len)
            SEQ_LEN="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --fp16)
            FP16="--fp16"
            shift
            ;;
        --grad_accum)
            GRAD_ACCUM="$2"
            shift 2
            ;;
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="output/${CONFIG}_seq${SEQ_LEN}_distributed"
fi

CONFIG_FILE="configs/${CONFIG}.json"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Find data directories
DATA_DIRS=""
if [ -d "data/processed/wikipedia" ]; then
    DATA_DIRS="$DATA_DIRS data/processed/wikipedia"
fi
if [ -d "data/processed/bookcorpus" ]; then
    DATA_DIRS="$DATA_DIRS data/processed/bookcorpus"
fi

if [ -z "$DATA_DIRS" ]; then
    echo "Error: No data found in data/processed/"
    echo "Run 'bash scripts/download_data.sh' first."
    exit 1
fi

# Build accelerate launch command
ACCELERATE_CMD="accelerate launch"
if [ "$NUM_GPUS" -gt 0 ]; then
    ACCELERATE_CMD="$ACCELERATE_CMD --num_processes $NUM_GPUS"
fi

echo "============================================"
echo "  BERT Pre-training (Distributed)"
echo "============================================"
echo "  Config:      $CONFIG_FILE"
echo "  Seq length:  $SEQ_LEN"
echo "  Batch size:  $BATCH_SIZE (per GPU)"
echo "  Grad accum:  $GRAD_ACCUM"
echo "  Epochs:      $EPOCHS"
echo "  Max steps:   $MAX_STEPS"
echo "  LR:          $LR"
echo "  FP16:        ${FP16:-disabled}"
echo "  Num GPUs:    ${NUM_GPUS} (0=all/auto)"
echo "  Output:      $OUTPUT_DIR"
echo "  Data dirs:   $DATA_DIRS"
echo "============================================"

$ACCELERATE_CMD src/pretrain_accelerate.py \
    --data_dirs $DATA_DIRS \
    --config_file "$CONFIG_FILE" \
    --max_seq_length "$SEQ_LEN" \
    --train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --num_train_epochs "$EPOCHS" \
    --max_steps "$MAX_STEPS" \
    --learning_rate "$LR" \
    --output_dir "$OUTPUT_DIR" \
    --logging_steps 100 \
    --save_steps 5000 \
    $FP16

echo ""
echo "Training complete! Model saved to: $OUTPUT_DIR"
