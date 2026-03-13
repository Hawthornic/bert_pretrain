#!/bin/bash
# =============================================================
# BERT Pre-training Evaluation Script
# =============================================================
# Usage:
#   bash scripts/evaluate.sh --model_path output/bert_base_seq128/final_model
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_PATH=""
DATA_DIR="data/processed/wikipedia"
SEQ_LEN=128
BATCH_SIZE=64
MAX_DOCS=10000

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --data_dir)
            DATA_DIR="$2"
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
        --max_docs)
            MAX_DOCS="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required"
    echo "Usage: bash scripts/evaluate.sh --model_path <path>"
    exit 1
fi

echo "============================================"
echo "  BERT Evaluation"
echo "============================================"
echo "  Model:     $MODEL_PATH"
echo "  Data:      $DATA_DIR"
echo "  Seq len:   $SEQ_LEN"
echo "  Max docs:  $MAX_DOCS"
echo "============================================"

python src/evaluate.py \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --max_seq_length "$SEQ_LEN" \
    --batch_size "$BATCH_SIZE" \
    --max_docs "$MAX_DOCS"
