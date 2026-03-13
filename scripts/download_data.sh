#!/bin/bash
# =============================================================
# BERT Pre-training Data Download & Preprocessing Script
# =============================================================
# Usage:
#   bash scripts/download_data.sh [--dataset wikipedia|bookcorpus|all] [--max_articles N]
#
# This script:
# 1. Installs Python dependencies
# 2. Downloads dataset(s) via HuggingFace Datasets
# 3. Preprocesses into document-level pickle shards
# =============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Default arguments
DATASET="wikipedia"
MAX_ARTICLES=0
SHARD_SIZE=100000
OUTPUT_DIR="data/processed"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --max_articles)
            MAX_ARTICLES="$2"
            shift 2
            ;;
        --shard_size)
            SHARD_SIZE="$2"
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

echo "============================================"
echo "  BERT Pre-training Data Preparation"
echo "============================================"
echo "  Dataset:      $DATASET"
echo "  Max articles: $MAX_ARTICLES (0=all)"
echo "  Shard size:   $SHARD_SIZE"
echo "  Output dir:   $OUTPUT_DIR"
echo "============================================"

# Step 1: Install dependencies
echo ""
echo "[Step 1/2] Checking dependencies..."
pip install -q datasets tqdm

# Step 2: Run preprocessing
echo ""
echo "[Step 2/2] Downloading and preprocessing data..."
python src/preprocess.py \
    --output_dir "$OUTPUT_DIR" \
    --dataset "$DATASET" \
    --max_articles "$MAX_ARTICLES" \
    --shard_size "$SHARD_SIZE"

echo ""
echo "============================================"
echo "  Data preparation complete!"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
echo ""
echo "Data files:"
find "$OUTPUT_DIR" -name "*.pkl" -exec ls -lh {} \;
echo ""
echo "Total size:"
du -sh "$OUTPUT_DIR"
