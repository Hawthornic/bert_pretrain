#!/bin/bash
# =============================================================
# Launch BERT Demo Web App (Gradio)
# Features: Fill-Mask, Semantic Similarity, Keyword Extraction
# =============================================================
# Usage:
#   bash scripts/run_app.sh                                          # default model
#   bash scripts/run_app.sh --model_path output/bert_medium/final_model  # custom path
# =============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL_PATH="output/bert_medium/checkpoint-final"

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model not found at: $MODEL_PATH"
    echo ""
    echo "Available checkpoints:"
    find output/ -name "config.json" -exec dirname {} \; 2>/dev/null || echo "  (none)"
    echo ""
    echo "Please train first: bash scripts/train_rtx3070.sh"
    exit 1
fi

echo "============================================"
echo "  BERT Demo Application"
echo "  Model: ${MODEL_PATH}"
echo "============================================"
echo ""
echo "Starting Gradio server..."
echo "  Local:  http://localhost:7860"
echo "  Public: (Gradio share link will appear below)"
echo ""

python src/app.py --model_dir "$MODEL_PATH" --share
