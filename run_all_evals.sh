#!/usr/bin/env bash
# Run pacman_eval.py for every KNOWN_MODELS entry, 5 games each, raw mode.
# Usage: bash run_all_evals.sh

set -e
cd "$(dirname "$0")"

RUNS=5

MODELS=(
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    "mlx-community/pixtral-12b-4bit"
    "mlx-community/SmolVLM-Instruct-4bit"
    "mlx-community/gemma-3-4b-it-4bit"
    "mlx-community/moondream2-bf16"
    "mlx-community/paligemma2-3b-pt-448-4bit"
    "mlx-community/llava-v1.6-mistral-7b-4bit"
)

TOTAL=${#MODELS[@]}
echo "========================================"
echo "  FULL EVAL: $TOTAL models x $RUNS games (raw mode)"
echo "========================================"
echo ""

for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    NUM=$((i + 1))
    SHORT="${MODEL##*/}"
    echo ""
    echo "========================================"
    echo "  MODEL $NUM/$TOTAL: $SHORT"
    echo "========================================"
    echo ""

    python3 pacman_eval.py \
        --model "$MODEL" \
        --runs "$RUNS" \
        --raw \
        --label "$SHORT (raw)" \
        || echo "  WARNING: $SHORT eval failed or was interrupted"

    echo ""
    echo "  Cooling down 10s before next model..."
    sleep 10
done

echo ""
echo "========================================"
echo "  ALL EVALS COMPLETE"
echo "========================================"
echo ""

echo "Running analysis..."
python3 pacman_analysis.py
echo "Done!"
