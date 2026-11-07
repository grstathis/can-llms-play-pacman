#!/usr/bin/env bash
# Run eval with v5 prompt (structured CoT) for key models, 5 games each.
# Nov 7 2026 — structured chain-of-thought prompt that fixes direction spam.

set -e
cd "$(dirname "$0")"

RUNS=5

MODELS=(
    "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    "vikhyatk/moondream2"
    "mlx-community/SmolVLM-Instruct-4bit"
    "mlx-community/gemma-3-4b-it-4bit"
    "mlx-community/pixtral-12b-4bit"
)

TOTAL=${#MODELS[@]}
echo "========================================"
echo "  V5 PROMPT EVAL: $TOTAL models x $RUNS games (raw mode)"
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
        --label "$SHORT (raw v5)" \
        || echo "  WARNING: $SHORT eval failed or was interrupted"

    echo ""
    echo "  Cooling down 10s before next model..."
    sleep 10
done

echo ""
echo "========================================"
echo "  ALL V5 EVALS COMPLETE"
echo "========================================"
