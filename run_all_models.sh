#!/bin/bash
# Run all known models with --raw (new tight prompt) — 5 games each
# Skips models that fail to load, kills Mesen between runs

cd "$(dirname "$0")"

MODELS=(
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    "vikhyatk/moondream2"
    "mlx-community/gemma-3-4b-it-4bit"
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
    "mlx-community/pixtral-12b-2409-4bit"
    "mlx-community/SmolVLM-Instruct-4bit"
    "mlx-community/moondream2-bf16"
    "mlx-community/paligemma2-3b-pt-448-4bit"
    "mlx-community/llava-v1.6-mistral-7b-4bit"
)

RUNS=5

for model in "${MODELS[@]}"; do
    short=$(echo "$model" | sed 's|.*/||')
    echo ""
    echo "=============================================="
    echo "  MODEL: $short (raw)"
    echo "=============================================="

    pkill -f Mesen 2>/dev/null
    sleep 2

    python3 pacman_eval.py \
        --runs "$RUNS" \
        --model "$model" \
        --raw \
        --label "$short (raw v2)"

    exitcode=$?
    if [ $exitcode -ne 0 ]; then
        echo "  FAILED (exit $exitcode) — moving on"
    fi

    pkill -f Mesen 2>/dev/null
    sleep 2
done

echo ""
echo "=============================================="
echo "  ALL MODELS DONE"
echo "=============================================="
