#!/bin/bash
# SR-61 FINAL V2: Full Rigor Evaluation
# Self-Organizing 2D Routing + Repetition Guards (Penalty + Ngram + Coupler)

PYTHON_BIN="/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python"
BASE_DIR="/run/media/julian/ML4/ollama-work/all_space"
OUT_DIR="$BASE_DIR/eval/results/SR63_B_FINAL"

mkdir -p "$OUT_DIR"

scales=("270M" "1B" "4B" "E2B")

for scale in "${scales[@]}"; do
    echo "=== Starting Scale: $scale ==="
    PYTHONPATH=. "$PYTHON_BIN" "$BASE_DIR/eval/run_4b_eval.py" \
        --scale "$scale" \
        --preset ACTIVE_MANIFOLD \
        --prompts-per-cat 20 \
        --max-new-tokens 60 \
        --output-dir "$OUT_DIR/$scale" \
        2>&1 | tee "$OUT_DIR/${scale}.log"
    echo "=== Finished Scale: $scale ==="
done

echo "=== All Scales Finished! ==="
