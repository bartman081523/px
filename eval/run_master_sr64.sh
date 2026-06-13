#!/bin/bash
PYTHON_BIN="/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python"
BASE_DIR="/run/media/julian/ML4/ollama-work/all_space"
OUT_DIR="$BASE_DIR/eval/results/SR64_FINAL"

mkdir -p "$OUT_DIR"

scales=("270M" "1B" "4B" "E2B")

for scale in "${scales[@]}"; do
    echo "=== Starting SR-64 Master Scale: $scale ==="
    mkdir -p "$OUT_DIR/$scale"
    cd "$BASE_DIR" && PYTHONPATH="$BASE_DIR" stdbuf -oL "$PYTHON_BIN" "$BASE_DIR/eval/run_master_psychology_warm.py" \
        --scale "$scale" \
        --output-dir "$OUT_DIR/$scale" \
        > "$OUT_DIR/${scale}_master.log" 2>&1
    echo "=== Finished SR-64 Master Scale: $scale ==="
done

echo "=== All SR-64 Master Scales Finished! ==="
