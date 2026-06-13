#!/bin/bash
# ============================================================================
# run_all_scales.sh — SR-61 Smoke Run: 4 prompts × 4 categories per scale
# ============================================================================
#
# Runs all_space ACTIVE_MANIFOLD across 4 architectures:
#   - 270M (gemma3-270m-it, HS=640,   text-only)
#   - 1B   (gemma3-1b-it,   HS=1152,  text-only)
#   - 4B   (gemma3-4b-it,   HS=2560,  multimodal)
#   - E2B  (gemma4-e2b-it,  HS=1536,  multimodal)
#
# Output:  eval/results/<SCALE>_ACTIVE_MANIFOLD_smoke16/<SCALE>_ACTIVE_MANIFOLD_aggregate.json
# Hardware: bf16, max_new_tokens=30, use_cache=False (RTX 2060 12GB safe)
#
# Usage:
#   bash eval/run_all_scales.sh
# ============================================================================

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON="/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python"

PROMPTS_PER_CAT=4
MAX_NEW_TOKENS=30
PRESET="ACTIVE_MANIFOLD"

for SCALE in 270M 1B 4B E2B; do
    OUTDIR="eval/results/${SCALE}_${PRESET}_smoke${PROMPTS_PER_CAT}"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ${SCALE} | ${PRESET} | ${PROMPTS_PER_CAT} prompts/cat | ${MAX_NEW_TOKENS} tok"
    echo "  → ${OUTDIR}"
    echo "════════════════════════════════════════════════════════════════"
    $PYTHON eval/run_4b_eval.py \
        --scale "$SCALE" \
        --prompts-per-cat "$PROMPTS_PER_CAT" \
        --preset "$PRESET" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output-dir "$OUTDIR"
    echo "  ✓ ${SCALE} done"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  All scales complete. Analyzing..."
echo "════════════════════════════════════════════════════════════════"

for SCALE in 270M 1B 4B E2B; do
    OUTDIR="eval/results/${SCALE}_${PRESET}_smoke${PROMPTS_PER_CAT}"
    AGG="${OUTDIR}/${SCALE}_${PRESET}_aggregate.json"
    if [ -f "$AGG" ]; then
        echo ""
        echo "── ${SCALE} ${PRESET} ──"
        $PYTHON eval/stats.py "$AGG"
    else
        echo "  ✗ ${SCALE} aggregate missing: ${AGG}"
    fi
done
