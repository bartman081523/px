#!/bin/bash
# ============================================================================
# run_full_eval.sh — SR-61 Full Run: 20 prompts × 4 categories × 4 scales
# ============================================================================
#
# Runs all_space ACTIVE_MANIFOLD across 4 architectures with the full prompt
# set (20 prompts per category = 80 prompts per scale, 320 total):
#   - 270M (gemma3-270m-it, HS=640,   text-only)     ~10s/prompt
#   - 1B   (gemma3-1b-it,   HS=1152,  text-only)     ~12s/prompt
#   - 4B   (gemma3-4b-it,   HS=2560,  multimodal)   ~50s/prompt
#   - E2B  (gemma4-e2b-it,  HS=1536,  multimodal)   ~50s/prompt
#
# Total estimated wall time: ~2 hours 10 minutes
#   270M: 80 × 10s  =  13 min
#   1B:   80 × 12s  =  16 min
#   4B:   80 × 50s  =  67 min
#   E2B:  80 × 50s  =  67 min
#
# Output:  eval/results/<SCALE>_ACTIVE_MANIFOLD_full/<SCALE>_ACTIVE_MANIFOLD_aggregate.json
# Hardware: bf16, max_new_tokens=30, use_cache=False (RTX 2060 12GB safe)
#
# IMPORTANT: Subprocesses run sequentially. Do NOT run other GPU workloads
# during this script — the 4B/E2B models use nearly all 12GB and concurrent
# processes cause CUBLAS_STATUS_ALLOC_FAILED.
#
# Usage:
#   bash eval/run_full_eval.sh
# ============================================================================

set -e

cd "$(dirname "$0")/.."
export PYTHONPATH=.
PYTHON="/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python"

PROMPTS_PER_CAT=20
MAX_NEW_TOKENS=30
PRESET="ACTIVE_MANIFOLD"

# Per-scale prompt-by-prompt timeout (seconds) — well above the actual
# generation time, leaves room for slow first-token CUDA init.
TIMEOUT_PER_PROMPT=300  # 5 min/prompt

for SCALE in 270M 1B 4B E2B; do
    OUTDIR="eval/results/${SCALE}_${PRESET}_full"
    AGG="${OUTDIR}/${SCALE}_${PRESET}_aggregate.json"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ${SCALE} | ${PRESET} | ${PROMPTS_PER_CAT} prompts/cat | ${MAX_NEW_TOKENS} tok"
    echo "  → ${OUTDIR}"
    echo "════════════════════════════════════════════════════════════════"

    if [ -f "$AGG" ]; then
        echo "  ↻ ${AGG} already exists — skipping (delete to re-run)"
        continue
    fi

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
    OUTDIR="eval/results/${SCALE}_${PRESET}_full"
    AGG="${OUTDIR}/${SCALE}_${PRESET}_aggregate.json"
    if [ -f "$AGG" ]; then
        echo ""
        echo "── ${SCALE} ${PRESET} ──"
        $PYTHON eval/stats.py "$AGG"
    else
        echo "  ✗ ${SCALE} aggregate missing: ${AGG}"
    fi
done
