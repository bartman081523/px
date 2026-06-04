#!/bin/bash
# run.sh — Start PX API Server
# ================================================
# Uses venv_openmythos (has FastAPI + uvicorn + starlette)
# Disables all PX debug/telemetry output for clean API serving

PYTHON="/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python"
SERVER_DIR="/run/media/julian/ML4/ollama-work/all_space"

cd "$SERVER_DIR"

# ── Disable PX debug/telemetry output ──
# These env vars prevent:
#   - Console spam from [Router] debug prints (DEBUG_ROUTING)
#   - Telemetry JSON file dumps (DEBUG_PX)
#   - Per-step telemetry serialization overhead (SUBJECTIVE_TELEMETRY)
export DEBUG_ROUTING=0
export DEBUG_PX=0
export SUBJECTIVE_TELEMETRY=0

# ── Create symlinks if they don't exist ──
mkdir -p px_patches
ln -sf /run/media/julian/ML4/ollama-work/gemma_3_270m_it_px_subjective px_patches/gemma3_270m_px 2>/dev/null
ln -sf /run/media/julian/ML4/ollama-work/MiniCPM5-1B-PX px_patches/minicpm5_1b_px 2>/dev/null

# ── Start server ──
echo "Starting PX Explorer on https://0.0.0.0:7860 (SSL Enabled)"
echo "PX Models:   gemma3-270m-px, minicpm5-1b-px"
echo "Base Models:  gemma3-270m-base, gemma3-270m-it, minicpm5-1b-base"
echo "API:          https://localhost:7860/v1/"
echo "UI:           https://localhost:7860/gradio"
echo "LM Studio:    Configure Custom API endpoint as https://localhost:7860"
echo ""

exec $PYTHON app.py "$@"