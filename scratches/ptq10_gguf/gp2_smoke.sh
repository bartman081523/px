#!/usr/bin/env bash
# G-P2/G-P4: llama-server Smoke (Laden, Greedy-Kohärenz) + VRAM-Peak.
# Aufruf: gp2_smoke.sh [GGUF] [PORT] [CTX]
set -u
BIN=/home/julian/llamacpp-build-ptq10/bin
SC=/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches/ptq10_gguf
GGUF=${1:-/home/julian/models/bonsai/Ternary-Bonsai-2-27B-PTQ1_0-recon.gguf}
PORT=${2:-8123}
CTX=${3:-8192}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256
"$BIN"/llama-server -m "$GGUF" -ngl 99 -c "$CTX" --port "$PORT" \
  --host 127.0.0.1 --threads 6 \
  > "$SC/gp2_server_${PORT}.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
READY=0
for _ in $(seq 1 100); do                        # max ~300 s Bootstrap
  if curl -s "http://127.0.0.1:$PORT/health" | grep -q '"ok"'; then
    READY=1; break
  fi
  sleep 3
done
echo "READY=$READY"
if [ "$READY" != 1 ]; then
  timeout 5 tail -c 3000 "$SC/gp2_server_${PORT}.log"
  exit 1
fi
# G-P4: VRAM-Peak waehrend Generierung
nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 2 \
  > "$SC/gp4_vram_${PORT}.log" &
VPID=$!
curl -s "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"In genau zwei Saetzen: Warum ist der Himmel blau?"}],"max_tokens":160,"temperature":0}'
echo
kill $VPID 2>/dev/null
sort -rn "$SC/gp4_vram_${PORT}.log" | head -1 | sed 's/^/VRAM_PEAK=/'
curl -s "http://127.0.0.1:$PORT/metrics" \
  | grep -E '^(llamacpp|llama)' | head -5 || true