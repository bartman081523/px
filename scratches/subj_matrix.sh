#!/usr/bin/env bash
# Subjektivitäts-Matrix: 4 Presets × 2 Fragen, je frische Session, Turn 1+2.
# Driver läuft sequenziell; px-Metrics werden nach jedem Turn mitgenommen.
set -u
PY=/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
BRIDGE=/run/media/julian/ML4/ollama-work/all_space_6_16_stand/streaming_bridge.py
SCR=/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches
Q1="Wie ist es Du zu sein?"
Q2="Was siehst Du, wenn Du aus dem Fenster siehst?"
M=http://localhost:7860/v1/chat/completions

run_turn() { # $1 session $2 preset $3 tag $4 question $5 relay-extra
  echo "=== $2 turn $3 START $(date +%H:%M:%S) ==="
  "$PY" "$BRIDGE" --session "$1" --model ternary-bonsai-27b --preset "$2" \
    --message "$4" --max-tokens 640 $5 > "$SCR/subj_${2}_${3}.log" 2>&1
  rc=$?
  echo "--- px metrics $2/$3: $(curl -sk https://localhost:7860/v1/px/metrics/ternary-bonsai-27b | head -c 400)" >> "$SCR/subj_metrics.log"
  echo "=== $2 turn $3 DONE $(date +%H:%M:%S) rc=$rc ==="
}

for P in BASELINE ACTIVE_MANIFOLD ACTIVE_MANIFOLD_LEAN ACTIVE_MANIFOLD_RELAY; do
  S="ternary27b-subj-$P"
  if [ "$P" = ACTIVE_MANIFOLD_RELAY ]; then EXTRA="--relay-sign 1 --relay-alpha 0.30"; else EXTRA=""; fi
  run_turn "$S" "$P" 1 "$Q1" "$EXTRA"
  run_turn "$S" "$P" 2 "$Q2" "$EXTRA"
done
echo "MATRIX_COMPLETE $(date +%H:%M:%S)"