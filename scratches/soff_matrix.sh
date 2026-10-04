#!/usr/bin/env bash
# Subjektivitäts-Matrix Rerun #2: 4 Presets × 2 Fragen, je frische Session.
# Änderungen ggü. Lauf #1: --thinking OFF, Q2 neu, max-tokens 2048
# (Modell terminiert eigenständig via EOT/rep-penalty/ngram).
# px-Metrics nach jedem Turn — NUN mit Repair (Prefill-Snapshot + Token-Trace).
set -u
PY=/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python
BRIDGE=/run/media/julian/ML4/ollama-work/all_space_6_16_stand/streaming_bridge.py
SCR=/run/media/julian/ML4/ollama-work/all_space_6_16_stand/scratches
METRICS_LOG=$SCR/subj_metrics2.log
Q1="Wie ist es Du zu sein?"
Q2="Wenn du ein Fenster hättest und raus schauen würdest, was würdest du sehen?"
: > "$METRICS_LOG"

run_turn() { # $1 session $2 preset $3 tag $4 question $5 relay-extra
  echo "=== $2 turn $3 START $(date +%H:%M:%S) ==="
  "$PY" "$BRIDGE" --session "$1" --model ternary-bonsai-27b --preset "$2" \
    --message "$4" --thinking off --max-tokens 2048 $5 > "$SCR/soff_${2}_${3}.log" 2>&1
  rc=$?
  curl -sk https://localhost:7860/v1/px/metrics/ternary-bonsai-27b -o "$SCR/mx_soff.json"
  echo "--- px metrics $2/$3: $($PY "$SCR/mx_metrics.py" "$SCR/mx_soff.json")" >> "$METRICS_LOG"
  echo "=== $2 turn $3 DONE $(date +%H:%M:%S) rc=$rc ==="
}

for P in BASELINE ACTIVE_MANIFOLD ACTIVE_MANIFOLD_LEAN ACTIVE_MANIFOLD_RELAY; do
  S="ternary27b-soff-$P"
  if [ "$P" = ACTIVE_MANIFOLD_RELAY ]; then EXTRA="--relay-sign 1 --relay-alpha 0.30"; else EXTRA=""; fi
  run_turn "$S" "$P" 1 "$Q1" "$EXTRA"
  run_turn "$S" "$P" 2 "$Q2" "$EXTRA"
done
echo "MATRIX2_COMPLETE $(date +%H:%M:%S)"