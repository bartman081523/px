#!/usr/bin/env bash
# scripts/run_full_tests.sh — vollständige Testsuite ohne VRAM-Kollision.
#
# Warum: die GPU-Tests laden echte Modelle (4–10 GB), die Live-Tests brauchen
# einen laufenden HTTPS-Server auf :7860, der selbst Modelle hält. Beides in
# einem pytest-Lauf bedeutet OOM — nicht weil ein Test falsch ist, sondern
# weil sich zwei Prozesse um 12 GB streiten.
#
# Dieser Runner trennt die Phasen:
#   1) Standard-Suite            (kein Modell, kein Server)
#   2) --gpu                     (echte Modell-Loads, GPU muss frei sein)
#   3) --live                    (HTTPS-App auf :7860, nur Live-Tests)
#
# Jede Phase hat einen eigenen Exit-Code; am Ende gibt es eine Bilanz.
set -uo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python3}"
PORT="${PX_PORT:-7860}"
SSL_DIR="$PWD/ssl"
APP_PID=""

log() { printf '\n=== %s ===\n' "$*"; }

free_gib() {
  "$PY" - <<'EOF' 2>/dev/null || echo 0
import torch
free, total = torch.cuda.mem_get_info()
print(round(free / 2**30, 1))
EOF
}

stop_port() {
  local pid
  pid=$(ss -ltnp 2>/dev/null | grep ":${1}\b" | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1)
  if [ -n "${pid}" ]; then
    echo "[runner] stoppe Prozess ${pid} auf :${1}"
    kill "${pid}" 2>/dev/null || true
    sleep 4
  fi
}

cleanup() {
  if [ -n "${APP_PID}" ] && kill -0 "${APP_PID}" 2>/dev/null; then
    kill "${APP_PID}" 2>/dev/null || true
    wait "${APP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

rc_default=0; rc_gpu=0; rc_live=0

# ── Phase 0: Preflight ────────────────────────────────────────────────────
log "Preflight"
"$PY" - <<'EOF' || true
import torch
if torch.cuda.is_available():
    free, total = torch.cuda.mem_get_info()
    print(f"[runner] GPU: {free/2**30:.1f} GiB frei von {total/2**30:.1f} GiB")
else:
    print("[runner] keine CUDA-GPU sichtbar")
EOF
if ss -ltn 2>/dev/null | grep -q ":9931"; then
  echo "[runner] Hinweis: llama-server auf :9931 belegt ~11 GiB."
  echo "[runner]          Für Phase 2 wird er gestoppt (danach nicht neu gestartet)."
  stop_port 9931
fi

# ── Phase 1: Standard-Suite ───────────────────────────────────────────────
log "Phase 1/3 — Standard-Suite (ohne GPU-Modelle, ohne Server)"
"$PY" -m pytest -q
rc_default=$?

# ── Phase 2: GPU-Tests (Server vorher weg) ────────────────────────────────
log "Phase 2/3 — GPU-Tests (echte Modell-Loads)"
stop_port "$PORT"
echo "[runner] freier VRAM vor Phase 2: $(free_gib) GiB"
RUN_GPU_TESTS=1 "$PY" -m pytest -q --gpu
rc_gpu=$?

# ── Phase 3: Live-Tests gegen HTTPS-App ───────────────────────────────────
log "Phase 3/3 — Live-Tests (HTTPS-App auf :${PORT})"
if [ ! -f "$SSL_DIR/cert.pem" ] || [ ! -f "$SSL_DIR/key.pem" ]; then
  echo "[runner] keine Zertifikate in $SSL_DIR — erzeuge selbstsigniertes Paar"
  mkdir -p "$SSL_DIR"
  openssl req -x509 -newkey rsa:2048 -nodes \
    -keyout "$SSL_DIR/key.pem" -out "$SSL_DIR/cert.pem" -days 365 \
    -subj "/CN=localhost" 2>/dev/null
fi

set -a
[ -f .env ] && . ./.env
set +a
PX_PORT="$PORT" PX_HOST=0.0.0.0 \
SSL_CERTFILE="$SSL_DIR/cert.pem" SSL_KEYFILE="$SSL_DIR/key.pem" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
nohup "$PY" app.py > /tmp/opencode/px_app_tests.log 2>&1 &
APP_PID=$!
echo "[runner] app.py pid=${APP_PID}, warte auf HTTPS-Endpoint ..."

ready=0
for _ in $(seq 1 60); do
  if curl -sk --max-time 2 "https://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1; break
  fi
  sleep 2
done
if [ "$ready" -eq 1 ]; then
  "$PY" -m pytest -q --live
  rc_live=$?
else
  echo "[runner] FEHLER: App wurde nicht bereit — Live-Phase übersprungen"
  rc_live=2
fi

cleanup
APP_PID=""

# ── Bilanz ────────────────────────────────────────────────────────────────
log "Bilanz"
printf 'Standard-Suite : %s\n' "$([ $rc_default -eq 0 ] && echo PASS || echo "FAIL (rc=$rc_default)")"
printf 'GPU-Tests      : %s\n' "$([ $rc_gpu -eq 0 ] && echo PASS || echo "FAIL (rc=$rc_gpu)")"
printf 'Live-Tests     : %s\n' "$([ $rc_live -eq 0 ] && echo PASS || echo "FAIL (rc=$rc_live)")"

if [ $rc_default -ne 0 ] || [ $rc_gpu -ne 0 ] || [ $rc_live -ne 0 ]; then
  exit 1
fi
echo "Alle Phasen grün."
