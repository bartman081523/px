#!/usr/bin/env bash
# ── Lokaler Space-Start mit allen Debugs + HTTPS ──
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

export DEBUG_ROUTING=1
export DEBUG_PX=1
export SUBJECTIVE_TELEMETRY=1
export PX_PORT=7860
export PX_HOST=0.0.0.0

# ── SSL-Zertifikate ──
SSL_DIR="${SCRIPT_DIR}/ssl"
export SSL_CERTFILE="${SSL_DIR}/cert.pem"
export SSL_KEYFILE="${SSL_DIR}/key.pem"

if [ ! -f "$SSL_CERTFILE" ] || [ ! -f "$SSL_KEYFILE" ]; then
    echo "[SSL] Generiere Selbstsigniertes Zertifikat..."
    mkdir -p "$SSL_DIR"
    openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$SSL_KEYFILE" \
        -out "$SSL_CERTFILE" \
        -days 365 \
        -subj "/CN=localhost" \
        2>/dev/null
    echo "[SSL] Zertifikat erstellt: $SSL_CERTFILE"
fi

echo "=== PX-DMT Local Debug Start (HTTPS) ==="
echo "HOST: $PX_HOST"
echo "PORT: $PX_PORT"
echo "DEBUG_ROUTING=$DEBUG_ROUTING"
echo "DEBUG_PX=$DEBUG_PX"
echo "SUBJECTIVE_TELEMETRY=$SUBJECTIVE_TELEMETRY"
echo "SSL_CERTFILE=$SSL_CERTFILE"
echo "========================================="

# Ggf. alten Prozess auf Port $PX_PORT killen
if lsof -ti:$PX_PORT >/dev/null 2>&1; then
    echo "[!] Port $PX_PORT belegt — kille alten Prozess..."
    kill $(lsof -ti:$PX_PORT) 2>/dev/null || true
    sleep 2
fi

PYTHON="/run/media/julian/ML4/open-mythos_p2/venv_openmythos/bin/python"
PYTHONUNBUFFERED=1 $PYTHON app.py 2>&1 | tee -a "${SCRIPT_DIR}/local_debug.log"
