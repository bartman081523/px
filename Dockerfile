# Dockerfile — px-explorer-v4 Docker-Migration (2026-10-09)
# Ziel (User-Mandat): komplettes Image, das ALLES vorgebacken enthält —
# deps, causal_conv1d-Wheel, GF3-Gewichte, Triton-Kernel-Cache — so dass
# zur LAUFZEIT nichts mehr kompiliert wird.
#
# Build-Zeit ist CPU-only (HF-Doku: keine GPU beim Build!) — der Triton-
# Cache wird daher lokal auf der RTX 2060 (sm_75 = T4-arch-identisch)
# in scratches/hfspace/bake_triton_cache.py erzeugt und als
# px_triton_cache/ am Repo-Root hier hinein-COPY't.
#
# HF-Konventionen: Container läuft als UID 1000, app_port 7860
# (README frontmatter sdk: docker + app_port: 7860).

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRITON_CACHE_DIR=/home/user/.triton/cache \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:256 \
    SPACE_ID=neuralworm/px-explorer-v4

# System-Bibliotheken: matplotlib/Gradio-Abhängigkeiten (libgl1, libglib2.0)
# + curl (Triton-Cache von px-wheels, build-time)
# + gcc + libc6-dev (C-Header): der Triton-Kernel-Launcher (.so, mit
#   python3.13-Tag) wird per C-Compiler gelinkt und LIEGT im Cache.
#   FALSIFIZIERT 2026-10-09: gcc allein reicht NICHT — python:3.13-slim
#   hat keine libc-Header (fatal error: stdlib.h, Bake-Versuch #1 im
#   HF-identischen Lokal-Image). Mit libc6-dev compilet der Launcher
#   sauber; im py3.13-nativen gebakten Cache ist das .so enthalten
#   → zur Laufzeit kein Compile mehr (gcc+Header bleiben als
#   Sicherheitsnetz für unerwartete Cache-Misses = Graceful Degradation).
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl gcc libc6-dev libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Container-User UID 1000 (HF-Konvention) VOR allen COPY-Layern anlegen
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# 1) Python-Abhängigkeiten (root-pip → system-weit; torch 2.12.0 PyPI-Bundle
#    bringt CUDA-13-user-space mit, Host-Treiber liefert HF).
#    --only-binary=:all: = Wheel-Reinheits-Assert: fehlt eine Wheel-Version
#    für cp313, failt der Build SOFORT mit klarer Meldung statt still ein
#    sdist ohne C-Compiler zu kompilieren (numpy==2.0.0-Beispiel, Build
#    2026-10-09 05:42).
COPY requirements.txt .
RUN pip install --no-cache-dir --only-binary=:all: -r requirements.txt

# 2) App-Code
COPY --chown=user:user . /home/user/app
#    FALSIFIZIERT 2026-10-09 (RUNTIME_ERROR, Build #23): WORKDIR legt
#    /home/user/app ROOT-owned an; COPY --chown chown't nur die kopierten
#    Inhalte NICHT das Zielverzeichnis → uid-1000-Runtime kann keine neuen
#    Unterordner anlegen → PermissionError '/home/user/app/telemetry'
#    (telemetry.py:50, import-Zeit). Fix: chown auf das DIR selbst (+ alle
#    Inhalte defensiv), telemetry/ + sessions/ voranlegen (ensure_session_dir
#    ist exists-guarded, FileExistsError unmöglich; infinite_context-l2
#    default None → nur bei aktivem Long-Context, dann abgedeckt).

# 3) GF3-Gewichte vorgebacken (~6 GB, im HF-hub-Cache-Layout).
#    FALSIFIZIERT 2026-10-09: angenommen war "public" — tatsächliche
#    anon-API: 401 (Repo privat). Das Space-Secret HF_TOKEN wird vom
#    HF-Docker-SDK automatisch als Build-Secret bereitgestellt
#    (id=Secretnach-Settings-Name) → hier mounten; im Image-Landet das
#    Token NICHT (nur als Build-Input gelesen).
RUN --mount=type=secret,id=HF_TOKEN,mode=0444,required=true \
    HF_TOKEN=$(cat /run/secrets/HF_TOKEN) \
    HF_HOME=/home/user/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download('neuralworm/ternary-bonsai-2-27b-hf'); print('weights baked')"

# 4) Triton-Kernel-Cache: lokal auf der RTX 2060 gebakt (sm_75 = T4-Arch-
#    Familie) — py3.13-NATIV (python 3.13.16 / torch 2.12.0+cu130 /
#    triton 3.7.0, 79 Entries = exakt der Kanon des py3.10-Bakes:
#    Phase-1 Δ+71, Phase-2 T=71 Δ+8, EXTRA T=54/59 je Δ0; Bake im
#    HF-identischen Image, 546,6 s — CACHE_MANIFEST in der Bake-Spur).
#    FALSIFIZIERT 2026-10-09: der py3.10-Cache versagt in py3.13 — der
#    Kernel-Launcher (.so) hat einen python-Tag im Namen
#    (cuda_utils.cpython-313-…) → py3.10-Launcher nicht ladbar →
#    Recompile-Versuch zur Laufzeit → C-Compiler-Pfad (war auf HF
#    "Failed to find C compiler", Warmup fail-soft, seeded=False).
#    HF-git lehnt Binärdateien im plain-push ab ("use xet") → der Cache
#    liegt als sha256-verifiziertes tar.gz im öffentlichen Repo
#    neuralworm/px-wheels und wird zur BUILD-Zeit geladen.
#    LAUFZEIT: keine Kompilation (gcc+libc6-dev nur als Sicherheitsnetz).
#    Beide Runtime-Verzeichnisse sind Root-geführten Layern entstanden
#    (snapshot_download als root, tar als root) → chown auf den
#    Container-User, sonst PermissionError beim ersten HF-Load
#    (.locks/etag-Writebacks) bzw. Triton-Lockfile-Warn-Spam.
RUN mkdir -p /home/user/.triton/cache \
    && curl -fL --retry 3 \
        "https://huggingface.co/neuralworm/px-wheels/resolve/main/px_triton_cache_py313_20261009.tar.gz" \
        -o /tmp/triton_cache.tar.gz \
    && echo "4e9e1097e0ac55f3aaf860250b72f31186ac8070d2da1cca7a7951efbbc2a706  /tmp/triton_cache.tar.gz" \
        | sha256sum -c - \
    && tar -xzf /tmp/triton_cache.tar.gz -C /home/user/.triton/cache \
    && rm /tmp/triton_cache.tar.gz \
    && chown -R user:user /home/user/.cache/huggingface /home/user/.triton/cache \
    && chmod -R u+rwX /home/user/.triton/cache /home/user/app \
    && mkdir -p /home/user/app/telemetry /home/user/app/sessions \
    && chown -R user:user /home/user/app \
    && echo "triton cache baked (sha verified), app runtime-writable"; \
    ls /home/user/.triton/cache | head -2

USER user

EXPOSE 7860

CMD ["python", "app.py"]