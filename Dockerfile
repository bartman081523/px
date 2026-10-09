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
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl libgl1 libglib2.0-0 \
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
#    Familie, 79 Entries = Warmup-Kanon; Fremd-T 54/56/81/155 je Δ=0 —
#    siehe CACHE_MANIFEST.txt). HF-git lehnt Binärdateien im plain-push
#    ab ("use xet") → der Cache liegt als sha256-verifiziertes tar.gz im
#    öffentlichen Repo neuralworm/px-wheels und wird zur BUILD-Zeit
#    geladen. LAUFZEIT: keine Kompilation.
#    Beide Runtime-Verzeichnisse sind Root-geführten Layern entstanden
#    (snapshot_download als root, tar als root) → chown auf den
#    Container-User, sonst PermissionError beim ersten HF-Load
#    (.locks/etag-Writebacks) bzw. Triton-Lockfile-Warn-Spam.
RUN mkdir -p /home/user/.triton/cache \
    && curl -fL --retry 3 \
        "https://huggingface.co/neuralworm/px-wheels/resolve/main/px_triton_cache_20261009.tar.gz" \
        -o /tmp/triton_cache.tar.gz \
    && echo "d5ddc996f7bd63b5a6b3265f9aeddf30a1ad2ab8ae44b771216fa1d64d5bf624  /tmp/triton_cache.tar.gz" \
        | sha256sum -c - \
    && tar -xzf /tmp/triton_cache.tar.gz -C /home/user/.triton/cache \
    && rm /tmp/triton_cache.tar.gz \
    && chown -R user:user /home/user/.cache/huggingface /home/user/.triton/cache \
    && chmod -R u+rwX /home/user/.triton/cache \
    && echo "triton cache baked (sha verified)"; \
    ls /home/user/.triton/cache | head -2

USER user

EXPOSE 7860

CMD ["python", "app.py"]