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
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Container-User UID 1000 (HF-Konvention) VOR allen COPY-Layern anlegen
RUN useradd -m -u 1000 user

WORKDIR /home/user/app

# 1) Python-Abhängigkeiten (root-pip → system-weit; torch 2.12.0 PyPI-Bundle
#    bringt CUDA-13-user-space mit, Host-Treiber liefert HF)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) App-Code
COPY --chown=user:user . /home/user/app

# 3) GF3-Gewichte vorgebacken (~6 GB, public repo, im HF-hub-Cache-Layout)
RUN HF_HOME=/home/user/.cache/huggingface \
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download('neuralworm/ternary-bonsai-2-27b-hf'); print('weights baked')"

# 4) Triton-Kernel-Cache (lokal auf sm_70-fähiger RTX 2060 gebakt, T4
#    ist sm_75 → gleiche Arch-Familie; Keys enthält keine Py-Version)
RUN mkdir -p /home/user/.triton/cache \
    && if [ -d /home/user/app/px_triton_cache ] && [ -n "$(ls -A /home/user/app/px_triton_cache 2>/dev/null)" ]; then \
         cp -a /home/user/app/px_triton_cache/. /home/user/.triton/cache/ \
         && chmod -R u+rwX /home/user/.triton/cache \
         && echo "triton cache baked"; \
    else \
         echo "WARNUNG: px_triton_cache leer/fehlt — Runtime-Compile wird stattfinden (Fallback)"; \
    fi

USER user

EXPOSE 7860

CMD ["python", "app.py"]