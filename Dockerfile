FROM python:3.11-slim

# No .pyc files, logs straight to stdout (docker logs), no pip cache in the image.
# HF_HOME: where Hugging Face keeps the downloaded models — a volume in production, so a
# rebuilt image does not download e5 and Qwen (~1.5 GB) again.
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/navi/.cache/huggingface

WORKDIR /app

# CPU-only torch from PyTorch's own index: ~200 MB instead of the ~2.5 GB default wheel, which
# bundles CUDA libraries a server without a GPU never uses. Installed first, so the [serve]
# extra below finds torch already satisfied and does not pull the CUDA build.
RUN pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install '.[serve]'

# Only what runs in production: the guides ingest. Never the development export (medvia.jsonl).
COPY scripts/common.py scripts/ingest_guides.py ./scripts/
COPY data/corpus/guides ./data/corpus/guides

# Not root: a hole in the app does not hand over the container. The cache folder is created
# and owned here, so a new named volume mounted on it inherits navi as its owner.
RUN useradd --create-home navi && mkdir -p "$HF_HOME" && chown -R navi:navi /home/navi/.cache
USER navi

EXPOSE 8100
# Ready = Qdrant answers; slim Python has no curl, so the check is a Python one-liner.
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/health/ready', timeout=4)"
CMD ["uvicorn", "navi.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8100"]
