# syntax=docker/dockerfile:1.6
#
# pseudo-maker-ml — web UI for Swedish/English PII anonymization of documents.
#
# Stage 1 builds the ONNX models (KB-BERT export needs torch; Rampart is
# downloaded). Stage 2 is the runtime image: onnxruntime + the model files,
# no torch, works fully offline.
#
#   docker compose build                         # KB-BERT + Rampart (+ spaCy fallback)
#   docker compose build --build-arg WITH_SPACY=0   # smaller image, no spaCy fallback

FROM python:3.11-slim AS models
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install ".[export]" \
    && PSEUDO_MAKER_MODELS_DIR=/models pseudo-maker fetch-models

FROM python:3.11-slim AS base

ARG WITH_SPACY=1
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PSEUDO_MAKER_MODELS_DIR=/opt/pseudo-maker/models \
    HF_HUB_OFFLINE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md NOTICE ./
COPY src ./src
RUN if [ "$WITH_SPACY" = "1" ]; then \
        pip install --no-cache-dir ".[spacy]" && python -m spacy download sv_core_news_lg; \
    else \
        pip install --no-cache-dir .; \
    fi

COPY --from=models /models /opt/pseudo-maker/models

# Non-root user with a writable home for any runtime caches.
RUN groupadd --system --gid 10001 app \
    && useradd  --system --uid 10001 --gid app --home /home/app --create-home app
USER app

ENV PSEUDO_MAKER_HOST=0.0.0.0 \
    PSEUDO_MAKER_PORT=8765

EXPOSE 8765

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["sh", "-c", "pseudo-maker serve --host ${PSEUDO_MAKER_HOST} --port ${PSEUDO_MAKER_PORT} --no-browser"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; urllib.request.urlopen(f'http://127.0.0.1:${PSEUDO_MAKER_PORT}/api/health', timeout=3).read() or sys.exit(1)" || exit 1
