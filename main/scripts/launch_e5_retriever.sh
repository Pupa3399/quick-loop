#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/main/models/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_ROOT/main/datasets/huggingface}"
export OURO_WIKI18_DIR="${OURO_WIKI18_DIR:-$PROJECT_ROOT/main/datasets/wiki18}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.venv/cache/uv}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DATASETS_OFFLINE="${DATASETS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${OURO_RETRIEVER_CUDA_VISIBLE_DEVICES:-0}"
export OURO_RETRIEVER_DEVICE="${OURO_RETRIEVER_DEVICE:-cuda:0}"
export OURO_FAISS_GPU="${OURO_FAISS_GPU:-1}"
export OURO_FAISS_DEVICE="${OURO_FAISS_DEVICE:-0}"
export OURO_FAISS_USE_FLOAT16="${OURO_FAISS_USE_FLOAT16:-0}"

if [[ "$HF_ENDPOINT" != "https://hf-mirror.com" ]]; then
  echo "HF_ENDPOINT must be https://hf-mirror.com" >&2
  exit 2
fi
if [[ "$HF_HUB_OFFLINE" != "1" || "$TRANSFORMERS_OFFLINE" != "1" || "$DATASETS_OFFLINE" != "1" ]]; then
  echo "Retriever must run with Hugging Face and datasets offline modes enabled" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
exec uv run --no-sync uvicorn ouro_search.retriever.e5_server:app --host 127.0.0.1 --port 8000
