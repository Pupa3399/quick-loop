#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data2/wuguanting/quick_loop/data/huggingface}"
export CUDA_VISIBLE_DEVICES="${OURO_RETRIEVER_CUDA_VISIBLE_DEVICES:-0}"
export OURO_RETRIEVER_DEVICE="${OURO_RETRIEVER_DEVICE:-cuda:0}"
export OURO_FAISS_GPU="${OURO_FAISS_GPU:-1}"
export OURO_FAISS_DEVICE="${OURO_FAISS_DEVICE:-0}"
export OURO_FAISS_USE_FLOAT16="${OURO_FAISS_USE_FLOAT16:-0}"

if [[ "$HF_ENDPOINT" != "https://hf-mirror.com" ]]; then
  echo "HF_ENDPOINT must be https://hf-mirror.com" >&2
  exit 2
fi

exec uv run --no-sync uvicorn retriever.e5_server:app --host 127.0.0.1 --port 8000
