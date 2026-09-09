#!/usr/bin/env bash
set -euo pipefail

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/data2/wuguanting/quick_loop/data/huggingface}"

if [[ "$HF_ENDPOINT" != "https://hf-mirror.com" ]]; then
  echo "HF_ENDPOINT must be https://hf-mirror.com" >&2
  exit 2
fi

exec uv run --no-sync uvicorn retriever.e5_server:app --host 127.0.0.1 --port 8000
