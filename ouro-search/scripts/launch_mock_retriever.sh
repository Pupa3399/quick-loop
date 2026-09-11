#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  UV_BIN="$PROJECT_ROOT/.uv-bootstrap/uv-0.9.9.data/scripts/uv"
fi

cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.uv-cache"
exec "$UV_BIN" run --frozen uvicorn retriever.server:app --host 127.0.0.1 --port 8000
