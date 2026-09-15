#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UV_BIN="${UV_BIN:-uv}"

cd "$PROJECT_ROOT"
export UV_CACHE_DIR="$PROJECT_ROOT/.venv/cache/uv"
exec "$UV_BIN" run --frozen uvicorn ouro_search.retriever.server:app --host 127.0.0.1 --port 8000
