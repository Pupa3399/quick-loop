#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
PYPI_INDEX="https://mirrors.aliyun.com/pypi/simple/"
VERL_MIRROR="https://gitee.com/mirrors/verl.git"
VERL_COMMIT="ddd86f527a4af75095e4677b02b5aa272913a088"
EXPERIMENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd)"
VERL_DIR="${VERL_DIR:-$PROJECT_ROOT/.venv/src/verl}"
VERL_PATCH="$EXPERIMENT_DIR/verl-ouro-r3.patch"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.venv/cache/uv}"

cd "$PROJECT_ROOT"
"$UV_BIN" sync --extra train --extra retriever --index "$PYPI_INDEX"
if [[ -e "$VERL_DIR" && ! -d "$VERL_DIR/.git" ]]; then
  echo "Existing veRL path is not a Git worktree: $VERL_DIR" >&2
  exit 2
fi
if [[ ! -d "$VERL_DIR/.git" ]]; then
  mkdir -p "$(dirname "$VERL_DIR")"
  git clone "$VERL_MIRROR" "$VERL_DIR"
fi
git -C "$VERL_DIR" fetch "$VERL_MIRROR" "$VERL_COMMIT"
git -C "$VERL_DIR" checkout --detach "$VERL_COMMIT"
if git -C "$VERL_DIR" apply --unidiff-zero --reverse --check "$VERL_PATCH" 2>/dev/null; then
  echo "veRL Ouro patch already applied"
else
  git -C "$VERL_DIR" apply --unidiff-zero --check "$VERL_PATCH"
  git -C "$VERL_DIR" apply --unidiff-zero "$VERL_PATCH"
fi
"$UV_BIN" pip install --no-deps --editable "$VERL_DIR" --index-url "$PYPI_INDEX"
