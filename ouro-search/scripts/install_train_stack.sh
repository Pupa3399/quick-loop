#!/usr/bin/env bash
set -euo pipefail

UV_BIN="${UV_BIN:-uv}"
PYPI_INDEX="https://mirrors.aliyun.com/pypi/simple/"
VERL_MIRROR="https://gitee.com/mirrors/verl.git"
VERL_COMMIT="ddd86f527a4af75095e4677b02b5aa272913a088"
VERL_DIR="${VERL_DIR:-third_party/verl}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERL_PATCH="$PROJECT_ROOT/patches/verl-ouro-r3.patch"

"$UV_BIN" sync --extra train --extra retriever --index "$PYPI_INDEX"
if [[ ! -d "$VERL_DIR/.git" ]]; then
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
