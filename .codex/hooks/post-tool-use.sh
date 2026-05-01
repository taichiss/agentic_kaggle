#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT" || exit 0

if ! command -v uv >/dev/null 2>&1; then
  echo "post-tool-use: uv is not installed; skipping auto-format." >&2
  exit 0
fi

if ! uv run python scripts/quality_gate.py format >/dev/null 2>&1; then
  echo "post-tool-use: auto-format failed; continuing without blocking." >&2
fi

exit 0
