#!/usr/bin/env bash
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required to bootstrap the quality harness." >&2
  exit 1
fi

uv sync --dev

if git rev-parse --show-toplevel >/dev/null 2>&1; then
  uv run pre-commit install
  echo "Quality harness bootstrapped: dev dependencies synced and pre-commit installed."
else
  echo "Dev dependencies synced."
  echo "Git repository is not initialized yet, so pre-commit was not installed."
  echo "After 'git init', run: uv run pre-commit install"
fi
