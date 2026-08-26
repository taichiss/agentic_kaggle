# Reproducible asset acquisition

All commands run from the repository root. Acquisition destinations are ignored by Git.

## 1. Kaggle authentication and joining

Install/sync the repository environment, authenticate with Kaggle CLI, and accept the competition
Rules in the browser. Never copy `kaggle.json` into this repository.

```bash
uv sync --extra dev
uv run kaggle config view
uv run kaggle competitions files biohub-cell-tracking-during-development
```

If the final command returns an authentication or permission error, complete Kaggle authentication
and join the competition before retrying. The fetch script does not attempt to accept Rules.

## 2. Competition data

```bash
uv run python competitions/biohub-cell-tracking-during-development/scripts/fetch_assets.py data
```

This downloads the 87.61 GB catalog into `data/downloads/` and extracts ZIP archives into
`data/raw/`. Existing extracted files are not overwritten unless `--force` is supplied. Confirm disk
space and the intended download before running it; the script never starts a data download as a
side effect of another command.

## 3. Organizer baseline and public notebooks

```bash
uv run python competitions/biohub-cell-tracking-during-development/scripts/fetch_assets.py baseline
uv run python competitions/biohub-cell-tracking-during-development/scripts/fetch_assets.py notebooks
```

Sources and destinations are versioned in `asset-sources.toml`. The baseline is a public GitHub
checkout. The pinned notebooks use Kaggle's public read API and can be recorded without credentials;
the response metadata and notebook source are saved together. Existing destinations are left
untouched, making retries safe. The authenticated Kaggle CLI remains available for additional
public notebooks that have no public API source configured.

To discover other public notebooks without downloading them:

```bash
uv run kaggle kernels list \
  --competition biohub-cell-tracking-during-development \
  --sort-by voteCount --page-size 20
```

Record reviewed additions in `asset-sources.toml`; do not bulk-copy arbitrary public solutions into
tracked source code.

## 4. Verify inventory

```bash
uv run python competitions/biohub-cell-tracking-during-development/scripts/inspect_data.py
git status --short --ignored
```

Only ignored `data/` paths should contain downloaded assets.
