# Biohub - Cell Tracking During Development

- Kaggle slug: `biohub-cell-tracking-during-development`
- Task: 3D+time microscopy cell detection, temporal linking, and division recovery
- Metric: `adjusted_edge_jaccard + 0.1 * division_jaccard` (`maximize`)
- Official page: <https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview>

This directory is the competition-specific workspace. Downloaded data, copied Kaggle notebooks,
predictions, and submissions remain under ignored runtime directories and are never committed.

## Bootstrap

From the repository root:

```bash
# Public organizer baseline; Kaggle credentials are not required.
uv run python competitions/biohub-cell-tracking-during-development/scripts/fetch_assets.py baseline

# Requires Kaggle authentication and prior acceptance of the competition rules.
uv run python competitions/biohub-cell-tracking-during-development/scripts/fetch_assets.py data
uv run python competitions/biohub-cell-tracking-during-development/scripts/fetch_assets.py notebooks

# Inspect the downloaded layout without reading the image volumes.
uv run python competitions/biohub-cell-tracking-during-development/scripts/inspect_data.py
```

See [the acquisition guide](docs/overview/acquisition.md) for authentication and reproducibility
details.

## Lightweight development loop

The first implementation milestone is deliberately small: generate a graph for each test dataset,
write the required mixed node/edge CSV rows, then validate its structure before spending a Kaggle
submission.

```bash
uv run python competitions/biohub-cell-tracking-during-development/src/submission.py \
  validate competitions/biohub-cell-tracking-during-development/submissions/submission.csv \
  --test-dir competitions/biohub-cell-tracking-during-development/data/raw/test
```

The competition is a Code Competition. Final submissions must be produced by a Kaggle Notebook
named `submission.csv`; the direct CSV `kaggle-lb submit` path is not the final submission path for
this competition. Use the notebook workflow in [notebooks/README.md](notebooks/README.md), then use
`kaggle competitions submissions biohub-cell-tracking-during-development` to inspect scoring.

## Source boundaries

- `docs/overview/`: official contract and organizer-maintained technical references.
- `docs/kernel/`: public Kaggle notebook references; copied notebook files live under ignored data.
- `strategy/`: hypotheses, experiment ledger, and TRUST-LB decisions.
- `configs/`: versioned experiment settings, never credentials or machine-specific paths.
- `src/`: competition-specific reusable code.
- `data/`, `artifacts/`, `submissions/`: local runtime state, ignored by Git.
