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

Build the pinned BioHub inspection/I/O environment:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment
```

The package policy and organizer-recommended tracker candidates are documented in
[the ecosystem decision](docs/overview/ecosystem.md) and the
[organizer Discussion note](docs/discussion/organizer-welcome.md).

The frozen topology-repair settings and the current precision/recall probes are
summarized in [the post-processing parameter inventory](docs/overview/postprocessing.md).

## Organizer baseline smoke run

The pinned organizer repository provides the joint 3D U-Net/temporal-attention detector and
cross-attention linker, sparse-supervision training, GEFF graph I/O, CSV conversion, metrics, and
Napari visualization. Run its smallest reproducible real-data contract check with:

```bash
uv sync \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --extra baseline
uv run \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --extra baseline \
  python competitions/biohub-cell-tracking-during-development/scripts/run_host_baseline_smoke.py
```

The smoke config uses one dataset, three frames, one training iteration, and a reduced model. It
checks CUDA training, checkpoint reload, inference, GEFF → CSV → GEFF, submission validation, and
the organizer metric. Outputs remain under ignored `artifacts/`, `submissions/`, and the ignored
organizer checkout's `weights/` directory.

To run the slightly longer W&B-traced check:

```bash
uv run \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --extra baseline --extra tracking \
  python competitions/biohub-cell-tracking-during-development/scripts/run_host_baseline_smoke.py \
  --config competitions/biohub-cell-tracking-during-development/configs/exp-0002-wandb-extended.toml
```

W&B receives configuration, parsed epoch metrics, runtime/system metrics, output counts, the local
score, and a checkpoint checksum. Competition data, predictions, submissions, and checkpoint files
remain local.

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
