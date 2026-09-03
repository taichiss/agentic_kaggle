# BioHub Python environment

The default environment covers data inspection and graph I/O without forcing a GPU framework:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment
```

Add the organizer-style PyTorch baseline only when training or inference is required:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment --extra baseline
```

Add online W&B experiment tracking together with the baseline when requested:

```bash
uv sync \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --extra baseline --extra tracking
```

W&B credentials remain in the user's standard credential store. Never place an API key in this
repository or an experiment config.

Install the pinned nnU-Net source for the controlled backbone A/B experiment:

```bash
uv sync \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --extra baseline --extra nnunet
```

The `nnunet` extra is pinned to an exact MIC-DKFZ/nnUNet Git revision. It is used as a feature
backbone, not as a standard dense-mask nnU-Net training pipeline.

Local native visualization is optional:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment --extra viz
```

Do not add all organizer-listed trackers to this environment. Create an experiment-specific project
after selecting one candidate in `strategy/current.md`.
