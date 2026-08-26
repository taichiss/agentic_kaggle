# BioHub Python environment

The default environment covers data inspection and graph I/O without forcing a GPU framework:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment
```

Add the organizer-style PyTorch baseline only when training or inference is required:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment --extra baseline
```

Local native visualization is optional:

```bash
uv sync --project competitions/biohub-cell-tracking-during-development/environment --extra viz
```

Do not add all organizer-listed trackers to this environment. Create an experiment-specific project
after selecting one candidate in `strategy/current.md`.
