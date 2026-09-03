# EXP-0009 milestone submission workflow

`run_temporal_graph_milestone_submission.py` connects each configured EXP-0009 periodic checkpoint
to the existing Dataset/Notebook packager and `kaggle-lb` ledger workflow. It accepts only completed
epoch 5 or 30, and reads checkpoint, Dataset, Kernel, variant, and post-processing identities from
`configs/exp-0009-host-tgraph3-residual-30e.toml`.

The default is a side-effect-free plan. It does not create a local bundle, publish a Kaggle Dataset
or Notebook, or spend a submission slot:

```bash
uv run \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --frozen \
  python competitions/biohub-cell-tracking-during-development/scripts/run_temporal_graph_milestone_submission.py \
  --completed-epoch 5 \
  --wait-for-checkpoint
```

Add `--execute` only when the milestone is authorized for submission. The executable workflow:

1. optionally waits for the configured atomic periodic checkpoint with a bounded timeout;
2. runs the temporal-graph packager in local validation mode;
3. publishes the private model Dataset and internet-disabled GPU Notebook;
4. polls `kaggle kernels status` until `COMPLETE`, failure, or timeout;
5. downloads the Kernel output and locates exactly one `submission.csv`;
6. runs both the Biohub structural validator and the platform manifest validator, requiring every
   sorted `data.test_dir/*.zarr` stem to be represented; and
7. invokes `kaggle-lb submit`, which waits for scoring and writes the submission ref, status, and
   public score to `strategy/lb-submissions.jsonl`.

Every completed step is atomically persisted under `artifacts/EXP-0009`. Restarting the same command
skips completed publication/download steps. A recorded submission ref is terminal for mutation:
the workflow will never spend a second slot for that epoch. A per-milestone file lock also prevents
two local workers from racing. The resolved Kaggle authentication environment is passed unchanged
to every workflow subprocess.
When `KAGGLE_API_TOKEN` is already set it is preserved. Otherwise the worker obtains the cached
OAuth access token with `kaggle auth print-access-token`, captures it without echoing it, and injects
it only into child-process environments; the token is never written to logs or workflow state.
Execution stops before any publish call if the configured local test directory contains no Zarr
datasets.

To launch both milestones beside training, use two independent workers. Epoch 5 proceeds as soon as
its checkpoint appears while epoch 30 remains in its checkpoint wait:

```bash
mkdir -p competitions/biohub-cell-tracking-during-development/artifacts/EXP-0009/submit-workers

nohup uv run \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --frozen \
  python competitions/biohub-cell-tracking-during-development/scripts/run_temporal_graph_milestone_submission.py \
  --completed-epoch 5 \
  --execute \
  --wait-for-checkpoint \
  --checkpoint-timeout 86400 \
  > competitions/biohub-cell-tracking-during-development/artifacts/EXP-0009/submit-workers/e5.log 2>&1 &

nohup uv run \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --frozen \
  python competitions/biohub-cell-tracking-during-development/scripts/run_temporal_graph_milestone_submission.py \
  --completed-epoch 30 \
  --execute \
  --wait-for-checkpoint \
  --checkpoint-timeout 86400 \
  > competitions/biohub-cell-tracking-during-development/artifacts/EXP-0009/submit-workers/e30.log 2>&1 &
```

Kernel and Leaderboard waits default to 12 hours. Override `--kernel-timeout`,
`--kernel-poll-interval`, `--leaderboard-timeout`, or `--leaderboard-poll-interval` when necessary.
If a bounded wait expires after Kaggle has already returned a submission ref, the ref remains in the
state file and a restart will not resubmit it.
