# Current Strategy: Biohub - Cell Tracking During Development

## Competition Contract

- slug: `biohub-cell-tracking-during-development`
- metric: adjusted edge Jaccard plus `0.1 * division Jaccard`
- direction: maximize
- submission: a Code Competition notebook must emit `submission.csv`
- critical constraints: sparse labels, anisotropic physical scale, 7 micrometer node matching,
  no internet at submission time, 12-hour CPU/GPU notebook limit

## Confirmed Facts

- Linking quality dominates the score weight; division Jaccard contributes 0.1.
- Over-predicting the total node count reduces the adjusted edge score.
- Train annotations are sparse, so unlabeled cells are not confirmed negatives.
- Submission rows mix nodes and edges and must include every test dataset.
- Official overview and organizer metric repository were checked on 2026-08-26.

## Open Questions

- [ ] Confirm the downloaded train/test file inventory and array metadata.
- [ ] Confirm dataset counts, volume shapes, dtypes, chunks, and estimated total-node metadata.
- [ ] Choose dataset-disjoint validation after inspecting dataset identities.
- [ ] Confirm the live daily submission quota and final notebook packaging requirements.
- [ ] Measure public baseline runtime and memory on a small real subset.

## Working Hypotheses

| id | hypothesis | evidence | falsification | status |
| --- | --- | --- | --- | --- |
| H001 | The organizer baseline provides the safest first end-to-end contract test | organizer-maintained code implements graph I/O and metric | it cannot round-trip one downloaded sample into a valid CSV within resource limits | proposed |
| H002 | Physical-distance linking is a higher-priority first baseline than division tuning | edge term has 10x division weight and matching is anisotropic | simple linking fails structurally or division omission dominates observed error | proposed |
| H003 | Node-count calibration must be tracked per dataset | official metric penalizes over-prediction per sample | local metric and first LB probes show no sensitivity in the relevant range | proposed |

## Priority Plan

1. Acquire and inventory the official data without loading full volumes.
2. Run the organizer baseline on one dataset/frame window and validate graph/CSV round trip.
3. Package the smallest notebook that processes every test dataset and emits a schema-valid file.
4. Establish an LB anchor, then vary one detection/linking decision per experiment.
5. Add division logic only after edge/linking and node-count failure modes are measured.

## Validation Plan

- split: dataset-disjoint; exact folds pending data inventory
- local metric: organizer `tracking_cellmot` implementation
- leakage checks: no frames from the same dataset across train/validation; no test-derived labels
- submission checks: exact header, consecutive IDs, row sentinel values, node references scoped by
  dataset, all test dataset names represented
- heavy CV: not a default gate under TRUST-LB

## Next Actions

- [ ] Authenticate Kaggle CLI and accept Rules.
- [ ] Run `fetch_assets.py data` and `inspect_data.py`.
- [ ] Record baseline dependency/runtime constraints as `EXP-0001`.
- [ ] Build a competition notebook only after the local one-sample round trip succeeds.
