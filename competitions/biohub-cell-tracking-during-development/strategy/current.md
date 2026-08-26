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
- Inference must track all cells even though evaluation uses a random sparse subset.
- The four public test clips are dummy notebook smoke inputs; the larger hidden test does not overlap
  train according to the organizer.
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
| H004 | Sparse-positive supervision is safer than treating unlabeled voxels/nodes as negatives | organizer states GT is sparse while inference must cover all cells | a controlled PU/semi-supervised alternative consistently improves hidden-LB evidence | accepted |
| H005 | Public dummy test success only predicts packaging reliability | organizer clarification says public clips may duplicate train and hidden test is larger/disjoint | organizer changes the public/hidden contract | accepted |

## Priority Plan

1. Acquire and inventory the official data without loading full volumes.
2. Run the organizer baseline on one dataset/frame window and validate graph/CSV round trip.
3. Establish the pinned Zarr/GEFF/tracksdata environment before adding external trackers.
4. Package the smallest notebook that processes every test dataset and emits a schema-valid file.
5. Establish an LB anchor, then vary one detection/linking decision per experiment.
6. Compare a single organizer-recommended tracker family only after the baseline failure is measured.
7. Add division logic after edge/linking and node-count failure modes are measured.

## Validation Plan

- split: dataset-disjoint; exact folds pending data inventory
- local metric: organizer `tracking_cellmot` implementation
- leakage checks: no frames from the same dataset across train/validation; no test-derived labels
- submission checks: exact header, consecutive IDs, row sentinel values, node references scoped by
  dataset, all test dataset names represented
- heavy CV: not a default gate under TRUST-LB

## Next Actions

- [x] Authenticate Kaggle CLI and start `fetch_assets.py data`.
- [ ] Complete data extraction and run `inspect_data.py`.
- [x] Sync the pinned BioHub core environment and smoke-test Zarr/tracksdata imports.
- [ ] Record baseline dependency/runtime constraints as `EXP-0001`.
- [ ] Build a competition notebook only after the local one-sample round trip succeeds.
