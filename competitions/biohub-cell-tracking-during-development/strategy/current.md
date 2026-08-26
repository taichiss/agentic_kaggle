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
- The 2026-08-26 local inventory matches the catalog: 24,886 files and 87,609,892,618 bytes.
- The extracted tree contains 199 train Zarr/GEFF pairs and four public test Zarr datasets.
- All 203 image arrays have shape `(100, 64, 256, 256)`, dtype `uint16`, and chunks
  `(1, 64, 256, 256)`.
- The downloaded `sample_submission.csv` passes the local schema/reference validator: four
  datasets, 12 nodes, and eight edges.
- EXP-0002 confirmed online W&B tracing and a five-epoch optimizer trend on ten real frames.
- EXP-0002 improved detection loss and validation node recall, but its fixed inference thresholds
  produced 7,711 nodes and zero edges; more epochs are not the next bottleneck.
- EXP-0003 verified private Notebook push, execution, output download, Code Competition submission,
  and public-score retrieval using Kaggle CLI only.
- The live submission command reported four submissions remaining after EXP-0003, implying a
  five-submission daily allowance at the time checked on 2026-08-26.

## Open Questions

- [x] Confirm the downloaded train/test file inventory and array metadata.
- [ ] Inspect estimated total-node metadata for node-count calibration.
- [ ] Choose dataset-disjoint validation after inspecting dataset identities.
- [x] Confirm the live daily submission quota and final notebook packaging requirements.
- [ ] Measure public baseline runtime and memory on a small real subset.

## Working Hypotheses

| id | hypothesis | evidence | falsification | status |
| --- | --- | --- | --- | --- |
| H001 | The organizer baseline provides the safest first end-to-end contract test | organizer-maintained code plus EXP-0001 train/infer/GEFF/CSV/metric pass | it cannot scale beyond the deliberately tiny smoke configuration within resource limits | accepted |
| H002 | Physical-distance linking is a higher-priority first baseline than division tuning | edge term has 10x division weight and matching is anisotropic | simple linking fails structurally or division omission dominates observed error | proposed |
| H003 | Node-count calibration must be tracked per dataset | official metric penalizes over-prediction per sample | local metric and first LB probes show no sensitivity in the relevant range | proposed |
| H004 | Sparse-positive supervision is safer than treating unlabeled voxels/nodes as negatives | organizer states GT is sparse while inference must cover all cells | a controlled PU/semi-supervised alternative consistently improves hidden-LB evidence | accepted |
| H005 | Public dummy test success only predicts packaging reliability | organizer clarification says public clips may duplicate train and hidden test is larger/disjoint | organizer changes the public/hidden contract | accepted |
| H006 | Checkpoint-specific node/edge calibration is required before longer training | EXP-0002 improved training recall but emitted 7,711 nodes and zero edges under fixed thresholds | a broad checkpoint sweep yields stable counts and non-empty edges without calibration | accepted |

## Priority Plan

1. Inspect estimated total-node metadata and calibrate detection/edge thresholds on EXP-0002.
2. Establish a dataset-disjoint validation split and run one complete held-out dataset.
3. Package the smallest notebook that processes every test dataset and emits a schema-valid file.
4. Establish an LB anchor, then vary one detection/linking decision per experiment.
5. Compare a single organizer-recommended tracker family only after the baseline failure is measured.
6. Add division logic after edge/linking and node-count failure modes are measured.

## Validation Plan

- split: dataset-disjoint; exact folds pending data inventory
- local metric: organizer `tracking_cellmot` implementation
- leakage checks: no frames from the same dataset across train/validation; no test-derived labels
- submission checks: exact header, consecutive IDs, row sentinel values, node references scoped by
  dataset, all test dataset names represented
- heavy CV: not a default gate under TRUST-LB

## Next Actions

- [x] Authenticate Kaggle CLI and start `fetch_assets.py data`.
- [x] Complete data extraction and run `inspect_data.py`.
- [x] Sync the pinned BioHub core environment and smoke-test Zarr/tracksdata imports.
- [x] Validate the official sample submission against all four public test dataset names.
- [x] Record baseline dependency/runtime constraints as `EXP-0001`.
- [x] Run the organizer baseline train/infer/GEFF/CSV/metric smoke path on three real frames.
- [x] Trace five epochs and five updates per epoch in W&B as `EXP-0002`.
- [x] Submit private Notebook version 1 through Kaggle CLI and record public score as `EXP-0003`.
- [ ] Calibrate node count and edge thresholds for the EXP-0002 checkpoint.
- [ ] Establish a dataset-disjoint baseline run before building the competition notebook.
