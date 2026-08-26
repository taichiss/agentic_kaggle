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
- EXP-0004 established the first model-based LB anchor: the epoch-5 organizer checkpoint scored
  0.787 public with fixed detection threshold 0.99 and edge threshold 0.5.
- EXP-0004 processed all four public test clips in 263.963 seconds and emitted 178,301 nodes plus
  135,077 edges, confirming that full offline model inference fits the Notebook contract.
- The best validation checkpoint through 25 completed epochs is zero-based epoch 19 (completed
  epoch 20), with proxy 0.9362. Its LB submission ref `55797775` is pending scoring.
- Artifact-free public-harness post-processing reduced that checkpoint from 200,201 to 179,571
  nodes, increased edge/node ratio from 0.783 to 0.953, and reduced division-like sources from
  5,898 to 681. Submission ref `55798388` is pending scoring.
- `Biohub Harness 0926 Probe` version 1 scored 0.926 publicly, but is an independent public
  Notebook reference rather than an EXP-0004 result.
- The live submission command reported zero submissions remaining after the post-processed
  epoch-19 submission on 2026-08-26 UTC.

## Open Questions

- [x] Confirm the downloaded train/test file inventory and array metadata.
- [ ] Inspect estimated total-node metadata for node-count calibration.
- [x] Choose dataset-disjoint validation after inspecting dataset identities.
- [x] Confirm the live daily submission quota and final notebook packaging requirements.
- [x] Measure public baseline runtime on all four public test clips.

## Working Hypotheses

| id | hypothesis | evidence | falsification | status |
| --- | --- | --- | --- | --- |
| H001 | The organizer baseline provides the safest first end-to-end contract test | organizer-maintained code plus EXP-0001 train/infer/GEFF/CSV/metric pass | it cannot scale beyond the deliberately tiny smoke configuration within resource limits | accepted |
| H002 | Physical-distance linking is a higher-priority first baseline than division tuning | edge term has 10x division weight and matching is anisotropic | simple linking fails structurally or division omission dominates observed error | proposed |
| H003 | Node-count calibration must be tracked per dataset | official metric penalizes over-prediction per sample | local metric and first LB probes show no sensitivity in the relevant range | proposed |
| H004 | Sparse-positive supervision is safer than treating unlabeled voxels/nodes as negatives | organizer states GT is sparse while inference must cover all cells | a controlled PU/semi-supervised alternative consistently improves hidden-LB evidence | accepted |
| H005 | Public dummy test success only predicts packaging reliability | organizer clarification says public clips may duplicate train and hidden test is larger/disjoint | organizer changes the public/hidden contract | accepted |
| H006 | Checkpoint-specific node/edge calibration is required before longer training | EXP-0002 improved training recall but emitted 7,711 nodes and zero edges under fixed thresholds | a broad checkpoint sweep yields stable counts and non-empty edges without calibration | accepted |
| H007 | An nnU-Net-configured spatial backbone improves difficult endpoint recall at a fixed detection budget | controlled configs keep target, loss, feature dimension, transformer, split, inference, and ILP fixed | EXP-0005B does not improve the recall-versus-node-count curve or downstream edge/division metrics over EXP-0005A | proposed |

## Priority Plan

1. Inspect estimated total-node metadata and calibrate detection/edge thresholds for the EXP-0004
   checkpoint without changing the model.
2. Complete the 50-epoch checkpoint series while preserving the embryo-grouped validation contract.
3. Vary one detection-count or linking decision per LB experiment from the 0.787 anchor.
4. Compare a single organizer-recommended tracker family only after the baseline failure is measured.
5. Add division logic after edge/linking and node-count failure modes are measured.

## Validation Plan

- split: embryo-prefix grouped fold 0; train group `6bba`, validation group `44b6`
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
- [x] Run the embryo-grouped organizer baseline and save five-epoch checkpoints as `EXP-0004`.
- [x] Package epoch 5 as an offline GPU Notebook and record the 0.787 public LB anchor.
- [ ] Calibrate node count and edge thresholds for the EXP-0004 checkpoint.
- [ ] Run the paired EXP-0005A/EXP-0005B backbone comparison after the shared GPU is available.
