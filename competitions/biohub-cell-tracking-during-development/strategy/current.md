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
  epoch 20), with proxy 0.9362. Its raw LB submission ref `55797775` scored 0.805 public.
- Artifact-free public-harness post-processing reduced that checkpoint from 200,201 to 179,571
  nodes, increased edge/node ratio from 0.783 to 0.953, and reduced division-like sources from
  5,898 to 681. Submission ref `55798388` scored 0.869 public: +0.064 over the identical raw
  checkpoint and +0.082 over the epoch-5 anchor.
- EXP-0004 completed all 50 epochs and wrote periodic checkpoints through completed epoch 50. The
  best full-run held-out proxy was 0.9381 at zero-based epoch 33 (completed epoch 34).
- The completed-epoch-34 best checkpoint and completed-epoch-50 final checkpoint scored 0.874 and
  0.877 public with the same artifact-free post-processing profile as refs `55805307` and
  `55805308`.
- The competition-metric-selected completed-epoch-30 checkpoint scored 0.890 public as ref
  `55810126`; this is the fixed control for post-processing calibration.
- EXP-0007's disjoint epoch-5 report screen scored 0.568826 for spatial arm A, 0.556438 for
  temporal arm B, and 0.457876 for temporal/predicted-proposal arm C, against 0.691999 for the
  frozen host reference. Arm A therefore won the controlled comparison but still trailed the host;
  its Kaggle submission scored 0.626 public as ref `55823762`.
- EXP-0009's one-dataset/two-transition GPU smoke passed the frozen-host candidate-cache contract
  with candidate recall 1.0 and a 1,182,793,728-byte peak CUDA reservation under deployment-matched
  four-view TTA. The full cache plus 30-epoch run is active in tmux session `biohub-exp0009`; W&B
  run <https://wandb.ai/salax0116-private-email/biohub-cell-tracking/runs/70f9278e> records it.
- EXP-0010 holds checkpoint, detection threshold, edge threshold, TTA, smoothing, and division
  logic fixed and tests two one-factor post-processing hypotheses. Corrected precision arm ref
  `55829542` changes minimum component size from 6 to 7; recall arm ref `55828801` changes only the
  relaxed Hungarian relink gate from 10 to 12 micrometers. Both are pending public scoring. An
  earlier precision child-Notebook ref `55828867` was rejected because it could not rerun inference
  on hidden test datasets and is packaging-failure evidence only.
- `Biohub Harness 0926 Probe` version 1 scored 0.926 publicly, but is an independent public
  Notebook reference rather than an EXP-0004 result.
- EXP-0010 straddled the 2026-08-28 00:00 UTC daily reset. The recall arm used the final pre-reset
  slot. The rejected child Notebook, corrected precision ref `55829542`, and accidental concurrent
  duplicate ref `55829582` consumed three post-reset slots; two submissions remain today. The
  duplicate is the identical precision condition and is excluded from hypothesis interpretation.

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
| H007 | An nnU-Net-configured spatial backbone improves difficult endpoint recall at a fixed detection budget | EXP-0007 keeps detector/linker contracts and the calibration/report split fixed across arms | EXP-0007A fails to improve the fixed report or LB result over the host baseline | rejected at epoch 5 |
| H008 | Artifact-free topology post-processing materially improves a fixed checkpoint | identical epoch-19 weights scored 0.805 raw and 0.869 with post-processing (+0.064) | a repeat on another checkpoint or hidden/private evidence removes the gain | accepted |
| H011 | A three-frame candidate-graph residual improves continuation links without retraining detection | frozen EXP-0004 e30 logits plus a bounded local residual preserve the 0.890 control at zero initialization | completed-epoch-5 and epoch-30 EXP-0009 LB do not exceed 0.890 | running |
| H012 | Remaining node-count penalty is driven partly by transient six-node tracks | epoch-30 control is penalized for excess nodes; min-7 removes 4,212 nodes while retaining division components | fixed-checkpoint public LB does not exceed 0.890 | submitted |
| H013 | Remaining edge error is recall-limited and benefits from a wider relaxed motion gate | epoch-30 screen recall 0.7849 trails precision 0.8538; 12 µm adds 2,676 relaxed links on public test clips | fixed-checkpoint public LB does not exceed 0.890 | submitted |

## Priority Plan

1. Complete the EXP-0009 cache/head run and submit the completed-epoch-5 and epoch-30 milestones.
2. Compare EXP-0010 precision and recall arms against the fixed epoch-30 public score 0.890.
3. Keep the winning direction and tune only one adjacent value after fresh quota is available.
4. Inspect estimated total-node metadata before changing detection thresholds.
5. Compare a single organizer-recommended tracker family only after the baseline failure is measured.

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
- [x] Complete all 50 EXP-0004 epochs; best held-out proxy 0.9381 at completed epoch 34.
- [x] Confirm post-processing improves the identical epoch-19 checkpoint from 0.805 to 0.869 public.
- [x] Submit completed epochs 34 and 50 with the proven post-processing profile.
- [x] Complete the EXP-0007A/B/C epoch-5 fixed report comparison and submit the winning A arm.
- [x] Pass the EXP-0009 one-dataset/two-transition candidate-cache smoke gate.
- [x] Submit the epoch-30 component-minimum 7 and relaxed-relink 12 µm one-factor hypotheses.
- [ ] Calibrate node count and edge thresholds for the EXP-0004 checkpoint.
- [ ] Complete EXP-0007A to 50 epochs and apply its pinned checkpoint-selection/report gate.
- [x] Start the W&B-traced EXP-0009 full cache plus 30-epoch run in a persistent tmux session.
- [ ] Submit the EXP-0009 completed-epoch-5 and epoch-30 checkpoints through the guarded Notebook workflow.
