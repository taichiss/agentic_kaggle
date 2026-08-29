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
- EXP-0009's full cache and 30-epoch residual-head run completed. Frozen-host validation accuracy
  was 0.922163; the MLP peaked at 0.922684 at epoch 3 and fell to 0.920704 at epoch 30. Its epoch-30
  Notebook submission ref `55843163` scored 0.891 public, +0.001 over the frozen 0.890 control.
- The identical-profile EXP-0009 epoch-3 best Notebook completed with 167,087 nodes and 159,380
  edges. Code Competition submission ref `55854853` scored 0.890, so local top-1 selection did not
  beat the epoch-30 LB-selected head at 0.891.
- EXP-0011 reused the immutable EXP-0009 cache and trained a four-head candidate-set attention
  residual for ten epochs. It peaked at 0.922372 at epoch 1 and ended at 0.920704, below the MLP
  best. At epoch 10 it fixed 52 host mistakes but regressed 66 host-correct links; a diagnostic
  0.2 residual multiplier reached 0.922580 but still did not beat the MLP best.
- EXP-0012 centers candidate residuals and smoothly bounds them to ±0.15. Its epoch-3 best reached
  0.923101 by fixing 12 host mistakes while regressing three host-correct links, exceeding the MLP
  best by four correct links. Epoch 10 remained above the frozen host at 0.922476.
- EXP-0013 exported the fixed EXP-0008 EMA epoch-30 state and transferred only the proven
  minimum-component 7 post-process. Notebook version 2 emitted 169,090 nodes and 161,789 edges;
  submission ref `55866168` is pending. EXP-0008 without min-7 has not been submitted, so this is
  not yet a paired estimate of the post-process effect.
- EXP-0010 holds checkpoint, detection threshold, edge threshold, TTA, smoothing, and division
  logic fixed and tests two one-factor post-processing hypotheses. Corrected precision arm ref
  `55829542` changes minimum component size from 6 to 7; recall arm ref `55828801` changes only the
  relaxed Hungarian relink gate from 10 to 12 micrometers. Min-7 scored 0.893 (+0.003); gate-12
  scored 0.884 (-0.006), so pruning is retained and the wider relink gate is rejected. An earlier
  precision child-Notebook ref `55828867` could not rerun inference on hidden test datasets and is
  packaging-failure evidence only.
- `Biohub Harness 0926 Probe` version 1 scored 0.926 publicly, but is an independent public
  Notebook reference rather than an EXP-0004 result.
- EXP-0010 straddled the 2026-08-28 00:00 UTC daily reset. The duplicate ref `55829582` is the
  identical precision condition and is excluded from hypothesis interpretation. EXP-0009 e3 ref
  `55854853` consumed the final slot before the next daily reset.

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
| H011 | A three-frame candidate-graph residual improves continuation links without retraining detection | frozen EXP-0004 e30 logits plus a bounded local residual preserve the 0.890 control at zero initialization | completed-epoch-5 and epoch-30 EXP-0009 LB do not exceed 0.890 | e30 accepted at 0.891; local-best e3 tied control at 0.890 |
| H014 | Candidate-set attention improves ambiguous parent selection over independent candidate scoring | 97.5% of validation rows have multiple candidates and attention can compare their host margin and motion jointly | ten-epoch best does not exceed EXP-0009 MLP best 0.922684 | rejected in tested form |
| H015 | Centering and bounding Attention residuals preserves confident host links while correcting ambiguous choices | unbounded e10 fixed 52 but regressed 66; a ±0.15 pairwise-safe correction limits destructive flips | bounded ten-epoch best does not exceed MLP best 0.922684 | locally accepted at 0.923101; report/LB unverified |
| H016 | The min-component-7 precision correction transfers to the independently trained EXP-0008 EMA checkpoint | the identical change improved EXP-0004 e30 from 0.890 to 0.893 | an identical-weight EXP-0008 min-6 control matches or beats min-7 | ref `55866168` pending; unpaired deployment probe |
| H012 | Remaining node-count penalty is driven partly by transient six-node tracks | epoch-30 control is penalized for excess nodes; min-7 removes 4,212 nodes while retaining division components | fixed-checkpoint public LB does not exceed 0.890 | accepted at 0.893 |
| H013 | Remaining edge error is recall-limited and benefits from a wider relaxed motion gate | epoch-30 screen recall 0.7849 trails precision 0.8538; 12 µm adds 2,676 relaxed links on public test clips | fixed-checkpoint public LB does not exceed 0.890 | rejected at 0.884 |

## Priority Plan

1. Read EXP-0013 ref `55866168`; submit the identical EXP-0008 EMA e30 min-6 control only after
   explicit approval if a paired post-process estimate is needed.
2. Retain EXP-0009 epoch 30 at 0.891; do not use local top-1 alone for checkpoint selection.
3. Retain EXP-0012 epoch 3 and verify it on an independent report subset before increasing
   temporal-head capacity.
4. Adopt EXP-0010 min-component 7 as the post-processing control; do not retain the 12 µm gate.
5. Design cache schema v2 before calling a future arm a temporal GRU or global GNN.
6. Inspect estimated total-node metadata before changing detection thresholds.

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
- [ ] Submit the EXP-0009 completed-epoch-5 checkpoint through the guarded Notebook workflow.
- [x] Submit the EXP-0009 completed-epoch-30 checkpoint as Notebook version 2 (ref `55843163`).
- [x] Submit the EXP-0009 best epoch-3 checkpoint as Notebook version 2 (ref `55854853`).
- [x] Run the EXP-0011 candidate-set attention probe for ten epochs with W&B tracing.
- [x] Retrain centered ±0.15 bounded Attention for ten epochs as EXP-0012.
- [ ] Verify EXP-0012 best epoch 3 on an independent report subset or LB after explicit approval.
- [x] Submit EXP-0008 EMA epoch 30 with min-component 7 as EXP-0013 (ref `55866168`).
