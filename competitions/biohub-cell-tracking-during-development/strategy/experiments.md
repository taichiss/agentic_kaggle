# Experiments: Biohub - Cell Tracking During Development

`LB` remains `not_submitted` until Kaggle reports a score. Artifacts stay outside Git; record a URI,
run ID, or checksum sufficient to identify them.

| id | date | hypothesis | data version | commit | config | seed/fold | CV | LB | artifact/run | takeaway | next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0001 | 2026-08-26 | H001 organizer baseline can establish an end-to-end contract | official 2026-08-26 inventory | `6f85aa7` | `configs/exp-0001-host-smoke.toml` | 20260826 / same-video smoke | 0.0000 (contract-only) | not_submitted | `artifacts/EXP-0001/result.json` | train/infer/GEFF/CSV/metric path passed | establish a real dataset-disjoint baseline run |
| EXP-0002 | 2026-08-26 | H006 a longer traced smoke exposes the next limiting stage | official 2026-08-26 inventory | `159c452` | `configs/exp-0002-wandb-extended.toml` | 20260826 / same-video trend check | 0.0000 (contract-only) | not_submitted | W&B `iyrrz897` | optimization improved, inference calibration failed | calibrate node and edge thresholds before adding epochs |
| EXP-0003 | 2026-08-26 | Code Competition submission works end to end through Kaggle CLI | official sample submission / public test | `9a80bb8` | private Notebook v1 | not applicable | not run | 0.000 public | Kaggle ref `55785839` | Notebook push/run/output/submit/score path passed | replace sample graph with calibrated inference |
| EXP-0004 | 2026-08-26 | H006/H008 full organizer baseline plus topology post-processing | official 2026-08-26 inventory | `9c3eba8` | `configs/exp-0004-host-baseline-fold0-50e.toml` | 20260826 / embryo fold 0 (`44b6` held out) | 0.9381 (50e best; completed epoch 34) | 0.787 e5; 0.805 e19 raw; 0.869 e19 post; e34/e50 pending | W&B `ud8rmowz`; Kaggle refs `55790493`, `55797775`, `55798388`, `55805307`, `55805308` | artifact-free post-processing added +0.064 on identical weights | compare e34/e50 and calibrate checkpoint-specific detection counts |

## Detailed Notes

### EXP-0001

- Organizer source: `royerlab/kaggle-cell-tracking-competition` at
  `075fc5f5a52d11077f9dc2b074644618f26939e2` (BSD-3-Clause).
- Runtime: NVIDIA GeForce RTX 5070 Ti, PyTorch 2.13.0+cu130; one dataset
  (`44b6_0113de3b`), first three frames, two windows, one epoch/iteration, seed 20260826.
- Reduced smoke architecture: 582,022 trainable parameters, channels `[4, 8]`, spatial downsample
  `(2, 8, 8)`, batch size 1. Training took 1.1 seconds after environment startup.
- Training result: detection loss 0.5728; edge/test loss, accuracy, and recall were 0.0. This is not
  a model-quality result because the run intentionally performs only one optimizer step.
- Inference contract settings: detection threshold 0.55, edge threshold 0.02, 30 µm local-max
  suppression, no TTA. These settings exist only to produce a bounded non-empty graph from the
  one-step model.
- Output: 92 nodes and 60 edges; GEFF → 152-row CSV → GEFF round trip passed the local submission
  validator. The organizer metric executed successfully and returned score 0.0000 because no
  predicted nodes matched sparse GT within the smoke slice.
- A first inference probe using 0.5 detection threshold and 5 µm suppression produced roughly 500
  candidates per frame and was stopped after two minutes; the all-pairs linker makes this unsuitable
  as a tiny smoke setting. The official high threshold 0.99 completed but produced an empty graph
  from the one-step model.
- Seed and cuDNN deterministic settings are enabled, but PyTorch reports that the organizer model's
  `max_pool3d` backward has no deterministic CUDA implementation. The wrapper therefore uses
  deterministic warning mode; exact one-step losses and graph counts may vary slightly by run.

Reproduce from the repository root:

```bash
uv run \
  --project competitions/biohub-cell-tracking-during-development/environment \
  --extra baseline \
  python competitions/biohub-cell-tracking-during-development/scripts/run_host_baseline_smoke.py
```

### EXP-0002

- W&B run: <https://wandb.ai/salax0116-private-email/biohub-cell-tracking/runs/iyrrz897>
  (`biohub-cell-tracking`, run ID `iyrrz897`, online sync completed).
- Runtime: NVIDIA GeForce RTX 5070 Ti, PyTorch 2.13.0+cu130 and W&B 0.28.2. The run used
  `6bba_09961292`, first ten frames, nine annotated windows, five epochs with five optimizer updates
  each, and the same 582,022-parameter reduced architecture as EXP-0001.
- Training took 6.404 seconds. Detection loss decreased from 0.5129 to 0.4415. Validation node
  recall increased from 0.0000 to 0.3148, accuracy reached 0.9962, and the organizer training proxy
  `accuracy * recall` reached 0.3136.
- Edge loss moved from 0.0082 to 0.0153 as detections began to match annotations; the initially zero
  validation edge signal means this should not be interpreted as a clean degradation curve.
- Inference with the predeclared smoke thresholds produced 7,711 nodes and zero edges. The GEFF →
  CSV → GEFF path and local metric still completed, but score remained 0.0000. This confirms the
  checkpoint needs detection-count and edge-softmax threshold calibration before adding epochs.
- W&B received five epoch records, system/runtime metrics, output counts, local score, source
  revision, and checkpoint SHA-256. Sync reported zero media and zero artifacts; data, predictions,
  CSV, and checkpoint contents remained local.

Checkpoint SHA-256:
`a17a387878822bededd8cd66def96a05829d8efa4ce7a1f41508974bda0007c8`.

### EXP-0003

- Private Notebook: <https://www.kaggle.com/code/suzukitaichi/biohub-cli-submission-smoke>,
  version 1, CPU, internet disabled, competition source attached.
- The Notebook copied the organizer-provided `sample_submission.csv` after checking the exact
  columns, consecutive IDs, both row types, and all four public test dataset names.
- Kaggle execution status: `COMPLETE`. The downloaded output passed the local validator with four
  datasets, 12 nodes, eight edges, and 20 total rows.
- Output SHA-256:
  `263dec32a126192f0ce4d5443b7432940fbeedf0d7ae4130156bf83657defc40`.
- CLI submission ref: `55785839`; status `COMPLETE`; public score `0.000`; private score unavailable;
  four submissions remained for the day after this submission.
- This is a submission-transport anchor, not a model baseline. A zero score is expected from the
  organizer sample graph and does not change the EXP-0002 calibration conclusion.

### EXP-0004

- Training uses the pinned organizer UNet+transformer at revision
  `075fc5f5a52d11077f9dc2b074644618f26939e2`, embryo-grouped fold 0, seed 20260826, batch size 8,
  BF16, and the versioned 50-epoch configuration. Training completed all 50 epochs and saved the
  requested five-epoch checkpoint series through `checkpoint_epoch_0050.pth`. The best full-run
  held-out `accuracy * node_recall` proxy was 0.9381 at zero-based epoch 33 (completed epoch 34).
  The first submitted snapshot was the periodic checkpoint after five completed epochs; its best
  held-out proxy was 0.9258.
- W&B run: <https://wandb.ai/salax0116-private-email/biohub-cell-tracking/runs/ud8rmowz>.
- Private model Dataset: <https://www.kaggle.com/datasets/suzukitaichi/biohub-exp-0004-host-baseline>,
  version 2. Private submission Notebook:
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0004-host-baseline-submit>, version 2, T4,
  internet disabled, competition source attached.
- Kaggle inference completed in 263.963 seconds and emitted all four public test datasets:
  178,301 nodes, 135,077 edges, and 313,378 total rows. The downloaded CSV passed the exact column,
  dataset coverage, row-sentinel, and edge-reference checks.
- CLI submission ref: `55790493`; status `COMPLETE`; public score `0.787`; private score unavailable.
- Periodic checkpoint SHA-256:
  `5af742c54fdbacaa458872ee9cbc66bed15c61bc0ec1090843fbacc1680840cc`.
  Flattened inference-weight SHA-256:
  `9f71b74210b568b282d6310f3bbea4c47a099861fa9446b033a22f3605ee992c`.
  Submission CSV SHA-256:
  `5149b67e28057c144e7e5ba7001c83e5cb71548fafb95910aa6b00c49f21d643`.
- `Biohub Harness 0926 Probe` version 1 scored 0.926 publicly. It is retained only as a public
  Notebook reference and is not recorded as an EXP-0004 result or treated as the optimization
  target for this baseline.
- Through the first 25 completed epochs, the maximum held-out proxy was 0.9362 at zero-based
  epoch 19. The corresponding `checkpoint_epoch_0020.pth` was packaged separately rather than
  replacing the epoch-5 anchor. Kaggle Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0004-best-through-epoch-25-submit>, version 1,
  completed in 297.728 seconds and emitted 200,201 nodes, 156,758 edges, and 356,959 total rows.
  Submission ref `55797775` completed with public score 0.805.
- Best-through-25 checkpoint SHA-256:
  `c68d5ddbbfd98089dba2feed646a81fff3d281f48b54e3b383d7213d2e75b69e`.
  Flattened inference-weight SHA-256:
  `7b19745d320a5aa96c2369df215e24a31c68ecdc017d3face36bcce09dc93322`.
  Submission CSV SHA-256:
  `80c7962c09fca84015abf54043011b2f9f9a70c797a9b281ce6eed9d96207913`.
- The same selected epoch-19 model was submitted again with only artifact-free post-processing
  adapted from the public harness: 6/10 µm two-pass Hungarian motion relinking, conservative
  divergence-confirmed divisions with frame/global caps, isolated and sub-six-node component
  pruning, and two-frame line-fit coordinate smoothing. The second-seed model, eight-view TTA,
  DeepCenter gate, and synthetic gap nodes were deliberately excluded.
- Post-processed Notebook:
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0004-best19-postprocess-v1-submit>, version 1.
  It completed in 334.181 seconds and emitted 179,571 nodes, 171,093 edges, and 350,664 total rows.
  Submission ref `55798388` completed with public score 0.869. Because the weights and base
  inference were identical to ref `55797775`, this is a +0.064 absolute post-processing gain; it is
  also +0.082 over the epoch-5 anchor. Submission CSV SHA-256:
  `b2a71f53a1994a40f63abd0ff16a96ee305fee78a7f8c7e02b58c77cbff78dab`.
- The best checkpoint across all 50 epochs was zero-based epoch 33 (completed epoch 34), proxy
  0.9381. Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0004-best34-postprocess-v1-submit>, version 1,
  completed with the proven post-processing profile in 379.807 seconds. Its validated output has
  171,542 nodes, 163,578 edges, and 335,120 rows. Submission ref `55805307` is pending scoring;
  CSV SHA-256 `dc534edb2d3cfebf6c88210c101c99d1f2b0e33e3df34dcd495b6cd3d74f3539`.
- The final completed-epoch-50 checkpoint was exported from `checkpoint_epoch_0050.pth`. Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0004-final50-postprocess-v1-submit>, version 1,
  completed with the same post-processing profile in 333.480 seconds. Its validated output has
  169,362 nodes, 161,631 edges, and 330,993 rows. Submission ref `55805308` is pending scoring;
  CSV SHA-256 `93e258699a86ea80e7ff807681d6337548023383062a92729ece9ddd17e5825a`.
