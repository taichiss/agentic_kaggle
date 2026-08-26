# Experiments: Biohub - Cell Tracking During Development

`LB` remains `not_submitted` until Kaggle reports a score. Artifacts stay outside Git; record a URI,
run ID, or checksum sufficient to identify them.

| id | date | hypothesis | data version | commit | config | seed/fold | CV | LB | artifact/run | takeaway | next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0001 | 2026-08-26 | H001 organizer baseline can establish an end-to-end contract | official 2026-08-26 inventory | `6f85aa7` | `configs/exp-0001-host-smoke.toml` | 20260826 / same-video smoke | 0.0000 (contract-only) | not_submitted | `artifacts/EXP-0001/result.json` | train/infer/GEFF/CSV/metric path passed | establish a real dataset-disjoint baseline run |
| EXP-0002 | 2026-08-26 | H006 a longer traced smoke exposes the next limiting stage | official 2026-08-26 inventory | `159c452` | `configs/exp-0002-wandb-extended.toml` | 20260826 / same-video trend check | 0.0000 (contract-only) | not_submitted | W&B `iyrrz897` | optimization improved, inference calibration failed | calibrate node and edge thresholds before adding epochs |

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
