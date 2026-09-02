# Experiments: Biohub - Cell Tracking During Development

`LB` remains `not_submitted` until Kaggle reports a score. Artifacts stay outside Git; record a URI,
run ID, or checksum sufficient to identify them.

| id | date | hypothesis | data version | commit | config | seed/fold | CV | LB | artifact/run | takeaway | next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-0001 | 2026-08-26 | H001 organizer baseline can establish an end-to-end contract | official 2026-08-26 inventory | `6f85aa7` | `configs/exp-0001-host-smoke.toml` | 20260826 / same-video smoke | 0.0000 (contract-only) | not_submitted | `artifacts/EXP-0001/result.json` | train/infer/GEFF/CSV/metric path passed | establish a real dataset-disjoint baseline run |
| EXP-0002 | 2026-08-26 | H006 a longer traced smoke exposes the next limiting stage | official 2026-08-26 inventory | `159c452` | `configs/exp-0002-wandb-extended.toml` | 20260826 / same-video trend check | 0.0000 (contract-only) | not_submitted | W&B `iyrrz897` | optimization improved, inference calibration failed | calibrate node and edge thresholds before adding epochs |
| EXP-0003 | 2026-08-26 | Code Competition submission works end to end through Kaggle CLI | official sample submission / public test | `9a80bb8` | private Notebook v1 | not applicable | not run | 0.000 public | Kaggle ref `55785839` | Notebook push/run/output/submit/score path passed | replace sample graph with calibrated inference |
| EXP-0004 | 2026-08-26 | H006/H008 full organizer baseline plus topology post-processing | official 2026-08-26 inventory | `9c3eba8` | `configs/exp-0004-host-baseline-fold0-50e.toml` | 20260826 / embryo fold 0 (`44b6` held out) | 0.9381 (50e best; completed epoch 34); metric screen selected e30 | 0.787 e5; 0.805 e20 raw; 0.869 e20 post; 0.874 e34; 0.877 e50; **0.890 e30** | W&B `ud8rmowz`; Kaggle refs `55790493`, `55797775`, `55798388`, `55805307`, `55805308`, `55810126` | checkpoint selection plus post-processing established the 0.890 fixed control | calibrate post-processing one factor at a time |
| EXP-0007 | 2026-08-27 | H007 controlled spatial/temporal/proposal backbone comparison | official 2026-08-26 inventory | `6b88560` | `configs/exp-0007{a,b,c}-*.toml` | 20260827 / fold 0 disjoint calibration/report | report: A 0.568826; B 0.556438; C 0.457876; host 0.691999 | A e5 0.626 public | `artifacts/EXP-0007{A,B,C}/`; Kaggle ref `55823762` | spatial A won the corrected arms, but all arms trailed the host and missed report divisions | continue only A to 50e under the pinned selection/report gate |
| EXP-0009 | 2026-08-28 | H011 three-frame residual improves continuation-edge selection over frozen e30 host logits | official 2026-08-26 inventory + frozen EXP-0004 e30 | `6b88560` | `configs/exp-0009-host-tgraph3-residual-30e.toml` | 20260828 / embryo fold 0 (`44b6` held out) | base 0.922163; best 0.922684 at e3 (+0.000521); e30 0.920704 (-0.001459) | e30 ref `55843163`: **0.891**; local-best e3 ref `55854853`: 0.890 | `artifacts/EXP-0009/`; W&B `70f9278e`; e30 Notebook v2 `345596422` | e30 beats both control and local-best e3, confirming proxy/LB mismatch | retain e30 and improve checkpoint ranking |
| EXP-0011 | 2026-08-28 | H014 candidate-set self-attention improves parent choice by modelling competition among the nearest eight candidates | frozen EXP-0009 cache `c5e97a56`; frozen EXP-0004 e30 logits | `7277812` + experiment working tree | `configs/exp-0011-tgraph3-candidate-attention-10e.toml` | 20260828 / identical EXP-0009 calibration split | base 0.922163; best 0.922372 at e1 (+2 links); e10 0.920704 (-14 links); MLP best 0.922684 (+5 links) | not_submitted | `artifacts/EXP-0011/`; W&B `693a7de8` | attention lowers CE more but does not beat the independent MLP; residual magnitude, not candidate interaction capacity, is the immediate limiter | test a bounded residual scale/gate before any cache-v2 temporal GNN/GRU |
| EXP-0012 | 2026-08-29 | H015 centered, smoothly bounded attention residuals preserve host-correct links while fixing ambiguous links | frozen EXP-0009 cache `c5e97a56`; frozen EXP-0004 e30 logits | `7277812` + experiment working tree | `configs/exp-0012-tgraph3-bounded-attention-10e.toml` | 20260828 / identical EXP-0009 calibration split | base 0.922163; best **0.923101 at e3** (+9 links); e10 0.922476 (+3 links); MLP best 0.922684 (+5 links) | not_submitted | `artifacts/EXP-0012/`; W&B `9d75368c` | valid-candidate centering plus ±0.15 tanh bound converts Attention from net regression to the best local parent-accuracy result | retain e3 best; verify on independent report/LB only after explicit submission request |
| EXP-0014 | 2026-09-02 | H017 complementary MLP and bounded-Attention link errors can improve the fixed Host graph through score-level combination or agreement gating | frozen EXP-0004 e30 Host + EXP-0009/0012 e3 heads; cache `c5e97a56` | `bf1ee23` | `configs/exp-0014-tgraph3-link-ensemble-abcd.toml` | 20260902 / identical EXP-0009 calibration split | Host 0.922163; MLP 0.922684; Attention **0.923101**; bounded 50:50 0.922372; gate 0.922476 | A `55943665`, B `55943911`, C `55943722`, D `55944373`; all pending | Dataset v1 + four private GPU Notebook v1 outputs under `artifacts/EXP-0014/` | all four output contracts passed; agreement gate exactly matches Host min7 on the public clips | read all four scores and compare directly with Host-only min7 ref `55829542` at 0.893 |
| EXP-0010 | 2026-08-28 | H012/H013 remaining error can be separated into short false tracks versus missed long-displacement links | frozen EXP-0004 e30 model / public test | `6b88560` | `configs/exp-0010-postprocess-ab.toml` | fixed checkpoint and inference profile | structural gate only; no new heavy CV | corrected min7 ref `55829542`: **0.893**; gate12 ref `55828801`: 0.884 | private Notebook v1 outputs under `artifacts/EXP-0010/` | pruning six-node tracks improves the fixed 0.890 control; widening the relink gate hurts | adopt min7 and reject gate12 |
| EXP-0013 | 2026-08-29 | H016 min-component 7 transfers to EXP-0008 EMA epoch 30 | official data + fixed EMA e30 wrapper `5a2d5fc` | `2207d45` + packaging working tree | `configs/exp-0013-exp0008-ema-e30-min7.toml` | 20260827 / fold 0; EMA proxy 0.929642 | competition screen pending until EXP-0008 reaches e50 | min-7 ref `55866168`: 0.879; min-6 ref `55877003` pending | Dataset v2; min-7 Notebook v2; min-6 Notebook v1; `artifacts/EXP-0013/` | min-7 transfer is below the established EXP-0004 controls; paired min-6 scoring will isolate the post-process effect | read min-6 LB and compare directly with 0.879 |

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
  171,542 nodes, 163,578 edges, and 335,120 rows. Submission ref `55805307` completed with public
  score 0.874;
  CSV SHA-256 `dc534edb2d3cfebf6c88210c101c99d1f2b0e33e3df34dcd495b6cd3d74f3539`.
- The final completed-epoch-50 checkpoint was exported from `checkpoint_epoch_0050.pth`. Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0004-final50-postprocess-v1-submit>, version 1,
  completed with the same post-processing profile in 333.480 seconds. Its validated output has
  169,362 nodes, 161,631 edges, and 330,993 rows. Submission ref `55805308` completed with public
  score 0.877;
  CSV SHA-256 `93e258699a86ea80e7ff807681d6337548023383062a92729ece9ddd17e5825a`.

### EXP-0007

- See `docs/overview/backbone-ab.md` for the corrected-v2 architecture, immutable split,
  calibration/report protocol, checkpoint hashes, and finalization gate.
- The epoch-5 report score was 0.568826 for spatial/GT-proposal arm A, 0.556438 for
  temporal/GT-proposal arm B, and 0.457876 for temporal/mixed-proposal arm C. The fixed host
  reference scored 0.691999 on the same report half.
- Arm A was the corrected-arm winner and its private Kaggle Notebook scored 0.626 public as ref
  `55823762`. This rejects an early benefit from the tested temporal MHA and aggressive predicted
  proposal curriculum; it does not establish an improvement over the host model.
- Only arm A is eligible for the 50-epoch continuation. Packaging remains blocked until the complete
  ten-checkpoint calibration sweep and one fixed report evaluation pass the pinned strict gate.

### EXP-0009

- See `docs/overview/temporal-graph-residual.md` for the frozen-source, candidate-graph, sparse-label,
  checkpoint, and submission contracts.
- Status: 30 epochs completed; completed epoch 30 scored 0.891 public as Kaggle ref `55843163`,
  +0.001 over the frozen 0.890 control. W&B run:
  <https://wandb.ai/salax0116-private-email/biohub-cell-tracking/runs/70f9278e>.
- The completed-epoch-30 organizer baseline is frozen at weights SHA-256
  `9e068669b861b0dd993483e3ce6fc636fde604004011d5ec1e08039ae2843337`, source
  checkpoint SHA-256 `729836a33f485eb9e90ffd97f4ad6defdae916ac0e0affad04de649ade91b79e`, and
  organizer revision `075fc5f5a52d11077f9dc2b074644618f26939e2`.
- `T_image=2` is unchanged. Adjacent frozen pair windows provide `T_graph=3`; a zero-initialized
  two-layer residual MLP refines only the right transition before softmax. Candidate parents are
  bounded to top 8 within 15 micrometers, while division and `public-applicable-v1`
  post-processing remain fixed.
- Cache and deployment proposal generation both use four-view XY detection TTA and probability
  threshold 0.99 (raw logit approximately 4.59512). A two-transition GPU smoke peaked at
  805,580,800 allocated bytes and 1,182,793,728 reserved bytes; candidate recall was 1.0 for the
  one available sparse parent.
- The full cache produced 105,327 training and 9,597 validation examples. Cache construction took
  1,975.8 seconds and the 30 residual-head epochs took 35.6 seconds. Periodic checkpoints were
  written at completed epochs 5, 10, 15, 20, 25, and 30.
- Frozen-host validation top-1 accuracy was 0.922163. The best refined result was 0.922684 at epoch
  3 (+0.000521, about +0.052 percentage points). Epoch 30 fell to 0.920704 (-0.001459, about
  -0.146 percentage points), while training accuracy rose from 0.954019 to 0.955643. This is a
  small early signal followed by mild overfitting, not a local validation improvement at epoch 30.
- Kaggle Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0009-tgraph3-e30-submit>, version 2,
  completed public-test inference and produced a valid four-dataset CSV with 167,116 nodes and
  159,438 edges. CSV SHA-256:
  `47e07feb66b1a92c1a6fe176017492738d22442ff796beab3d6ae49e15b9baa5`.
- Best-epoch Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0009-tgraph3-best-e3-submit>, version 2,
  completed public-test inference and produced a valid four-dataset CSV with 167,087 nodes and
  159,380 edges. CSV SHA-256:
  `2037bffd71aebe44f98c3823a20ff06ec4aac4aad94fcd76a0367c25de6f0a68`.
  Code Competition submission ref `55854853` scored 0.890, tying the frozen control and trailing
  the epoch-30 residual head by 0.001 despite its better local top-1 score.

### EXP-0011

- This is a controlled ten-epoch architecture probe over the immutable EXP-0009 cache, not a new
  image or graph cache. The shared cache manifest SHA-256 is
  `6296240ca44fa9b6b9ce3d98ba56e774621b67b2e460a2b0f26ece6264f56542`; its 105,327 training and
  9,597 validation rows, seed, split, base logits, candidates, loss, optimizer settings, and
  checkpoint cadence are unchanged.
- One four-head candidate-set attention block replaces only the independent residual MLP. It has
  40,597 trainable parameters versus 11,285 for the MLP and attends over the nearest-eight parent
  candidates. The attention axis is candidate competition, not time; explicit node identities and
  transition boundaries are not present in cache schema v1.
- The ten epochs completed in 23.6 seconds. W&B run:
  <https://wandb.ai/salax0116-private-email/biohub-cell-tracking/runs/693a7de8>.
  Checkpoints were written at epochs 5 and 10, with the best checkpoint at epoch 1.
- Best attention accuracy was 0.922372: seven frozen-host mistakes were fixed and five host-correct
  links regressed, for a net gain of two links. The EXP-0009 MLP best fixed 14 and regressed nine,
  for a net gain of five links and higher accuracy 0.922684.
- At attention epoch 10, validation CE improved to 0.28930, but 52 host mistakes fixed were offset
  by 66 regressions, leaving 0.920704 accuracy. A diagnostic residual multiplier of 0.2 recovered
  0.922580 (18 fixed, 14 regressed), still below the MLP best. The tested attention architecture is
  therefore rejected as the next capacity increase; a bounded residual gate is the next local test.
- A true temporal GRU or global spatiotemporal GNN remains out of scope for this cache. Cache schema
  v2 must retain `t_start`, node identities, candidate source indices, transition grouping, and a
  genuine previous-parent appearance token before those names are technically accurate.

### EXP-0012

- This rerun changes only the candidate-attention residual transform. For each target it subtracts
  the valid-candidate mean, then applies `0.15 * tanh(centered / 0.15)`. Common offsets cannot alter
  parent selection, and the bound limits the maximum pairwise logit change to 0.30 while retaining
  unit slope at zero. Cache, split, seed, architecture, optimizer, and ten-epoch budget match
  EXP-0011.
- The run completed in 13.4 seconds and wrote best, epoch-5, epoch-10, and last checkpoints. W&B:
  <https://wandb.ai/salax0116-private-email/biohub-cell-tracking/runs/9d75368c>.
- The retained epoch-3 `best_model.pth` SHA-256 is
  `f72a6446a73831f30795f8599f6e80b2e52ff3038367c7226ae5de045e66899d`.
- Best epoch 3 reached 0.923101: it fixed 12 frozen-host mistakes and regressed three host-correct
  links, a net gain of nine. This exceeds the EXP-0009 MLP best by four correct links and the
  unbounded Attention best by seven. Epoch 10 remained above the host at 0.922476 (20 fixed, 17
  regressed), rather than falling below it as unbounded Attention did.
- The result supports bounded correction capacity as the immediate mechanism. It does not yet
  establish a competition-metric gain because the local objective is sparse parent top-1 rather
  than the post-processed graph metric.

### EXP-0014

- This is one four-arm deployment comparison, not four independently tuned experiments. All arms
  freeze EXP-0004 completed-epoch-30 detections and Host logits, top-8/15 µm candidates,
  `T_image=2`, `T_graph=3`, thresholds, TTA, division handling, and the proven min-component-7
  post-process. Host-only min-7 ref `55829542` at 0.893 is the direct control.
- Arm A uses EXP-0009 local-best MLP completed epoch 3; arm B uses EXP-0012 bounded Attention
  completed epoch 3. Arm C centers the 50:50 mean candidate residual and applies
  `0.15 * tanh(delta / 0.15)`. Arm D keeps Host logits unless both heads select the same non-Host
  valid-candidate parent, then applies only the centered ±0.15 Attention correction.
- Both checkpoints bind the same frozen Host SHA-256
  `9e068669b861b0dd993483e3ce6fc636fde604004011d5ec1e08039ae2843337` and cache fingerprint
  `c5e97a56ac6a9cabe8fb62496ec40a44f49a8abe22eefd16afd1c010c354eeeb`. MLP checkpoint SHA-256 is
  `a295c3cba5a06c9a7b3893f7998934143ccfde2cebd41285f4a8db3cf35dc808`; Attention checkpoint
  SHA-256 is `f72a6446a73831f30795f8599f6e80b2e52ff3038367c7226ae5de045e66899d`.
- On the 9,597-row calibration cache, the heads disagree on only 30 rows. Local top-1 is 0.922684
  for MLP, 0.923101 for Attention, 0.922372 for bounded 50:50, and 0.922476 for the agreement gate,
  against Host 0.922163. These are diagnostics, not the competition selection metric.
- Shared private Dataset
  <https://www.kaggle.com/datasets/suzukitaichi/biohub-exp-0014-tgraph3-link-abcd>, version 1,
  binds manifest SHA-256 `9e9457c34a414b7d5aa055deb01e5d27de73f1d4f17779e8b56d950ad4f51cbf`.
- Arm A Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0014-tgraph3-mlp-e3-submit>, version 1,
  emitted 162,881 nodes and 155,875 edges in 320.605 inference seconds. CSV SHA-256 is
  `62699d43be4d9040f7344fcfbce5f3a5d0620de46eb2b9df9e4613f619d6310e`; submission ref
  `55943665` scored 0.893 public.
- Arm B Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0014-tgraph3-bounded-attn-e3-submit>, version 1,
  emitted 162,882 nodes and 155,877 edges in 320.751 seconds. CSV SHA-256 is
  `b33e01cd77f8280eee94fd57ddf8a99e5d72584b0db928b55d376f65d6274614`; submission ref
  `55943911` scored 0.893 public.
- Arm C Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0014-tgraph3-bounded-logit-5050-submit>, version 1,
  emitted 162,872 nodes and 155,868 edges in 341.934 seconds. CSV SHA-256 is
  `159b6170eb971001609f2e6aaced013c4fd026ac53a0b73ce68a7542cd0911b5`; submission ref
  `55943722` scored 0.893 public.
- Arm D Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0014-tgraph3-agreement-gate-submit>, version 1,
  emitted 162,863 nodes and 155,865 edges in 307.066 seconds. CSV SHA-256 is
  `68e9ef756433c6778e7560027997e0a5b04a829807b285c8bd2cb0cc56abedba`; submission ref
  `55944373` scored 0.893 public. The CSV is byte-identical to Host-only min7 ref `55829542` on the public
  clips, so the agreement gate made no public-input graph changes.
- All four link policies tied the Host-only min7 control at 0.893. H017 is rejected for these
  checkpoints: architectural diversity and the 30 local disagreements did not translate into a
  measurable LB difference. EXP-0015 therefore changes temporal evidence itself with
  `T_graph=4`, while retaining arm C as the fixed score-combination policy.

### EXP-0015

- This is the controlled `T_graph=4` successor to EXP-0014 C. The image model remains
  `T_image=2`; detections, Host logits, top-8/15 µm candidates, division handling, and
  min-component-7 post-processing stay frozen. The first transition uses Host, the second uses
  the exact EXP-0014 C T3 bounded-logit ensemble, and later transitions use the T4 ensemble.
- Cache schema v2 adds four features to the T3 width: a normalized constant-acceleration target
  residual in xyz and the probability mass with valid second-history support. The probability is
  re-normalized over valid prior parents before both historical expectations are computed; a row
  with no second-history mass receives four exact zeros. The resulting feature width is 110.
- The full cache contains 104,350 training and 9,496 validation examples. Construction took
  1,973.579 seconds and retained sparse-parent candidate recall of 0.998211 and 0.998633. Cache
  fingerprint is `44db840466c86e93b1b76ba5a5b5a04ad4d3c3160ee745d9689e3ceaabea871d`;
  enriched manifest SHA-256 is
  `78f01de4b4659b8e34690097616bc360be9742b377d97489b6451787a80f38e2`.
- The MLP completed ten epochs in 12.372 seconds and retained epoch 8 at 0.922704 (W&B
  `c892282b`, checkpoint SHA-256
  `669d6104e2c8a2ab476a278bdcf9f88a792717a6dc02c79061dff50cf8585b56`). The bounded Attention
  head completed ten epochs in 13.022 seconds and retained epoch 5 at 0.923126 (W&B `5981b97c`,
  checkpoint SHA-256
  `b93091c28f6171015842c633a7b6c79db0273a6cc7e8fd5fa9ae672574a42ba1`).
- On the exact 9,496-row validation cache, Host gets 8,758 links correct; MLP gets 8,762,
  Attention gets 8,766, and the fixed centered ±0.15 bounded-logit 50:50 ensemble gets 8,759.
  The deployment ensemble fixes two Host mistakes and regresses one Host-correct link. The heads
  disagree on 74 rows. This passes the direction-only local gate, but its +1-link margin is too
  small to claim a robust gain before the controlled Kaggle comparison.
- The packaged private Dataset binds both T4 heads, both immutable T3 startup-fallback heads, the
  frozen Host, and integrated inference with content-addressed manifest SHA-256
  `37df05d112df7b750782906fb573663a7ab2f6d8270a95ffcd2219802df43f38`. Dataset
  `suzukitaichi/biohub-exp-0015-tgraph4-link-5050` version 1 is attached to Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0015-tgraph4-bounded-logit-5050-submit>,
  version 2.
- Notebook inference completed in 321.186 seconds and emitted 162,877 nodes plus 155,874 edges
  across all four datasets. The 318,751-row CSV passed both competition-specific and generic
  validators; SHA-256 is
  `70ce6dcfa9c01fd0dfe107c094e65daf9b32c32b3c0da527c383355131e24c1b`. The normal file-upload
  API was rejected because this is a Code Competition; Notebook version 2 was then submitted via
  the Code Submission API as ref `55949925` and scored 0.893 public, tying the direct EXP-0014 C
  comparison ref `55943722` and Host-only min7. H018 is rejected on the current LB evidence.

### EXP-0016

- This is the controlled `T_graph=5` successor to EXP-0015. The Host remains `T_image=2`; image
  weights, detections, Host logits, candidates, thresholds, division handling, and
  min-component-7 post-processing remain frozen. Startup dispatch is Host for the first
  transition, T3 50:50 for the second, T4 50:50 for the third, and T5 50:50 thereafter.
- Cache schema v3 retains the complete 110-column T4 vector as an exact prefix and appends a
  normalized constant-jerk target residual in xyz plus deepest-history path mass. The resulting
  width is 114. Missing third-history support yields four exact zeros; schema, feature revision,
  width, pair adjacency, source checkpoint, and cache fingerprint are fail-closed.
- One-dataset/three-transition smoke completed on GPU with train/validation candidate recall 1.0,
  finite 114-column payloads, and 805,580,800 peak allocated bytes. W&B run `d7892627` records the
  smoke. The full schema-v3 cache contains 103,342 training and 9,401 validation examples; its
  candidate recall is 0.998194 and 0.998619. Construction took 1,875.990 seconds under W&B run
  `952d942a`. Cache fingerprint is
  `0aa5c2d2924e4c940f284918013026c7e016e7e93f82629991ab58b48fd5a739`; manifest SHA-256 is
  `a56e8143b51dca75101ccd809ae2d05a0c1fab7448eb319315a9944a6033741a`.
- The T5 MLP completed ten epochs in 11.816 seconds and retained epoch 9 at 0.922455 (W&B
  `d4e31ff8`, checkpoint SHA-256
  `186159abf90c584e39a8da3781641fe24f6063c801e2b23eb36292cf6812833a`). Bounded Attention
  completed ten epochs in 13.425 seconds and retained epoch 6 at 0.922987 (W&B `e2d2de10`,
  checkpoint SHA-256
  `6f8cd3ccff4a1b266bdec400cba7f099726e6263d95dac9ed117dcd8b9283190`). Epoch-5 and epoch-10
  periodic checkpoints were retained for both heads.
- On the exact 9,401-row cache, Host, MLP, bounded Attention, and the fixed centered bounded-logit
  50:50 ensemble get 8,668, 8,672, 8,677, and 8,669 parent links correct. The deployment ensemble
  fixes two Host mistakes and regresses one Host-correct link, again leaving only a +1-link local
  signal. MLP and Attention disagree on 76 rows.
- The fixed deployment contains six compact heads: T5 MLP/Attention primary, immutable EXP-0015
  T4 MLP/Attention fallback, and immutable EXP-0009/EXP-0012 T3 MLP/Attention fallback. Every
  tier uses centered ±0.15 bounded-logit 50:50 combination. The content-addressed bundle binds
  all six checkpoints and the frozen Host in manifest SHA-256
  `94e63f697ddf348a2abc80f0ead4fade70b21254ef3b354468802fe5bd740058`.

### EXP-0013

- The fixed source is EXP-0008 EMA completed epoch 30, wrapper SHA-256
  `5a2d5fc84e4aebd485945016350273992fc457f30c575e1dd0078d546a6809b0`. Its exported 136-tensor
  inference state SHA-256 is
  `dfd6a3d57a768080f9d1344614132ca284b909525984962c4ddce610daca4d91`.
- EMA was selected over raw for this single-slot deployment probe because the epoch-30 adjacent-pair
  proxy was 0.929642 versus 0.914262. The competition-metric checkpoint screen has not run yet, so
  this is a pragmatic choice rather than a proven competition-metric selection.
- The only post-process change is division-preserving minimum component size 6 to 7, transferred
  from EXP-0010 ref `55829542` at 0.893. Detection, TTA, linking, division repair, and smoothing are
  unchanged. EXP-0008 EMA e30 min-6 has not been submitted, so the transferred effect is unpaired.
- Dataset `suzukitaichi/biohub-exp-0008-ema-e30` version 2 binds the Kaggle-expanded model archive.
  Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0008-ema-e30-min7-submit>, version 2,
  completed in 318.302 inference seconds and emitted 169,090 nodes plus 161,789 edges. CSV SHA-256:
  `8bfca1b8ee2f561e4847245aa5ad1b6d1abc276657f4f5a4340b03c42fee4105`.
  Code Competition submission ref `55866168` scored 0.879. This is below the EXP-0004 e30 min-6
  score 0.890 and min-7 score 0.893, but does not isolate the cause because the checkpoint changed.
- Notebook version 1 stopped before inference because the generic Dataset manifest did not bind
  Kaggle-expanded ZIP members. No submission slot was consumed. The packager now records every
  archive member hash, and version 2 passed the same fail-closed verification.
- The paired min-6 control uses the identical Dataset version 2, checkpoint SHA, detection, TTA,
  linking, division, and smoothing settings with an empty patch list. Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0008-ema-e30-min6-submit>, version 1,
  completed in 321.910 inference seconds and emitted 173,512 nodes plus 165,474 edges. This is
  4,422 nodes and 3,685 edges above min-7. CSV SHA-256:
  `6ab71c474177189b603fd07061182174a9f21b2694c91e25976263a0fc526998`.
  Code Competition submission ref `55877003` is pending.

### EXP-0010

- Fixed control: completed epoch 30, checkpoint SHA-256
  `9e068669b861b0dd993483e3ce6fc636fde604004011d5ec1e08039ae2843337`, detection
  threshold 0.99, edge threshold 0.5, 5 µm pooling, four-view XY TTA, and
  `public-applicable-v1`; public score 0.890 (ref `55810126`).
- All hard-coded control parameters are enumerated in
  `configs/exp-0010-postprocess-ab.toml`: 6/10 µm two-pass Hungarian relinking,
  conservative capped division repair, division-preserving minimum-six component pruning, and
  two-frame linear coordinate smoothing with fitted/raw weights 0.8/0.2.
- Precision/H012 changes only the minimum retained component size from 6 to 7 and still preserves
  any component containing a division. Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0010-precision-min7-full-submit>, version 1,
  produced a structurally valid four-dataset CSV with 162,863 nodes and 155,865 edges. Relative to
  the control output this removes 4,212 nodes and 3,510 edges. CSV SHA-256:
  `68e9ef756433c6778e7560027997e0a5b04a829807b285c8bd2cb0cc56abedba`.
  Corrected submission ref `55829542` scored 0.893 public, +0.003 over the fixed control.
- Recall/H013 changes only the relaxed Hungarian relink gate from 10 to 12 µm. Notebook
  <https://www.kaggle.com/code/suzukitaichi/biohub-exp-0010-recall-gate12-submit>, version 1,
  produced a structurally valid four-dataset CSV with 171,734 nodes and 164,530 edges. Relative to
  control this adds 4,659 nodes and 5,155 edges; relaxed-pass links increase from 7,302 to 9,978.
  CSV SHA-256 `a41b3dd92aa9aa0cb7feed8f8bddf6cbc5ed565863554397169dd3c2b4ca4144`.
  Submission ref `55828801` scored 0.884 public, -0.006 below the fixed control.
- The two submissions deliberately move in opposite precision/recall directions and must each be
  interpreted against the same 0.890 control, not only against one another.
- The first precision attempt, ref `55828867`, was a CPU child Notebook over the public-test output.
  Kaggle correctly rejected it as `incorrect format` because a Code Competition submission must
  rerun inference on hidden test datasets; it carries no model-quality evidence and was replaced.
- Ref `55829582` is an accidental concurrent duplicate of corrected precision ref `55829542` using
  the identical Notebook version and condition. Its identical 0.893 score is excluded from the A/B
  interpretation.
- `scripts/prepare_postprocess_ab_notebooks.py` now reproduces both valid submitted execution modes
  as full GPU inference from the frozen model Dataset. The corrected min-7 public output is
  byte-identical to the locally transformed control, and the gate-12 patched inference script is
  byte-identical to its submitted artifact.
