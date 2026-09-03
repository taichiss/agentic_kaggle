# BioHub backbone A/B overview

This document is the implementation and evaluation pointer for the controlled nnU-Net and temporal
backbone study. EXP-0007 uses one explicit `corrected_v2` contract so that spatial/temporal and
proposal-strategy changes can be compared without silently changing the detector, linker, sparse-GT
semantics, decoder, validation split, or inference calibration protocol.

Keywords: `PlainConvUNet`, 3D detection heatmap, per-voxel temporal attention, sparse GT,
positive-unlabelled supervision, physical coordinates, parent/null softmax, division head,
predicted-node curriculum, disjoint calibration/report split, content-addressed finalization.

## EXP-0007 corrected-v2 architecture and controlled study

EXP-0007 replaces the transitional EXP-0006 wiring with one explicit `corrected_v2` model contract.
All corrected arms use the same nnU-Net `PlainConvUNet`, detector, node representation, parent/null
linker, division head, sparse-label losses, and graph decoder. The controlled question is split into
two comparisons:

- EXP-0007A versus EXP-0007B changes only deep-stage temporal fusion from `identity` to
  `per_voxel_mha`.
- EXP-0007B versus EXP-0007C changes only node proposals from ground-truth proposals to the detached
  predicted-node curriculum.

```text
augmented image window [B,T,Z,Y,X]
             ↓ shared nnU-Net encoder
   identity (A) or temporal MHA at stages 2/3 (B/C)
             ↓ shared decoder
      32-channel features + detection logits
             ↓ proposals created after augmentation
appearance + physical position (µm) + spatial embedding
    + detached detection/division confidence + frame role + Δt
             ↓ candidate graph: radius 15 µm, nearest top 32
       parent softmax (including null) + division logits
```

The temporal block uses four-head per-voxel self-attention, a learned relative-time embedding, and a
zero-initialized residual gate. `common_head_seed=20260827` gives A/B/C identical initialization for
the common detector, division, linker, and null-parent heads; deterministic augmentation derives its
RNG from `(seed, epoch, item index)`. Image shape and voxel size remain sample-specific tensors, so
physical-coordinate features preserve anisotropy after batching.

Sparse-GT handling is explicit in this contract:

- a target with one known parent supervises that parent; a natural missing proposal or zero-parent
  sparse label remains unknown rather than being converted to a negative;
- source dropout (`0.05`) deliberately removes a uniquely annotated parent and is the only source of
  reliable null-parent positives; duplicate and unmatched proposals remain unknown;
- division labels are three-state: more than one child is positive, exactly one child is a weak
  negative, and zero annotated children is unknown. Positive and negative terms are normalized
  separately as `L_pos + 0.1 * L_neg`;
- a positive-parent training label outside the 15 µm/top-32 candidate set is masked from parent CE.
  Validation does not hide this failure and separately records candidate recall;
- predicted proposals are local-max/top-K capped at 96 before physical one-to-one GT matching. The
  EXP-0007C five-epoch detached-prediction ratios are `[0, 0.25, 0.50, 0.75, 0.75]`.

Full-state checkpoint envelopes include optimizer state, history, RNG state, config fingerprint,
and validation-manifest hash for stateful resume and provenance. This still is not full nnU-Net
training: automatic plans, spacing-driven kernels/strides, foreground oversampling, deep
supervision, Dice+CE, standard nnU-Net augmentation, cascade, and ensemble remain out of scope.

## Fixed calibration/report protocol

The 71 held-out `44b6` datasets are frozen in
`artifacts/EXP-0007/validation_split.json` (SHA-256
`2fcede60f353645de37768be3e735a332f552fa9253d51d1154564ba4ce5cc1c`). Seed `20260827`
stratifies by division presence and annotated-edge-density quartile into:

- `calibration`: 35 datasets, 43 densest-plus-all-division transitions, 167 GT edges, 13 GT
  divisions; threshold and checkpoint selection are allowed only here;
- `report`: 36 disjoint datasets, 46 transitions, 205 GT edges, 13 GT divisions; the selected saved
  inference profile is applied once without another sweep.

Both halves use 7 µm optimal node matching, 5 µm pooling, four-view XY detection TTA, and the exact
competition aggregation. They are local isolated-transition screens, so none of these values is an
absolute public-LB estimate. The host epoch-5 reference uses its deployed `det=0.99`, `edge=0.50`
profile on the same report half.

| Epoch-5 report model | det | edge | null | division | score | node recall | edge precision | edge recall | division Jaccard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Host baseline | 0.99 | 0.50 | — | — | **0.691999** | **0.970109** | **0.868852** | **0.775610** | **0.117647** |
| EXP-0007A, spatial/GT proposals | 0.10 | 0.15 | 0.25 | 0.75 | **0.568826** | 0.949628 | 0.761905 | 0.702439 | 0.000000 |
| EXP-0007B, temporal/GT proposals | 0.10 | 0.15 | 0.25 | 0.75 | 0.556438 | 0.921019 | 0.747368 | 0.692683 | 0.000000 |
| EXP-0007C, temporal/mixed proposals | 0.10 | 0.25 | 0.25 | 0.75 | 0.457876 | 0.833925 | 0.705521 | 0.560976 | 0.000000 |

At five epochs A is the corrected-arm winner. B trails A by `0.012388`, so this temporal MHA does
not show an early benefit under the controlled contract. C trails A by `0.110950`; the tested
curriculum suppresses node and edge recall, but this rejects only the aggressive five-epoch schedule,
not predicted-node adaptation in general. A still trails the host reference by `0.123173`, and all
three corrected arms miss all 13 report divisions at their calibration-selected threshold. Division
supervision/calibration therefore remains unresolved.

The host report score `0.691999` is an architecture-comparison target only. It is not the
finalization or submission gate. The gate is hard-pinned to the prior EXP-0007A epoch-5 fixed-report
score `0.5688260117`, as described below; neither the host score nor the public-LB score may be
substituted for it.

## Reproducibility pointers and run status

| Arm | Training config | Calibration config | Report config | Report artifact |
| --- | --- | --- | --- | --- |
| A | `configs/exp-0007a-corrected-spatial.toml` | `configs/exp-0007a-calibration-screen.toml` | `configs/exp-0007a-report-screen.toml` | `artifacts/EXP-0007A/report-screen-epoch5/summary.json` |
| B | `configs/exp-0007b-corrected-temporal.toml` | `configs/exp-0007b-calibration-screen.toml` | `configs/exp-0007b-report-screen.toml` | `artifacts/EXP-0007B/report-screen-epoch5/summary.json` |
| C | `configs/exp-0007c-corrected-temporal-curriculum.toml` | `configs/exp-0007c-calibration-screen.toml` | `configs/exp-0007c-report-screen.toml` | `artifacts/EXP-0007C/report-screen-epoch5/summary.json` |

Each arm's full training record is `artifacts/EXP-0007A/result.json`,
`artifacts/EXP-0007B/result.json`, or `artifacts/EXP-0007C/result.json`; its selected profile and
calibration sweep are in the adjacent `calibration-screen-epoch5/` directory. The host reference is
defined by `configs/exp-0007-host-e5-report-screen.toml` and recorded in
`artifacts/EXP-0007/HOST-E5-report-screen/summary.json`.

- Epoch-5 checkpoint SHA-256: A
  `f8e0e1e5719f1653a8e3bc6da5d2c332d9cc851df2adee722c8abad614e0e9cd`, B
  `fc410305226068431c2cd85089d84dc6aa9be12dfa25e022c45f4349f8695c4f`, and C
  `58d2dd00f7cfd32d57fb5fe67fdb3b148f889a6ae3278440765bd02035b836ab`.
- EXP-0007A epoch 5 was submitted as Kaggle reference `55823762` and completed with public score
  `0.626`. Its generated bundle is under `artifacts/EXP-0007A/kaggle-epoch5-output/`.
- The winning A arm uses the continuation config
  `configs/exp-0007a-corrected-spatial-50e.toml`. Final 50-epoch evidence must be generated by the
  complete selection/report procedure below; it must not be inferred from the epoch-5 result.

## Final 50-epoch selection and submission gate

After all ten periodic checkpoints exist, run the complete calibration sweep and the single selected
report evaluation with the pinned 50-epoch templates:

```bash
COMP_ROOT="competitions/biohub-cell-tracking-during-development"
SELECT_ROOT="$COMP_ROOT/artifacts/EXP-0007A/checkpoint-selection-calibration"

uv run --project "$COMP_ROOT/environment" --extra baseline --extra nnunet \
  python "$COMP_ROOT/scripts/select_exp7a_calibration_checkpoints.py" \
  --calibration-template "$COMP_ROOT/configs/exp-0007a-calibration-screen-epoch50.toml" \
  --report-template "$COMP_ROOT/configs/exp-0007a-report-screen-epoch50.toml" \
  --output-root "$SELECT_ROOT" \
  --epochs 5 10 15 20 25 30 35 40 45 50
```

The required candidate set is exactly completed epochs `[5, 10, 15, 20, 25, 30, 35, 40, 45,
50]`; the selector and finalizer refuse an incomplete sweep. Existing candidate and report summaries
are reused only after their checkpoint, profile, generated screen config, split manifest, and
content hashes are revalidated. The selector writes one immutable `selection.json`, copies the
selected profile, and evaluates the report half once with that fixed profile.

Packaging is allowed only when the selected report score is strictly greater than the prior
EXP-0007A epoch-5 fixed-report score `0.5688260117`. The implemented comparison is
`selected_score > 0.5688260117 + explicit_tolerance`; tolerance `0` means exact strict comparison,
so equality is rejected. A positive tolerance requests that additional absolute improvement margin.
The tolerance argument is required explicitly. The packager and Kaggle runtime both reject a
baseline value other than this pinned epoch-5 score.

```bash
SELECTED_EPOCH="$(uv run --project "$COMP_ROOT/environment" python -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["completed_epoch"])' \
  "$SELECT_ROOT/selection.json")"
printf -v SELECTED_EPOCH4 '%04d' "$SELECTED_EPOCH"
REPORT_SUMMARY="$SELECT_ROOT/selected-report-epoch-$SELECTED_EPOCH4/summary.json"
PACKAGE_ROOT="$COMP_ROOT/data/kaggle-submission-EXP-0007A-epoch$SELECTED_EPOCH"
DATASET_ID="suzukitaichi/biohub-exp-0007a-epoch$SELECTED_EPOCH"
KERNEL_ID="suzukitaichi/biohub-exp-0007a-epoch$SELECTED_EPOCH-submit"

uv run --project "$COMP_ROOT/environment" --extra baseline --extra nnunet \
  python "$COMP_ROOT/scripts/prepare_backbone_ab_submission.py" \
  --selection-json "$SELECT_ROOT/selection.json" \
  --report-summary "$REPORT_SUMMARY" \
  --output-root "$PACKAGE_ROOT" \
  --dataset-id "$DATASET_ID" \
  --kernel-id "$KERNEL_ID" \
  --require-report-score-above 0.5688260117 \
  --report-score-tolerance 0
```

The corrected-v2 packager has no manual checkpoint/profile fallback: it derives the experiment
config, selected checkpoint, completed epoch, and profile only from the validated selection/report
chain. There is no epoch-5/default-bundle fallback. The generated notebook pins the package-manifest
SHA-256, and runtime rechecks every direct file or Kaggle-expanded ZIP member plus the selected
epoch, profile, split, and report-score gate. Only after that command succeeds should the newly named
Dataset and Notebook be pushed:

```bash
uv run kaggle datasets create -p "$PACKAGE_ROOT/dataset"
uv run kaggle kernels push -p "$PACKAGE_ROOT/kernel"
uv run kaggle kernels status "$KERNEL_ID"
```

If that exact Dataset ID already exists, use `kaggle datasets version` instead of `create`; do not
change the ID to an epoch-5/default bundle. After the Notebook run succeeds, use that Notebook
version's competition submission action, then inspect the recorded submission through
`uv run kaggle competitions submissions biohub-cell-tracking-during-development`.

## Known limitations / next ablations

These limitations do not invalidate the current A/B/C comparison or the resumed EXP-0007A run.
The three arms share the same affected components, initialization controls, split, and evaluation
protocol; A/B still isolates temporal fusion and B/C still isolates proposal strategy. Finish the
current spatial 50-epoch control before changing these contracts.

- The 15 µm/top-32 candidate mask is applied to edge logits only after the organizer
  `SimpleNodeTransformer` has run. Its global attention can therefore still mix context from valid
  non-candidate nodes. A candidate-aware attention mask or sparse neighbourhood linker is a separate
  ablation from changing the candidate radius/top-K.
- Detection voxels marked unknown use `unknown_weight=0.05`, so they are weak negatives rather than
  a true positive-unlabelled (PU) mask. Compare this contract with masked-unknown, PU-risk, or
  confidence-filtered pseudo-label objectives while holding proposals and calibration fixed.
- The graph decoder greedily accepts edges in descending probability order subject to parent/child
  caps. It does not optimize the transition graph globally; compare the same calibrated logits with
  bipartite/min-cost-flow or ILP decoding before attributing all remaining errors to the model.
- Full RNG and optimizer state make resume stateful and auditable, but ordinary CUDA execution is
  not guaranteed to be bit-exact because nondeterministic GPU kernels remain enabled. Treat a
  resumed run as semantically reproducible, not byte-for-byte identical to an uninterrupted run.
- Temporal attention sees only `T=2`. It models a single adjacent-frame transition, not long-range
  history; longer windows, recurrent state, or cached temporal attention require a distinct ablation
  with matched compute and proposal settings.
