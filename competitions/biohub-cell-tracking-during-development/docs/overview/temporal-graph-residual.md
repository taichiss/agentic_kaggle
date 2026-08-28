# EXP-0009 three-frame temporal graph residual

## Decision

EXP-0009 tests H011: a small three-frame candidate-graph model can improve continuation-edge
selection after the image model without changing cell detection, division generation, or the proven
submission post-processing. The experiment freezes the EXP-0004 completed-epoch-30 host model and
learns only an additive residual for its edge logits.

```text
image windows of two frames (T_image = 2)
                 |
                 v
frozen EXP-0004 epoch-30 host detector and edge scorer
                 |
                 v
three-frame candidate graph (T_graph = 3)
  radius <= 15 um, nearest top 8 candidates per target
                 |
                 v
local temporal residual MLP -> additive edge-logit residual
                 |
                 v
host logit + learned residual
                 |
                 v
fixed division policy + public-applicable-v1 post-processing
```

The host image network still sees only adjacent frame pairs. The graph window combines the two
adjacent transitions around a middle frame, allowing the residual scorer to use displacement and
constant-velocity consistency without increasing the 3D image window or retraining the detector.

## Pinned source model

The immutable source is organizer revision `075fc5f5a52d11077f9dc2b074644618f26939e2` and the
EXP-0004 completed-epoch-30 weights. The flattened inference weights have SHA-256
`9e068669b861b0dd993483e3ce6fc636fde604004011d5ec1e08039ae2843337`; their original periodic
checkpoint has SHA-256 `729836a33f485eb9e90ffd97f4ad6defdae916ac0e0affad04de649ade91b79e`.
The run fails before cache generation if either recorded identity does not match.

## Candidate graph and residual

For each three-frame window `(t-1, t, t+1)`, candidates are restricted in physical coordinates to
15 micrometers and the nearest eight sources per target. Features contain the frozen host logit,
physical displacement, constant-velocity residual, both frozen appearance views of the shared
middle node, target appearance, previous-parent entropy, and a history-availability flag.

A two-layer, 64-dimensional MLP emits one additive logit residual per candidate. Feature
construction and scoring stay O(NK). The final layer is zero-initialized, so epoch zero reproduces
the host logits exactly. The head cannot create division edges; decoding and all
`public-applicable-v1` topology repairs remain downstream and unchanged.

## Sparse-label contract

Only targets with one known annotated parent are supervised. Alternative bounded candidates are
negative only for those targets; unknown targets stay masked. A known parent outside the radius or
top-K set is counted as a candidate-recall failure and omitted rather than mislabeled. The frozen
host model is always in evaluation mode and receives no gradient.

## Runtime and tracking

The host pair pass runs once with image batch size 1 and stores compact CPU caches. The small graph
head then trains for 30 epochs with batch size 2048, AdamW at `1e-4`, and checkpoints every five
completed epochs. Before running beside another GPU job, cache generation is measured on one
dataset and two transitions; the full pass is parallelized only if the measured memory peak leaves
a safe margin.

W&B online tracking is required in project `biohub-cell-tracking`. It records config, cache
fingerprint and recall, train/validation loss and parent accuracy, residual magnitudes, runtime, and
checkpoint hashes. An authentication failure is fatal rather than silently switching offline.

Cache construction and deployment both use the fixed four-view XY detection TTA from the `0.890`
control. The configured detection probability threshold is `0.99`, equivalent to a raw-logit
threshold of approximately `4.59512`; the cache converts it explicitly before calling the
organizer detector. Both paths also keep the pooling kernel at `5.0` micrometers, edge threshold
`0.5`, degree limits `1/2`, and `public-applicable-v1` post-processing.

## TRUST-LB milestones

After schema, unit, and one-dataset smoke gates pass, one continuous run produces epoch-5 and
epoch-30 checkpoints. Both are packaged as internet-disabled Kaggle Notebooks against the same
base weights and post-processing. Epoch 5 is the early direction probe; epoch 30 is the fixed
duration comparison against the `0.890` host anchor.

EXP-0009 is not a detector, segmentation model, LSTM, division head, global ILP, gap-node, or
long-image-window experiment. Those require separate experiment IDs.
