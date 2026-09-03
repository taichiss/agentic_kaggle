# Post-processing parameter inventory

The fixed control is EXP-0004 completed epoch 30 with
`public-applicable-v1`, submission ref `55810126`, and public score `0.890`.
On identical completed-epoch-20 weights, the same profile improved public LB
from `0.805` raw to `0.869`, so topology repair is a confirmed high-impact
stage.

## Frozen `public-applicable-v1` parameters

| stage | parameter | value |
| --- | --- | ---: |
| inference | detection / edge threshold | 0.99 / 0.50 |
| inference | pooling / detection TTA | 5 µm / four XY views |
| motion | tight / relaxed Hungarian gate | 6 µm / 10 µm |
| motion | velocity extrapolation | 0.5 |
| motion | cost | motion distance + 0.05 × raw distance − learned probability |
| division | parent / sister / existing-child maximum | 8 / 11 / 10 µm |
| division | minimum next-frame sister divergence | 2.25 µm |
| division | frame / global fraction cap | 0.0076 / 0.00375 |
| pruning | minimum component nodes | 6 |
| pruning | division-containing components | always retained |
| smoothing | neighborhood / raw-to-fit blend | ±2 frames / 0.2:0.8 |

The control public output has 167,075 nodes, 159,375 edges, 326,450 rows,
and an edge/node ratio of 0.9539. Its CSV SHA-256 is
`61bb2abda986176efa602dbd7ebe5e2cc5cc35c6dee7a028714073915bb4b64a`.

## EXP-0010 one-factor probes

All model weights, inference thresholds, TTA, division logic, and smoothing
remain fixed. Each arm changes exactly one scalar.

| hypothesis | variant | only change | nodes | edges | edge/node | adopted submission |
| --- | --- | --- | ---: | ---: | ---: | --- |
| H012 precision | `precision_min7` | minimum component size 6 → 7 | 162,863 | 155,865 | 0.9570 | ref `55829542`, pending |
| H013 recall | `recall_gate12` | relaxed motion gate 10 → 12 µm | 171,734 | 164,530 | 0.9581 | ref `55828801`, pending |

H012 predicts that transient six-node non-division tracks drive enough of the
remaining node-count penalty that removing them beats `0.890`. H013 predicts
that missing long-displacement links dominate enough of the remaining error
that expanding the relaxed gate beats `0.890`.

Interpretation is predeclared:

- only H012 improves: node-count overprediction dominates;
- only H013 improves: missing motion links dominate;
- both improve: useful edge density can be raised from both directions;
- neither improves: retain the frozen profile at this checkpoint.

## Submission audit

- Ref `55828867` used a CPU child Notebook that pruned a fixed public-test CSV.
  Its CSV passed all local structural checks, but the Code Competition hidden
  rerun could not generate the active hidden datasets and returned
  `Submission Scoring Error`. It is an invalid packaging attempt, not an H012
  result.
- Ref `55829542` corrects H012 by running the pinned epoch-30 model and min-7
  post-processing on the active competition test mount. Its public output is
  byte-identical to the validated min-7 output, SHA-256
  `68e9ef756433c6778e7560027997e0a5b04a829807b285c8bd2cb0cc56abedba`.
- Ref `55829582` is an accidental duplicate of the corrected H012 submission,
  created by overlapping submit processes. It must not be treated as a third
  experimental condition.

The exact frozen settings and patch strings are versioned in
[`configs/exp-0010-postprocess-ab.toml`](../../configs/exp-0010-postprocess-ab.toml).
