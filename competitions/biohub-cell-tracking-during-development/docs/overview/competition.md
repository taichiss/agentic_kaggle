# Official competition contract

Checked on 2026-08-26 against the [Kaggle overview][overview]. The live Kaggle pages and accepted
Rules remain authoritative if they change.

## Objective

Detect cells in 3D microscopy data, associate them across time, identify division events, and
reconstruct cell lineages. Ground truth is sparse rather than an exhaustive label of every cell.

## Metric

The score is maximized:

```text
score = adjusted_edge_jaccard + 0.1 * division_jaccard
```

Predicted and ground-truth nodes are matched independently per timepoint by optimal bipartite
assignment using scaled centroid distance, with a maximum of 7.0 micrometers. Spatial scale is
`(z, y, x) = (1.625, 0.40625, 0.40625)` micrometers per voxel. A predicted edge is correct when its
matched endpoints correspond to a ground-truth edge. The edge Jaccard is adjusted by a penalty for
over-predicting total node count. Division events are nodes with at least two outgoing edges and are
evaluated with a local lineage window. Scores may exceed 1.0.

For implementation detail, use the organizer-maintained [metric specification][metric-spec] and
[baseline repository][baseline]. These technical references clarify the official overview but do
not replace the competition Rules.

## Submission schema

The required CSV header is:

```text
id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
```

- `node` rows provide integer voxel centroids and use `-1` for `source_id,target_id`.
- `edge` rows provide node references and use `-1` for `node_id,t,z,y,x`.
- `id` is a consecutive throwaway index.
- `dataset` is each test directory name without `.zarr`; every test dataset must appear.
- The submitted notebook must create a file named `submission.csv`.

## Timeline and execution constraints

- Start: 2026-06-29.
- Entry and team merger deadline: 2026-09-22 23:59 UTC.
- Final submission deadline: 2026-09-29 23:59 UTC.
- Submission is through Kaggle Notebooks.
- CPU and GPU notebook runtime are each limited to 12 hours.
- Internet must be disabled during submitted notebook execution.
- Freely and publicly available external data and pretrained models are allowed.

The daily submission quota and all eligibility/licensing details must be checked on the live Rules
page after joining; they are intentionally not inferred here.

[overview]: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview
[metric-spec]: https://github.com/royerlab/kaggle-cell-tracking-competition/blob/main/metrics.md
[baseline]: https://github.com/royerlab/kaggle-cell-tracking-competition
