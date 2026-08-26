# Data contract

## Confirmed structure

The competition uses 3D+time microscopy images and tracking graphs:

- Images are Zarr v3 arrays with axes `(T, Z, Y, X)` and `uint16` values. The data catalog shows a
  typical shape of `(100, 64, 256, 256)`; individual arrays still need local inspection.
- Spatial voxel scale `(Z, Y, X)` is `(1.625, 0.40625, 0.40625)` micrometers.
- Training tracking annotations use GEFF graphs. Nodes contain time and centroid coordinates; edges
  link a cell across time, with a fork representing division.
- The annotations are sparse. Absence of an annotation must not be interpreted as confirmed
  background.
- The competition data description states that train and test embryos are disjoint.
- The official data catalog reports 87.61 GB across 24,886 files under the CC0 public-domain
  license. Downloading is therefore an explicit authenticated step, not an automatic bootstrap.

The organizer baseline documents the expected Kaggle mount as `train/` pairs of `{name}.zarr` and
`{name}.geff`, plus test `{name}.zarr` images. Treat the actual downloaded tree as authoritative and
record any discrepancy before implementing loaders.

## Local placement

```text
data/
├── downloads/        Kaggle archives
├── raw/              extracted competition data
│   ├── train/
│   └── test/
├── public-notebooks/ Kaggle kernel copies
└── external/         organizer baseline checkout
```

Everything under `data/` is ignored by Git. Do not commit symlinks to private data, archives,
credentials, GEFF labels, image chunks, or notebook outputs.

## Inspection boundary

Run `scripts/inspect_data.py` after download. It only inventories names, suffixes, and sizes; it does
not load OME-Zarr arrays into memory. Dataset-specific loaders should be implemented only after this
inventory confirms the real layout.
