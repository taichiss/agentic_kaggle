# BioHub tracking ecosystem decisions

This document turns the organizer Discussion resource list into an environment policy. It avoids a
single oversized environment containing mutually incompatible tracking stacks.

## Adopt now

- Zarr v3 for image arrays.
- The GEFF specification through the organizer-pinned `tracksdata` implementation for sparse graph
  annotations and predictions. The GEFF repository itself is not installed as a Python package.
- NumPy, SciPy, pandas, and blosc2 for inspection, matching, tabulation, and chunk decoding.
- The organizer baseline at the revision pinned in `asset-sources.toml` for metric and I/O behavior.

The reproducible environment is in `environment/pyproject.toml`. Git dependencies are pinned to
commits observed on 2026-08-26 rather than moving `main` branches.

## Optional profiles

- `baseline`: PyTorch model execution and organizer-style training.
- `viz`: napari and ndv for native local visualization.
- GPU image processing is intentionally separate because the CuPy package must match the local CUDA
  major version. Add the matching CuPy/cuCIM build only after `nvidia-smi` and the runtime agree.

## Evaluate as isolated experiments

`ultrack`, `trackastra`, `byotrack`, `laptrack`, `CELLECT`, `ASCENT`, `OrganoidTracker`, and `motile`
are research candidates. Each can bring a large or conflicting dependency graph, so a selected method
gets its own experiment environment or Kaggle Notebook image. Selection requires:

1. support for 3D+time data or a documented adaptation;
2. lineage/division output that can map to GEFF;
3. offline execution with all weights and packages attached;
4. completion within the 12-hour notebook limit;
5. a valid `submission.csv` on every public dummy test dataset.

## External data

The Cell Tracking Challenge is the first external-data candidate because the organizer explicitly
recommended it. Before use, record the exact dataset, license, axes/scale adaptation, and evidence that
it is freely and publicly available under the current Rules.
