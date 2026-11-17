#!/usr/bin/env bash
# Run the local-only frozen 0.477 CE-rank ablation on the external-80 panel.
set -euo pipefail

COMP="$(cd "$(dirname "$0")/../.." && pwd)"
PANEL="$COMP/data/derived/candidate-rescue-corrected-v2"
BUILD="$COMP/data/derived/ce-views-20261117/rank-only-local-build"
SCRATCH="${CE_RANK_ONLY_SCRATCH:-$COMP/data/derived/ce-views-20261117/rank-only-local-mount-v1}"
NOTEBOOK="$BUILD/ce-analog-rank-only-local.ipynb"
RECEIPT="$BUILD/build-receipt.json"

test -f "$NOTEBOOK"
test -f "$RECEIPT"
test "$(sha256sum /kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet | cut -d ' ' -f 1)" = \
  "$(sha256sum "$PANEL/test.parquet" | cut -d ' ' -f 1)"
mkdir -p "$SCRATCH/input" "$SCRATCH/working"
test ! -e "$SCRATCH/working/submission.csv"

export COMP PANEL BUILD SCRATCH NOTEBOOK RECEIPT
unshare -m --propagation private bash -c '
  set -euo pipefail
  mount --bind /kaggle/input "$SCRATCH/input"
  mount --rbind "$SCRATCH" /kaggle
  test "$(readlink -f /kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet)" = "$PANEL/test.parquet"
  cd "$COMP"
  export KAGGLE_IS_COMPETITION_RERUN=1
  uv run --offline --with torch==2.10.0+cu129 --with pandas --with pyarrow \
    --with numpy --with rdkit==2026.3.3 --with numba --with lightgbm \
    --with matplotlib --with seaborn --with ipython --with scikit-learn --with scipy \
    --with networkx --with joblib --with tqdm --with pip \
    --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple \
    python src/public_notebooks/run_ce_analog_rank_only_local.py \
      --notebook "$NOTEBOOK" --receipt "$RECEIPT" --working "$SCRATCH/working"
' 2>&1 | tee "$SCRATCH/run.log"
