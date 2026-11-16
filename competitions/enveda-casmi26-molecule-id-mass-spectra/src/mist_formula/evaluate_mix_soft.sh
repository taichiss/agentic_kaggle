#!/usr/bin/env bash
set -euo pipefail

# Run after `run_mix_soft.sh full` from the CASMI competition directory.
output=data/derived/mist-mixup-20261116/M3-full
checkpoint="$output/final.pt"
test -f "$checkpoint" || { echo "Missing M3 checkpoint: $checkpoint" >&2; exit 2; }
uv run --offline --with torch==2.10.0+cu129 --with pandas --with pyarrow --with numpy \
  --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple \
  python src/mist_formula/evaluate_checkpoint.py \
  --checkpoint "$checkpoint" \
  --cache data/derived/saner-multipositive/cache-parallel \
  --precise-mzs data/derived/mist-formula-20261107/precise-mzs.bin \
  --panel data/derived/model-comparison-20261107/common-panel.parquet \
  --queries 1000 --device cuda --query-ranks \
  > "$output/evaluate.log" 2>&1
python3 - "$output" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
training = json.loads((out / "manifest.json").read_text())["validation"]
inference = json.loads((out / "validation-inference-parity.json").read_text())
for metric in ("queries", "mrr25", "top1", "top25"):
    assert abs(training[metric] - inference[metric]) < 1e-12, (metric, training, inference)
assert len(inference["query_mols"]) == len(inference["ranks"]) == 1000
PY
uv run --offline --with numpy --index https://pypi.org/simple \
  python src/mist_formula/compare_mixup_runs.py \
  --baseline-jsonl data/derived/mist-bcf-20261110/panel-evaluation/per-molecule.jsonl \
  --treatment-json "$output/validation-inference-parity.json" \
  --output "$output/paired-vs-A.json" \
  > "$output/paired-vs-A.log"
cat "$output/paired-vs-A.json"
