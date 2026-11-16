#!/usr/bin/env bash
set -euo pipefail

# Run from the CASMI competition directory after the 12k strict checkpoint exists.
base=data/derived/mist-mixup-20261116
new="$base/M1-ce-strict-sharedmerge"
old="$base/M1-strict"
test -f "$new/manifest.json"
test -f "$new/final.pt"
test -f "$new/pair-audit.json"

uv run --offline --with torch==2.10.0+cu129 --with pandas --with pyarrow \
  --with numpy --with rdkit --index https://download.pytorch.org/whl/cu129 \
  --index https://pypi.org/simple \
  python src/mist_formula/evaluate_mixup_parent.py \
  --control data/derived/exp0015-cv/submission-control.csv \
  --test data/derived/candidate-rescue-corrected-v2/test.parquet \
  --labels data/derived/candidate-rescue-corrected-v2/labels-local-only.csv \
  --model-source src/mist_formula/frozen/a_0479_model.py \
  --model A=data/derived/mist-duration-20261107/dataset-12k/final.pt \
  --model M1_original="$old/final.pt" \
  --model M1_ce_strict="$new/final.pt" \
  --prior-a data/derived/mist-bcf-20261110/parent-proxy-evaluation/per-molecule.jsonl \
  --weight 0.25 --output "$new/parent-proxy" \
  > "$new/parent-proxy.log" 2>&1

uv run --offline --with numpy --index https://pypi.org/simple \
  python src/mist_formula/compare_m1_ce_strict.py \
  --A-panel-ranks data/derived/mist-bcf-20261110/panel-evaluation/per-molecule.jsonl \
  --old-manifest "$old/manifest.json" --new-manifest "$new/manifest.json" \
  --parent-report "$new/parent-proxy/report.json" \
  --parent-per-molecule "$new/parent-proxy/per-molecule.jsonl" \
  --pair-audit "$new/pair-audit.json" \
  --old-checkpoint "$old/final.pt" --new-checkpoint "$new/final.pt" \
  --output "$new/comparison-report.json"
