#!/usr/bin/env bash
set -euo pipefail

# Run from the CASMI competition directory. Original M1 files remain untouched.
case "${1:-}" in
  smoke0)
    output=data/derived/mist-mixup-20261116/M1-ce-strict-sharedmerge-sidecar-smoke0
    steps=1
    eval_queries=1
    mix_probability=0
    ;;
  full)
    output=data/derived/mist-mixup-20261116/M1-ce-strict-sharedmerge
    steps=12000
    eval_queries=1000
    mix_probability=0.25
    test -f "$output/pair-audit.json" || {
      echo "Pair replay audit is required before full training" >&2
      exit 2
    }
    ;;
  *)
    echo "usage: $0 {smoke0|full}" >&2
    exit 2
    ;;
esac

if test -e "$output/manifest.json"; then
  echo "Completed output already exists: $output" >&2
  exit 2
fi
mkdir -p "$output"
uv run --offline --with torch==2.10.0+cu129 --with pandas --with pyarrow --with numpy \
  --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple \
  python src/mist_formula/train_mix_same_ce_strict.py \
  --raw-train data/input/train.parquet \
  --cache data/derived/saner-multipositive/cache-parallel \
  --precise-mzs data/derived/mist-formula-20261107/precise-mzs.bin \
  --source-metadata data/derived/mist-mixup-20261116/source-metadata \
  --formula-csv data/derived/mist-bcf-20261110/B/original-0479-chno-compositions.csv \
  --panel data/derived/model-comparison-20261107/common-panel.parquet \
  --output "$output" --device cuda --seed 20261107 --steps "$steps" \
  --batch-size 128 --lr 0.0003 --rank-weight 0.5 --width 128 --layers 2 \
  --ppm 20 --mix-probability "$mix_probability" --mix-alpha-min 0.25 \
  --mix-alpha-max 0.75 --eval-queries "$eval_queries" --log-every 250 \
  --save-every 12000 --milestone-steps "$steps" > "$output/train.log" 2>&1
tail -3 "$output/train.log"
