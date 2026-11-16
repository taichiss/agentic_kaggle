#!/usr/bin/env bash
set -euo pipefail

# Run from the CASMI competition directory. M3 uses frozen A inputs and the
# source-aligned partner index shared with M2.
case "${1:-}" in
  smoke0)
    output=data/derived/mist-mixup-20261116/M3-smoke0
    steps=1
    eval_queries=1
    mix_probability=0.0
    device=cpu
    ;;
  smoke0gpu)
    output=data/derived/mist-mixup-20261116/M3-smoke0-gpu
    steps=1
    eval_queries=1
    mix_probability=0.0
    device=cuda
    ;;
  smoke)
    output=data/derived/mist-mixup-20261116/M3-smoke
    steps=1
    eval_queries=1
    mix_probability=1.0
    device=cpu
    ;;
  full)
    output=data/derived/mist-mixup-20261116/M3-full
    steps=12000
    eval_queries=1000
    mix_probability=0.25
    device=cuda
    ;;
  *)
    echo "usage: $0 {smoke0|smoke0gpu|smoke|full}" >&2
    exit 2
    ;;
esac

metadata=data/derived/mist-mixup-20261116/source-metadata
pair_audit=data/derived/mist-mixup-20261116/M2-pair-coverage.json
test -f "$metadata/manifest.json" || {
  echo "Source-aligned mixup metadata is incomplete: $metadata" >&2
  exit 2
}
test -f "$pair_audit" || {
  echo "Cross-molecule pair audit is incomplete: $pair_audit" >&2
  exit 2
}
if test -e "$output/manifest.json"; then
  echo "Completed output already exists: $output" >&2
  exit 2
fi
mkdir -p "$output"
uv run --offline --with torch==2.10.0+cu129 --with pandas --with pyarrow --with numpy \
  --index https://download.pytorch.org/whl/cu129 --index https://pypi.org/simple \
  python src/mist_formula/train_mix_soft.py \
  --raw-train data/input/train.parquet \
  --cache data/derived/saner-multipositive/cache-parallel \
  --precise-mzs data/derived/mist-formula-20261107/precise-mzs.bin \
  --formula-csv data/derived/mist-bcf-20261110/B/original-0479-chno-compositions.csv \
  --a-checkpoint data/derived/mist-duration-20261107/dataset-12k/final.pt \
  --panel data/derived/model-comparison-20261107/common-panel.parquet \
  --metadata-dir "$metadata" --pair-audit "$pair_audit" --output "$output" \
  --device "$device" --seed 20261107 --steps "$steps" --batch-size 128 --lr 0.0003 \
  --rank-weight 0.5 --width 128 --layers 2 --ppm 20 \
  --mix-probability "$mix_probability" --eval-queries "$eval_queries" \
  --log-every 250 --save-every 12000 --milestone-steps "$steps" \
  > "$output/train.log" 2>&1
tail -3 "$output/train.log"
