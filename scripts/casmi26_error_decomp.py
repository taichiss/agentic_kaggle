"""CASMI26 誤差分解 CLI (kaggle_flow_v1.md §4 Week 1 誤差分解ゲート)。

使い方:
    uv run python scripts/casmi26_error_decomp.py \
        --truth truth.parquet \
        --stage pool=pool.parquet --stage fp=fp.parquet \
        --stage simulator=sim.parquet --stage ranker=final.parquet \
        --reference-keys ref_keys.parquet --database-keys db_keys.parquet \
        --out experiments/EXP-0001/error_decomp

入力:
    truth          : molecule_id + (truth_key | truth_smiles)
    --stage        : 段名=parquet。molecule_id, rank + (candidate_key | smiles)
                     指定順が上流→下流。最後の段が最終提出の順位
    --*-keys       : key 列 1 本の parquet (Class 1/2/3 ラベル付け用、任意)

出力 (--out 配下):
    report.md, per_molecule.parquet, stage_table.csv, rank_table.csv, summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi26.analysis import (
    StageSpec,
    assign_classes,
    attach_canonical_keys,
    decompose,
    default_stages,
    to_markdown,
)
from casmi26.metrics import canonical_inchikey14


def _load_stage(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "candidate_key" not in df.columns:
        if "smiles" not in df.columns:
            raise SystemExit(f"{path}: candidate_key か smiles 列が必要")
        df = attach_canonical_keys(df, smiles_col="smiles")
    return df


def _load_truth(path: Path) -> dict[str, str | None]:
    df = pd.read_parquet(path)
    if "truth_key" not in df.columns:
        if "truth_smiles" not in df.columns:
            raise SystemExit(f"{path}: truth_key か truth_smiles 列が必要")
        df["truth_key"] = df["truth_smiles"].map(canonical_inchikey14)
    return dict(zip(df["molecule_id"].astype(str), df["truth_key"], strict=True))


def _load_keys(path: Path | None) -> set[str]:
    if path is None:
        return set()
    df = pd.read_parquet(path)
    return set(df.iloc[:, 0].dropna().astype(str))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--stage", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--fp-top-n", type=int, default=100)
    ap.add_argument("--reference-keys", type=Path)
    ap.add_argument("--database-keys", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    known = {s.name: s for s in default_stages(args.fp_top_n)}
    stages: list[StageSpec] = []
    outputs: dict[str, pd.DataFrame] = {}
    for item in args.stage:
        name, _, path = item.partition("=")
        if not path:
            raise SystemExit(f"--stage は NAME=PATH 形式: {item}")
        # 既定 4 段以外の名前は「含有」判定の段として扱う
        stages.append(known.get(name, StageSpec(name, None)))
        outputs[name] = _load_stage(Path(path))
    # 最後の段は最終提出の順位なので rank 1 を通過条件にする
    stages[-1] = StageSpec(stages[-1].name, 1)

    truth = _load_truth(args.truth)
    classes = None
    if args.reference_keys or args.database_keys:
        classes = assign_classes(
            truth,
            reference_keys=_load_keys(args.reference_keys),
            database_keys=_load_keys(args.database_keys),
        )

    res = decompose(truth, outputs, stages=stages, classes=classes)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.md").write_text(to_markdown(res), encoding="utf-8")
    res.per_molecule.to_parquet(args.out / "per_molecule.parquet", index=False)
    res.stage_table.to_csv(args.out / "stage_table.csv", index=False)
    res.rank_table.to_csv(args.out / "rank_table.csv")
    summary = {
        "mrr": res.mrr,
        "count": res.count,
        "stages": [{"name": s.name, "pass_rank": s.pass_rank} for s in res.stages],
        "ceilings": res.ceilings,
        "loss_share_by_stage": res.stage_table.groupby("bucket")["loss_share"].sum().to_dict(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"MRR@25={res.mrr:.4f} U={res.count} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
