"""EXP-0009: 保存済みの候補スコア (features.parquet) から、決め打ちの並べ方を比べる。

重みは調整しない。prior は 0.25 × (ln(1+SID) + ln(1+PMID)) (公開 kernel と同じ)。
同点は半分ずつ前に数える期待順位で MRR@25 を出す。

使い方:
    uv run python scripts/casmi26_rank_variants.py <EXP-0009 の条件フォルダ> ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

K = 25


def _z(x: pd.Series) -> pd.Series:
    sd = x.std(ddof=0)
    return (x - x.mean()) / sd if sd > 1e-9 else x * 0.0


def variants(f: pd.DataFrame) -> dict[str, pd.Series]:
    fused = f["lib"] + f["analog"] + 0.25 * f["frag"]
    zf = fused.groupby(f["molecule_id"]).transform(_z)
    za = f["analog"].groupby(f["molecule_id"]).transform(_z)
    zfr = f["frag"].groupby(f["molecule_id"]).transform(_z)
    return {
        "A 固定融合": fused,
        "B analog": f["analog"],
        "prior だけ": f["prior"],
        "A + prior": zf + f["prior"],
        "B + prior": za + f["prior"],
        "B + frag + prior": za + 0.5 * zfr + f["prior"],
    }


def mrr(f: pd.DataFrame, score: pd.Series, truth: dict[str, str], n_mol: int) -> float:
    d = f[["molecule_id", "candidate_key"]].assign(s=score.to_numpy())
    tot = 0.0
    for mid, sub in d.groupby("molecule_id"):
        t = truth.get(mid)
        hit = sub[sub["candidate_key"] == t]
        if hit.empty:
            continue
        s = float(hit["s"].iloc[0])
        r = 1 + (sub["s"] > s).sum() + 0.5 * ((sub["s"] == s).sum() - 1)
        tot += 1.0 / r if r <= K else 0.0
    return tot / n_mol


def main() -> int:
    out = {}
    for p in map(Path, sys.argv[1:]):
        f = pd.read_parquet(p / "features.parquet")
        t = pd.read_parquet(p / "truth.parquet")
        truth = dict(zip(t["molecule_id"], t["truth_key"], strict=True))
        out[p.name] = {k: round(float(mrr(f, v, truth, len(t))), 4) for k, v in variants(f).items()}
        print(p.name, out[p.name], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
