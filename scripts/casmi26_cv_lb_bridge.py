"""CV と LB の橋渡し: クラス別 MRR (CV 実測) と 2 本の LB から hidden test のクラス構成を逆算する。

モデル:
    LB_s = p1 * M_s(class1) + p2 * M_s(class2) + p3 * 0      (s ∈ {A, B})
    p1 + p2 + p3 = 1
    Class 3 (候補 pool に正解なし) はどの順位付けでも 0 点。
    M_s(class2) は EXP-0001 <split>_keep (reference を全部除いた CV)、
    M_s(class1) は EXP-0002 <split>_other (他ソースの reference を残した CV) の class1 分子。

2 式 2 未知数 (p1, p2) を解き、LB のノイズ (SE) と CV 側の分子 bootstrap で不確かさを出す。

使い方:
    uv run python scripts/casmi26_cv_lb_bridge.py --split np
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP1 = Path("enveda-casmi26/experiments/EXP-0001")
EXP2 = Path("enveda-casmi26/experiments/EXP-0002")
LB = {"A": 0.191, "B": 0.247}
STAGE = {"A": "ranker", "B": "channel_analog"}
LB_SE = 0.03  # Public LB 約 124 分子での 1 本あたりの標準誤差 (EXP-0001 README)
K = 25


def per_molecule_rr(run: Path) -> pd.DataFrame:
    """run の分子ごとの reciprocal rank (A / B) とクラス。"""
    per = pd.read_parquet(run / "per_molecule.parquet")[["molecule_id", "truth_key", "class"]]
    for s, stage in STAGE.items():
        df = pd.read_parquet(run / f"stage_{stage}.parquet")
        hit = df.merge(per[["molecule_id", "truth_key"]], on="molecule_id")
        hit = hit[hit["candidate_key"] == hit["truth_key"]].groupby("molecule_id")["rank"].min()
        rank = per["molecule_id"].map(hit)
        per[f"rr_{s}"] = np.where(rank <= K, 1.0 / rank, 0.0)
    return per


def solve(m: dict[str, dict[str, float]], lb: dict[str, float]) -> tuple[float, float, float]:
    """[[A1, A2], [B1, B2]] @ [p1, p2] = [LB_A, LB_B] を解く。"""
    a = np.array([[m["A"]["class1"], m["A"]["class2"]], [m["B"]["class1"], m["B"]["class2"]]])
    p1, p2 = np.linalg.solve(a, np.array([lb["A"], lb["B"]]))
    return float(p1), float(p2), float(1.0 - p1 - p2)


def class_mrr(c1: pd.DataFrame, c2: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {s: {"class1": c1[f"rr_{s}"].mean(), "class2": c2[f"rr_{s}"].mean()} for s in STAGE}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["np", "fold0"], default="np")
    ap.add_argument("--out", type=Path, default=EXP2)
    ap.add_argument("--n-boot", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args()

    keep = per_molecule_rr(EXP1 / f"{args.split}_keep")
    other = per_molecule_rr(EXP2 / f"{args.split}_other")
    c2 = keep  # reference を全部除いた CV は全分子 class2 (pool=keep)
    c1 = other[other["class"] == "class1"]
    m = class_mrr(c1, c2)
    p1, p2, p3 = solve(m, LB)

    rng = np.random.default_rng(args.seed)
    draws = []
    for _ in range(args.n_boot):
        b1 = c1.sample(len(c1), replace=True, random_state=rng.integers(1 << 31))
        b2 = c2.sample(len(c2), replace=True, random_state=rng.integers(1 << 31))
        lb = {s: LB[s] + rng.normal(0.0, LB_SE) for s in LB}
        try:
            draws.append(solve(class_mrr(b1, b2), lb))
        except np.linalg.LinAlgError:
            continue
    d = np.array(draws)

    def band(x: np.ndarray) -> list[float]:
        return [float(v) for v in np.quantile(x, [0.1, 0.5, 0.9])]

    # p1 を固定したときに A / B それぞれが要求する p3 (図用の直線)
    grid = np.linspace(0.0, 0.5, 51)
    lines = {
        s: [
            {"p1": float(g), "p3": float(1 - g - (LB[s] - g * m[s]["class1"]) / m[s]["class2"])}
            for g in grid
        ]
        for s in STAGE
    }
    res = {
        "split": args.split,
        "lb": LB,
        "lb_se": LB_SE,
        "n_class1": int(len(c1)),
        "n_class2": int(len(c2)),
        "n_other_run": int(len(other)),
        "class_mrr": m,
        "estimate": {"p1": p1, "p2": p2, "p3": p3},
        "interval_10_50_90": {"p1": band(d[:, 0]), "p2": band(d[:, 1]), "p3": band(d[:, 2])},
        "feasible_share": float(((d >= 0) & (d <= 1)).all(axis=1).mean()),
        "lines": lines,
        "other_run_mrr": {s: float(other[f"rr_{s}"].mean()) for s in STAGE},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"bridge_{args.split}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False)
    )
    print(json.dumps({k: v for k, v in res.items() if k != "lines"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
