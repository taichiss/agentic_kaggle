"""EXP-0008: 候補 pool の改善前後の差分を集計する (レポート用 JSON)。

段階:
    v17       — COCONUT ∪ train、±8.5 ppm (EXP-0001 / EXP-0002)
    pcl       — + PubChemLite (EXP-0003 / EXP-0007)
    improved  — + PubChemLite、m/z のずれ補正、±max(5 ppm, 2 mDa) (EXP-0008)

使い方:
    uv run python scripts/casmi26_pool_diff.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

E = Path("enveda-casmi26/experiments")
K = 25
RUNS = {
    "np_keep": {
        "v17": E / "EXP-0001/np_keep",
        "pcl": E / "EXP-0003/np_keep_pcl",
        "improved": E / "EXP-0008/np_keep",
    },
    "np_drop": {
        "v17": E / "EXP-0001/np_drop",
        "pcl": E / "EXP-0003/np_drop_pcl",
        "improved": E / "EXP-0008/np_drop",
    },
    "np_other": {
        "v17": E / "EXP-0002/np_other",
        "pcl": E / "EXP-0007/np_other_pcl",
        "improved": E / "EXP-0008/np_other",
    },
    "fold0_keep": {
        "v17": E / "EXP-0001/fold0_keep",
        "pcl": E / "EXP-0007/fold0_keep_pcl",
        "improved": E / "EXP-0008/fold0_keep",
    },
}
STAGES = {"A": "ranker", "B": "channel_analog"}


def truth_ranks(run: Path) -> pd.DataFrame:
    per = pd.read_parquet(run / "per_molecule.parquet").set_index("molecule_id")
    out = pd.DataFrame(index=per.index)
    out["in_pool"] = per["rank_pool"].notna()
    for s, stage in STAGES.items():
        df = pd.read_parquet(run / f"stage_{stage}.parquet")
        hit = df.merge(per[["truth_key"]].reset_index(), on="molecule_id")
        hit = hit[hit["candidate_key"] == hit["truth_key"]].groupby("molecule_id")["rank"].min()
        out[f"rank_{s}"] = hit.reindex(per.index)
    diag = pd.read_parquet(run / "diag.parquet").set_index("molecule_id")
    out["n_cand"] = diag["n_candidates"].reindex(per.index)
    return out


def rr(rank: pd.Series) -> pd.Series:
    return (1.0 / rank).where(rank <= K, 0.0).fillna(0.0)


def bins(rank: pd.Series, in_pool: pd.Series) -> dict[str, int]:
    r = rank.where(in_pool)
    return {
        "1": int((r == 1).sum()),
        "2-3": int(r.between(2, 3).sum()),
        "4-10": int(r.between(4, 10).sum()),
        "11-25": int(r.between(11, K).sum()),
        "26+": int((r > K).sum()),
        "pool外": int((~in_pool).sum()),
    }


def main() -> int:
    res: dict = {}
    for cond, stages in RUNS.items():
        tr = {v: truth_ranks(p) for v, p in stages.items()}
        c: dict = {"versions": {}}
        for v, t in tr.items():
            c["versions"][v] = {
                "n": len(t),
                "coverage": float(t["in_pool"].mean()),
                "n_cand_median": float(t["n_cand"].median()),
                "n_cand_p90": float(t["n_cand"].quantile(0.9)),
                **{f"mrr_{s}": float(rr(t[f"rank_{s}"]).mean()) for s in STAGES},
                **{f"bins_{s}": bins(t[f"rank_{s}"], t["in_pool"]) for s in STAGES},
            }
        a, b = tr["v17"], tr["improved"]
        c["pool_change"] = {
            "gained": int((~a["in_pool"] & b["in_pool"]).sum()),
            "lost": int((a["in_pool"] & ~b["in_pool"]).sum()),
            "kept": int((a["in_pool"] & b["in_pool"]).sum()),
            "never": int((~a["in_pool"] & ~b["in_pool"]).sum()),
        }
        for s in STAGES:
            x, y = rr(a[f"rank_{s}"]), rr(b[f"rank_{s}"])
            c[f"rank_change_{s}"] = {
                "better": int((y > x).sum()),
                "worse": int((y < x).sum()),
                "same": int((y == x).sum()),
            }

            # 散布図用: 改善前後の正解順位 (圏外は 26、pool 外は 0 で表す)
            def code(r: pd.Series, pool: pd.Series) -> list[int]:
                return [
                    0 if not p else (int(v) if v <= K else 26) for v, p in zip(r, pool, strict=True)
                ]

            c[f"pairs_{s}"] = list(
                zip(
                    code(a[f"rank_{s}"], a["in_pool"]),
                    code(b[f"rank_{s}"], b["in_pool"]),
                    strict=True,
                )
            )
        res[cond] = c
    out = E / "EXP-0008" / "pool_diff.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    for cond, c in res.items():
        print(
            cond,
            {
                v: (
                    round(x["coverage"], 3),
                    x["n_cand_median"],
                    round(x["mrr_A"], 4),
                    round(x["mrr_B"], 4),
                )
                for v, x in c["versions"].items()
            },
            c["pool_change"],
        )
    print("→", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
