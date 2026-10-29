"""EXP-0009: 正解構造が PubChem の窓の中で文献量の何位にいるか (上位 N 件のカバー率曲線)。

パイプラインを回す前の安価な確認。窓は EXP-0008 と同じ (m/z のずれ補正 + ±max(5 ppm, 2 mDa))。
窓内の全 PubChem 構造に標準 InChIKey の先頭 14 文字を付け、正解の InChIKey14 と照合する。

使い方:
    uv run python scripts/casmi26_pubchem_coverage.py
"""

from __future__ import annotations

import importlib.util
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from casmi26.baseline.chem import corrected_mz, neutral_mass
from casmi26.baseline.infer import POOL_EXP0008
from casmi26.baseline.pubchem import PubChemIndex

RDLogger.DisableLog("rdApp.*")  # type: ignore[attr-defined]
INDEX = Path("data/working/casmi26/pubchem_index")
OUT = Path("enveda-casmi26/experiments/EXP-0009")
NS = (10, 30, 100, 300, 1000, 3000)


def ik14(smi: str) -> str:
    m = Chem.MolFromSmiles(smi) if smi else None
    k = Chem.MolToInchiKey(m) if m is not None else ""
    return k[:14]


def _diag_module():
    spec = importlib.util.spec_from_file_location("d", "scripts/casmi26_ms_diagnostics.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    d = _diag_module()
    pc = PubChemIndex(INDEX)
    res: dict = {}
    for split in ("np", "fold0"):
        q, st = d.select_queries(split, 400)
        st1 = st.drop_duplicates("scorer_key").set_index("scorer_key")
        pmz = corrected_mz(
            q["precursor_mz"].to_numpy(float),
            q["ionization_mode"].to_numpy(),
            POOL_EXP0008["mz_offset_pos"],
            POOL_EXP0008["mz_offset_neg"],
        )
        q = q.assign(nm=neutral_mass(pmz, q["adduct"].to_numpy()))
        rows, jobs = [], []
        for key, sub in q.groupby("key", sort=True):
            nms = sub["nm"].to_numpy()
            nms = nms[np.isfinite(nms)]
            t = float(np.median(nms))
            half = max(t * POOL_EXP0008["pool_ppm"] / 1e6, POOL_EXP0008["pool_min_da"])
            lo = int(np.searchsorted(pc.mass, t - half, "left"))
            hi = int(np.searchsorted(pc.mass, t + half, "right"))
            pri = np.asarray(pc.prior[lo:hi])
            order = np.argsort(-pri, kind="stable")
            cids = np.asarray(pc.cid[lo:hi])[order]
            rows.append({"key": key, "truth_ik": st1.loc[key, "inchikey14"], "n_window": hi - lo})
            jobs.append([pc.smiles_of(int(c)) for c in cids])
        flat = [s for j in jobs for s in j]
        with mp.get_context("spawn").Pool(12) as p:
            keys = p.map(ik14, flat, chunksize=512)
        pos = 0
        for r, j in zip(rows, jobs, strict=True):
            ks = keys[pos : pos + len(j)]
            pos += len(j)
            hit = [i for i, k in enumerate(ks) if k == r["truth_ik"]]
            r["rank"] = hit[0] + 1 if hit else None
        df = pd.DataFrame(rows)
        n = len(df)
        res[split] = {
            "molecules": n,
            "window_median": float(df["n_window"].median()),
            "window_p90": float(df["n_window"].quantile(0.9)),
            "in_pubchem_window": float(df["rank"].notna().mean()),
            "coverage_top_n": {str(k): float((df["rank"] <= k).sum() / n) for k in NS},
            "rank_median_when_found": float(df["rank"].median()),
        }
        df.to_csv(OUT / f"pubchem_rank_{split}.csv", index=False)
        print(split, json.dumps(res[split], ensure_ascii=False), flush=True)
    (OUT / "pubchem_coverage.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
