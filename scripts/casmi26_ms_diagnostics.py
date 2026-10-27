"""EXP-0003: 質量スペクトルの原理から、候補 pool とランカーの失敗要因を切り分ける。

入力 (既存の実験出力を再利用):
    EXP-0002/<split>_other/stage_pool.parquet   候補窓 (candidate_key, smiles)。pool=keep と同じ
    EXP-0001/<split>_keep/stage_{ranker,channel_analog}.parquet   A / B の順位
query の抽出は scripts/casmi26_baseline_cv.py と同じ乱数・同じ手順で再現する。

出力: enveda-casmi26/experiments/EXP-0003/ms_diag_<split>.json と .md

使い方:
    uv run python scripts/casmi26_ms_diagnostics.py --split np
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, DataStructs
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from casmi26.analysis.ms_diag import (
    CATEGORIES,
    Peaks,
    ScoreVariant,
    annotate,
    exact_match_ppm,
    explain_prepared,
    prepare_for,
    prepare_peaks,
)
from casmi26.baseline.chem import ADDUCTS, morgan_fp, neutral_mass
from casmi26.baseline.fragments import fragment_masses

TRAIN = Path("data/input/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
FOLDS = Path("data/input/external/casmi26/casmi26-identity-disjoint-validation-folds")
COCO = Path("data/input/external/casmi26/coconut-casmi26-candidates")
EXP = Path("enveda-casmi26/experiments")
SEED = 20260930
MAX_SPECTRA = 9
K = 25
_EMPTY = np.zeros(0)
BIG_BONDS = 120  # 結合数上限を実質外した列挙
CAP3 = 45  # 3 結合切断は結合数 45 以下の分子だけ (組合せ爆発を避ける)

VARIANTS = [
    ScoreVariant("V0 V17 そのまま"),
    ScoreVariant("V1 結合数上限を撤廃", max_bonds=BIG_BONDS),
    ScoreVariant("V2 +許容差 10 ppm", max_bonds=BIG_BONDS, abs_da=None, ppm=10.0),
    ScoreVariant(
        "V3 +同位体ピーク除去", max_bonds=BIG_BONDS, abs_da=None, ppm=10.0, deisotope=True
    ),
    ScoreVariant(
        "V4 +中性ロス・連続ロス",
        max_bonds=BIG_BONDS,
        abs_da=None,
        ppm=10.0,
        deisotope=True,
        neutral_losses=True,
    ),
    ScoreVariant(
        "V5 +付加イオン担体",
        max_bonds=BIG_BONDS,
        abs_da=None,
        ppm=10.0,
        deisotope=True,
        neutral_losses=True,
        adduct_carriers=True,
    ),
    ScoreVariant(
        "V6 V5 で H 移動を ±1 に制限",
        max_bonds=BIG_BONDS,
        abs_da=None,
        ppm=10.0,
        h_shifts=(-1, 0, 1),
        deisotope=True,
        neutral_losses=True,
        adduct_carriers=True,
    ),
    ScoreVariant("V7 V0 で H 移動を ±1 に制限", h_shifts=(-1, 0, 1)),
]

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def select_queries(split: str, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """casmi26_baseline_cv.py と同じ query 行を選ぶ。"""
    meta = pq.read_table(
        TRAIN,
        columns=["ingest_lib", "instrument_type", "adduct", "ionization_mode", "precursor_mz"],
    ).to_pandas()
    sf = pd.read_parquet(FOLDS / "spectra_folds.parquet")
    st = pd.read_parquet(FOLDS / "structures.parquet")
    key_of_ik = dict(zip(st["inchikey14"], st["scorer_key"], strict=True))
    meta["key"] = sf["inchikey14"].map(key_of_ik).to_numpy(dtype=object)
    usable = meta["adduct"].isin(list(ADDUCTS))
    if split == "np":
        q_rows = meta[(meta["ingest_lib"] == "enveda-np-examples") & usable]
    else:
        cand = meta[(sf["fold_identity"] == 0) & (meta["instrument_type"] == "timsTOF") & usable]
        keys = np.sort(cand["key"].unique())
        rng = np.random.default_rng(SEED)
        pick = set(rng.choice(keys, size=min(n, len(keys)), replace=False))
        q_rows = cand[cand["key"].isin(pick)]
    rng = np.random.default_rng(SEED)
    q_rows = (
        q_rows.assign(_r=rng.random(len(q_rows)))
        .sort_values(["key", "_r"])
        .groupby("key")
        .head(MAX_SPECTRA)
    )
    return q_rows, st


def read_rows(rows: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray, float]]:
    want = np.sort(rows)
    pf = pq.ParquetFile(TRAIN)
    out: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    base = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        sel = want[(want >= base) & (want < base + n)]
        if len(sel):
            t = pf.read_row_group(
                g, columns=["ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev"]
            ).take(sel - base)
            for r, m, i, ce in zip(
                sel,
                t.column("ms2_mzs").to_pylist(),
                t.column("ms2_normalized_intensities").to_pylist(),
                t.column("collision_energy_ev").to_pylist(),
                strict=True,
            ):
                cev = float(np.mean(ce)) if ce else float("nan")
                out[int(r)] = (np.asarray(m, float), np.asarray(i, float), cev)
        base += n
    return out


def formula(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi) if smi else None
    return CalcMolFormula(m) if m is not None else None


def n_bonds(smi: str) -> int:
    m = Chem.MolFromSmiles(smi) if smi else None
    return m.GetNumBonds() if m is not None else 0


def pmap(fn, items: list, workers: int, chunksize: int = 16) -> list:
    with mp.get_context("spawn").Pool(workers) as p:
        return p.map(fn, items, chunksize=chunksize)


def truth_rank(scores: np.ndarray, truth_idx: int) -> float:
    """同点は半分ずつ前に数える期待順位。"""
    s = scores[truth_idx]
    return 1.0 + float((scores > s).sum()) + 0.5 * float((scores == s).sum() - 1)


def rr(rank: float) -> float:
    return 1.0 / rank if rank <= K else 0.0


def q(x: list[float] | np.ndarray, qs: tuple[float, ...] = (0.1, 0.5, 0.9)) -> list[float]:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return [float(v) for v in np.quantile(a, qs)] if len(a) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["np", "fold0"], required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", type=Path, default=EXP / "EXP-0003")
    args = ap.parse_args()

    q_rows, st = select_queries(args.split, args.n)
    raw = read_rows(q_rows.index.to_numpy())
    st1 = st.drop_duplicates("scorer_key").set_index("scorer_key")
    log(f"{args.split}: {q_rows['key'].nunique()} molecules / {len(q_rows)} spectra")

    pool = pd.read_parquet(EXP / "EXP-0002" / f"{args.split}_other" / "stage_pool.parquet")
    ranker = pd.read_parquet(EXP / "EXP-0001" / f"{args.split}_keep" / "stage_ranker.parquet")
    analog = pd.read_parquet(
        EXP / "EXP-0001" / f"{args.split}_keep" / "stage_channel_analog.parquet"
    )
    top1 = {
        "A": ranker[ranker["rank"] == 1].set_index("molecule_id")["candidate_key"].to_dict(),
        "B": analog[analog["rank"] == 1].set_index("molecule_id")["candidate_key"].to_dict(),
    }

    # ---- 1. precursor 質量と候補窓 -------------------------------------------------
    nm = neutral_mass(q_rows["precursor_mz"].to_numpy(float), q_rows["adduct"].to_numpy())
    true_m = st1.loc[q_rows["key"], "mono_mass"].to_numpy(float)
    delta = nm - true_m
    ppm = delta / true_m * 1e6
    per_mol = (
        pd.DataFrame({"key": q_rows["key"].to_numpy(), "nm": nm, "true": true_m})
        .groupby("key")
        .agg(nm=("nm", "median"), true=("true", "first"))
    )
    mol_ppm = (per_mol["nm"] - per_mol["true"]) / per_mol["true"] * 1e6
    cover = {f"{x:g}": float((mol_ppm.abs() <= x).mean()) for x in (1, 2, 3, 5, 8.5, 10, 15, 20)}
    iso_mis = int((np.abs(np.abs(delta) - 1.0033548) < 0.01).sum())
    by_adduct = (
        pd.DataFrame({"adduct": q_rows["adduct"].to_numpy(), "ppm": ppm})
        .groupby("adduct")["ppm"]
        .agg(["count", "median", lambda s: float(s.abs().quantile(0.9))])
        .rename(columns={"<lambda_0>": "abs_p90"})
    )
    co_mass = np.load(COCO / "coco_mass.npy")
    pool_mass = np.sort(np.concatenate([co_mass, st1["mono_mass"].to_numpy(float)]))
    win = {}
    for x in (3, 5, 8.5, 15):
        lo = np.searchsorted(pool_mass, per_mol["nm"].to_numpy() * (1 - x / 1e6))
        hi = np.searchsorted(pool_mass, per_mol["nm"].to_numpy() * (1 + x / 1e6))
        win[f"{x:g}"] = float(np.median(hi - lo))
    log("precursor mass done")

    # ---- 2. 候補窓の分子式構成と、1 位の誤り方 ---------------------------------------
    smiles_all = sorted(set(pool["smiles"]) | set(st1.loc[q_rows["key"].unique(), "smiles"]))
    form = dict(zip(smiles_all, pmap(formula, smiles_all, args.workers, 64), strict=True))
    nb = dict(zip(smiles_all, pmap(n_bonds, smiles_all, args.workers, 64), strict=True))
    pool["formula"] = pool["smiles"].map(form)
    comp = []
    confusion: dict[str, list[dict]] = {"A": [], "B": []}
    for key, sub in pool.groupby("molecule_id"):
        tkey = key.removeprefix("v_")
        tsmi = st1.loc[tkey, "smiles"]
        tf = form.get(tsmi)
        comp.append(
            {
                "n": len(sub),
                "n_formulas": sub["formula"].nunique(),
                "n_same_formula": int((sub["formula"] == tf).sum()),
            }
        )
        k2s = dict(zip(sub["candidate_key"], sub["smiles"], strict=False))
        for s in ("A", "B"):
            ck = top1[s].get(key)
            if ck is None or ck == tkey:
                continue
            csmi = k2s.get(ck, "")
            a, b = morgan_fp(tsmi), morgan_fp(csmi)
            tan = DataStructs.TanimotoSimilarity(a, b) if a is not None and b is not None else 0.0
            confusion[s].append({"same_formula": form.get(csmi) == tf, "tanimoto": tan})
    comp_df = pd.DataFrame(comp)
    conf = {
        s: {
            "n_wrong_top1": len(v),
            "same_formula": float(np.mean([x["same_formula"] for x in v])) if v else None,
            "tanimoto_q10_50_90": q([x["tanimoto"] for x in v]),
        }
        for s, v in confusion.items()
    }
    log("formula composition done")

    # ---- 3. フラグメント列挙 --------------------------------------------------------
    cand_smiles = sorted(set(pool["smiles"]) | set(st1.loc[q_rows["key"].unique(), "smiles"]))
    frag2 = dict(
        zip(
            cand_smiles,
            pmap(
                partial(fragment_masses, max_breaks=2, max_bonds=BIG_BONDS),
                cand_smiles,
                args.workers,
            ),
            strict=True,
        )
    )
    log(f"2-break fragments for {len(cand_smiles):,} structures")
    pair_smiles = set(st1.loc[q_rows["key"].unique(), "smiles"])
    for key in pool["molecule_id"].unique():
        ck = top1["A"].get(key)
        sub = pool[(pool["molecule_id"] == key) & (pool["candidate_key"] == ck)]
        pair_smiles |= set(sub["smiles"])
    pair_list = sorted(s for s in pair_smiles if nb.get(s, 0) <= CAP3)
    frag3 = dict(
        zip(
            pair_list,
            pmap(
                partial(fragment_masses, max_breaks=3, max_bonds=CAP3), pair_list, args.workers, 2
            ),
            strict=True,
        )
    )
    log(f"3-break fragments for {len(pair_list):,} truth/decoy structures")

    # ---- 4. ピーク帰属 (正解構造 vs A の誤答 1 位) ------------------------------------
    rows_ann = []
    ppm_exact: list[float] = []
    base_rows = []
    for key, sub in q_rows.groupby("key", sort=True):
        mid = f"v_{key}"
        tsmi = st1.loc[key, "smiles"]
        ck = top1["A"].get(mid)
        dsub = pool[(pool["molecule_id"] == mid) & (pool["candidate_key"] == ck)]
        dsmi = dsub["smiles"].iloc[0] if len(dsub) and ck != key else None
        for r, ad, pmz, mode in zip(
            sub.index,
            sub["adduct"],
            sub["precursor_mz"],
            sub["ionization_mode"],
            strict=True,
        ):
            mz, it, ce = raw[int(r)]
            pk = prepare_peaks(mz, it, float(pmz), str(ad), 1.0 if mode == "positive" else -1.0)
            if len(pk.mz) == 0:
                continue
            for who, smi in (("truth", tsmi), ("decoy", dsmi)):
                if smi is None:
                    continue
                cat = annotate(pk, frag2.get(smi, np.zeros(0)), frag3.get(smi))
                share = np.bincount(cat, weights=pk.intensity, minlength=len(CATEGORIES))
                cnt = np.bincount(cat, minlength=len(CATEGORIES)) / len(cat)
                rows_ann.append(
                    {
                        "who": who,
                        "adduct": str(ad),
                        "ce": ce,
                        "big": nb.get(smi, 0) > 34,
                        **{f"i_{c}": share[i] for i, c in enumerate(CATEGORIES)},
                        **{f"n_{c}": cnt[i] for i, c in enumerate(CATEGORIES)},
                    }
                )
                if who == "truth":
                    ppm_exact.extend(exact_match_ppm(pk, frag2.get(smi, np.zeros(0))).tolist())
                    bi = int(np.argmax(pk.intensity))
                    prec_i = pk.intensity[np.abs(pk.mz - pk.precursor_mz) <= 0.01].sum()
                    p = pk.intensity[pk.intensity > 0]
                    base_rows.append(
                        {
                            "base_cat": CATEGORIES[cat[bi]],
                            "prec_rel": float(prec_i / pk.intensity.max()),
                            "entropy": float(-(p * np.log(p)).sum()),
                            "n_peaks": len(pk.mz),
                            "ce": ce,
                        }
                    )
    ann = pd.DataFrame(rows_ann)
    base = pd.DataFrame(base_rows)
    log("peak annotation done")

    # ---- 5. MetFrag-lite 変種ごとの、断片チャネル単独の MRR ---------------------------
    pk_by_mol: dict[str, list[Peaks]] = {}
    for key, sub in q_rows.groupby("key", sort=True):
        pk_by_mol[f"v_{key}"] = [
            prepare_peaks(
                raw[int(r)][0],
                raw[int(r)][1],
                float(pmz),
                str(ad),
                1.0 if mode == "positive" else -1.0,
            )
            for r, ad, pmz, mode in zip(
                sub.index, sub["adduct"], sub["precursor_mz"], sub["ionization_mode"], strict=True
            )
        ]
    prepared = {
        mid: [[prepare_for(pk, v) for pk in pks] for v in VARIANTS]
        for mid, pks in pk_by_mol.items()
    }
    var_res: dict[str, list[float]] = {v.name: [] for v in VARIANTS}
    var_big: dict[str, list[float]] = {v.name: [] for v in VARIANTS}
    for mid, sub in pool.groupby("molecule_id"):
        tkey = mid.removeprefix("v_")
        keys = sub["candidate_key"].to_numpy()
        if tkey not in set(keys):
            for v in VARIANTS:
                var_res[v.name].append(0.0)
            continue
        ti = int(np.flatnonzero(keys == tkey)[0])
        smis = sub["smiles"].to_numpy()
        big = nb.get(st1.loc[tkey, "smiles"], 0) > 34
        for v, preps in zip(VARIANTS, prepared[mid], strict=True):
            sc = np.array(
                [
                    max(
                        (
                            explain_prepared(pp, frag2.get(s, _EMPTY), nb.get(s, 0), v)
                            for pp in preps
                        ),
                        default=0.0,
                    )
                    for s in smis
                ]
            )
            r = rr(truth_rank(sc, ti))
            var_res[v.name].append(r)
            if big:
                var_big[v.name].append(r)
    log("score variants done")

    cats = list(CATEGORIES)
    ann_mean = ann.groupby("who")[[f"i_{c}" for c in cats] + [f"n_{c}" for c in cats]].mean()
    ann_big = ann[ann["who"] == "truth"].groupby("big")[[f"i_{c}" for c in cats]].mean()
    ann_adduct = ann[ann["who"] == "truth"].groupby("adduct")[[f"i_{c}" for c in cats]].mean()
    ann_adduct["n_spectra"] = ann[ann["who"] == "truth"].groupby("adduct").size()
    ce_bin = pd.cut(base["ce"], [-1, 15, 30, 45, 1000], labels=["≤15", "15-30", "30-45", ">45"])
    base_tab = (
        base.assign(ce_bin=ce_bin)
        .groupby("ce_bin", observed=True)
        .agg(
            n=("prec_rel", "size"),
            prec_rel_median=("prec_rel", "median"),
            base_is_precursor=("base_cat", lambda s: float((s == "precursor").mean())),
            entropy_median=("entropy", "median"),
        )
    )
    res = {
        "split": args.split,
        "molecules": int(q_rows["key"].nunique()),
        "spectra": int(len(q_rows)),
        "precursor": {
            "spectrum_ppm_q10_50_90": q(ppm),
            "molecule_abs_ppm_q50_90_99": q(mol_ppm.abs(), (0.5, 0.9, 0.99)),
            "coverage_by_window_ppm": cover,
            "isotope_misassigned_spectra": iso_mis,
            "median_window_size_by_ppm": win,
            "by_adduct": by_adduct.round(3).reset_index().to_dict("records"),
        },
        "window": {
            "n_candidates_median": float(comp_df["n"].median()),
            "n_formulas_median": float(comp_df["n_formulas"].median()),
            "n_same_formula_as_truth_median": float(comp_df["n_same_formula"].median()),
            "share_truth_formula_unique": float((comp_df["n_same_formula"] <= 1).mean()),
        },
        "confusion_top1": conf,
        "truth_bonds_over_34": float(
            np.mean([nb.get(st1.loc[k, "smiles"], 0) > 34 for k in q_rows["key"].unique()])
        ),
        "annotation_mean": ann_mean.round(4).to_dict("index"),
        "annotation_truth_by_size": {
            str(k): v for k, v in ann_big.round(4).to_dict("index").items()
        },
        "annotation_truth_by_adduct": ann_adduct.round(4).to_dict("index"),
        "exact_match_ppm_q50_90_95_99": q(ppm_exact, (0.5, 0.9, 0.95, 0.99)),
        "base_peak_category_share": base["base_cat"]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
        "base_peak_by_ce": base_tab.round(4)
        .reset_index()
        .astype({"ce_bin": str})
        .to_dict("records"),
        "frag_channel_mrr": {k: float(np.mean(v)) for k, v in var_res.items()},
        "frag_channel_mrr_big_molecules": {
            k: (float(np.mean(v)) if v else None) for k, v in var_big.items()
        },
        "n_big_molecules": len(next(iter(var_big.values()))),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"ms_diag_{args.split}.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float)
    )
    log(f"→ {args.out / f'ms_diag_{args.split}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
