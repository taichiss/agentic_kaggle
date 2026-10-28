"""fold-safe baseline を検証 split で回し、誤差分解まで出す (Week 1 Phase D)。

split:
    fold0 — identity-disjoint fold 0 から timsTOF スペクトルを持つ構造を N 個抽出
    np    — enveda-np-examples の全構造 (hidden test に最も近い分布)

どちらもホールドアウト構造の identity group に属する全スペクトルを
ライブラリ / analog 代表から除く (cv_manifest.json 準拠)。候補 pool に構造を残すかは
--pool-heldout で選ぶ (keep = 既知構造で reference なし / drop = DB 未収録)。
--reference other はホールドアウト構造のスペクトルのうち query と同じソース
(np: enveda-np-examples / fold0: timsTOF) のものだけ除き、他装置・他ライブラリの
reference を残す (Class 1 の模擬。天然物パネルの "Other reference" 状況に相当)。

使い方:
    uv run python scripts/casmi26_baseline_cv.py --split np --pool-heldout keep \
        --out enveda-casmi26/experiments/EXP-0001/np_keep
    uv run python scripts/casmi26_baseline_cv.py --split fold0 --n 400 --pool-heldout keep \
        --out enveda-casmi26/experiments/EXP-0001/fold0_keep
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.analysis import StageSpec, assign_classes, decompose, to_markdown
from casmi26.baseline.chem import ADDUCTS, corrected_mz, neutral_mass
from casmi26.baseline.pipeline import (
    BaselineConfig,
    CandidatePool,
    Query,
    SpectralIndex,
    run_baseline,
)
from casmi26.baseline.spectra import CleanConfig, clean_library

TRAIN = Path("data/input/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
FOLDS = Path("data/input/external/casmi26/casmi26-identity-disjoint-validation-folds")
COCO = Path("data/input/external/casmi26/coconut-casmi26-candidates")
PCL = Path("data/input/external/casmi26/casmi26-pubchemlite-cc0-candidate-fingerprints")
CACHE = Path("data/working/casmi26/library_clean.npz")
SEED = 20260930
MAX_SPECTRA = 9  # test の 1 分子あたり最大スペクトル数

T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def read_rows(rows: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """train.parquet の指定行の生スペクトル。"""
    want = np.sort(rows)
    pf = pq.ParquetFile(TRAIN)
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    base = 0
    for g in range(pf.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        sel = want[(want >= base) & (want < base + n)]
        if len(sel):
            t = pf.read_row_group(g, columns=["ms2_mzs", "ms2_normalized_intensities"])
            t = t.take(sel - base)
            mzs = t.column("ms2_mzs").to_pylist()
            its = t.column("ms2_normalized_intensities").to_pylist()
            for r, m, i in zip(sel, mzs, its, strict=True):
                out[int(r)] = (np.asarray(m, np.float32), np.asarray(i, np.float32))
        base += n
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["fold0", "np"], required=True)
    ap.add_argument("--n", type=int, default=400, help="fold0 の抽出分子数")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument(
        "--pool-heldout",
        choices=["keep", "drop"],
        default="keep",
        help="keep: ホールドアウト構造を候補 pool に残す (既知構造・reference なし) / "
        "drop: pool からも除く (DB 未収録)",
    )
    ap.add_argument(
        "--precursor-window",
        type=float,
        default=0.0,
        help="> 0 なら m/z > precursor - W のピークを類似度計算から除く (0 = V17 と同じ)",
    )
    ap.add_argument(
        "--pool-window",
        choices=["v17", "corrected"],
        default="v17",
        help="v17: ±8.5 ppm / corrected: m/z のずれ (正 +0.70, 負 +0.25 mDa) を補正して"
        " ±max(5 ppm, 2 mDa) (EXP-0008)",
    )
    ap.add_argument(
        "--extra-pool",
        choices=["none", "pubchemlite"],
        default="none",
        help="候補 pool に追加する構造 DB",
    )
    ap.add_argument(
        "--reference",
        choices=["none", "other"],
        default="none",
        help="none: ホールドアウト構造の全スペクトルを除く / other: query と同じソースだけ除く",
    )
    args = ap.parse_args()
    win = (
        dict(pool_ppm=5.0, pool_min_da=2e-3, mz_offset_pos=0.70e-3, mz_offset_neg=0.25e-3)
        if args.pool_window == "corrected"
        else {}
    )
    cfg = BaselineConfig(
        workers=args.workers,
        clean=CleanConfig(precursor_window=args.precursor_window),
        **win,  # type: ignore[arg-type]
    )
    cache = CACHE
    if args.precursor_window > 0:
        cache = CACHE.with_name(f"library_clean_pw{args.precursor_window:g}.npz")

    meta = pq.read_table(
        TRAIN,
        columns=["ingest_lib", "instrument_type", "adduct", "ionization_mode", "precursor_mz"],
    ).to_pandas()
    sf = pd.read_parquet(FOLDS / "spectra_folds.parquet")
    st = pd.read_parquet(FOLDS / "structures.parquet")
    assert len(meta) == len(sf) and (sf["row"].to_numpy() == np.arange(len(sf))).all()
    key_of_ik = dict(zip(st["inchikey14"], st["scorer_key"], strict=True))
    row_keys = sf["inchikey14"].map(key_of_ik).to_numpy(dtype=object)
    meta["key"] = row_keys
    meta["group"] = sf["identity_group"].to_numpy()
    log(f"loaded metadata: {len(meta):,} spectra / {len(st):,} structures")

    usable = meta["adduct"].isin(list(ADDUCTS))
    if args.split == "np":
        q_mask = (meta["ingest_lib"] == "enveda-np-examples") & usable
        held_groups = set(meta.loc[meta["ingest_lib"] == "enveda-np-examples", "group"])
        q_rows = meta[q_mask]
    else:
        held_groups = set(st.loc[st["fold_identity"] == 0, "identity_group"])
        cand = meta[(sf["fold_identity"] == 0) & (meta["instrument_type"] == "timsTOF") & usable]
        keys = np.sort(cand["key"].unique())
        rng = np.random.default_rng(SEED)
        pick = set(rng.choice(keys, size=min(args.n, len(keys)), replace=False))
        q_rows = cand[cand["key"].isin(pick)]
    rng = np.random.default_rng(SEED)
    q_rows = (
        q_rows.assign(_r=rng.random(len(q_rows)))
        .sort_values(["key", "_r"])
        .groupby("key")
        .head(MAX_SPECTRA)
    )
    log(f"split={args.split}: {q_rows['key'].nunique()} molecules / {len(q_rows)} spectra")

    raw = read_rows(q_rows.index.to_numpy())
    queries = []
    for key, sub in q_rows.groupby("key", sort=True):
        pmz = corrected_mz(
            sub["precursor_mz"].to_numpy(np.float64),
            sub["ionization_mode"].to_numpy(),
            cfg.mz_offset_pos,
            cfg.mz_offset_neg,
        )
        nm = neutral_mass(pmz, sub["adduct"].to_numpy())
        mode = float(np.mean([1.0 if m == "positive" else -1.0 for m in sub["ionization_mode"]]))
        queries.append(
            Query(
                molecule_id=f"v_{key}",
                truth_key=str(key),
                spectra=[raw[int(r)] for r in sub.index],
                neutral_masses=nm,
                ion_mode=mode,
                precursor_mzs=sub["precursor_mz"].to_numpy(np.float64),
            )
        )

    lib = clean_library(TRAIN, cache, cfg.clean)
    held = meta["group"].isin(held_groups)
    if args.reference == "other":
        same_source = (
            meta["ingest_lib"] == "enveda-np-examples"
            if args.split == "np"
            else meta["instrument_type"] == "timsTOF"
        )
        held &= same_source
    allowed = ~held.to_numpy()
    index = SpectralIndex(lib, allowed, row_keys)
    log(f"library: {len(index.rows):,} spectra / analog reps: {len(index.rep_rows):,}")

    cm = pickle.load(open(COCO / "coco_meta.pkl", "rb"))
    co_mass = np.load(COCO / "coco_mass.npy")
    # coconut-casmi26-candidates は train 構造を除いた COCONUT なので、ホールドアウト構造を
    # pool に残すかどうかは train 側の構造表で決める (TECHNICAL_NOTES Validation Protocol 2)
    st_pool = st if args.pool_heldout == "keep" else st[~st["identity_group"].isin(held_groups)]
    st_ok = st_pool.drop_duplicates("scorer_key")
    ex_mass, ex_smiles, ex_keys = np.zeros(0), np.zeros(0, object), np.zeros(0, object)
    if args.extra_pool == "pubchemlite":
        with open(PCL / "pubchemlite_meta.pkl", "rb") as f:
            pm = pickle.load(f)
        ex_mass = np.load(PCL / "pubchemlite_mass.npy")
        ex_smiles = np.asarray(pm["smiles"], object)
        ex_keys = np.asarray(pm["keys"], object)
    pool = CandidatePool(
        mass=np.concatenate([co_mass, ex_mass, st_ok["mono_mass"].to_numpy()]),
        smiles=np.concatenate(
            [np.asarray(cm["smiles"], object), ex_smiles, st_ok["smiles"].to_numpy(object)]
        ),
        known_key=np.concatenate(
            [
                np.full(len(co_mass) + len(ex_mass), None, object),
                st_ok["scorer_key"].to_numpy(object),
            ]
        ),
        source=np.array(
            ["coconut"] * len(co_mass) + [args.extra_pool] * len(ex_mass) + ["train"] * len(st_ok),
            object,
        ),
    )
    log(
        f"pool: {len(co_mass):,} COCONUT + {len(ex_mass):,} {args.extra_pool}"
        f" + {len(st_ok):,} train"
    )
    key_to_smiles = dict(zip(st["scorer_key"], st["smiles"], strict=False))

    res = run_baseline(queries, index, pool, key_to_smiles, cfg, log=log)
    log("baseline done")

    truth = {q.molecule_id: q.truth_key for q in queries}
    classes = assign_classes(
        truth,
        reference_keys=index.keys,
        database_keys=res.database_keys | set(cm["keys"]) | set(ex_keys) | set(st_ok["scorer_key"]),
    )
    stages = [StageSpec("pool", None), StageSpec("ranker", 1)]
    dec = decompose(truth, res.stages, stages=stages, classes=classes)
    channel_mrr = {
        name: decompose(truth, res.stages, stages=[StageSpec(name, 1)]).mrr
        for name in ("channel_lib", "channel_analog", "channel_frag")
    }

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    for name, df in res.stages.items():
        df.to_parquet(out / f"stage_{name}.parquet", index=False)
    pd.DataFrame({"molecule_id": list(truth), "truth_key": list(truth.values())}).to_parquet(
        out / "truth.parquet", index=False
    )
    res.diag.to_parquet(out / "diag.parquet", index=False)
    if res.features is not None:
        res.features.to_parquet(out / "features.parquet", index=False)
    dec.per_molecule.to_parquet(out / "per_molecule.parquet", index=False)
    md = to_markdown(
        dec,
        title=(
            f"CASMI26 誤差分解 — fold-safe baseline / {args.split}"
            f" / pool={args.pool_heldout} / ref={args.reference}"
        ),
    )
    md += "\n## 5. 単一チャネルだけで並べたときの MRR@25\n\n"
    md += "\n".join(f"- {k}: {v:.4f}" for k, v in channel_mrr.items()) + "\n"
    n_cand = res.diag["n_candidates"]
    md += (
        f"\n## 6. 候補数\n\n- median {n_cand.median():.0f} / mean {n_cand.mean():.1f}"
        f" / 0 件 {int((n_cand == 0).sum())}\n"
    )
    (out / "report.md").write_text(md, encoding="utf-8")
    summary = {
        "split": args.split,
        "pool_heldout": args.pool_heldout,
        "reference": args.reference,
        "precursor_window": args.precursor_window,
        "extra_pool": args.extra_pool,
        "pool_window": args.pool_window,
        "molecules": dec.count,
        "spectra": int(len(q_rows)),
        "mrr": dec.mrr,
        "top1": float((dec.per_molecule["final_rank"] == 1).mean()),
        "pool_coverage": dec.ceilings.get("pool"),
        "class_counts": dec.per_molecule["class"].value_counts().to_dict(),
        "loss_share_by_stage": dec.stage_table.groupby("bucket")["loss_share"].sum().to_dict(),
        "channel_mrr": channel_mrr,
        "config": asdict(cfg),
        "seed": SEED,
        "elapsed_sec": time.time() - T0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log(f"MRR@25={dec.mrr:.4f} coverage={dec.ceilings.get('pool'):.3f} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
