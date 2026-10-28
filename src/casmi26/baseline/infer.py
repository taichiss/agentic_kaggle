"""test 推論: train 全体をライブラリ・pool にして submission.csv を作る。

CV (scripts/casmi26_baseline_cv.py) との違いはホールドアウトがないことだけで、
ライブラリ / analog 代表 / 候補 pool / 特徴量 / 融合はすべて同じコードを通る。
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi26.baseline.chem import corrected_mz, neutral_mass
from casmi26.baseline.pipeline import (
    BaselineConfig,
    BaselineOutput,
    CandidatePool,
    Query,
    SpectralIndex,
    run_baseline,
)
from casmi26.baseline.spectra import clean_library

TOPN = 25
PAD_SMILES = "CCO"

# 提出モード → 使う段出力
MODES = {"fusion": "ranker", "analog": "channel_analog"}


@dataclass(frozen=True)
class InputPaths:
    train: Path
    test: Path
    sample_submission: Path
    coconut_dir: Path  # coco_meta.pkl / coco_mass.npy
    structures: Path  # casmi26-identity-disjoint-validation-folds/structures.parquet


def build_queries(test: pd.DataFrame, cfg: BaselineConfig) -> list[Query]:
    queries = []
    for mid, sub in test.groupby("molecule_id", sort=True):
        pmz = corrected_mz(
            sub["precursor_mz"].to_numpy(np.float64),
            sub["ionization_mode"].to_numpy(),
            cfg.mz_offset_pos,
            cfg.mz_offset_neg,
        )
        nm = neutral_mass(pmz, sub["adduct"].to_numpy())
        mode = float(np.mean([1.0 if m == "positive" else -1.0 for m in sub["ionization_mode"]]))
        spectra = [
            (np.asarray(m, np.float32), np.asarray(i, np.float32))
            for m, i in zip(sub["ms2_mzs"], sub["ms2_normalized_intensities"], strict=True)
        ]
        queries.append(
            Query(
                molecule_id=str(mid),
                truth_key="",
                spectra=spectra,
                neutral_masses=nm,
                ion_mode=mode,
                precursor_mzs=sub["precursor_mz"].to_numpy(np.float64),
            )
        )
    return queries


def run_test(
    paths: InputPaths,
    cfg: BaselineConfig,
    *,
    cache_path: Path | None = None,
    log: Callable[[str], None] = print,
) -> BaselineOutput:
    st = pd.read_parquet(
        paths.structures, columns=["inchikey14", "scorer_key", "smiles", "mono_mass"]
    )
    key_of_ik = dict(zip(st["inchikey14"], st["scorer_key"], strict=True))
    train_ik = pq.read_table(paths.train, columns=["inchikey14"]).column("inchikey14").to_pylist()
    row_keys = np.asarray([key_of_ik.get(k) for k in train_ik], dtype=object)
    log(f"train: {len(row_keys):,} spectra / {len(st):,} structures")

    lib = clean_library(paths.train, cache_path, cfg.clean)
    allowed = np.array([k is not None for k in row_keys])
    index = SpectralIndex(lib, allowed, row_keys)
    log(f"library: {len(index.rows):,} spectra / analog reps: {len(index.rep_rows):,}")

    with open(paths.coconut_dir / "coco_meta.pkl", "rb") as f:
        cm = pickle.load(f)
    co_mass = np.load(paths.coconut_dir / "coco_mass.npy")
    st_ok = st.drop_duplicates("scorer_key")
    pool = CandidatePool(
        mass=np.concatenate([co_mass, st_ok["mono_mass"].to_numpy()]),
        smiles=np.concatenate([np.asarray(cm["smiles"], object), st_ok["smiles"].to_numpy(object)]),
        known_key=np.concatenate(
            [np.full(len(co_mass), None, object), st_ok["scorer_key"].to_numpy(object)]
        ),
        source=np.array(["coconut"] * len(co_mass) + ["train"] * len(st_ok), object),
    )
    log(f"pool: {len(co_mass):,} COCONUT + {len(st_ok):,} train structures")

    test = pd.read_parquet(paths.test)
    queries = build_queries(test, cfg)
    log(f"test: {len(queries)} molecules / {len(test)} spectra")
    key_to_smiles = dict(zip(st["scorer_key"], st["smiles"], strict=False))
    return run_baseline(queries, index, pool, key_to_smiles, cfg, log=log)


def to_submission(stage: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """段出力 (long format) → submission。候補が 25 本に満たない分子は PAD_SMILES で埋める。"""
    top = stage[stage["rank"] <= TOPN].sort_values(["molecule_id", "rank"])
    lists = top.groupby("molecule_id")["smiles"].apply(list).to_dict()
    rows = []
    for mid in sample["molecule_id"]:
        smis = list(lists.get(mid, []))[:TOPN]
        smis += [PAD_SMILES] * (TOPN - len(smis))
        rows.append((mid, ";".join(smis)))
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def validate_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:
    assert list(sub.columns) == ["molecule_id", "smiles"]
    assert len(sub) == len(sample)
    assert (sub["molecule_id"].to_numpy() == sample["molecule_id"].to_numpy()).all()
    assert sub["smiles"].notna().all()
    assert (sub["smiles"].str.split(";").str.len() == TOPN).all()
