from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from casmi26.baseline.pubchem import PRIOR_WEIGHT, PubChemIndex

ROOT = Path(__file__).resolve().parents[2]


def _build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """CID 1–6 の小さな Extras から索引を作る (CID 4 は欠番)。"""
    src, out = tmp_path / "src", tmp_path / "idx"
    src.mkdir()
    files = {
        "CID-Mass.gz": [
            "1\tC2H6O\t150.0\t150.0",
            "2\tC3H8O\t150.001\t150.001",
            "3\tC4H10N+\t150.0005\t150.0005",  # 荷電種 → 除外
            "5\tC5H12O\t300.0\t300.0",
            "6\tC2H6O\t150.0002\t150.0002",
        ],
        "CID-SMILES.gz": ["1\tCCO", "2\tCCCO", "3\tCCC[NH3+]", "5\tCCCCCO", "6\tCCO.Cl"],
        "CID-SID.gz": ["1\t10", "1\t11", "2\t12", "5\t13"],
        "CID-PMID.gz": ["1\t100\t1", "2\t101\t1", "2\t102\t1", "2\t103\t1"],
    }
    for name, lines in files.items():
        with gzip.open(src / name, "wt") as f:
            f.write("\n".join(lines) + "\n")
    spec = importlib.util.spec_from_file_location(
        "build_idx", ROOT / "scripts/casmi26_build_pubchem_index.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(sys, "argv", ["x", "--src", str(src), "--out", str(out)])
    assert mod.main() == 0
    return out


def test_index_keeps_single_neutral_components_sorted_by_mass(tmp_path, monkeypatch):
    idx = PubChemIndex(_build(tmp_path, monkeypatch))
    # CID 3 (荷電種) と CID 6 (複数成分) は除外、CID 4 は欠番
    assert list(idx.cid) == [1, 2, 5]
    assert list(idx.mass) == pytest.approx([150.0, 150.001, 300.0])
    assert [idx.smiles_of(c) for c in (1, 2, 5)] == ["CCO", "CCCO", "CCCCCO"]
    assert idx.smiles_of(4) == ""


def test_top_n_orders_by_literature_prior(tmp_path, monkeypatch):
    idx = PubChemIndex(_build(tmp_path, monkeypatch))
    hits = idx.top_n(150.0005, 0.002, 5)
    assert [h.cid for h in hits] == [2, 1]  # CID 2: SID 1 + PMID 3 > CID 1: SID 2 + PMID 1
    assert hits[0].prior == pytest.approx(PRIOR_WEIGHT * (np.log1p(1) + np.log1p(3)), rel=1e-5)
    assert [h.cid for h in idx.top_n(150.0005, 0.002, 1)] == [2]
    assert idx.top_n(500.0, 0.002, 5) == []
    assert idx.window_size(150.0005, 0.002) == 2


def test_prior_promotes_documented_candidate():
    from casmi26.baseline.pipeline import (
        BaselineConfig,
        CandidatePool,
        Query,
        SpectralIndex,
        run_baseline,
    )
    from casmi26.baseline.spectra import CleanLibrary
    from casmi26.metrics import canonical_inchikey14

    peaks = np.array([60.0, 80.0, 100.0], np.float32)
    inten = np.array([0.2, 0.3, 0.5], np.float32)
    lib = CleanLibrary(off=np.array([0, 3]), mz=peaks, it=inten, nm=np.array([999.0]))
    index = SpectralIndex(lib, np.array([True]), np.array(["X"], object))
    smiles = np.array(["CCO", "COC"], object)  # 同じ質量の異性体
    pool = CandidatePool(
        mass=np.array([46.0419, 46.0419]),
        smiles=smiles,
        known_key=np.array([None, None], object),
        source=np.array(["pubchem", "pubchem"], object),
        prior=np.array([0.5, 3.0], np.float32),
    )
    q = Query("q", "", [(peaks, inten)], np.array([46.0419]), 1.0)
    out = run_baseline([q], index, pool, {}, BaselineConfig(workers=1), log=lambda _: None)
    top = out.stages["ranker_prior"].sort_values("rank").iloc[0]["candidate_key"]
    assert top == canonical_inchikey14("COC")
    assert out.features is not None and set(out.features["prior"]) == {0.5, 3.0}
