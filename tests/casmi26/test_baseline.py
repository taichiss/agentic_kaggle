from __future__ import annotations

import numpy as np
import pytest

from casmi26.baseline.chem import PROTON, neutral_mass
from casmi26.baseline.fragments import explain_score, fragment_masses
from casmi26.baseline.pipeline import (
    BaselineConfig,
    CandidatePool,
    Query,
    SpectralIndex,
    run_baseline,
)
from casmi26.baseline.spectra import CleanConfig, CleanLibrary, clean, clean_flat, entropy_sim
from casmi26.metrics import canonical_inchikey14


def test_neutral_mass_inverts_common_adducts():
    m = 300.0
    mz = np.array([m + PROTON, m - PROTON, (2 * m + PROTON), 1.0])
    out = neutral_mass(mz, np.array(["[M+H]+", "[M-H]-", "[2M+H]+", "[M+Unknown]"], object))
    assert out[:3] == pytest.approx([m, m, m])
    assert np.isnan(out[3])


def test_entropy_sim_is_one_for_identical_and_zero_for_disjoint():
    mz = np.array([100.0, 150.0, 200.0], np.float32)
    it = np.array([0.2, 0.3, 0.5], np.float32)
    assert entropy_sim(mz, it, mz, it, 0.01) == pytest.approx(1.0, abs=1e-6)
    other = mz + 7.0
    assert entropy_sim(mz, it, other, it, 0.01) == pytest.approx(0.0, abs=1e-6)


def test_clean_flat_matches_per_spectrum_clean():
    rng = np.random.default_rng(0)
    specs = [
        (np.sort(rng.uniform(50, 500, n)).astype(np.float32), rng.random(n).astype(np.float32))
        for n in (5, 0, 300, 40)
    ]
    off = np.zeros(len(specs) + 1, np.int64)
    off[1:] = np.cumsum([len(s[0]) for s in specs])
    mz = np.concatenate([s[0] for s in specs])
    it = np.concatenate([s[1] for s in specs])
    cfg = CleanConfig()
    o2, m2, i2 = clean_flat(off, mz, it, cfg)
    for r, (a, b) in enumerate(specs):
        em, ei = clean(a, b, cfg.floor, cfg.topk, cfg.power, cfg.ent_weight)
        np.testing.assert_allclose(m2[o2[r] : o2[r + 1]], em)
        np.testing.assert_allclose(i2[o2[r] : o2[r + 1]], ei)
    assert o2[3] - o2[2] == cfg.topk


def test_fragment_explains_protonated_precursor():
    frags = fragment_masses("CCO")
    mz = np.array([float(frags.max()) + PROTON])
    assert explain_score(frags, mz, np.array([1.0]), mode=1.0) == pytest.approx(1.0)


def _toy_library():
    """3 構造 × 1 スペクトル。row 0 が正解構造と同じキー (ホールドアウト対象)。"""
    peaks = np.array([60.0, 80.0, 100.0], np.float32)
    inten = np.array([0.2, 0.3, 0.5], np.float32)
    off = np.array([0, 3, 6, 9], np.int64)
    mz = np.concatenate([peaks, peaks, peaks + 5.0])
    it = np.concatenate([inten] * 3)
    lib = CleanLibrary(off=off, mz=mz, it=it, nm=np.array([46.0419, 46.0419, 46.0419]))
    keys = np.array(["T", "A", "B"], object)
    return lib, keys, peaks, inten


def test_spectral_index_excludes_heldout_rows():
    lib, keys, peaks, inten = _toy_library()
    index = SpectralIndex(lib, np.array([False, True, True]), keys)
    assert "T" not in index.keys
    hits = index.lib_hits([(peaks, inten)], 46.0419, BaselineConfig())
    assert "T" not in hits
    assert hits["A"] == pytest.approx(1.0, abs=1e-6)


def test_run_baseline_ranks_library_match_first():
    lib, keys, peaks, inten = _toy_library()
    index = SpectralIndex(lib, np.array([False, True, True]), keys)
    smiles = np.array(["CCO", "COC"], object)
    key_a = canonical_inchikey14("COC")
    # ライブラリ側キーを pool の構造キーに合わせる
    index.row_keys[1] = key_a
    pool = CandidatePool(
        mass=np.array([46.0419, 46.0419]),
        smiles=smiles,
        known_key=np.array([canonical_inchikey14("CCO"), key_a], object),
        source=np.array(["train", "train"], object),
    )
    q = Query(
        molecule_id="q",
        truth_key=key_a,
        spectra=[(peaks, inten)],
        neutral_masses=np.array([46.0419]),
        ion_mode=1.0,
    )
    out = run_baseline([q], index, pool, {}, BaselineConfig(workers=1), log=lambda _: None)
    ranker = out.stages["ranker"].sort_values("rank")
    assert ranker.iloc[0]["candidate_key"] == key_a
    assert set(out.stages["pool"]["candidate_key"]) == {key_a, canonical_inchikey14("CCO")}


def test_to_submission_pads_and_keeps_sample_order():
    import pandas as pd

    from casmi26.baseline.infer import PAD_SMILES, TOPN, to_submission, validate_submission

    sample = pd.DataFrame({"molecule_id": ["m2", "m1", "m3"], "smiles": ["CCO"] * 3})
    stage = pd.DataFrame(
        [("m1", f"K{i}", i, f"C{'C' * i}O") for i in range(1, 31)] + [("m2", "K", 1, "CN")],
        columns=["molecule_id", "candidate_key", "rank", "smiles"],
    )
    sub = to_submission(stage, sample)
    validate_submission(sub, sample)
    assert list(sub["molecule_id"]) == ["m2", "m1", "m3"]
    m1 = sub.set_index("molecule_id").loc["m1", "smiles"].split(";")
    assert m1[0] == "CCO" and len(m1) == TOPN and m1[-1] == f"C{'C' * TOPN}O"
    m2 = sub.set_index("molecule_id").loc["m2", "smiles"].split(";")
    assert m2[0] == "CN" and m2[1:] == [PAD_SMILES] * (TOPN - 1)
    assert sub.set_index("molecule_id").loc["m3", "smiles"] == ";".join([PAD_SMILES] * TOPN)
