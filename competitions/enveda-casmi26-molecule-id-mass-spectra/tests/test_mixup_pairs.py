"""Chemical and split invariants for training-only spectrum mixing."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/mist_formula"))
from mixup_pairs import PartnerPool, merge_peaks  # noqa: E402


def test_merge_keeps_measured_mz_and_sums_linear_intensity():
    mz, intensity = merge_peaks(
        np.array([100.0000, 150.0]), np.array([1.0, 0.2]),
        np.array([100.0010, 175.0]), np.array([0.5, 0.4]),
        0.9,
    )
    assert len(mz) == 3
    assert mz[0] == 100.0
    assert np.isclose(intensity[0], 0.9 * 1.0 + 0.1 * 0.5)


def test_partner_pool_never_crosses_split_or_measurement_group(tmp_path):
    np.save(tmp_path / "ingest_id.npy", np.array([0, 0, 1, 0, 0, 0, 0], np.int32))
    np.save(tmp_path / "instrument_id.npy", np.array([0, 0, 0, 1, 0, 0, 0], np.int32))
    np.save(tmp_path / "ce_ev.npy", np.array([20, 25, 20, 20, 40, 21, 20], np.float32))
    np.save(tmp_path / "ce_kind.npy", np.ones(7, np.int8))
    (tmp_path / "categories.json").write_text(json.dumps({
        "ingest_lib": ["a", "b"], "instrument_type": ["tof", "orbitrap"]
    }))
    data = SimpleNamespace(
        spec=pd.DataFrame({"split": ["train"] * 5 + ["validation", "train"]}),
        mol=pd.DataFrame({"formula": ["C5H6O2"] * 5}),
        mol_idx=np.array([0, 1, 2, 3, 1, 2, 4]),
        adduct=np.ones(7, np.int64),
        meta=np.array([[0.123, 0, 0, 1]] * 6 + [[0.123, 0, 0, -1]], np.float32),
    )
    pool = PartnerPool(data, tmp_path, mode="cross")
    rng = np.random.default_rng(1)
    assert pool.sample_partner(0, rng) == 1
    assert pool.sample_partner(2, rng) is None  # different library
    assert pool.sample_partner(3, rng) is None  # different instrument
    assert pool.sample_partner(4, rng) is None  # CE beyond 10 eV
    assert pool.sample_partner(5, rng) is None  # validation is absent from index
    assert pool.sample_partner(6, rng) is None  # opposite polarity
