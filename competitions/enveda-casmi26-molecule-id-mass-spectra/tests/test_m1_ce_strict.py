"""Synthetic metadata regression for the repaired same-molecule mixup policy."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src/mist_formula"))
from mixup_pairs import merge_peaks  # noqa: E402
from train_mix_same_ce_strict import Cache  # noqa: E402


class Pick:
    def __init__(self, index: int):
        self.index = index

    def integers(self, high: int) -> int:
        assert self.index < high
        return self.index


def toy_cache(tmp_path: Path) -> Cache:
    root = tmp_path / "cache"
    source = tmp_path / "source"
    root.mkdir()
    source.mkdir()
    pd.DataFrame({
        "split": ["train"], "formula": ["C2H6O"], "key": ["toy"],
    }).to_parquet(root / "molecules.parquet")
    pd.DataFrame({
        "split": ["train"] * 6, "length": [1] * 6,
        "adduct_id": [1] * 6, "mol_idx": [0] * 6,
    }).to_parquet(root / "spectra.parquet")
    peaks = np.zeros((6, 96, 4), dtype="<f2")
    peaks[2, 0, 1] = np.sqrt(0.2)
    peaks[3, 0, 1] = np.sqrt(0.8)
    peaks.tofile(root / "peaks.bin")
    mz = np.zeros((6, 96), dtype="<f4")
    mz[2, 0], mz[3, 0] = 100.0, 100.001
    precise = tmp_path / "precise-mzs.bin"
    mz.tofile(precise)
    meta = np.zeros((6, 4), dtype="<f4")
    meta[:, 0] = 0.2  # precursor m/z 200
    meta[:, 3] = 1
    meta[2:, 2] = 1  # kind 1 or 2 has a numeric legacy CE flag
    meta[2, 1], meta[3, 1], meta[5, 1] = 0.95, 0.01, 0.95
    meta.tofile(root / "metadata.bin")
    np.save(root / "fingerprints.npy", np.zeros((1, 2048), np.uint8))
    np.save(source / "ingest_id.npy", np.zeros(6, np.int32))
    np.save(source / "instrument_id.npy", np.zeros(6, np.int32))
    np.save(source / "ce_kind.npy", np.array([0, 0, 1, 1, 2, 1], np.int8))
    np.save(source / "ce_ev.npy", np.array([np.nan, np.nan, 20, 25, 22, 40], np.float32))
    (source / "categories.json").write_text(json.dumps({"ingest_lib": ["toy"]}))
    return Cache(root, precise, source)


def test_ce_kind_excludes_ambiguous_and_never_crosses_kinds(tmp_path):
    data = toy_cache(tmp_path)
    keys = data.same_mol_adduct_source_instrument_kind_polarity
    assert sorted(sorted(rows) for rows in keys.values()) == [[0, 1], [2, 3, 5]]
    assert data.partner(4, Pick(0)) == -1  # kind 2 cannot be an anchor
    assert data.partner(0, Pick(1)) == 1   # missing CE pairs with missing CE
    assert data.partner(2, Pick(1)) == 3   # sidecar 20 and 25 eV pass
    assert data.partner(2, Pick(2)) == -1  # sidecar 20 and 40 eV fail
    # The legacy meta CE values above disagree with sidecar 20/25 eV; the
    # accepted pair proves that the explicit sidecar governs kind-1 matching.


def test_nearby_peak_keeps_shared_strongest_measured_mz(tmp_path):
    data = toy_cache(tmp_path)
    feature, selected_mz, n = data.mixed_spectrum(2, 3, 0.5)
    merged_mz, _ = merge_peaks(
        data.mz[2, :1], np.asarray(data.x[2, :1, 1], np.float32) ** 2,
        data.mz[3, :1], np.asarray(data.x[3, :1, 1], np.float32) ** 2,
        0.5,
    )
    assert n == 1
    assert selected_mz[0] == merged_mz[0] == data.mz[3, 0]
    assert feature.shape == (96, 4)

    class RecordingCatalog:
        def __init__(self):
            self.received = []

        def assign(self, peak_mz, *_args, **_kwargs):
            self.received.append(np.asarray(peak_mz).copy())
            return tuple(np.zeros((96, 1), np.float32) for _ in range(4))

    catalog = RecordingCatalog()
    batch = data.batch(
        np.array([2]), catalog, "cpu", partners=np.array([3]), alphas=np.array([0.5])
    )
    np.testing.assert_array_equal(catalog.received[0], merged_mz)
    assert batch[1].item() == 1
