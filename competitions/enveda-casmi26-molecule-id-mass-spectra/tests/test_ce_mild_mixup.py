"""Generator-side CE views preserve the frozen parent rank-feature path."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src/public_notebooks/ce_mild_mixup.py"
)
spec = importlib.util.spec_from_file_location("ce_mild_mixup", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def measured(ce, *, adduct="[M+H]+", mode=1, prec=200.0, ce_n=1, instrument="timsTOF"):
    return {
        "mz": np.array([80.0, 100.0, 150.0], dtype=np.float32),
        "it": np.array([0.5, 1.0, 0.25], dtype=np.float32),
        "mode": mode,
        "adduct": adduct,
        "prec": prec,
        "ce": ce,
        "ce_n": ce_n,
        "instrument": instrument,
    }


def test_only_matching_single_ce_measured_views_are_mixed():
    spectra = [
        measured(20), measured(40), measured(60),
        measured(40, ce_n=3),  # An already merged 20/40/60 acquisition.
        measured(20, adduct="[M+Na]+"),
        measured(40, adduct="[M+Na]+", prec=201),  # 5,000 ppm away.
        measured(40, instrument="Orbitrap"),
    ]
    views, count = module.augmented_views(spectra, include_mixup=True)
    assert count == 3
    assert views[: len(spectra)] == spectra
    synthetic = views[len(spectra) :]
    assert all(row["adduct"] == "[M+H]+" for row in synthetic)
    assert all(row["instrument"] == "timsTOF" for row in synthetic)
    assert all(row["ce_n"] == 2 and row["ce_mixup"] for row in synthetic)
    assert all(20 <= row["ce"] <= 60 for row in synthetic)
    assert all(np.isclose(row["it"].max(), 1) for row in synthetic)
    again, again_count = module.augmented_views(spectra, include_mixup=True)
    assert again_count == count
    for left, right in zip(synthetic, again[len(spectra) :], strict=True):
        np.testing.assert_array_equal(left["mz"], right["mz"])
        np.testing.assert_array_equal(left["it"], right["it"])


def test_peak_merge_retains_measured_mass_and_linear_intensity():
    anchor = measured(20)
    partner = measured(40)
    anchor["mz"] = np.array([100.0, 200.0], dtype=np.float32)
    anchor["it"] = np.array([1.0, 0.5], dtype=np.float32)
    partner["mz"] = np.array([100.001, 175.0], dtype=np.float32)
    partner["it"] = np.array([1.0, 1.0], dtype=np.float32)
    mz, intensity = module._mix_peaks(anchor, partner, 0.9)
    np.testing.assert_allclose(mz, [100.0, 175.0, 200.0])
    np.testing.assert_allclose(intensity, [1.0, 0.1, 0.45], atol=1e-6)


class FakeEngine:
    def __init__(self):
        self.cfg = SimpleNamespace(analog_query="merged", max_query_spectra=16, gen_n_analog=2)
        self._last_an = None
        self.search_calls = []

    def analogs(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1):
        self.search_calls.append((self.cfg.analog_query, len(spectra), exclude_sid, exclude_lib))
        if self.cfg.analog_query == "merged":
            result = [(1, 0.6, 0.0, False)]
        elif any(row.get("ce_mixup") for row in spectra):
            result = [(3, 0.8, 0.0, False)]
        else:
            result = [(2, 0.7, 0.0, False)]
        self._last_an = result
        return result

    def generate(self, analogs, target, exclude_keys):
        return [item[0] for item in analogs]

    def run(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1):
        analogs = self.analogs(spectra, target, exclude, exclude_sid, exclude_lib)
        generated = self.generate(analogs, target, set())
        return {
            "rank_feature_analog": analogs[0][0],
            "recorded_analog": self._last_an[0][0],
            "generated_analog": generated[0],
            "mode_after": self.cfg.analog_query,
        }


@pytest.mark.parametrize("include_mixup,expected_parent,expected_views", [
    (False, 2, 3), (True, 3, 6),
])
def test_generator_only_query_does_not_change_rank_analogs(
    include_mixup, expected_parent, expected_views
):
    engine = FakeEngine()
    module.install(engine, include_mixup=include_mixup)
    result = engine.run([measured(20), measured(40), measured(60)], 200, exclude_sid=9)
    assert result == {
        "rank_feature_analog": 1,
        "recorded_analog": 1,
        "generated_analog": expected_parent,
        "mode_after": "merged",
    }
    assert engine.search_calls == [("merged", 3, 9, -1), ("each", expected_views, 9, -1)]
    assert engine._ce_view_history[-1]["mixed_views"] == (3 if include_mixup else 0)
    assert engine._ce_view_history[-1]["generated_candidates"] == 1
    with pytest.raises(ValueError, match="already installed"):
        module.install(engine, include_mixup=include_mixup)


def test_mixed_views_skip_when_only_composite_or_view_cap_reached():
    composite = measured(40, ce_n=3)
    assert module.augmented_views([composite], include_mixup=True)[1] == 0
    single = [measured(20), measured(40)]
    assert module.augmented_views(single, include_mixup=True, max_views=2)[1] == 0
    without_instrument = [measured(20, instrument=""), measured(40, instrument="")]
    assert module.augmented_views(without_instrument, include_mixup=True)[1] == 0
