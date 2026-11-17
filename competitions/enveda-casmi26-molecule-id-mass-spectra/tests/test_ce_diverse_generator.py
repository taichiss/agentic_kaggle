"""The CE-diverse adapter changes only generator parents, with an exact fallback."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src/public_notebooks/ce_diverse_generator.py"
spec = importlib.util.spec_from_file_location("ce_diverse_generator", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def measured(
    ce: float, *, instrument: str = "timsTOF", adduct: str = "[M+H]+",
    mode: int = 1, prec: float = 200.0, ce_n: int = 1,
) -> dict:
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


def analogs(*sids: int) -> list[tuple]:
    return [(sid, 0.9 - index * 0.01, 0.0, False) for index, sid in enumerate(sids)]


def test_compatible_single_ce_views_and_composite_exclusion():
    spectra = [
        measured(20), measured(40), measured(60),
        measured(40, ce_n=3),
        measured(20, instrument="Orbitrap"),
        measured(40, adduct="[M+Na]+"),
        measured(20, prec=201.0),
    ]
    selected = module.choose_ce_views(spectra)
    assert [ce for ce, _ in selected] == [20, 40, 60]
    assert all(row["instrument"] == "timsTOF" for _, row in selected)
    assert all(row["adduct"] == "[M+H]+" for _, row in selected)
    assert module.choose_ce_views([measured(20), measured(40, instrument="Orbitrap")]) == []
    assert module.choose_ce_views([measured(20), measured(40, prec=201.0)]) == []
    assert module.choose_ce_views([measured(20), measured(40, ce_n=3)]) == []


def test_duplicate_ce_uses_precursor_compatible_pair():
    high_quality_wrong_precursor = measured(20, prec=201.0)
    high_quality_wrong_precursor["mz"] = np.array([70, 80, 100, 150], dtype=np.float32)
    high_quality_wrong_precursor["it"] = np.ones(4, dtype=np.float32)
    selected = module.choose_ce_views([
        high_quality_wrong_precursor, measured(20), measured(40)
    ])
    assert [ce for ce, _ in selected] == [20, 40]
    assert [row["prec"] for _, row in selected] == [200.0, 200.0]


def test_deterministic_round_robin_preserves_baseline_leader_and_sid_uniqueness():
    selected, provenance = module.round_robin_parents(
        analogs(1, 2, 3, 4, 5, 6),
        [(20, analogs(1, 7, 8)), (40, analogs(1, 9, 10)), (60, analogs(11, 12))],
        limit=6, min_similarity=0.3,
    )
    assert [item[0] for item in selected] == [1, 7, 9, 11, 8, 10]
    assert provenance == [
        {"sid": 1, "ce": "baseline"}, {"sid": 7, "ce": 20},
        {"sid": 9, "ce": 40}, {"sid": 11, "ce": 60},
        {"sid": 8, "ce": 20}, {"sid": 10, "ce": 40},
    ]


class FakeEngine:
    def __init__(self):
        self.cfg = SimpleNamespace(
            analog_query="merged", max_query_spectra=16, gen_n_analog=6,
            gen_min_sim=0.3,
        )
        self._last_an = None
        self.search_calls = []

    def analogs(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1):
        self.search_calls.append((self.cfg.analog_query, len(spectra), exclude_sid, exclude_lib))
        if self.cfg.analog_query == "merged":
            result = analogs(1, 2, 3, 4, 5, 6)
        else:
            ce = spectra[0]["ce"]
            result = {20: analogs(1, 7, 8), 40: analogs(1, 9, 10),
                      60: analogs(11, 12)}[ce]
        self._last_an = result
        return result

    def generate(self, parents, target, exclude_keys):
        return [item[0] for item in parents]

    def run(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1):
        merged_analogs = self.analogs(spectra, target, exclude, exclude_sid, exclude_lib)
        generated = self.generate(merged_analogs, target, set())
        return {
            "rank_analog": merged_analogs[0][0],
            "recorded_analog": self._last_an[0][0],
            "generator_parents": generated,
            "mode_after": self.cfg.analog_query,
        }


def test_real_run_hook_preserves_merged_rank_analogs_and_exclusion():
    engine = FakeEngine()
    module.install(engine)
    result = engine.run([measured(20), measured(40), measured(60)], 200, exclude_sid=9)
    assert result == {
        "rank_analog": 1,
        "recorded_analog": 1,
        "generator_parents": [1, 7, 9, 11, 8, 10],
        "mode_after": "merged",
    }
    assert engine.search_calls == [
        ("merged", 3, 9, -1), ("each", 1, 9, -1),
        ("each", 1, 9, -1), ("each", 1, 9, -1),
    ]
    history = engine._ce_diverse_history[-1]
    assert history["baseline_parent_sids"] == [1, 2, 3, 4, 5, 6]
    assert history["generated_candidates"] == 6
    assert len(history["generator_parents"]) == 6
    with pytest.raises(ValueError, match="already installed"):
        module.install(engine)


def test_fewer_than_two_compatible_ce_views_uses_original_parents():
    engine = FakeEngine()
    module.install(engine)
    result = engine.run([measured(20), measured(40, instrument="Orbitrap")], 200)
    assert result["generator_parents"] == [1, 2, 3, 4, 5, 6]
    assert engine.search_calls == [("merged", 2, -1, -1)]
    history = engine._ce_diverse_history[-1]
    assert history["eligible_ce_views"] == []
    assert [row["ce"] for row in history["generator_parents"]] == ["baseline"] * 6
