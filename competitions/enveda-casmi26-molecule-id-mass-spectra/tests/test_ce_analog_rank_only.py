"""Guard the causal boundary between CE analog ranking and generation."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace


ADAPTER = Path(__file__).resolve().parents[1] / "src/public_notebooks/ce_analog_rank_only.py"
SPEC = importlib.util.spec_from_file_location("ce_analog_rank_only", ADAPTER)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeEngine:
    def __init__(self):
        self.cfg = SimpleNamespace(analog_query="merged", max_query_spectra=16)
        self._last_an = None
        self._last_gen = []
        self.queries = []

    def analogs(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1):
        self.queries.append((self.cfg.analog_query, list(spectra), exclude_sid, exclude_lib))
        result = [(1 if self.cfg.analog_query == "merged" else 2, 0.5, 0.0, False)]
        self._last_an = result
        return result

    def generate(self, analogs, target, exclude_keys):
        products = [{"key": f"generated-by-{analogs[0][0]}"}]
        self._last_gen = products
        return products

    def run(self, spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1):
        analogs = self.analogs(spectra, target, exclude, exclude_sid, exclude_lib)
        generated = self.generate(analogs, target, {"pool-candidate"})
        return generated, self._last_an, self._last_gen


def test_merged_generation_but_each_rank_context():
    engine = FakeEngine()
    MODULE.install(engine)
    candidates, recorded_analogs, recorded_generated = engine.run(
        ["spectrum"], 300.0, exclude_sid=17, exclude_lib=4
    )
    assert [item["key"] for item in candidates] == ["generated-by-1"]
    assert candidates == recorded_generated
    assert recorded_analogs[0][0] == 2
    assert engine.cfg.analog_query == "each"
    assert [query[0] for query in engine.queries] == ["each", "merged"]
    assert all(query[2:] == (17, 4) for query in engine.queries)
    assert engine._ce_rank_only_history[0]["generated_keys"] == ["generated-by-1"]


def test_refuses_repeat_install():
    engine = FakeEngine()
    MODULE.install(engine)
    try:
        MODULE.install(engine)
    except ValueError as error:
        assert "already installed" in str(error)
    else:
        raise AssertionError("Repeated installation should be rejected")
