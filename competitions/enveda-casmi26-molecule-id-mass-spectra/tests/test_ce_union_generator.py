"""The CE union keeps frozen baseline products before adding new candidates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/public_notebooks"))
from ce_union_generator import union_generated  # noqa: E402


def fake_generate(parents, target, excluded):
    assert target == 300.0
    return [
        {"key": f"k{parent[0]}", "parent": parent[0]}
        for parent in parents
        if f"k{parent[0]}" not in excluded
    ]


def test_union_keeps_baseline_products_and_adds_ce_products_under_cap():
    baseline = [(1, 0.9, 0, False), (2, 0.8, 0, False)]
    per_ce = [
        (20.0, [(3, 0.8, 0, False)]),
        (60.0, [(4, 0.7, 0, False)]),
    ]
    generated, provenance, baseline_count = union_generated(
        fake_generate, baseline, per_ce, 300.0, set(),
        limit=3, min_similarity=0.5, max_total=3,
    )
    assert baseline_count == 2
    assert [row["key"] for row in generated] == ["k1", "k2", "k3"]
    assert [row["sid"] for row in provenance] == [1, 3, 4]


def test_no_ce_view_returns_exact_baseline_generation():
    baseline = [(1, 0.9, 0, False), (2, 0.8, 0, False)]
    generated, provenance, baseline_count = union_generated(
        fake_generate, baseline, [], 300.0, set(),
        limit=3, min_similarity=0.5, max_total=3,
    )
    assert baseline_count == 2
    assert generated == fake_generate(baseline, 300.0, set())
    assert provenance == []


def test_hooked_union_retains_baseline_context_and_derivative_support():
    class Hook:
        def __init__(self):
            self._derlog = []
            self._last_gen = []

        def generate(self, parents, target, excluded):
            out = []
            for sid, sim, *_ in parents:
                key = f"k{sid}"
                if key in excluded:
                    self._derlog.append((sid, sim, (1, 2), "pool", "key", key))
                else:
                    out.append({"key": key, "parent": sid, "raw": key, "combo": (1, 2)})
            self._last_gen = out
            return out

    hook = Hook()
    baseline = [(1, 0.9, 0, False), (2, 0.8, 0, False)]
    per_ce = [(20.0, [(3, 0.8, 0, False)])]
    generated, _, n_base = union_generated(
        hook.generate, baseline, per_ce, 300.0, set(),
        limit=3, min_similarity=0.5, max_total=10, hook_state=hook,
    )
    assert n_base == 2
    assert [row["key"] for row in generated] == ["k1", "k2", "k3"]
    assert hook._last_gen == generated
    assert hook._derlog == [
        (1, 0.9, (1, 2), "gen", "key", "k1"),
        (2, 0.8, (1, 2), "gen", "key", "k2"),
    ]
