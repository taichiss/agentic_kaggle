"""Retain baseline rule-generated candidates and add distinct CE-view products."""

from __future__ import annotations

from typing import Any

try:
    from ce_diverse_generator import choose_ce_views, round_robin_parents
except ImportError:
    from .ce_diverse_generator import choose_ce_views, round_robin_parents


def union_generated(original_generate, baseline, per_ce, target, exclude_keys, *,
                    limit, min_similarity, max_total, hook_state=None):
    """Keep baseline products and their hooked derivative evidence intact."""
    base = original_generate(baseline, target, exclude_keys)
    baseline_keys = {row["key"] for row in base}
    derlog_start = len(hook_state._derlog) if hook_state is not None else 0
    if not per_ce or len(base) >= max_total:
        if hook_state is not None:
            hook_state._last_gen = base
        return base, [], len(base)
    selected, provenance = round_robin_parents(
        baseline, per_ce, limit=limit, min_similarity=min_similarity)
    seen = set(exclude_keys)
    seen.update(baseline_keys)
    extra = original_generate(selected, target, seen)
    generated = base + extra[:max_total - len(base)]
    if hook_state is not None:
        # The hooked generator treats every exclusion as a pool member. A
        # baseline-generated key is excluded from pass two, but stays a
        # generated candidate when derivative supports are attributed.
        for index in range(derlog_start, len(hook_state._derlog)):
            row = hook_state._derlog[index]
            if row[3] == "pool" and row[4] == "key" and row[5] in baseline_keys:
                hook_state._derlog[index] = (*row[:3], "gen", *row[4:])
        hook_state._last_gen = generated
    return generated, provenance, len(base)


def install(engine: Any) -> None:
    """Keep the original analog rank features and augment only generation."""
    if getattr(engine, "_ce_union_installed", False):
        raise ValueError("CE-union generator adapter is already installed")
    if engine.cfg.analog_query != "merged":
        raise ValueError("Expected frozen 0.477 merged analog-query control")
    original_run = engine.run
    original_generate = engine.generate
    state: dict[str, Any] = {"active": False}
    engine._ce_union_history = []

    def generate(analogs, target, exclude_keys):
        if not state["active"]:
            raise RuntimeError("CE-union generation called outside Engine.run")
        selected = choose_ce_views(state["spectra"])
        saved_last = getattr(engine, "_last_an", None)
        per_ce = []
        try:
            engine.cfg.analog_query = "each"
            for ce, spectrum in selected:
                per_ce.append((ce, engine.analogs(
                    [spectrum], target, state["exclude"],
                    state["exclude_sid"], state["exclude_lib"])))
        finally:
            engine.cfg.analog_query = "merged"
            engine._last_an = saved_last
        generated, provenance, n_base = union_generated(
            original_generate, analogs, per_ce, target, exclude_keys,
            limit=engine.cfg.gen_n_analog,
            min_similarity=engine.cfg.gen_min_sim,
            max_total=engine.cfg.gen_max_total, hook_state=engine)
        engine._ce_union_history.append({
            "target": float(target),
            "eligible_ce_views": [ce for ce, _ in selected],
            "baseline_generated": n_base,
            "extra_generated": len(generated) - n_base,
            "generated_candidates": len(generated),
            "ce_parent_provenance": provenance,
        })
        return generated

    def run(spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1,
            drop_pid=-1):
        if state["active"]:
            raise RuntimeError("Recursive CE-union Engine.run is unsupported")
        state.update({
            "active": True,
            "spectra": list(spectra[:engine.cfg.max_query_spectra]),
            "exclude": exclude,
            "exclude_sid": exclude_sid,
            "exclude_lib": exclude_lib,
        })
        try:
            return original_run(
                spectra, target, exclude=exclude, exclude_sid=exclude_sid,
                exclude_lib=exclude_lib, drop_pid=drop_pid)
        finally:
            state.clear()
            state["active"] = False

    engine.generate = generate
    engine.run = run
    engine._ce_union_installed = True
