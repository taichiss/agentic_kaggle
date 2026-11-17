"""Use individual CE spectra for analog rank features, but frozen merged generation.

Install after ``v1engine.V1FE`` has installed ``hooks_v1``.  The parent Engine
remains serial.  This adapter is for a local causal diagnostic, not submission.
"""

from __future__ import annotations

from typing import Any


def install(engine: Any) -> None:
    if getattr(engine, "_ce_rank_only_installed", False):
        raise ValueError("CE analog rank-only adapter is already installed")
    if engine.cfg.analog_query != "merged":
        raise ValueError("Expected frozen 0.477 merged analog-query control")

    original_run = engine.run
    original_generate = engine.generate
    state: dict[str, Any] = {"active": False}
    engine._ce_rank_only_history = []
    engine.cfg.analog_query = "each"

    def generate(analogs, target, exclude_keys):
        if not state["active"]:
            raise RuntimeError("Generation called outside Engine.run")
        rank_analogs = list(analogs)
        try:
            engine.cfg.analog_query = "merged"
            merged_analogs = engine.analogs(
                state["spectra"], target, state["exclude"],
                state["exclude_sid"], state["exclude_lib"],
            )
        finally:
            engine.cfg.analog_query = "each"
            # V1FE.context reads the hook's most recent analog list; that must
            # describe the ranking arm, not the generation-only lookup.
            engine._last_an = rank_analogs
        generated = original_generate(merged_analogs, target, exclude_keys)
        engine._ce_rank_only_history.append({
            "target": float(target),
            "rank_analog_sids": [int(item[0]) for item in rank_analogs[:6]],
            "generator_analog_sids": [int(item[0]) for item in merged_analogs[:6]],
            "generated_keys": [str(item["key"]) for item in generated],
        })
        return generated

    def run(
        spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1
    ):
        if state["active"]:
            raise RuntimeError("Recursive Engine.run is unsupported")
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
                exclude_lib=exclude_lib, drop_pid=drop_pid,
            )
        finally:
            state.clear()
            state["active"] = False

    engine.generate = generate
    engine.run = run
    engine._ce_rank_only_installed = True
