"""Choose rule-generator parents across measured CE views without changing ranks.

Install after ``v1engine.V1FE(E, ...)``. The original merged-view analog list
continues to supply the 0.477 rank features. Only the six parents passed to
the rule-based derivative generator are diversified across real, single-CE
measurements of the same query molecule/adduct/polarity/instrument.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations, product
from typing import Any

import numpy as np

CE_LEVELS = (20.0, 40.0, 60.0)
PRECURSOR_PPM = 20.0


def measured_ce(spectrum: dict[str, Any]) -> float | None:
    try:
        if int(spectrum.get("ce_n", 1)) != 1:
            return None
        ce = float(spectrum["ce"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(ce):
        return None
    return next((level for level in CE_LEVELS if abs(ce - level) <= 0.01), None)


def choose_ce_views(spectra: list[dict[str, Any]]) -> list[tuple[float, dict[str, Any]]]:
    """Take the best measured row at each CE from one compatible ion group."""
    groups: dict[
        tuple[int, str, str], dict[float, list[tuple[int, int, dict[str, Any]]]]
    ] = defaultdict(lambda: defaultdict(list))
    for index, spectrum in enumerate(spectra):
        level = measured_ce(spectrum)
        instrument = str(spectrum.get("instrument") or "").strip()
        if level is None or not instrument:
            continue
        key = (int(spectrum["mode"]), str(spectrum["adduct"]), instrument)
        mz = np.asarray(spectrum["mz"])
        intensity = np.asarray(spectrum["it"])
        if mz.shape != intensity.shape:
            raise ValueError("Measured m/z and intensity arrays differ in length")
        quality = int(np.count_nonzero(
            np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        ))
        if quality:
            groups[key][level].append((quality, index, spectrum))
    options = []
    for key, levels in groups.items():
        if len(levels) < 2:
            continue
        available = tuple(level for level in CE_LEVELS if level in levels)
        for count in (3, 2):
            for subset in combinations(available, count):
                for rows in product(*(levels[level] for level in subset)):
                    precursors = np.asarray([float(row[2]["prec"]) for row in rows])
                    if not np.isfinite(precursors).all() or np.min(precursors) <= 0:
                        continue
                    ppm_span = np.ptp(precursors) / np.median(precursors) * 1e6
                    if ppm_span > PRECURSOR_PPM:
                        continue
                    options.append((
                        count, sum(row[0] for row in rows), key,
                        tuple(row[1] for row in rows),
                        [(level, row[2]) for level, row in zip(subset, rows, strict=True)],
                    ))
    if not options:
        return []
    options.sort(key=lambda row: (-row[0], -row[1], row[2], row[3]))
    return options[0][4]


def round_robin_parents(
    baseline: list[tuple], per_ce: list[tuple[float, list[tuple]]],
    *, limit: int, min_similarity: float,
) -> tuple[list[tuple], list[dict[str, int | float | str]]]:
    """Keep the baseline leader; alternate eligible CE parents within six slots."""
    if not per_ce:
        parents = [item for item in baseline if item[1] >= min_similarity][:limit]
        return parents, [{"sid": int(item[0]), "ce": "baseline"} for item in parents]
    chosen = [baseline[0]] if baseline and baseline[0][1] >= min_similarity else []
    seen = {int(item[0]) for item in chosen}
    provenance = [{"sid": int(item[0]), "ce": "baseline"} for item in chosen]
    cursors = [0] * len(per_ce)
    while len(chosen) < limit:
        advanced = False
        for index, (ce, analogs) in enumerate(per_ce):
            while cursors[index] < len(analogs):
                item = analogs[cursors[index]]
                cursors[index] += 1
                if item[1] < min_similarity:
                    break
                sid = int(item[0])
                if sid in seen:
                    continue
                chosen.append(item)
                seen.add(sid)
                provenance.append({"sid": sid, "ce": ce})
                advanced = True
                break
            if len(chosen) >= limit:
                break
        if not advanced:
            break
    for item in baseline:
        if len(chosen) >= limit:
            break
        sid = int(item[0])
        if sid not in seen and item[1] >= min_similarity:
            chosen.append(item)
            seen.add(sid)
            provenance.append({"sid": sid, "ce": "baseline"})
    return chosen, provenance


def install(engine: Any) -> None:
    """Preserve merged analog rank features and diversify only generator parents."""
    if getattr(engine, "_ce_diverse_installed", False):
        raise ValueError("CE-diverse generator adapter is already installed")
    if engine.cfg.analog_query != "merged":
        raise ValueError("Expected frozen 0.477 merged analog-query control")
    original_run = engine.run
    original_generate = engine.generate
    state: dict[str, Any] = {"active": False}
    engine._ce_diverse_history = []

    def generate(analogs, target, exclude_keys):
        if not state["active"]:
            raise RuntimeError("CE-diverse generation called outside Engine.run")
        selected = choose_ce_views(state["spectra"])
        saved_last = getattr(engine, "_last_an", None)
        per_ce = []
        try:
            engine.cfg.analog_query = "each"
            for ce, spectrum in selected:
                per_ce.append((ce, engine.analogs(
                    [spectrum], target,
                    state["exclude"], state["exclude_sid"], state["exclude_lib"]
                )))
        finally:
            engine.cfg.analog_query = "merged"
            engine._last_an = saved_last
        chosen, provenance = round_robin_parents(
            analogs, per_ce, limit=engine.cfg.gen_n_analog,
            min_similarity=engine.cfg.gen_min_sim,
        )
        if not chosen:
            chosen, provenance = round_robin_parents(
                analogs, [], limit=engine.cfg.gen_n_analog,
                min_similarity=engine.cfg.gen_min_sim,
            )
        generated = original_generate(chosen, target, exclude_keys)
        engine._ce_diverse_history.append({
            "target": float(target),
            "eligible_ce_views": [ce for ce, _ in selected],
            "baseline_parent_sids": [
                int(item[0]) for item in analogs
                if item[1] >= engine.cfg.gen_min_sim
            ][:engine.cfg.gen_n_analog],
            "generator_parents": provenance,
            "generated_candidates": len(generated),
        })
        return generated

    def run(spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1):
        if state["active"]:
            raise RuntimeError("Recursive CE-diverse Engine.run is unsupported")
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
    engine._ce_diverse_installed = True
