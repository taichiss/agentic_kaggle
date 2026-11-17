"""Add measured CE views to analog-based candidate generation only.

Install this adapter *after* ``v1engine.V1FE(E, ...)`` installs its recording
hooks. The parent calls ``E.run`` once per molecule. Its original merged-view
analogs continue to supply ranker features; the generator alone receives
analogs searched from separate measured views, optionally with gentle mixtures
of adjacent, single-CE measurements of the same query molecule.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

import numpy as np

CE_LEVELS = (20.0, 40.0, 60.0)
PRECURSOR_PPM = 20.0
PEAK_PPM = 20.0
MAX_NEW_VIEWS = 3
MAX_PEAKS = 128


def _measured_ce(spectrum: dict[str, Any]) -> float | None:
    """Recognize an individual 20/40/60 eV acquisition, not a CE composite."""
    try:
        if int(spectrum.get("ce_n", 1)) != 1:
            return None
        ce = float(spectrum["ce"])
    except (KeyError, TypeError, ValueError):
        return None
    if not np.isfinite(ce):
        return None
    return next((level for level in CE_LEVELS if abs(ce - level) <= 0.01), None)


def _valid_peaks(spectrum: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    mz = np.asarray(spectrum["mz"], dtype=np.float64).reshape(-1)
    intensity = np.asarray(spectrum["it"], dtype=np.float64).reshape(-1)
    if mz.shape != intensity.shape:
        raise ValueError("Measured m/z and intensity arrays differ in length")
    keep = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    return mz[keep], intensity[keep]


def _fraction(anchor: dict[str, Any], partner: dict[str, Any]) -> float:
    """Stable 0.85–0.95 anchor fraction, independent of process hash seeds."""
    digest = hashlib.blake2b(digest_size=8)
    for spectrum in (anchor, partner):
        metadata = np.asarray(
            [float(spectrum["prec"]), float(spectrum["ce"])], dtype="<f8"
        )
        digest.update(metadata.tobytes())
        digest.update(np.asarray(spectrum["mz"], dtype="<f4").tobytes())
    return 0.85 + 0.10 * int.from_bytes(digest.digest(), "little") / (2**64 - 1)


def _mix_peaks(
    anchor: dict[str, Any], partner: dict[str, Any], fraction: float
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.85 <= fraction <= 0.95:
        raise ValueError("Anchor fraction must be in [0.85, 0.95]")
    mz_a, it_a = _valid_peaks(anchor)
    mz_b, it_b = _valid_peaks(partner)
    mz = np.concatenate((mz_a, mz_b))
    intensity = np.concatenate((fraction * it_a, (1 - fraction) * it_b))
    if not len(mz):
        raise ValueError("Cannot mix two empty measured peak lists")
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    out_mz: list[float] = []
    out_intensity: list[float] = []
    start = 0
    for end in range(1, len(mz) + 1):
        if end < len(mz) and mz[end] - mz[end - 1] <= mz[end - 1] * PEAK_PPM * 1e-6:
            continue
        block = intensity[start:end]
        out_mz.append(float(mz[start + int(block.argmax())]))
        out_intensity.append(float(block.sum()))
        start = end
    chosen = np.argsort(-np.asarray(out_intensity), kind="stable")[:MAX_PEAKS]
    chosen.sort()
    result_mz = np.asarray(out_mz, dtype=np.float32)[chosen]
    result_it = np.asarray(out_intensity, dtype=np.float32)[chosen]
    result_it /= result_it.max()
    return result_mz, result_it


def augmented_views(
    spectra: list[dict[str, Any]], *, include_mixup: bool, max_views: int = 16
) -> tuple[list[dict[str, Any]], int]:
    """Append up to three adjacent-CE same-adduct views to one molecule's input.

    The 0.477 caller supplies spectra for one molecule. Only single-CE rows
    are eligible; composite acquisitions remain untouched in the original list.
    """
    original = list(spectra[:max_views])
    if not include_mixup or len(original) >= max_views:
        return original, 0
    groups: dict[
        tuple[int, str, str], dict[float, tuple[int, dict[str, Any]]]
    ] = defaultdict(dict)
    for index, spectrum in enumerate(original):
        level = _measured_ce(spectrum)
        instrument = str(spectrum.get("instrument") or "").strip()
        if level is None or not instrument:
            continue
        key = (int(spectrum["mode"]), str(spectrum["adduct"]), instrument)
        quality = len(_valid_peaks(spectrum)[0])
        prior = groups[key].get(level)
        if quality and (prior is None or quality > len(_valid_peaks(prior[1])[0])):
            groups[key][level] = (index, spectrum)
    mixed: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        levels = groups[group_key]
        for level in CE_LEVELS:
            if level not in levels:
                continue
            partners = [
                other for other in CE_LEVELS if other in levels and abs(other - level) == 20
            ]
            if not partners:
                continue
            partner_level = min(partners)
            anchor = levels[level][1]
            partner = levels[partner_level][1]
            precursor = float(anchor["prec"])
            partner_precursor = float(partner["prec"])
            if not np.isfinite(precursor) or precursor <= 0 or not np.isfinite(partner_precursor):
                continue
            if abs(precursor - partner_precursor) / precursor * 1e6 > PRECURSOR_PPM:
                continue
            fraction = _fraction(anchor, partner)
            mz, intensity = _mix_peaks(anchor, partner, fraction)
            mixed.append({
                "mz": mz,
                "it": intensity,
                "mode": group_key[0],
                "adduct": group_key[1],
                "instrument": group_key[2],
                "prec": precursor,
                "ce": fraction * level + (1 - fraction) * partner_level,
                "ce_n": 2,
                "ce_mixup": True,
            })
            if len(mixed) >= min(MAX_NEW_VIEWS, max_views - len(original)):
                return original + mixed, len(mixed)
    return original + mixed, len(mixed)


def install(engine: Any, *, include_mixup: bool) -> None:
    """Search generator parents with separate CE views while preserving rank features.

    Run this after ``V1FE`` installs its hooks. The engine is serial in the
    0.477 parent; concurrent or recursive ``run`` calls are intentionally refused.
    """
    if getattr(engine, "_ce_view_adapter_installed", False):
        raise ValueError("CE view adapter is already installed")
    if engine.cfg.analog_query != "merged":
        raise ValueError("Expected the frozen 0.477 merged analog-query control")
    original_run = engine.run
    original_generate = engine.generate
    state: dict[str, Any] = {"active": False}
    engine._ce_view_history = []

    def generate(analogs, target, exclude_keys):
        if not state["active"]:
            raise RuntimeError("CE view generation called outside Engine.run")
        spectra = state["spectra"]
        views, mixed_count = augmented_views(
            spectra, include_mixup=include_mixup, max_views=engine.cfg.max_query_spectra
        )
        saved_last = getattr(engine, "_last_an", None)
        try:
            engine.cfg.analog_query = "each"
            new_analogs = engine.analogs(
                views,
                target,
                state["exclude"],
                state["exclude_sid"],
                state["exclude_lib"],
            )
        finally:
            engine.cfg.analog_query = "merged"
            engine._last_an = saved_last
        if not new_analogs:
            new_analogs = analogs
        generated = original_generate(new_analogs, target, exclude_keys)
        engine._ce_view_history.append({
            "target": float(target),
            "measured_views": len(spectra),
            "mixed_views": mixed_count,
            "baseline_analog_sids": [int(item[0]) for item in analogs[:engine.cfg.gen_n_analog]],
            "generator_analog_sids": [
                int(item[0]) for item in new_analogs[:engine.cfg.gen_n_analog]
            ],
            "generated_candidates": len(generated),
        })
        return generated

    def run(
        spectra, target, exclude=None, exclude_sid=-1, exclude_lib=-1, drop_pid=-1
    ):
        if state["active"]:
            raise RuntimeError("Recursive CE view Engine.run is unsupported")
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
                exclude_lib=exclude_lib, drop_pid=drop_pid
            )
        finally:
            state.clear()
            state["active"] = False

    engine.generate = generate
    engine.run = run
    engine._ce_view_adapter_installed = True
