"""Deterministic dataset-level calibration/report split for EXP-0007."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _density_quartiles(records: list[dict[str, Any]]) -> dict[str, int]:
    ordered = sorted(
        records,
        key=lambda row: (int(row["max_annotated_edges"]), str(row["dataset"])),
    )
    total = len(ordered)
    return {
        str(row["dataset"]): min(3, index * 4 // max(total, 1))
        for index, row in enumerate(ordered)
    }


def build_calibration_report_split(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int,
    calibration_size: int,
) -> dict[str, Any]:
    """Split whole datasets while balancing division presence and density quartile."""
    materialized = [dict(row) for row in records]
    names = [str(row["dataset"]) for row in materialized]
    if len(names) != len(set(names)):
        raise ValueError("validation split records contain duplicate datasets")
    if not 0 < calibration_size < len(materialized):
        raise ValueError("calibration_size must leave both subsets non-empty")

    quartiles = _density_quartiles(materialized)
    strata: dict[tuple[bool, int], list[dict[str, Any]]] = defaultdict(list)
    for row in materialized:
        dataset = str(row["dataset"])
        row["density_quartile"] = quartiles[dataset]
        row["has_division"] = int(row["annotated_divisions"]) > 0
        strata[(bool(row["has_division"]), quartiles[dataset])].append(row)

    allocations = {key: len(rows) // 2 for key, rows in strata.items()}
    remaining = calibration_size - sum(allocations.values())
    odd_strata = [key for key, rows in strata.items() if len(rows) % 2]
    odd_strata.sort(key=lambda key: _stable_key(seed, repr(key)))
    if remaining < 0 or remaining > len(odd_strata):
        raise ValueError("requested calibration size cannot preserve half-stratum balance")
    for key in odd_strata[:remaining]:
        allocations[key] += 1

    calibration: list[str] = []
    report: list[str] = []
    annotated_records: list[dict[str, Any]] = []
    for key in sorted(strata):
        rows = sorted(
            strata[key],
            key=lambda row: _stable_key(seed, str(row["dataset"])),
        )
        n_calibration = allocations[key]
        for index, row in enumerate(rows):
            subset = "calibration" if index < n_calibration else "report"
            dataset = str(row["dataset"])
            (calibration if subset == "calibration" else report).append(dataset)
            annotated_records.append({**row, "subset": subset})

    calibration.sort()
    report.sort()
    annotated_records.sort(key=lambda row: str(row["dataset"]))
    if len(calibration) != calibration_size:
        raise AssertionError("calibration allocation diverged from requested size")
    if set(calibration) & set(report):
        raise AssertionError("calibration/report datasets overlap")

    return {
        "schema_version": 1,
        "seed": seed,
        "stratification": ["has_division", "annotated_edge_density_quartile"],
        "calibration": calibration,
        "report": report,
        "records": annotated_records,
    }
