"""Count label-free G4/G5 CE-selector eligibility on frozen test spectra."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/public_notebooks"))
from ce_diverse_generator import choose_ce_views  # noqa: E402

MODEL_CODE = (
    Path("/kaggle/input/casmi26-attributed-v4g-reproduction-assets")
    / "casmi26-v4b-models/code"
)
sys.path.insert(0, str(MODEL_CODE))
from casmi import chem  # noqa: E402

SELECTOR = ROOT / "src/public_notebooks/ce_diverse_generator.py"
SELECTOR_SHA = "178adad68f7af279464c3786bd25d2b13777cd070e1c0ee426177e3b594cd13a"
FULL_TEST = ROOT / "data/input/test.parquet"
FULL_SUBMISSION = ROOT / "data/input/sample_submission.csv"
EXTERNAL80 = ROOT / "data/derived/candidate-rescue-corrected-v2/test.parquet"
VISIBLE12 = ROOT / "data/derived/ce-views-20261117/output-g5-v2/ce-union-generator-audit.json"
OUTPUT = ROOT / "data/derived/ce-views-20261117/g5-selector-test-coverage"
MAX_QUERY_SPECTRA = 16


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spectra_for_group(group: pd.DataFrame) -> list[dict]:
    """Match frozen 0.477 cell 39's test-row construction before Engine.run."""
    spectra = []
    for row in group.itertuples():
        ce = row.collision_energy_ev
        ce_arr = (
            np.atleast_1d(np.asarray(ce, dtype=float))
            if ce is not None else np.zeros(0)
        )
        spectra.append({
            "mz": np.asarray(row.ms2_mzs, np.float64),
            "it": np.asarray(row.ms2_normalized_intensities, np.float64),
            "mode": 1 if row.ionization_mode == "positive" else -1,
            "adduct": row.adduct,
            "prec": float(row.precursor_mz),
            "ce": float(ce_arr.mean()) if len(ce_arr) else np.nan,
            "ce_n": int(len(ce_arr)) if len(ce_arr) else 1,
            "instrument": str(row.instrument_type),
        })
    return spectra


def panel(path: Path) -> tuple[list[dict], dict]:
    frame = pd.read_parquet(path)
    frame["_neutral_mass"] = chem.neutral_mass(
        frame.precursor_mz.to_numpy(dtype=np.float64), frame.adduct.values
    )
    records = []
    for molecule_id, group in frame.groupby("molecule_id", sort=False):
        spectra = spectra_for_group(group)
        selected = choose_ce_views(spectra[:MAX_QUERY_SPECTRA])
        ce_levels = [float(level) for level, _ in selected]
        records.append({
            "molecule_id": str(molecule_id),
            "n_spectra": len(spectra),
            "has_finite_target": bool(np.isfinite(group._neutral_mass).any()),
            "selected_ce_views": ce_levels,
            "eligible": len(ce_levels) >= 2,
        })
    n_view = Counter(len(row["selected_ce_views"]) for row in records)
    patterns = Counter(
        "/".join(str(int(ce)) for ce in row["selected_ce_views"])
        for row in records if row["eligible"]
    )
    summary = {
        "molecules": len(records),
        "spectra": len(frame),
        "molecules_with_finite_target": sum(row["has_finite_target"] for row in records),
        "eligible_molecules": sum(row["eligible"] for row in records),
        "selected_view_count": {str(key): n_view[key] for key in sorted(n_view)},
        "selected_ce_pattern": dict(sorted(patterns.items())),
        "max_spectra_per_molecule": max(row["n_spectra"] for row in records),
        "input_sha256": sha(path),
    }
    return records, summary


def main() -> None:
    if sha(SELECTOR) != SELECTOR_SHA:
        raise ValueError("CE selector differs from frozen G5 v2")
    full_rows, full = panel(FULL_TEST)
    external_rows, external = panel(EXTERNAL80)
    sample = pd.read_csv(FULL_SUBMISSION, usecols=["molecule_id"])
    if len(full_rows) != 400 or len(sample) != 400:
        raise ValueError("Expected 400 test and submission molecule IDs")
    if set(sample.molecule_id.astype(str)) != {row["molecule_id"] for row in full_rows}:
        raise ValueError("Full test and submission molecule IDs differ")
    if len(external_rows) != 80:
        raise ValueError("Expected 80 external validation groups")

    receipt = json.loads(VISIBLE12.read_text())
    visible_12 = [row["selected_ce_views"] for row in full_rows[:12]]
    receipt_12 = [row["eligible_ce_views"] for row in receipt["queries"]]
    if visible_12 != receipt_12:
        raise ValueError("Frozen Kaggle visible-12 CE-view sequence differs")
    if len(receipt_12) != 12 or sum(bool(row) for row in visible_12) != 10:
        raise ValueError("Frozen Kaggle visible-12 eligibility differs")
    if external["eligible_molecules"] != 8:
        raise ValueError("External80 CE eligibility differs from frozen audit")

    report = {
        "scope": "label-free CE-view metadata eligibility, not generation success or score",
        "frozen_selector_sha256": SELECTOR_SHA,
        "full_test": full,
        "external80": external,
        "visible12": {
            "molecules": 12,
            "eligible_molecules": 10,
            "ce_view_sequence_matches_private_v2": True,
            "private_audit_sha256": sha(VISIBLE12),
        },
        "sample_submission_sha256": sha(FULL_SUBMISSION),
        "evaluator_sha256": sha(Path(__file__)),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (OUTPUT / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in full_rows)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
