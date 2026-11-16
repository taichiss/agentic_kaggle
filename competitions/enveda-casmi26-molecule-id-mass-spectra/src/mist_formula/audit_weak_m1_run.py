"""Verify that weak M1 changes only the mixture coefficient interval."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m1-manifest", type=Path, required=True)
    parser.add_argument("--weak-manifest", type=Path, required=True)
    parser.add_argument("--m1-history", type=Path, required=True)
    parser.add_argument("--weak-history", type=Path, required=True)
    parser.add_argument("--m1-source-provenance", type=Path, required=True)
    parser.add_argument("--train-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    m1 = json.loads(args.m1_manifest.read_text())
    weak = json.loads(args.weak_manifest.read_text())
    source = json.loads(args.m1_source_provenance.read_text())
    if sha(args.train_source) != source["trainer_sha256"]:
        raise AssertionError("The trainer source differs from strict M1")
    immutable = (
        "model", "formula_assignment", "ion_channels", "non_chno", "formula_mass",
        "data", "precise_mzs", "source_metadata", "raw_train_sha256",
        "precise_mzs_alignment", "aligned_spectra", "cache_spectra_sha256",
        "cache_molecules_sha256", "precise_mzs_sha256", "ingest_id_sha256",
        "instrument_id_sha256", "formula_csv_sha256", "formula_rows",
        "panel_sha256", "split", "seed", "steps", "batch_size", "learning_rate",
    )
    if any(m1[key] != weak[key] for key in immutable):
        raise AssertionError("A non-alpha training input differs from strict M1")
    expected_mix = ("kind", "probability", "attempted", "applied", "attempts_by_source",
                    "applied_by_source", "observed_eligible_spectra",
                    "observed_eligible_molecules", "max_partner_search_draws",
                    "intensity_reconstruction", "merged_mz_ppm")
    if any(m1["mixup"][key] != weak["mixup"][key] for key in expected_mix):
        raise AssertionError("Mixing partner stream or policy differs from strict M1")
    if m1["mixup"]["alpha_range"] != [0.25, 0.75]:
        raise AssertionError("Unexpected strict M1 alpha interval")
    if weak["mixup"]["alpha_range"] != [0.85, 0.95]:
        raise AssertionError("Unexpected weak M1 alpha interval")
    m1_events = json.loads(args.m1_history.read_text())
    weak_events = json.loads(args.weak_history.read_text())
    stream = ("step", "mix_attempted", "mix_applied")
    if len(m1_events) != len(weak_events) or any(
        any(left[key] != right[key] for key in stream)
        for left, right in zip(m1_events, weak_events, strict=True)
    ):
        raise AssertionError("Logged partner stream differs from strict M1")
    report = {
        "same_non_alpha_inputs": True,
        "same_trainer_source_sha256": sha(args.train_source),
        "same_logged_partner_stream": True,
        "same_final_mix_coverage_and_source_breakdown": True,
        "strict_alpha_range": m1["mixup"]["alpha_range"],
        "weak_alpha_range": weak["mixup"]["alpha_range"],
        "mix_attempted": weak["mixup"]["attempted"],
        "mix_applied": weak["mixup"]["applied"],
        "checkpoints_sha256": {
            "M1": m1["checkpoint_sha256"],
            "WEAK": weak["checkpoint_sha256"],
        },
        "input_sha256": {
            name: sha(getattr(args, name))
            for name in (
                "m1_manifest", "weak_manifest", "m1_history", "weak_history",
                "m1_source_provenance", "train_source",
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
