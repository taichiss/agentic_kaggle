"""Validate a full-parent CE analog Notebook output before competition submission."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EXPECTED_COLUMNS = ["molecule_id", "smiles"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidates(value: str) -> list[str]:
    items = value.split(";")
    if not 1 <= len(items) <= 25 or not all(items) or len(items) != len(set(items)):
        raise ValueError("A row must contain 1–25 distinct, nonempty SMILES")
    return items


def validate(sample: Path, control: Path, variant: Path, *, expected_rows: int) -> dict:
    frames = {
        name: pd.read_csv(path, dtype=str, keep_default_na=False)
        for name, path in (("sample", sample), ("control", control), ("variant", variant))
    }
    for name, frame in frames.items():
        if list(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(f"{name} columns differ from submission contract")
        if len(frame) != expected_rows or frame.molecule_id.duplicated().any():
            raise ValueError(f"{name} row count or molecule IDs are invalid")
        if frame.molecule_id.isna().any() or not frame.molecule_id.astype(bool).all():
            raise ValueError(f"{name} has an empty molecule ID")
    if not frames["sample"].molecule_id.equals(frames["control"].molecule_id):
        raise ValueError("Control molecule order differs from sample")
    if not frames["sample"].molecule_id.equals(frames["variant"].molecule_id):
        raise ValueError("Variant molecule order differs from sample")
    control_lists = [candidates(value) for value in frames["control"].smiles]
    variant_lists = [candidates(value) for value in frames["variant"].smiles]
    changed_rows = [
        index
        for index, (before, after) in enumerate(zip(control_lists, variant_lists, strict=True))
        if before != after
    ]
    changed_candidate_sets = sum(
        set(control_lists[index]) != set(variant_lists[index]) for index in changed_rows
    )
    changed_top1 = sum(
        control_lists[index][0] != variant_lists[index][0] for index in changed_rows
    )
    return {
        "expected_rows": expected_rows,
        "changed_rows": len(changed_rows),
        "changed_candidate_sets": changed_candidate_sets,
        "changed_top1": changed_top1,
        "control_csv_sha256": sha256(control),
        "variant_csv_sha256": sha256(variant),
        "sample_csv_sha256": sha256(sample),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.sample, args.control, args.variant, expected_rows=args.expected_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
