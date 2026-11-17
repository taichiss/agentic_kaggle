"""The generator output may change candidates, but must remain a valid submission."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "src/public_notebooks/validate_ce_analog_output.py"
)
spec = importlib.util.spec_from_file_location("validate_ce_analog_output", SOURCE)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def write_rows(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["molecule_id", "smiles"])
        writer.writerows(rows)


def test_generator_may_add_a_candidate_and_change_top_rank(tmp_path: Path) -> None:
    sample, control, variant = (
        tmp_path / name for name in ("sample.csv", "control.csv", "variant.csv")
    )
    write_rows(sample, [("m1", ""), ("m2", "")])
    write_rows(control, [("m1", "CCO;CCN"), ("m2", "CCC")])
    write_rows(variant, [("m1", "CNC;CCO"), ("m2", "CCC")])
    result = module.validate(sample, control, variant, expected_rows=2)
    assert result["changed_rows"] == 1
    assert result["changed_candidate_sets"] == 1
    assert result["changed_top1"] == 1


def test_wrong_order_or_duplicate_candidate_is_rejected(tmp_path: Path) -> None:
    sample, control, variant = (
        tmp_path / name for name in ("sample.csv", "control.csv", "variant.csv")
    )
    write_rows(sample, [("m1", ""), ("m2", "")])
    write_rows(control, [("m1", "CCO"), ("m2", "CCC")])
    write_rows(variant, [("m2", "CCC"), ("m1", "CCO")])
    with pytest.raises(ValueError, match="order"):
        module.validate(sample, control, variant, expected_rows=2)
    write_rows(variant, [("m1", "CCO;CCO"), ("m2", "CCC")])
    with pytest.raises(ValueError, match="distinct"):
        module.validate(sample, control, variant, expected_rows=2)
