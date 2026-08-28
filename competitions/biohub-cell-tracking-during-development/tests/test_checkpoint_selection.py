from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "checkpoint_selection.py"
SPEC = importlib.util.spec_from_file_location("biohub_checkpoint_selection", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
checkpoint_selection = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checkpoint_selection
SPEC.loader.exec_module(checkpoint_selection)


def _record(
    epoch: int,
    competition_score: float,
    adjusted_edge_jaccard: float = 0.7,
    edge_jaccard: float = 0.6,
    node_recall: float = 0.9,
) -> dict[str, float | int]:
    return {
        "completed_epoch": epoch,
        "competition_score": competition_score,
        "adjusted_edge_jaccard": adjusted_edge_jaccard,
        "edge_jaccard": edge_jaccard,
        "node_recall": node_recall,
    }


def test_selects_highest_competition_score_before_diagnostics() -> None:
    records = [
        _record(5, 0.81, adjusted_edge_jaccard=0.99),
        _record(10, 0.82, adjusted_edge_jaccard=0.70),
    ]

    assert checkpoint_selection.select_best_checkpoint(records) is records[1]


@pytest.mark.parametrize(
    ("field", "winner_value"),
    [
        ("adjusted_edge_jaccard", 0.71),
        ("edge_jaccard", 0.61),
        ("node_recall", 0.91),
    ],
)
def test_score_ties_use_ordered_quality_metrics(field: str, winner_value: float) -> None:
    first = _record(5, 0.82)
    second = _record(10, 0.82)
    second[field] = winner_value

    assert checkpoint_selection.select_best_checkpoint([first, second]) is second


def test_complete_tie_selects_earlier_completed_epoch() -> None:
    later = _record(20, 0.82)
    earlier = _record(10, 0.82)

    assert checkpoint_selection.select_best_checkpoint([later, earlier]) is earlier


def test_annotation_marks_one_copy_without_mutating_records() -> None:
    records = [_record(5, 0.81), _record(10, 0.82)]

    annotated = checkpoint_selection.annotate_checkpoint_selection(records)

    assert [record["selected"] for record in annotated] == [False, True]
    assert all("selected" not in record for record in records)


@pytest.mark.parametrize("bad_value", [None, float("nan"), float("inf")])
def test_rejects_missing_or_non_finite_competition_score(bad_value: float | None) -> None:
    record = _record(5, 0.81)
    if bad_value is None:
        del record["competition_score"]
    else:
        record["competition_score"] = bad_value

    with pytest.raises(ValueError, match="competition_score"):
        checkpoint_selection.select_best_checkpoint([record])
