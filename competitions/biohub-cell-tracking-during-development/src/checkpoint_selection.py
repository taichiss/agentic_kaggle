"""Deterministic checkpoint selection using the Biohub competition metric."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

SCORE_FIELDS = (
    "competition_score",
    "adjusted_edge_jaccard",
    "edge_jaccard",
    "node_recall",
)
EPOCH_FIELD = "completed_epoch"


def _finite_number(record: Mapping[str, Any], field: str, index: int) -> float:
    if field not in record:
        raise ValueError(f"checkpoint record {index} is missing {field!r}")

    value = record[field]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"checkpoint record {index} has a non-numeric {field!r}: {value!r}"
        )

    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(
            f"checkpoint record {index} has a non-finite {field!r}: {value!r}"
        )
    return numeric_value


def _selection_key(record: Mapping[str, Any], index: int) -> tuple[float, ...]:
    scores = tuple(_finite_number(record, field, index) for field in SCORE_FIELDS)
    completed_epoch = _finite_number(record, EPOCH_FIELD, index)
    return (*scores, -completed_epoch)


def select_best_checkpoint(
    records: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the checkpoint with the best official score and deterministic ties.

    Ties are resolved by higher adjusted edge Jaccard, higher edge Jaccard,
    higher node recall, and finally the earlier completed epoch, in that order.
    Every selection metric must be present, numeric, and finite.
    """

    if not records:
        raise ValueError("at least one checkpoint record is required")

    ranked = [(_selection_key(record, index), record) for index, record in enumerate(records)]
    return max(ranked, key=lambda item: item[0])[1]


def annotate_checkpoint_selection(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Copy records and mark exactly one competition-selected checkpoint."""

    selected = select_best_checkpoint(records)
    selected_id = id(selected)
    selected_marked = False
    annotated: list[dict[str, Any]] = []

    for record in records:
        is_selected = not selected_marked and id(record) == selected_id
        annotated.append({**record, "selected": is_selected})
        selected_marked = selected_marked or is_selected

    return annotated
