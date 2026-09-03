from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "src" / "submission.py"
SPEC = importlib.util.spec_from_file_location("biohub_submission", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
submission = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = submission
SPEC.loader.exec_module(submission)


def test_submission_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    submission.write_submission(
        path,
        [
            submission.Node("sample-a", 1, 0, 2, 3, 4),
            submission.Node("sample-a", 2, 1, 2, 4, 4),
        ],
        [submission.Edge("sample-a", 1, 2)],
    )

    assert submission.validate_submission(path, ["sample-a"]) == {
        "datasets": 1,
        "nodes": 2,
        "edges": 1,
    }


def test_submission_rejects_missing_edge_node(tmp_path: Path) -> None:
    path = tmp_path / "submission.csv"
    submission.write_submission(
        path,
        [submission.Node("sample-a", 1, 0, 2, 3, 4)],
        [submission.Edge("sample-a", 1, 2)],
    )

    with pytest.raises(ValueError, match="missing nodes"):
        submission.validate_submission(path)
