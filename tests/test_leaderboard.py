from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from agentic_kaggle.leaderboard import (
    Submission,
    choose_submission,
    submit_and_wait,
    validate_submission,
)


def _completed(command: list[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, 0, stdout=stdout, stderr="")


def test_validate_submission_checks_manifest_columns(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,prediction\n1,0.2\n2,0.8\n", encoding="utf-8")
    manifest = {
        "submission": {
            "id_columns": ["id"],
            "prediction_columns": ["prediction"],
            "expected_rows": 2,
        }
    }

    metadata = validate_submission(submission, manifest)

    assert metadata["rows"] == 2
    assert len(metadata["sha256"]) == 64


def test_validate_submission_rejects_missing_column(tmp_path: Path) -> None:
    submission = tmp_path / "submission.csv"
    submission.write_text("id,wrong\n1,0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing manifest columns"):
        validate_submission(
            submission,
            {"submission": {"id_columns": ["id"], "prediction_columns": ["prediction"]}},
        )


def test_choose_submission_prefers_new_matching_ref() -> None:
    old = Submission("1", "submission.csv", "old", "baseline", "complete", "0.1", None)
    new = Submission("2", "submission.csv", "new", "baseline", "pending", None, None)

    selected = choose_submission(
        [new, old],
        previous_refs={"1"},
        file_name="submission.csv",
        message="baseline",
    )

    assert selected == new


def test_submit_waits_for_score_and_appends_ledger(tmp_path: Path) -> None:
    competition = "sample-competition"
    workspace = tmp_path / "competitions" / competition
    (workspace / "strategy").mkdir(parents=True)
    (workspace / "competition.toml").write_text(
        """
[competition]
slug = "sample-competition"

[submission]
id_columns = ["id"]
prediction_columns = ["prediction"]
expected_rows = 2
""".strip(),
        encoding="utf-8",
    )
    submission_path = tmp_path / "submission.csv"
    submission_path.write_text("id,prediction\n1,0.2\n2,0.8\n", encoding="utf-8")

    pending = [
        {
            "ref": "new-ref",
            "fileName": "submission.csv",
            "date": "2026-08-26",
            "description": "baseline",
            "status": "pending",
            "publicScore": None,
            "privateScore": None,
        }
    ]
    complete = [{**pending[0], "status": "complete", "publicScore": "0.8123"}]
    submission_responses = iter([[], pending, complete])

    def runner(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if command[:3] == ["kaggle", "competitions", "submissions"]:
            return _completed(command, json.dumps(next(submission_responses)))
        if command[:3] == ["kaggle", "competitions", "submit"]:
            return _completed(command, "Successfully submitted")
        if command[:3] == ["git", "rev-parse", "--short"]:
            return _completed(command, "abc123\n")
        raise AssertionError(f"unexpected command: {command}")

    clock = [0.0]

    def monotonic() -> float:
        return clock[0]

    def sleep(seconds: float) -> None:
        clock[0] += seconds

    result, ledger = submit_and_wait(
        tmp_path,
        competition,
        file_path=submission_path,
        experiment_id="EXP-0001",
        message="baseline",
        timeout=30,
        poll_interval=1,
        runner=runner,
        monotonic=monotonic,
        sleep=sleep,
    )

    assert result.public_score == "0.8123"
    record = json.loads(ledger.read_text(encoding="utf-8"))
    assert record["experiment_id"] == "EXP-0001"
    assert record["commit"] == "abc123"
    assert record["public_score"] == "0.8123"
