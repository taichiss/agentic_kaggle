from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import time
import tomllib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_kaggle.paths import find_repo_root

Runner = Callable[..., subprocess.CompletedProcess[str]]
SUCCESS_STATUSES = {"complete", "completed"}
FAILURE_STATUSES = {"error", "failed", "cancelled", "canceled"}


@dataclass(frozen=True)
class Submission:
    ref: str
    file_name: str
    date: str
    description: str
    status: str
    public_score: str | None
    private_score: str | None

    @classmethod
    def from_api(cls, item: dict[str, Any]) -> Submission:
        return cls(
            ref=str(item.get("ref", "")),
            file_name=str(item.get("fileName", "")),
            date=str(item.get("date", "")),
            description=str(item.get("description", "")),
            status=str(item.get("status", "")),
            public_score=_optional_text(item.get("publicScore")),
            private_score=_optional_text(item.get("privateScore")),
        )


def _optional_text(value: Any) -> str | None:
    if value is None or str(value).strip() in {"", "None", "null"}:
        return None
    return str(value)


def load_manifest(root: Path, competition: str) -> tuple[Path, dict[str, Any]]:
    workspace = root / "competitions" / competition
    manifest_path = workspace / "competition.toml"
    if not manifest_path.is_file():
        raise ValueError(f"competition manifest was not found: {manifest_path}")
    with manifest_path.open("rb") as file:
        manifest = tomllib.load(file)
    manifest_slug = manifest.get("competition", {}).get("slug")
    if manifest_slug != competition:
        raise ValueError(
            f"competition slug mismatch: argument={competition!r}, manifest={manifest_slug!r}"
        )
    return workspace, manifest


def validate_submission(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"submission file was not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"submission file is empty: {path}")

    submission_contract = manifest.get("submission", {})
    required_columns = [
        *submission_contract.get("id_columns", []),
        *submission_contract.get("prediction_columns", []),
    ]
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError(f"submission CSV has no header: {path}") from exc
        row_count = sum(1 for _ in reader)

    if not header or any(not column.strip() for column in header):
        raise ValueError("submission CSV contains an empty column name")
    if row_count == 0:
        raise ValueError("submission CSV has no prediction rows")
    missing = [column for column in required_columns if column not in header]
    if missing:
        raise ValueError(f"submission CSV is missing manifest columns: {missing}")

    expected_rows = submission_contract.get("expected_rows", 0)
    if expected_rows and row_count != expected_rows:
        raise ValueError(
            f"submission row count mismatch: expected {expected_rows}, found {row_count}"
        )
    return {"columns": header, "rows": row_count, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_kaggle(args: Sequence[str], runner: Runner = subprocess.run) -> str:
    command = ["kaggle", *args]
    try:
        result = runner(command, text=True, capture_output=True, check=False)
    except FileNotFoundError as exc:
        raise RuntimeError("Kaggle CLI is unavailable; run `uv sync` first") from exc
    if result.returncode != 0:
        details = (result.stderr or result.stdout).strip()
        raise RuntimeError(f"Kaggle CLI failed ({' '.join(command)}): {details}")
    return result.stdout


def fetch_submissions(competition: str, runner: Runner = subprocess.run) -> list[Submission]:
    output = _run_kaggle(
        [
            "competitions",
            "submissions",
            competition,
            "--format",
            "json",
            "--page-size",
            "200",
            "--quiet",
        ],
        runner,
    ).strip()
    if not output or output == "No submissions found":
        return []
    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Kaggle CLI returned invalid JSON: {output[:300]}") from exc
    if not isinstance(payload, list):
        raise RuntimeError("Kaggle CLI submissions response must be a JSON list")
    return [Submission.from_api(item) for item in payload]


def choose_submission(
    submissions: Sequence[Submission],
    *,
    previous_refs: set[str] | None = None,
    file_name: str | None = None,
    message: str | None = None,
    submission_ref: str | None = None,
) -> Submission | None:
    candidates = list(submissions)
    if submission_ref is not None:
        candidates = [item for item in candidates if item.ref == submission_ref]
    if previous_refs is not None:
        new_candidates = [item for item in candidates if item.ref not in previous_refs]
        if new_candidates:
            candidates = new_candidates
    if file_name is not None:
        matching_file = [item for item in candidates if item.file_name == Path(file_name).name]
        if matching_file:
            candidates = matching_file
    if message is not None:
        matching_message = [item for item in candidates if item.description == message]
        if matching_message:
            candidates = matching_message
    return candidates[0] if candidates else None


def wait_for_submission(
    competition: str,
    *,
    previous_refs: set[str],
    file_name: str,
    message: str,
    timeout: float,
    poll_interval: float,
    runner: Runner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Submission:
    deadline = monotonic() + timeout
    last_submission: Submission | None = None
    while True:
        submissions = fetch_submissions(competition, runner)
        current = choose_submission(
            submissions,
            previous_refs=previous_refs,
            file_name=file_name,
            message=message,
        )
        if current is not None:
            last_submission = current
            normalized_status = current.status.lower()
            if normalized_status in SUCCESS_STATUSES or current.public_score is not None:
                return current
            if normalized_status in FAILURE_STATUSES:
                return current
        if monotonic() >= deadline:
            return Submission(
                ref="" if last_submission is None else last_submission.ref,
                file_name=Path(file_name).name,
                date="" if last_submission is None else last_submission.date,
                description=message,
                status="timeout",
                public_score=None if last_submission is None else last_submission.public_score,
                private_score=None if last_submission is None else last_submission.private_score,
            )
        sleep(poll_interval)


def append_ledger(
    workspace: Path,
    *,
    experiment_id: str,
    submission: Submission,
    file_path: Path,
    file_metadata: dict[str, Any],
    commit: str,
) -> Path:
    ledger_path = workspace / "strategy" / "lb-submissions.jsonl"
    record = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "commit": commit,
        "file": str(file_path),
        "file_sha256": file_metadata["sha256"],
        "rows": file_metadata["rows"],
        **asdict(submission),
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return ledger_path


def _git_commit(root: Path, runner: Runner = subprocess.run) -> str:
    result = runner(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def submit_and_wait(
    root: Path,
    competition: str,
    *,
    file_path: Path,
    experiment_id: str,
    message: str,
    timeout: float,
    poll_interval: float,
    runner: Runner = subprocess.run,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[Submission, Path]:
    workspace, manifest = load_manifest(root, competition)
    file_path = file_path.resolve()
    file_metadata = validate_submission(file_path, manifest)

    before = fetch_submissions(competition, runner)
    previous_refs = {submission.ref for submission in before}
    _run_kaggle(
        [
            "competitions",
            "submit",
            competition,
            "--file",
            str(file_path),
            "--message",
            message,
        ],
        runner,
    )
    submission = wait_for_submission(
        competition,
        previous_refs=previous_refs,
        file_name=file_path.name,
        message=message,
        timeout=timeout,
        poll_interval=poll_interval,
        runner=runner,
        monotonic=monotonic,
        sleep=sleep,
    )
    ledger = append_ledger(
        workspace,
        experiment_id=experiment_id,
        submission=submission,
        file_path=file_path,
        file_metadata=file_metadata,
        commit=_git_commit(root, runner),
    )
    return submission, ledger


def _print_submission(submission: Submission) -> None:
    print(f"ref: {submission.ref}")
    print(f"status: {submission.status}")
    print(f"public_score: {submission.public_score or 'not_available'}")
    print(f"private_score: {submission.private_score or 'not_available'}")
    print(f"file: {submission.file_name}")
    print(f"message: {submission.description}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-lb", description="Submit to Kaggle and retrieve Leaderboard scores."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="submit a file and wait for scoring")
    submit_parser.add_argument("competition", help="competition slug registered in competitions/")
    submit_parser.add_argument("--file", required=True, type=Path, help="submission CSV path")
    submit_parser.add_argument(
        "--experiment-id", required=True, help="experiment ledger identifier"
    )
    submit_parser.add_argument("--message", required=True, help="Kaggle submission description")
    submit_parser.add_argument(
        "--timeout", type=float, default=43200, help="wait timeout in seconds"
    )
    submit_parser.add_argument(
        "--poll-interval", type=float, default=30, help="status polling interval in seconds"
    )

    status_parser = subparsers.add_parser(
        "status", help="show a submission score without submitting"
    )
    status_parser.add_argument("competition", help="competition slug registered in competitions/")
    selection = status_parser.add_mutually_exclusive_group()
    selection.add_argument("--ref", dest="submission_ref", help="Kaggle submission ref")
    selection.add_argument("--latest", action="store_true", help="show the latest submission")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = find_repo_root()
    try:
        load_manifest(root, args.competition)
        if args.command == "status":
            submission = choose_submission(
                fetch_submissions(args.competition), submission_ref=args.submission_ref
            )
            if submission is None:
                raise RuntimeError("no matching Kaggle submission was found")
            _print_submission(submission)
            return 0

        submission, ledger = submit_and_wait(
            root,
            args.competition,
            file_path=args.file,
            experiment_id=args.experiment_id,
            message=args.message,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    _print_submission(submission)
    print(f"ledger: {ledger}")
    if submission.status.lower() in FAILURE_STATUSES | {"timeout"}:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
