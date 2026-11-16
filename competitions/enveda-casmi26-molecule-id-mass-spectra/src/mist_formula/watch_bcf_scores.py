"""Record Kaggle's eventual B/C/F scores without creating new submissions."""

from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASMI = ROOT.parent
DATA = ROOT / "data/derived/mist-bcf-20261110"
STATUS_FILE = DATA / "score-watch-status.json"
LEDGER = ROOT / "strategy/lb-submissions.jsonl"
AUTH = ROOT / "src/mist_formula/kaggle_auth_exec.py"
REFS = {"B": "58205324", "C": "58208249", "F": "58206790"}


def latest_records() -> dict[str, dict]:
    result = {}
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["ref"])] = row
    return result


def status(ref: str) -> dict[str, str]:
    proc = subprocess.run(
        [
            "python3",
            str(AUTH),
            "uv",
            "run",
            "--with",
            "kaggle==2.2.5",
            "kaggle-lb",
            "status",
            "enveda-casmi26-molecule-id-mass-spectra",
            "--ref",
            ref,
        ],
        cwd=CASMI,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError(proc.stderr[-500:])
    return dict(line.split(": ", 1) for line in proc.stdout.splitlines() if ": " in line)


def record(state: str, **fields: object) -> None:
    STATUS_FILE.write_text(
        json.dumps({"time": datetime.now(UTC).isoformat(), "state": state, **fields}, indent=2)
    )


def append_score(ref: str, result: dict[str, str]) -> bool:
    old = latest_records()[ref]
    if old.get("public_score") is not None:
        return False
    updated = dict(old)
    updated["status"] = result["status"]
    updated["public_score"] = float(result["public_score"])
    updated["private_score"] = (
        None if result.get("private_score") == "not_available" else float(result["private_score"])
    )
    updated["recorded_at"] = datetime.now(UTC).isoformat()
    with LEDGER.open("a") as handle:
        handle.write(json.dumps(updated, ensure_ascii=False) + "\n")
    return True


def refresh_dashboard() -> None:
    proc = subprocess.run(
        ["uv", "run", "python", "src/dashboard/refresh.py"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode:
        raise RuntimeError("dashboard refresh failed: " + proc.stderr[-500:])


def main() -> None:
    for _ in range(288):  # at most 24 hours at five-minute intervals
        seen = {}
        changed = False
        for name, ref in REFS.items():
            result = status(ref)
            seen[name] = {
                "status": result.get("status"),
                "public_score": result.get("public_score"),
            }
            if result.get("status") == "SubmissionStatus.COMPLETE" and result.get(
                "public_score"
            ) not in (None, "not_available"):
                changed = append_score(ref, result) or changed
            elif result.get("status") not in (
                "SubmissionStatus.PENDING",
                "SubmissionStatus.RUNNING",
            ):
                raise RuntimeError(f"{name} submission failed: {result}")
        if changed:
            refresh_dashboard()
        record(
            "complete"
            if all(item["public_score"] not in (None, "not_available") for item in seen.values())
            else "waiting",
            scores=seen,
        )
        if all(item["public_score"] not in (None, "not_available") for item in seen.values()):
            return
        time.sleep(300)
    raise TimeoutError("B/C/F scoring did not finish within 24 hours")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        record("failed", error=repr(exc))
        raise
