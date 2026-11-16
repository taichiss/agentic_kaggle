"""Watch newly submitted Mixup variants and record their eventual Kaggle scores."""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from watch_bcf_scores import append_score, latest_records, refresh_dashboard, status

ROOT = Path(__file__).resolve().parents[2]
STATUS_FILE = ROOT / "data/derived/mist-mixup-20261116/adaptive-score-watch.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", action="append", required=True)
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--max-hours", type=float, default=12)
    args = parser.parse_args()
    if args.interval < 10 or args.max_hours <= 0:
        parser.error("interval must be >=10 seconds and max-hours must be positive")
    latest = latest_records()
    chosen = {}
    for row in latest.values():
        if row.get("experiment_id") in args.experiment_id:
            chosen[row["experiment_id"]] = str(row["ref"])
    missing = set(args.experiment_id) - set(chosen)
    if missing:
        parser.error(f"No submission record for {sorted(missing)}")
    deadline = time.monotonic() + args.max_hours * 3600
    while time.monotonic() < deadline:
        results = {name: status(ref) for name, ref in chosen.items()}
        changed = False
        for name, ref in chosen.items():
            result = results[name]
            state, score = result.get("status"), result.get("public_score")
            if state == "SubmissionStatus.COMPLETE" and score not in (None, "not_available"):
                changed = append_score(ref, result) or changed
            elif state not in ("SubmissionStatus.PENDING", "SubmissionStatus.RUNNING"):
                raise RuntimeError(f"{name}: {result}")
        if changed:
            refresh_dashboard()
        summary = {
            name: {"status": item.get("status"), "public_score": item.get("public_score")}
            for name, item in results.items()
        }
        complete = all(
            item["public_score"] not in (None, "not_available") for item in summary.values()
        )
        STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATUS_FILE.write_text(
            json.dumps(
                {
                    "time": datetime.now(UTC).isoformat(),
                    "state": "complete" if complete else "waiting",
                    "scores": summary,
                },
                indent=2,
            )
        )
        print(json.dumps(summary), flush=True)
        if complete:
            return
        time.sleep(args.interval)
    raise TimeoutError("Adaptive Mixup scores did not finish before the deadline")


if __name__ == "__main__":
    main()
