"""Watch the three submitted Mixup runs and record scores without resubmitting."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from watch_bcf_scores import append_score, refresh_dashboard, status

ROOT = Path(__file__).resolve().parents[2]
STATUS_FILE = ROOT / "data/derived/mist-mixup-20261116/score-watch-status.json"
REFS = {"M1": "58435231", "M2": "58434412", "M3": "58435174"}


def main() -> None:
    for _ in range(720):  # 12 hours at one-minute intervals
        results = {name: status(ref) for name, ref in REFS.items()}
        changed = False
        for name, ref in REFS.items():
            result = results[name]
            state = result.get("status")
            score = result.get("public_score")
            if state == "SubmissionStatus.COMPLETE" and score not in (None, "not_available"):
                changed = append_score(ref, result) or changed
            elif state not in ("SubmissionStatus.PENDING", "SubmissionStatus.RUNNING"):
                raise RuntimeError(f"{name} submission failed: {result}")
        if changed:
            refresh_dashboard()
        summary = {
            name: {"status": item.get("status"), "public_score": item.get("public_score")}
            for name, item in results.items()
        }
        complete = all(
            item["public_score"] not in (None, "not_available") for item in summary.values()
        )
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
        time.sleep(60)
    raise TimeoutError("Mixup scoring did not finish within 12 hours")


if __name__ == "__main__":
    main()
