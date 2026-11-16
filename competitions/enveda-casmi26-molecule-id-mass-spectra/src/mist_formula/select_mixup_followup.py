"""Select the fifth Mixup submission from completed public LB records.

This implements the rule recorded in docs/mist-spectrum-mixup-results-20261116.md
before the initial M1/M2/M3 scores were available. It only reads the ledger.
"""

from __future__ import annotations

import argparse
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "strategy/lb-submissions.jsonl"
REFERENCES = {
    "A": ("58077626", "EXP-0025-mist-12k"),
    "M1": ("58435231", "EXP-20261116-mixup-m1"),
    "M2": ("58434412", "EXP-20261116-mixup-m2"),
    "M3": ("58435174", "EXP-20261116-mixup-m3"),
}


def latest_by_ref(rows: list[dict]) -> dict[str, dict]:
    """Use the final ledger entry for each immutable Kaggle submission ref."""
    return {str(row["ref"]): row for row in rows}


def score(row: dict, name: str, experiment_id: str) -> Decimal:
    if row.get("experiment_id") != experiment_id:
        raise ValueError(f"{name}: submission ref belongs to a different experiment")
    if row.get("status") != "SubmissionStatus.COMPLETE":
        raise ValueError(f"{name}: public LB is not complete")
    raw = row.get("public_score")
    try:
        value = Decimal(str(raw))
    except InvalidOperation as exc:
        raise ValueError(f"{name}: invalid public score {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"{name}: non-finite public score")
    return value


def select(rows: list[dict]) -> dict:
    """Return the preregistered decision; never infer a score from PENDING rows."""
    latest = latest_by_ref(rows)
    scores: dict[str, Decimal] = {}
    for name, (ref, experiment_id) in REFERENCES.items():
        if ref not in latest:
            raise ValueError(f"{name}: submission ref {ref} is missing")
        scores[name] = score(latest[ref], name, experiment_id)

    eligible = [
        name
        for name in ("M2", "M3")
        if scores[name] > scores["A"] and scores[name] > scores["M1"]
    ]
    if eligible:
        # M2 wins an exact displayed-score tie, as preregistered.
        winner = max(eligible, key=lambda name: (scores[name], name == "M2"))
        selected = f"A75+{winner}25"
        reason = f"{winner} exceeds both A and M1 on the displayed public LB"
    else:
        winner = None
        selected = "weak_M1"
        reason = "Neither M2 nor M3 exceeds both A and M1 on the displayed public LB"
    return {
        "state": "ready",
        "decision_rule": "predeclared-mixup-fifth-submission-20261117",
        "selected": selected,
        "winner": winner,
        "reason": reason,
        "public_lb_displayed": {name: str(value) for name, value in scores.items()},
        "submission_refs": {name: ref for name, (ref, _) in REFERENCES.items()},
        "score_precision_note": (
            "Displayed Kaggle scores are rounded to three decimals; a tie is inconclusive."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.ledger.read_text().splitlines() if line.strip()]
    try:
        result = select(rows)
    except ValueError as exc:
        print(json.dumps({"state": "waiting", "reason": str(exc)}, ensure_ascii=False))
        raise SystemExit(2) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
