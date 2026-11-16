"""Verify a complete private Kaggle Mixup Notebook against its staged artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        table = csv.DictReader(stream)
        if table.fieldnames != ["molecule_id", "smiles"]:
            raise ValueError(f"Unexpected CSV columns in {path}")
        return list(table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--staged-notebook", type=Path, required=True)
    parser.add_argument("--staged-checkpoint", type=Path, required=True)
    parser.add_argument("--remote-output", type=Path, required=True)
    parser.add_argument("--local-expected-submission", type=Path, required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = json.loads(args.stage_receipt.read_text())
    build = json.loads(args.build_receipt.read_text())
    model = build["model"]
    remote = args.remote_output
    submission_path = remote / "submission.csv"
    control_path = remote / f"{model}-control.csv"
    audit_path = remote / f"{model}-mixup-audit.json"
    overlay_path = remote / f"{model}-overlay-report.json"
    audit = json.loads(audit_path.read_text())
    overlay = json.loads(overlay_path.read_text())
    submitted = rows(submission_path)
    control = rows(control_path)
    if (
        stage["checkpoint_sha256"] != build["checkpoint_sha256"]
        or sha(args.staged_checkpoint) != stage["checkpoint_sha256"]
        or sha(args.staged_notebook) != stage["notebook_sha256"]
        or stage["notebook_sha256"] != build["final_notebook_sha256"]
        or build["dataset"] != stage["dataset_id"]
        or build["kernel"] != stage["kernel_id"]
        or build.get("experiment_id") != stage.get("experiment_id")
        or len(submitted) != len(control)
        or len(control) != 400
        or [row["molecule_id"] for row in submitted]
        != [row["molecule_id"] for row in control]
    ):
        raise AssertionError("Staged checkpoint/notebook or remote CSV shape differs")
    members = "\n".join(
        row["molecule_id"] + "\t" + ";".join(sorted(row["smiles"].split(";")))
        for row in control
    )
    candidate_sha = hashlib.sha256(members.encode()).hexdigest()
    changed = 0
    for old, new in zip(control, submitted, strict=True):
        before, after = old["smiles"].split(";"), new["smiles"].split(";")
        if sorted(before) != sorted(after) or before[0] != after[0]:
            raise AssertionError("Candidate set or top-ranked structure changed")
        changed += before != after
    if (
        candidate_sha != build["candidate_set_sha256"]
        or sha(control_path) != build["parent_output_sha256"]
        or sha(submission_path) != sha(args.local_expected_submission)
        or audit["parent_notebook_sha256"] != build["control_sha256"]
        or audit.get("experiment_id") != stage.get("experiment_id")
        or audit["parent_output_sha256"] != sha(control_path)
        or audit["candidate_set_sha256"] != candidate_sha
        or audit["checkpoint_sha256"] != stage["checkpoint_sha256"]
        or audit["inference_sha256"] != build["module_sha256"]
        or audit["rank_fusion_weight"] != stage["rank_fusion_weight"]
        or audit["submission_sha256"] != sha(submission_path)
        or audit["rows"] != 400
        or audit["changed"] != changed
        or audit["top1_changed"] != 0
        or audit["errors"] != 0
        or overlay["errors"] != []
        or overlay["submission_sha256"] != sha(submission_path)
        or overlay["checkpoint_sha256"] != stage["checkpoint_sha256"]
        or overlay["protected_top_n"] != 1
        or not overlay["within_formula"]
    ):
        raise AssertionError("Remote audit, parent, or local output disagrees")
    report = {
        "kernel_id": stage["kernel_id"],
        "experiment_id": stage.get("experiment_id"),
        "dataset_id": stage["dataset_id"],
        "kernel_version": args.version,
        "kernel_status": "COMPLETE",
        "competition_submitted": False,
        "rows": len(submitted),
        "changed_from_parent": changed,
        "top1_changed": 0,
        "errors": 0,
        "parent_notebook_sha256": build["control_sha256"],
        "parent_output_sha256": sha(control_path),
        "candidate_set_sha256": candidate_sha,
        "checkpoint_sha256": sha(args.staged_checkpoint),
        "staged_notebook_sha256": sha(args.staged_notebook),
        "submission_sha256": sha(submission_path),
        "local_expected_submission_sha256": sha(args.local_expected_submission),
        "remote_audit_sha256": sha(audit_path),
        "remote_overlay_report_sha256": sha(overlay_path),
        "verification": "all hashes, row-level candidates/top1, and remote audit fields agree",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
