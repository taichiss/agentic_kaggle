"""Idempotently register the post-hoc four-source multi-candidate panel in CV history."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

SPLIT_ID = "natural-sources-5425-variable-candidates-posthoc-v1"
EXPERIMENTS = {
    "A": "EXP-20261110-bcf-a",
    "M1": "EXP-20261116-mixup-m1",
    "M2": "EXP-20261116-mixup-m2",
    "M3": "EXP-20261116-mixup-m3",
    "WEAK": "EXP-20261117-mixup-m1-weak-085095",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--followup", type=Path, required=True)
    parser.add_argument("--cv-results", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    followup = json.loads(args.followup.read_text())
    primary = report["metrics"]["multi_candidate_primary"]
    audit = followup["audit"]
    if (primary["molecule_source_pairs"] != 5425
            or primary["unique_structure_keys"] != 4108
            or audit["train_query_structure_key_overlap"] != 0
            or report["query_training_structure_overlap"] != 0
            or report["metrics"]["source_x_timsTOF_only"]["enveda-np-examples"][
                "molecule_source_pairs"
            ] != 24):
        raise AssertionError("natural-source audit differs from the fixed panel")
    note = (
        "Post-hoc supplementary four-source panel on the molecule-disjoint validation split. "
        "Only 5425 multi-candidate molecule-source pairs are scored; 1687 forced-rank-1 "
        "singletons excluded. Candidate lists vary from 2 to 25 same-full-formula cache "
        "structures and are not the hidden competition candidates. Bootstrap resamples "
        "structure keys, retaining all source measurements together. Query/true-label "
        "train structure overlap 0; train structures can occur as decoy candidates only. "
        "Only 24 multi-candidate timsTOF queries occur, all from enveda-np-examples. "
        "Do not compare absolute MRR with earlier fixed-25 panels or Public LB."
    )
    entries = []
    for name, experiment_id in EXPERIMENTS.items():
        if name == "A":
            value = primary["A_mrr25"]
            top1 = primary["A_top1"]
            delta = 0.0
            interval = [0.0, 0.0]
        else:
            metrics = primary[name]
            value = metrics["mrr25"]
            top1 = metrics["top1"]
            delta = metrics["delta_mrr25_vs_A"]
            interval = metrics["delta_cluster_bootstrap_95pct"]
        entries.append({
            "experiment_id": experiment_id,
            "split_id": SPLIT_ID,
            "metric": "MRR@25",
            "value": value,
            "comparison_kind": "synthetic_diagnostic",
            "comparison_note": note,
            "metrics_sha256": sha(args.report),
            "followup_sha256": sha(args.followup),
            "panel_sha256": report["panel_sha256"],
            "checkpoint_sha256": report["checkpoints_sha256"][name],
            "n_groups": primary["molecule_source_pairs"],
            "n_unique_structures": primary["unique_structure_keys"],
            "single_candidate_excluded": 1687,
            "top1": top1,
            "baseline_value": primary["A_mrr25"],
            "delta_mrr25_vs_A": delta,
            "paired_bootstrap_95pct_delta_vs_A": interval,
            "paired_bootstrap_unit": "structure_key",
            "paired_bootstrap_seed": 20261117,
            "timsTOF_multi_candidate_queries": 24,
        })
    written = []
    with args.cv_results.open("a+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        stream.seek(0)
        previous = stream.read()
        existing = {
            (row["experiment_id"], row.get("split_id"), row.get("metric")): row
            for line in previous.splitlines()
            if (row := json.loads(line))
        }
        stream.seek(0, 2)
        if previous and not previous.endswith("\n"):
            stream.write("\n")
        for entry in entries:
            key = (entry["experiment_id"], entry["split_id"], entry["metric"])
            if key in existing:
                if existing[key] != entry:
                    raise ValueError(f"conflicting CV result exists for {key}")
                continue
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written.append(entry["experiment_id"])
        stream.flush()
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    receipt = {"split_id": SPLIT_ID, "written": written,
               "cv_results_sha256": sha(args.cv_results), "entries": len(entries)}
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
