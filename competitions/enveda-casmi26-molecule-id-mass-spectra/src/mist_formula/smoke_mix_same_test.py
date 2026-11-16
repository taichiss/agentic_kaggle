"""Locally reproduce A and M1 reranks on the fixed 400-row test parent."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bcf_rerank import rerank


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_hash(table: pd.DataFrame) -> str:
    members = "\n".join(
        str(mol_id) + "\t" + ";".join(sorted(smiles.split(";")))
        for mol_id, smiles in zip(table.molecule_id, table.smiles, strict=True)
    )
    return hashlib.sha256(members.encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "test", "A-submission", "A-checkpoint", "M1-checkpoint",
                 "frozen-model", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--condition-name", default="M1")
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.frozen_model.parent))
    from model import score_candidates  # noqa: PLC0415

    control = pd.read_csv(args.control, dtype=str, keep_default_na=False)
    expected_a = pd.read_csv(args.A_submission, dtype=str, keep_default_na=False)
    if list(control.columns) != ["molecule_id", "smiles"] or len(control) != 400:
        raise AssertionError("fixed 400-row parent changed")
    if not control.molecule_id.equals(expected_a.molecule_id):
        raise AssertionError("saved A submission IDs differ from parent")
    spectra = pd.read_parquet(args.test)
    groups = {
        str(key): group.to_dict("records")
        for key, group in spectra.groupby("molecule_id", sort=False)
    }
    output = control.copy()
    changes = []
    a_reproduced = []
    for row in control.itertuples(index=False):
        names = row.smiles.split(";")
        if len(names) == 1:
            a_reproduced.append(row.smiles)
            continue
        group = groups.get(str(row.molecule_id))
        if not group:
            raise AssertionError(f"test spectra missing for {row.molecule_id}")
        scores_a = np.asarray(score_candidates(group, names, args.A_checkpoint, "cuda"), float)
        scores_m1 = np.asarray(score_candidates(group, names, args.M1_checkpoint, "cuda"), float)
        order_a, audit_a = rerank(names, scores_a, weight=0.25)
        order_m1, audit_m1 = rerank(names, scores_m1, weight=0.25)
        result_a = [names[index] for index in order_a]
        result_m1 = [names[index] for index in order_m1]
        if result_a[0] != names[0] or result_m1[0] != names[0]:
            raise AssertionError("first candidate changed")
        if sorted(result_a) != sorted(names) or sorted(result_m1) != sorted(names):
            raise AssertionError("candidate membership changed")
        a_reproduced.append(";".join(result_a))
        output.loc[output.molecule_id == row.molecule_id, "smiles"] = ";".join(result_m1)
        changes.append({"molecule_id": row.molecule_id,
                        "A_vs_M1_changed": result_m1 != result_a,
                        "parent_vs_M1_changed": result_m1 != names,
                        "A_missing_groups": audit_a["missing_groups"],
                        "M1_missing_groups": audit_m1["missing_groups"]})
    if a_reproduced != expected_a.smiles.tolist():
        raise AssertionError("frozen A inference did not reproduce saved A submission")
    if candidate_hash(output) != candidate_hash(control):
        raise AssertionError("M1 candidate set changed")
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "submission.csv"
    output.to_csv(path, index=False)
    report = {"rows": len(output), "multi_candidate_rows": len(changes),
              "condition_name": args.condition_name,
              "control_sha256": sha(args.control),
              "test_sha256": sha(args.test),
              "A_checkpoint_sha256": sha(args.A_checkpoint),
              "M1_checkpoint_sha256": sha(args.M1_checkpoint),
              "frozen_model_sha256": sha(args.frozen_model),
              "A_saved_sha256": sha(args.A_submission),
              "M1_submission_sha256": sha(path),
              "M1_differs_from_A": output.smiles.tolist() != expected_a.smiles.tolist(),
              "candidate_set_sha256": candidate_hash(output),
              "first_candidate_unchanged": True,
              "A_vs_M1_changed_rows": sum(row["A_vs_M1_changed"] for row in changes),
              "parent_vs_M1_changed_rows": sum(row["parent_vs_M1_changed"] for row in changes),
              "M1_missing_groups": sum(row["M1_missing_groups"] for row in changes)}
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    (args.output / "per-molecule.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in changes)
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
