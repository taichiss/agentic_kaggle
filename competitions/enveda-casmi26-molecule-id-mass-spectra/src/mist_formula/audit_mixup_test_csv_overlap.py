"""Audit local test-submission rank changes and overlap without hidden labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

EXPECTED_PARENT_SHA = "39309f6c46d25e9c934dad8871d211ad8afebda07f4441c596714cbe7b6fd9b2"
EXPECTED_CANDIDATE_SET_SHA = "e8de1f1b07ab6d56685805d01878185241a9a06bc1603b9fef13e537fc408542"
CONDITIONS = ("A", "M1", "M2", "M3", "M4", "WEAK_M1")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_submission(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["molecule_id", "smiles"]:
            raise ValueError(f"Unexpected submission schema: {path}")
        rows = list(reader)
    if len(rows) != 400 or len({row["molecule_id"] for row in rows}) != 400:
        raise ValueError(f"Expected 400 unique molecule IDs: {path}")
    if any(not row["smiles"] for row in rows):
        raise ValueError(f"Empty candidate ranking: {path}")
    return rows


def candidate_set_sha(rows: list[dict[str, str]]) -> str:
    members = "\n".join(
        row["molecule_id"] + "\t" + ";".join(sorted(row["smiles"].split(";"))) for row in rows
    )
    return hashlib.sha256(members.encode()).hexdigest()


def occurrence_tokens(smiles: str) -> list[tuple[str, int]]:
    seen: dict[str, int] = defaultdict(int)
    tokens = []
    for value in smiles.split(";"):
        tokens.append((value, seen[value]))
        seen[value] += 1
    return tokens


def movement(original: str, changed: str) -> dict | None:
    if original == changed:
        return None
    base = occurrence_tokens(original)
    final = occurrence_tokens(changed)
    if Counter(base) != Counter(final):
        raise ValueError("Candidate multiset changed")
    where = {value: index + 1 for index, value in enumerate(final)}
    permutation = [where[value] for value in base]
    shifts = [abs(index + 1 - position) for index, position in enumerate(permutation)]
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return {
        "first_changed_rank": next(
            index + 1
            for index, (left, right) in enumerate(zip(base, final, strict=True))
            if left != right
        ),
        "moved_candidates": sum(shift > 0 for shift in shifts),
        "sum_abs_rank_shift": sum(shifts),
        "max_abs_rank_shift": max(shifts),
        "pairwise_rank_inversions": inversions,
        "permutation_from_A": permutation,
    }


def summarize_changes(details: list[dict]) -> dict:
    if not details:
        return {"changed_rows": 0, "sum_abs_rank_shift": 0, "pairwise_rank_inversions": 0}
    return {
        "changed_rows": len(details),
        "sum_abs_rank_shift": sum(item["sum_abs_rank_shift"] for item in details),
        "mean_abs_rank_shift_per_changed_row": sum(item["sum_abs_rank_shift"] for item in details)
        / len(details),
        "max_abs_rank_shift": max(item["max_abs_rank_shift"] for item in details),
        "pairwise_rank_inversions": sum(item["pairwise_rank_inversions"] for item in details),
        "first_changed_rank_min": min(item["first_changed_rank"] for item in details),
    }


def run(args: argparse.Namespace) -> dict:
    parsed = [value.split("=", 1) for value in args.csv]
    if len(parsed) != len(CONDITIONS) or {name for name, _ in parsed} != set(CONDITIONS):
        raise ValueError("Provide exactly A, M1, M2, M3, M4 and WEAK_M1")
    paths = {name: Path(path) for name, path in parsed}
    if sha(args.parent_control) != EXPECTED_PARENT_SHA:
        raise ValueError("Parent output is not the frozen 0.477 control")
    parent = read_submission(args.parent_control)
    rows = {name: read_submission(paths[name]) for name in CONDITIONS}
    parent_ids = [row["molecule_id"] for row in parent]
    if candidate_set_sha(parent) != EXPECTED_CANDIDATE_SET_SHA:
        raise ValueError("Parent candidate set differs")
    for name, ranked in rows.items():
        if [row["molecule_id"] for row in ranked] != parent_ids:
            raise ValueError(f"{name} molecule ordering differs from parent")
        if candidate_set_sha(ranked) != EXPECTED_CANDIDATE_SET_SHA:
            raise ValueError(f"{name} candidate multiset differs from parent")
        for base, candidate in zip(parent, ranked, strict=True):
            if base["smiles"].split(";")[0] != candidate["smiles"].split(";")[0]:
                raise ValueError(f"{name} changed protected top1 at {base['molecule_id']}")

    changed: dict[str, set[str]] = {}
    per_molecule: dict[str, dict] = {}
    summaries = {}
    for name in CONDITIONS[1:]:
        changed[name] = set()
        details = []
        for base, candidate in zip(rows["A"], rows[name], strict=True):
            difference = movement(base["smiles"], candidate["smiles"])
            if difference is None:
                continue
            mid = base["molecule_id"]
            changed[name].add(mid)
            details.append(difference)
            per_molecule.setdefault(
                mid,
                {"molecule_id": mid, "candidate_count": len(base["smiles"].split(";"))},
            )[name] = difference
        summaries[name] = {**summarize_changes(details), "changed_ids_vs_A": sorted(changed[name])}

    pairwise = {}
    for left, right in combinations(CONDITIONS[1:], 2):
        shared = changed[left] & changed[right]
        union = changed[left] | changed[right]
        exact_same = [
            row_left["molecule_id"]
            for row_left, row_right in zip(rows[left], rows[right], strict=True)
            if row_left["smiles"] == row_right["smiles"]
        ]
        direct_movement = [
            difference
            for row_left, row_right in zip(rows[left], rows[right], strict=True)
            if (difference := movement(row_left["smiles"], row_right["smiles"])) is not None
        ]
        pairwise[left + "__" + right] = {
            "both_changed_vs_A": len(shared),
            "left_only_changed_vs_A": len(changed[left] - changed[right]),
            "right_only_changed_vs_A": len(changed[right] - changed[left]),
            "changed_set_jaccard": len(shared) / len(union) if union else None,
            "identical_ranking_rows": len(exact_same),
            "different_ranking_rows": 400 - len(exact_same),
            "identical_among_both_changed": sum(mid in exact_same for mid in shared),
            "direct_sum_abs_rank_shift": sum(
                item["sum_abs_rank_shift"] for item in direct_movement
            ),
            "direct_rank_inversions": sum(
                item["pairwise_rank_inversions"] for item in direct_movement
            ),
        }

    focus = {
        **pairwise["M4__WEAK_M1"],
        "shared_changed_ids": sorted(changed["M4"] & changed["WEAK_M1"]),
        "M4_only_changed_ids": sorted(changed["M4"] - changed["WEAK_M1"]),
        "WEAK_M1_only_changed_ids": sorted(changed["WEAK_M1"] - changed["M4"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail = args.output.parent / "per-molecule.jsonl"
    detail.write_text("".join(json.dumps(per_molecule[mid]) + "\n" for mid in sorted(per_molecule)))
    report = {
        "scope": "Unlabeled 400-test submission ranking audit; no accuracy or LB claim",
        "parent_output_sha256": EXPECTED_PARENT_SHA,
        "candidate_set_sha256": EXPECTED_CANDIDATE_SET_SHA,
        "input_csv_sha256": {name: sha(path) for name, path in paths.items()},
        "molecules": 400,
        "all_conditions_same_ids_candidates_and_top1": True,
        "A_changed_from_parent": sum(
            a["smiles"] != p["smiles"] for a, p in zip(rows["A"], parent, strict=True)
        ),
        "conditions_vs_A": summaries,
        "pairwise": pairwise,
        "M4_vs_WEAK_M1": focus,
        "union_changed_ids_vs_A": sorted(per_molecule),
        "per_molecule_detail": detail.name,
        "per_molecule_detail_sha256": sha(detail),
        "interpretation_limit": (
            "Only ranking diversity is observable. Hidden correct structures are unavailable; "
            "a larger or more distinct movement does not imply higher accuracy."
        ),
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summary": summaries, "M4_vs_WEAK_M1": focus}))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-control", type=Path, required=True)
    parser.add_argument("--csv", action="append", required=True, help="NAME=CSV")
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
