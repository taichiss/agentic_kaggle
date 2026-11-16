"""Compare which held-out molecules M1, M2 and M3 help or hurt versus A."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_a(path: Path) -> dict[int, int]:
    rows = (json.loads(line) for line in path.read_text().splitlines())
    result = {int(row["mol_idx"]): int(row["A_rank"]) for row in rows}
    if len(result) != 1000:
        raise ValueError("Expected 1000 distinct A panel queries")
    return result


def load_treatment(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text())
    result = payload.get("validation", payload)
    ids = [int(i) for i in result["query_mols"]]
    ranks = [int(i) for i in result["ranks"]]
    if len(ids) != len(ranks) or len(ids) != len(set(ids)):
        raise ValueError(f"Repeated or missing query ranks: {path}")
    return dict(zip(ids, ranks, strict=True))


def run(paths: dict[str, Path], output: Path) -> dict:
    arrays = {"A": load_a(paths["A"])}
    for arm in ("M1", "M2", "M3"):
        arrays[arm] = load_treatment(paths[arm])
        if set(arrays[arm]) != set(arrays["A"]):
            raise ValueError(f"{arm} query molecules differ from A")
    ids = sorted(arrays["A"])
    ranks = {arm: np.asarray([values[i] for i in ids]) for arm, values in arrays.items()}
    gains = {arm: 1 / ranks[arm] - 1 / ranks["A"] for arm in ("M1", "M2", "M3")}
    improved = {arm: set(np.asarray(ids)[ranks[arm] < ranks["A"]]) for arm in gains}
    worsened = {arm: set(np.asarray(ids)[ranks[arm] > ranks["A"]]) for arm in gains}
    report = {
        "queries": len(ids),
        "source_paths": {name: str(path) for name, path in paths.items()},
        "arms": {
            arm: {
                "mrr25": float(np.mean(1 / values)),
                "top1": float(np.mean(values == 1)),
                "delta_mrr25_vs_A": float(np.mean(gains[arm])) if arm != "A" else 0.0,
                "improved": len(improved[arm]) if arm != "A" else 0,
                "worsened": len(worsened[arm]) if arm != "A" else 0,
            }
            for arm, values in ranks.items()
        },
        "pairwise": {},
    }
    for left, right in (("M1", "M2"), ("M1", "M3"), ("M2", "M3")):
        report["pairwise"][f"{left}_{right}"] = {
            "both_improved_vs_A": len(improved[left] & improved[right]),
            "only_left_improved": len(improved[left] - improved[right]),
            "only_right_improved": len(improved[right] - improved[left]),
            "left_improved_right_worsened": len(improved[left] & worsened[right]),
            "right_improved_left_worsened": len(improved[right] & worsened[left]),
            "mean_delta_reciprocal_rank_correlation": float(
                np.corrcoef(gains[left], gains[right])[0, 1]
            ),
        }
    report["largest_arm_disagreements"] = [
        {
            "mol_idx": int(ids[i]),
            "A_rank": int(ranks["A"][i]),
            "M1_rank": int(ranks["M1"][i]),
            "M2_rank": int(ranks["M2"][i]),
            "M3_rank": int(ranks["M3"][i]),
        }
        for i in np.argsort(-np.ptp(np.stack([ranks[x] for x in ("M1", "M2", "M3")]), axis=0))[:20]
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("A", "M1", "M2", "M3"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run({name: getattr(args, name) for name in ("A", "M1", "M2", "M3")}, args.output)


if __name__ == "__main__":
    main()
