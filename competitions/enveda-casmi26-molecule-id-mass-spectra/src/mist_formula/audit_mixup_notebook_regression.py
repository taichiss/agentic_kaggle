"""Rebuild staged Mixup Notebooks in separate directories and compare bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from build_mixup_notebook import build, sha
from stage_mixup_submission import BASE, INFERENCE, PARENT, PARENT_METADATA


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checks = []
    for label in ("m1", "m2", "m3", "m1weak085095", "m1strictshared"):
        stage = json.loads((BASE / f"stage-{label}.json").read_text())
        old_notebook = Path(stage["notebook_dir"]) / "trained-model-overlay.ipynb"
        checkpoint = Path(stage["dataset_dir"]) / "final.pt"
        rebuilt_dir = args.output / label
        rebuilt = build(SimpleNamespace(
            parent=PARENT,
            parent_metadata=PARENT_METADATA,
            inference_module=INFERENCE,
            checkpoint=checkpoint,
            dataset=stage["dataset_id"],
            kernel=stage["kernel_id"],
            model_name=f"mist-mixup-{label}",
            output=rebuilt_dir,
            weight=stage["rank_fusion_weight"],
            experiment_id=stage.get("experiment_id"),
        ))
        record = {
            "label": label,
            "experiment_id": stage.get("experiment_id"),
            "expected_notebook_sha256": stage["notebook_sha256"],
            "staged_notebook_sha256": sha(old_notebook),
            "rebuilt_notebook_sha256": rebuilt["final_notebook_sha256"],
            "rebuilt_path": str(rebuilt_dir / "trained-model-overlay.ipynb"),
        }
        record["exact_match"] = (
            record["expected_notebook_sha256"]
            == record["staged_notebook_sha256"]
            == record["rebuilt_notebook_sha256"]
        )
        checks.append(record)
        print(json.dumps(record), flush=True)
    report = {"all_exact": all(item["exact_match"] for item in checks),
              "checks": checks}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["all_exact"]:
        raise AssertionError("At least one Notebook changed after regeneration")


if __name__ == "__main__":
    main()
