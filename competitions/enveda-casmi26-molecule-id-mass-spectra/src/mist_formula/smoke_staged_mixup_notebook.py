"""Run the two staged overlay/audit cells against a saved local 400-row parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    parser.add_argument("--parent-control", type=Path, required=True)
    parser.add_argument("--competition-input", type=Path, required=True)
    parser.add_argument("--independent-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stage = json.loads(args.stage_receipt.read_text())
    build = json.loads(args.build_receipt.read_text())
    notebook_dir = Path(stage["notebook_dir"])
    checkpoint = Path(stage["dataset_dir"]) / "final.pt"
    notebook = notebook_dir / "trained-model-overlay.ipynb"
    if (
        sha(checkpoint) != stage["checkpoint_sha256"]
        or sha(notebook) != stage["notebook_sha256"]
        or stage["notebook_sha256"] != build["final_notebook_sha256"]
        or sha(args.parent_control) != build["parent_output_sha256"]
    ):
        raise ValueError("Staged notebook, checkpoint, or parent control differs")
    cells = json.loads(notebook.read_text())["cells"]
    overlay, audit = ("".join(cells[index]["source"]) for index in (-2, -1))
    work = args.output / "working"
    inp = args.output / "input" / stage["dataset_id"].split("/", 1)[1]
    work.mkdir(parents=True, exist_ok=True)
    inp.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.parent_control, work / "submission.csv")
    shutil.copyfile(checkpoint, inp / "final.pt")
    scope = {"COMP": args.competition_input.resolve()}
    for source in (overlay, audit):
        source = source.replace("/kaggle/working", str(work)).replace(
            "/kaggle/input", str(inp.parent)
        )
        exec(compile(source, "staged-mixup-local-smoke", "exec"), scope)
    model = build["model"]
    report = json.loads((work / f"{model}-mixup-audit.json").read_text())
    if (
        report["parent_notebook_sha256"] != build["control_sha256"]
        or report.get("experiment_id") != stage.get("experiment_id")
        or report["parent_output_sha256"] != build["parent_output_sha256"]
        or report["candidate_set_sha256"] != build["candidate_set_sha256"]
        or report["checkpoint_sha256"] != stage["checkpoint_sha256"]
        or report["rank_fusion_weight"] != stage["rank_fusion_weight"]
        or report["submission_sha256"] != sha(args.independent_submission)
        or report["rows"] != 400
        or report["top1_changed"] != 0
        or report["errors"] != 0
    ):
        raise AssertionError("Staged local overlay disagrees with the independent output")
    report["independent_submission_sha256"] = sha(args.independent_submission)
    (args.output / "local-smoke-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
