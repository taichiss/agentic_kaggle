"""Execute a pinned rank-only local Notebook inside an isolated /kaggle mount.

The Notebook source is built by build_ce_analog_rank_only_local.py. This worker
reads no labels and writes predictions only; evaluation is a separate step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text())
    if digest(args.notebook) != receipt["variant_notebook_sha256"]:
        raise ValueError("Local variant Notebook differs from build receipt")
    if os.environ.get("KAGGLE_IS_COMPETITION_RERUN") != "1":
        raise ValueError("Expected full competition-rerun mode")
    if not os.path.samefile("/kaggle/working", args.working):
        raise ValueError("Isolated working mount differs")
    notebook = json.loads(args.notebook.read_text())
    scope = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        print(f"LOCAL CE RANK-ONLY CELL {index} START", flush=True)
        exec(compile("".join(cell["source"]), f"rank_only_cell_{index}", "exec"), scope)
        print(f"LOCAL CE RANK-ONLY CELL {index} DONE", flush=True)
    import pandas as pd

    submission = Path("/kaggle/working/submission.csv")
    frame = pd.read_csv(submission, dtype=str)
    if len(frame) != 80 or frame.molecule_id.nunique() != 80:
        raise ValueError("Expected 80 external query molecules")
    print("LOCAL CE RANK-ONLY COMPLETE", digest(submission), flush=True)


if __name__ == "__main__":
    main()
