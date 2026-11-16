"""Stage a private checkpoint Dataset and exact-A-parent Kaggle Notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data/derived/mist-mixup-20261116"
PARENT = ROOT / (
    "data/derived/kaggle-public-0477-control/lb-0-477-two-0-472-fixes-merged-the-right-way.ipynb"
)
PARENT_METADATA = ROOT / "data/derived/mist-duration-20261107/kernel-12k/kernel-metadata.json"
INFERENCE = ROOT / "src/mist_formula/frozen/a_0479_model.py"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-id")
    parser.add_argument("--kernel-id")
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    if not args.label.isascii() or not args.label.replace("-", "").isalnum():
        parser.error("label must be ASCII alphanumeric with optional hyphens")
    if not args.checkpoint.is_file():
        parser.error("checkpoint not found")
    owner = "suzukitaichi"
    dataset_id = args.dataset_id or f"{owner}/casmi26-mixup-{args.label}-20261116"
    kernel_id = args.kernel_id or f"{owner}/casmi26-mixup-{args.label}-rerank-20261116"
    name = f"mist-mixup-{args.label}"
    dataset_dir = BASE / "kaggle-datasets" / args.label
    notebook_dir = BASE / "kaggle-notebooks" / args.label
    dataset_dir.mkdir(parents=True, exist_ok=True)
    notebook_dir.mkdir(parents=True, exist_ok=True)
    staged = dataset_dir / "final.pt"
    if staged.is_file() and sha(staged) != sha(args.checkpoint):
        raise ValueError("staged checkpoint differs; use a new label")
    if not staged.is_file():
        shutil.copy2(args.checkpoint, staged)
    title = f"CASMI26 MIST spectrum mixup {args.label} 20261116"
    if len(title) > 50:
        title = f"CASMI26 MIST mixup {args.label} 20261117"
    description = (
        "Private checkpoint of a CASMI26 train-only spectrum mixup variant "
        "for the fixed 0.479 formula Transformer comparison."
    )
    if args.experiment_id:
        description += f" Experiment ID: {args.experiment_id}."
    metadata = {
        "title": title,
        "id": dataset_id,
        "licenses": [{"name": "CC-BY-NC-SA-4.0"}],
        "description": description,
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))
    command = [
        "uv",
        "run",
        "python",
        str(ROOT / "src/mist_formula/build_mixup_notebook.py"),
        "--parent",
        str(PARENT),
        "--parent-metadata",
        str(PARENT_METADATA),
        "--inference-module",
        str(INFERENCE),
        "--checkpoint",
        str(staged),
        "--dataset",
        dataset_id,
        "--kernel",
        kernel_id,
        "--model-name",
        name,
        "--output",
        str(notebook_dir),
        "--weight",
        str(args.weight),
    ]
    if args.experiment_id:
        command.extend(("--experiment-id", args.experiment_id))
    subprocess.run(command, cwd=ROOT, check=True)
    receipt = {
        "label": args.label,
        "checkpoint_sha256": sha(staged),
        "dataset_id": dataset_id,
        "kernel_id": kernel_id,
        "rank_fusion_weight": args.weight,
        "dataset_dir": str(dataset_dir),
        "notebook_dir": str(notebook_dir),
        "notebook_sha256": sha(notebook_dir / "trained-model-overlay.ipynb"),
    }
    if args.experiment_id:
        receipt["experiment_id"] = args.experiment_id
    (BASE / f"stage-{args.label}.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
