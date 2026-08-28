#!/usr/bin/env python
"""Prepare a private model Dataset and GPU Notebook for Kaggle submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
import zipfile
from pathlib import Path

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0004-host-baseline-fold0-50e.toml"
DEFAULT_OUTPUT = COMPETITION_ROOT / "data/kaggle-submission-EXP-0004"
DEFAULT_DATASET_ID = "suzukitaichi/biohub-exp-0004-host-baseline"
DEFAULT_KERNEL_ID = "suzukitaichi/biohub-exp-0004-host-baseline-submit"
DEFAULT_DATASET_TITLE = "Biohub EXP-0004 Host Baseline"
DEFAULT_KERNEL_TITLE = "Biohub EXP-0004 Host Baseline Submit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _notebook(
    dataset_id: str,
    title: str,
    postprocess_profile: str = "none",
) -> dict[str, object]:
    source = [
        "from pathlib import Path\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
        "weights = list(Path('/kaggle/input').rglob('edge_predictor_best.pth'))\n",
        "assert len(weights) == 1, f'expected one model checkpoint, found {weights}'\n",
        "bundle = weights[0].parent\n",
        "test_candidates = [\n",
        "    Path('/kaggle/input/competitions/biohub-cell-tracking-during-development/test'),\n",
        "    Path('/kaggle/input/biohub-cell-tracking-during-development/test'),\n",
        "]\n",
        "test_dirs = [p for p in test_candidates if p.is_dir() and any(p.glob('*.zarr'))]\n",
        "if not test_dirs:\n",
        "    test_dirs = sorted(\n",
        "        {p.parent for p in Path('/kaggle/input').rglob('*.zarr')\n",
        "         if p.parent.name == 'test'}\n",
        "    )\n",
        "assert len(test_dirs) == 1, (\n",
        "    f'expected one competition test directory, found {test_dirs}'\n",
        ")\n",
        "test_dir = test_dirs[0]\n",
        "output = Path('/kaggle/working/submission.csv')\n",
        "command = [\n",
        "    sys.executable, str(bundle / 'run_kaggle_inference.py'),\n",
        "    '--bundle-dir', str(bundle),\n",
        "    '--test-dir', str(test_dir),\n",
        "    '--output', str(output),\n",
        "    '--det-threshold', '0.99',\n",
        "    '--edge-threshold', '0.5',\n",
        "]\n",
        "subprocess.run(command, check=True)\n",
        "assert output.exists() and output.stat().st_size > 0\n",
        "print(f'submission ready: {output} ({output.stat().st_size:,} bytes)')\n",
    ]
    if postprocess_profile != "none":
        command_start = source.index("command = [\n")
        command_end = source.index("]\n", command_start)
        source.insert(
            command_end,
            f"    '--postprocess-profile', '{postprocess_profile}',\n",
        )
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    "\n",
                    "Offline inference with the locally trained organizer "
                    "UNet+transformer checkpoint."
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source,
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "agentic_kaggle": {"dataset_source": dataset_id},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def prepare(
    config_path: Path,
    output_root: Path,
    trained_dir_override: Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    kernel_id: str = DEFAULT_KERNEL_ID,
    dataset_title: str = DEFAULT_DATASET_TITLE,
    kernel_title: str = DEFAULT_KERNEL_TITLE,
    postprocess_profile: str = "none",
) -> dict[str, object]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    source_repo = COMPETITION_ROOT / config["source"]["repository_path"]
    method = config["train"]["method"]
    fold = int(config["data"]["fold"])
    trained_dir = (
        trained_dir_override
        if trained_dir_override is not None
        else source_repo / "weights" / method / f"split_{fold}"
    )
    weights_path = trained_dir / "edge_predictor_best.pth"
    model_config_path = trained_dir / "config.json"
    if not weights_path.is_file() or not model_config_path.is_file():
        raise FileNotFoundError(f"completed checkpoint is missing from {trained_dir}")

    dataset_dir = output_root / "dataset"
    kernel_dir = output_root / "kernel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    copy_pairs = [
        (weights_path, dataset_dir / weights_path.name),
        (model_config_path, dataset_dir / model_config_path.name),
        (
            COMPETITION_ROOT / "scripts/run_kaggle_inference.py",
            dataset_dir / "run_kaggle_inference.py",
        ),
        (source_repo / "LICENSE", dataset_dir / "ORGANIZER-LICENSE"),
    ]
    checkpoint_metadata = trained_dir / "checkpoint-metadata.json"
    if checkpoint_metadata.is_file():
        copy_pairs.append(
            (checkpoint_metadata, dataset_dir / checkpoint_metadata.name)
        )
    for source, destination in copy_pairs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination)

    model_archive = dataset_dir / "tracking_cellmot_models.zip"
    model_sources = [
        source_repo / "src/tracking_cellmot/__init__.py",
        source_repo / "src/tracking_cellmot/models/__init__.py",
        source_repo / "src/tracking_cellmot/models/temporal_unet.py",
        source_repo / "src/tracking_cellmot/models/simple_node_transformer.py",
    ]
    with zipfile.ZipFile(model_archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for source in model_sources:
            relative = source.relative_to(source_repo / "src")
            archive.write(source, relative.as_posix())
    copied.append(model_archive)

    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "organizer_revision": config["source"]["revision"],
        "fold": fold,
        "train_parameters": config["train"],
        "runtime": config.get("runtime", {}),
        "checkpoint": config.get("checkpoint", {}),
        "postprocess_profile": postprocess_profile,
        "files": {
            str(path.relative_to(dataset_dir)): {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in copied
        },
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (dataset_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_title,
                "id": dataset_id,
                "licenses": [{"name": "other"}],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    notebook_name = f"{kernel_id.split('/', 1)[-1]}.ipynb"
    (kernel_dir / notebook_name).write_text(
        json.dumps(
            _notebook(dataset_id, kernel_title, postprocess_profile), indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": kernel_id,
                "title": kernel_title,
                "code_file": notebook_name,
                "language": "python",
                "kernel_type": "notebook",
                "is_private": "true",
                "enable_gpu": "true",
                "enable_tpu": "false",
                "enable_internet": "false",
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": [dataset_id],
                "competition_sources": ["biohub-cell-tracking-during-development"],
                "kernel_sources": [],
                "model_sources": [],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    result = {
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "weights_sha256": _sha256(weights_path),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trained-dir", type=Path, default=None)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--kernel-id", default=DEFAULT_KERNEL_ID)
    parser.add_argument("--dataset-title", default=DEFAULT_DATASET_TITLE)
    parser.add_argument("--kernel-title", default=DEFAULT_KERNEL_TITLE)
    parser.add_argument(
        "--postprocess-profile",
        choices=("none", "public-applicable-v1"),
        default="none",
    )
    args = parser.parse_args()
    prepare(
        args.config.resolve(),
        args.output_root.resolve(),
        args.trained_dir.resolve() if args.trained_dir is not None else None,
        dataset_id=args.dataset_id,
        kernel_id=args.kernel_id,
        dataset_title=args.dataset_title,
        kernel_title=args.kernel_title,
        postprocess_profile=args.postprocess_profile,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
