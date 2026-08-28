#!/usr/bin/env python
"""Package an A/B checkpoint as an offline Kaggle model Dataset and GPU Notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from dataclasses import replace
from pathlib import Path

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = COMPETITION_ROOT / "src"
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0005b-backbone-ab-nnunet.toml"
DEFAULT_OUTPUT = COMPETITION_ROOT / "data/kaggle-submission-EXP-0005B-epoch5"
DEFAULT_DATASET_ID = "suzukitaichi/biohub-exp-0005b-nnunet-epoch5"
DEFAULT_KERNEL_ID = "suzukitaichi/biohub-exp-0005b-nnunet-epoch5-submit"
EXP7A_EPOCH5_REPORT_BASELINE = 0.5688260117

_PACKAGE_DIRECTORY_NAMES = ("dataset", "kernel")
_KAGGLE_SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

sys.path.insert(0, str(SOURCE_ROOT))
from backbone_ab.checkpointing import (  # noqa: E402
    canonical_json_sha256,
    load_checkpoint,
    load_inference_profile,
    sha256_file,
    write_inference_profile,
)
from backbone_ab.config import load_and_validate_config  # noqa: E402
from backbone_ab.finalization import (  # noqa: E402
    FinalizationBinding,
    validate_selection_report_binding,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_exp7_provenance(
    provenance: dict[str, object],
    *,
    experiment_config_sha256: str,
) -> None:
    required_hashes = {
        "checkpoint_sha256",
        "inference_profile_sha256",
        "experiment_config_sha256",
        "manifest_sha256",
        "selection_sha256",
        "report_summary_sha256",
        "report_config_sha256",
    }
    missing = sorted(required_hashes - provenance.keys())
    if missing:
        raise ValueError(f"selection provenance is missing SHA-256 bindings: {missing}")
    for key in required_hashes:
        value = provenance[key]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError(f"selection provenance {key} must be lowercase SHA-256")
    if provenance["experiment_config_sha256"] != experiment_config_sha256:
        raise ValueError("selection provenance experiment config SHA-256 mismatch")
    if provenance.get("candidate_epochs") != list(range(5, 51, 5)):
        raise ValueError("selection provenance must cover the complete checkpoint sweep")
    score = float(provenance.get("report_score", float("nan")))
    baseline = float(provenance.get("report_score_baseline", float("nan")))
    tolerance = float(provenance.get("report_score_tolerance", float("nan")))
    gate = float(provenance.get("report_score_gate_exclusive", float("nan")))
    if (
        not all(math.isfinite(value) for value in (score, baseline, tolerance, gate))
        or baseline != EXP7A_EPOCH5_REPORT_BASELINE
        or tolerance < 0.0
        or gate != baseline + tolerance
        or score <= gate
    ):
        raise ValueError("selection provenance report score gate was not satisfied")


def _fresh_package_directories(output_root: Path) -> tuple[Path, Path]:
    """Recreate only the two package directories directly below ``output_root``."""
    if output_root.is_symlink():
        raise ValueError(f"refusing to clean a symlinked output root: {output_root}")
    resolved_root = output_root.resolve()
    protected_roots = {
        Path.home().resolve(),
        COMPETITION_ROOT.resolve(),
        *(parent.resolve() for parent in COMPETITION_ROOT.parents[:3]),
    }
    if resolved_root.parent == resolved_root or resolved_root in protected_roots:
        raise ValueError(f"refusing to clean a broad output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    directories = []
    for name in _PACKAGE_DIRECTORY_NAMES:
        directory = output_root / name
        if directory.is_symlink():
            raise ValueError(f"refusing to replace symlinked package directory: {directory}")
        if directory.resolve().parent != resolved_root:
            raise ValueError(f"package directory escapes output root: {directory}")
        if directory.exists():
            if directory.is_dir():
                shutil.rmtree(directory)
            else:
                directory.unlink()
        directory.mkdir()
        directories.append(directory)
    return directories[0], directories[1]


def _assert_clean_package_tree(output_root: Path) -> None:
    forbidden = sorted(
        path
        for path in output_root.rglob("*")
        if path.name == "__pycache__" or path.suffix.lower() in {".pyc", ".pyo"}
    )
    if forbidden:
        rendered = ", ".join(str(path) for path in forbidden)
        raise RuntimeError(f"generated package contains Python cache artifacts: {rendered}")


def _write_deterministic_zip_member(
    archive: zipfile.ZipFile,
    source: Path,
    archive_name: str,
) -> None:
    info = zipfile.ZipInfo(archive_name, date_time=_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(
        info,
        source.read_bytes(),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    )


def _zip_tree(archive_path: Path, source_root: Path, package_name: str) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        sources = sorted(
            (
                source
                for source in source_root.rglob("*.py")
                if "__pycache__" not in source.parts
                and source.suffix.lower() not in {".pyc", ".pyo"}
            ),
            key=lambda source: source.relative_to(source_root).as_posix(),
        )
        for source in sources:
            relative = source.relative_to(source_root)
            _write_deterministic_zip_member(
                archive,
                source,
                (Path(package_name) / relative).as_posix(),
            )


def _kaggle_title(identifier: str) -> tuple[str, str]:
    """Return a title whose Kaggle slug is exactly the requested identifier slug."""
    parts = identifier.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Kaggle identifier must have the form owner/slug: {identifier!r}")
    slug = parts[1]
    if _KAGGLE_SLUG.fullmatch(slug) is None:
        raise ValueError(
            "Kaggle slug must contain lowercase ASCII letters, digits, and single hyphens: "
            f"{slug!r}"
        )
    return slug.replace("-", " "), slug


def _notebook(
    title: str,
    *,
    require_selection_provenance: bool = False,
    expected_manifest_sha256: str | None = None,
) -> dict[str, object]:
    source = [
        "from pathlib import Path\n",
        "import subprocess\n",
        "import sys\n",
        "\n",
    ]
    if require_selection_provenance:
        if expected_manifest_sha256 is None:
            raise ValueError("finalized notebook requires a package manifest SHA-256")
        source.extend(
            [
                "import hashlib\n",
                "provenance = list(Path('/kaggle/input').rglob('selection-provenance.json'))\n",
                "assert len(provenance) == 1, "
                "f'expected one finalized bundle, found {provenance}'\n",
                "bundle = provenance[0].parent\n",
                "assert (bundle / 'edge_predictor_best.pth').is_file()\n",
                "manifest = bundle / 'manifest.json'\n",
                f"expected_manifest_sha256 = {expected_manifest_sha256!r}\n",
                "actual_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()\n",
                "assert actual_manifest_sha256 == expected_manifest_sha256, "
                "'package manifest SHA-256 mismatch'\n",
            ]
        )
    else:
        source.extend(
            [
                "weights = list(Path('/kaggle/input').rglob('edge_predictor_best.pth'))\n",
                "assert len(weights) == 1, f'expected one checkpoint, found {weights}'\n",
                "bundle = weights[0].parent\n",
            ]
        )
    source.extend(
        [
            "test_dir = Path('/kaggle/input/competitions') / "
            "'biohub-cell-tracking-during-development/test'\n",
            "output = Path('/kaggle/working/submission.csv')\n",
            "command = [\n",
            "    sys.executable, str(bundle / 'run_kaggle_inference.py'),\n",
            "    '--bundle-dir', str(bundle),\n",
            "    '--test-dir', str(test_dir),\n",
            "    '--output', str(output),\n",
            "]\n",
            "subprocess.run(command, check=True)\n",
            "assert output.exists() and output.stat().st_size > 0\n",
            "print(f'submission ready: {output} ({output.stat().st_size:,} bytes)')\n",
        ]
    )
    return {
        "cells": [
            {
                "id": "package-overview",
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", "\n", "Offline nnU-Net backbone inference."],
            },
            {
                "id": "run-inference",
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
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def prepare(
    config_path: Path,
    checkpoint_path: Path,
    output_root: Path,
    dataset_id: str,
    kernel_id: str,
    inference_profile_path: Path | None = None,
    *,
    selection_provenance: dict[str, object] | None = None,
    expected_completed_epoch: int | None = None,
) -> dict[str, object]:
    import dynamic_network_architectures
    import torch

    config = load_and_validate_config(config_path)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.state_dict
    completed_epochs = int(checkpoint.metadata.get("completed_epochs", 0))
    model_api = str(
        config["inference"].get(
            "model_api", config["backbone"].get("contract", "legacy")
        )
    )

    profile = None
    source_profile_sha256 = None
    if model_api == "corrected_v2":
        if inference_profile_path is None or selection_provenance is None:
            raise ValueError(
                "corrected_v2 packaging requires a validated selection and inference profile"
            )
        _validate_exp7_provenance(
            selection_provenance,
            experiment_config_sha256=sha256_file(config_path),
        )
        profile = load_inference_profile(inference_profile_path)
        if profile.checkpoint_sha256 != checkpoint.sha256:
            raise ValueError("inference profile checkpoint hash does not match packaged weights")
        if profile.experiment_config_sha256 != sha256_file(config_path):
            raise ValueError("inference profile experiment config hash does not match")
        if checkpoint.metadata.get("experiment_id") != config.get("experiment_id"):
            raise ValueError("checkpoint experiment_id does not match experiment config")
        if checkpoint.metadata.get("model_contract") != "corrected_v2":
            raise ValueError("checkpoint does not use the corrected_v2 model contract")
        resume_config = json.loads(json.dumps(config))
        resume_config["train"].pop("epochs", None)
        if checkpoint.metadata.get("resume_fingerprint") != canonical_json_sha256(
            resume_config
        ):
            raise ValueError("checkpoint resume fingerprint does not match experiment config")
        if expected_completed_epoch is None or completed_epochs != expected_completed_epoch:
            raise ValueError("checkpoint completed epoch does not match final selection")
        if selection_provenance.get("completed_epoch") != completed_epochs:
            raise ValueError("selection provenance completed epoch mismatch")
        if selection_provenance.get("checkpoint_sha256") != checkpoint.sha256:
            raise ValueError("selection provenance checkpoint SHA-256 mismatch")
        if selection_provenance.get("inference_profile_sha256") != profile.sha256:
            raise ValueError("selection provenance inference profile SHA-256 mismatch")
        if (
            checkpoint.metadata.get("validation_subset_manifest_sha256")
            != selection_provenance.get("manifest_sha256")
        ):
            raise ValueError("checkpoint split manifest SHA-256 mismatch")
        source_profile_sha256 = profile.sha256

    organizer = COMPETITION_ROOT / config["source"]["organizer_repository_path"]
    actual_revision = subprocess.check_output(
        ["git", "-C", str(organizer), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_revision != config["source"]["organizer_revision"]:
        raise ValueError("organizer source revision does not match experiment config")
    required_sources = [
        COMPETITION_ROOT / "scripts/run_kaggle_inference.py",
        COMPETITION_ROOT / "src/backbone_ab/backbones.py",
        organizer / "src/tracking_cellmot/models/simple_node_transformer.py",
    ]
    missing_sources = [str(path) for path in required_sources if not path.is_file()]
    if missing_sources:
        raise FileNotFoundError(f"submission runtime source is missing: {missing_sources}")
    dataset_title, dataset_slug = _kaggle_title(dataset_id)
    kernel_title, kernel_slug = _kaggle_title(kernel_id)

    # All source bindings are checked before cleaning a previously generated package.
    dataset_dir, kernel_dir = _fresh_package_directories(output_root)

    weights_path = dataset_dir / "edge_predictor_best.pth"
    torch.save(state_dict, weights_path)
    model_config = {
        "experiment_id": config["experiment_id"],
        "backbone_type": config["backbone"]["name"],
        "backbone": config["backbone"],
        "downsample": config["train"]["downsample"],
        "window_size": config["data"]["window_size"],
        "pool_kernel_um": config["train"]["pool_kernel_um"],
        "model_api": model_api,
        "inference": config["inference"],
    }
    config_out = dataset_dir / "config.json"
    config_out.write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(COMPETITION_ROOT / "scripts/run_kaggle_inference.py", dataset_dir)

    profile_out = dataset_dir / "inference_profile.json"
    if profile is not None:
        profile = replace(profile, checkpoint_sha256=_sha256(weights_path))
        write_inference_profile(profile_out, profile)

    tracking_archive = dataset_dir / "tracking_cellmot_models.zip"
    with zipfile.ZipFile(
        tracking_archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for relative in (
            "tracking_cellmot/__init__.py",
            "tracking_cellmot/models/__init__.py",
            "tracking_cellmot/models/temporal_unet.py",
            "tracking_cellmot/models/simple_node_transformer.py",
        ):
            _write_deterministic_zip_member(
                archive,
                organizer / "src" / relative,
                relative,
            )

    backbone_root = COMPETITION_ROOT / "src/backbone_ab"
    backbone_archive = dataset_dir / "backbone_ab.zip"
    _zip_tree(backbone_archive, backbone_root, "backbone_ab")
    dynamic_root = Path(dynamic_network_architectures.__path__[0])
    dynamic_archive = dataset_dir / "dynamic_network_architectures.zip"
    _zip_tree(dynamic_archive, dynamic_root, "dynamic_network_architectures")

    metadata_path = dataset_dir / "checkpoint-metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "completed_epochs": completed_epochs,
                "best_score_at_save": (
                    float(checkpoint.metadata["best_score"])
                    if "best_score" in checkpoint.metadata
                    else None
                ),
                "source_checkpoint": checkpoint_path.name,
                "source_checkpoint_sha256": checkpoint.sha256,
                "checkpoint_format": checkpoint.source_format,
                "inference_profile_sha256": profile.sha256 if profile else None,
                "source_inference_profile_sha256": source_profile_sha256,
                "experiment_config_sha256": sha256_file(config_path),
                "validation_subset_manifest_sha256": checkpoint.metadata.get(
                    "validation_subset_manifest_sha256"
                ),
                "organizer_revision": config["source"]["organizer_revision"],
                "nnunet_revision": config["source"]["nnunet_revision"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    provenance_path = None
    if selection_provenance is not None:
        provenance_path = dataset_dir / "selection-provenance.json"
        provenance_path.write_text(
            json.dumps(selection_provenance, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    copied = [
        weights_path,
        config_out,
        dataset_dir / "run_kaggle_inference.py",
        tracking_archive,
        backbone_archive,
        dynamic_archive,
        metadata_path,
    ]
    if profile is not None:
        copied.append(profile_out)
    if provenance_path is not None:
        copied.append(provenance_path)
    manifest = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "completed_epochs": completed_epochs,
        "files": {},
    }
    for path in copied:
        entry: dict[str, object] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                entry["members"] = {
                    info.filename: {
                        "bytes": info.file_size,
                        "sha256": hashlib.sha256(archive.read(info)).hexdigest(),
                    }
                    for info in archive.infolist()
                    if not info.is_dir()
                }
        manifest["files"][path.name] = entry
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest_sha256 = _sha256(dataset_dir / "manifest.json")
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

    notebook_name = f"{kernel_slug}.ipynb"
    (kernel_dir / notebook_name).write_text(
        json.dumps(
            _notebook(
                kernel_title,
                require_selection_provenance=selection_provenance is not None,
                expected_manifest_sha256=(
                    manifest_sha256 if selection_provenance is not None else None
                ),
            ),
            indent=2,
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
    _assert_clean_package_tree(dataset_dir)
    _assert_clean_package_tree(kernel_dir)
    result = {
        "dataset_id": dataset_id,
        "dataset_slug": dataset_slug,
        "dataset_dir": str(dataset_dir),
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "completed_epochs": completed_epochs,
        "weights_sha256": _sha256(weights_path),
        "inference_profile_sha256": profile.sha256 if profile else None,
        "manifest_sha256": manifest_sha256,
    }
    print(json.dumps(result, indent=2))
    return result


def _validate_exp7_identifier(identifier: str, epoch: int, *, kernel: bool) -> None:
    _, slug = _kaggle_title(identifier)
    required = ("exp-0007a", f"epoch{epoch}")
    if any(
        re.search(rf"(?:^|-){re.escape(token)}(?:-|$)", slug) is None
        for token in required
    ):
        raise ValueError(
            f"Kaggle identifier must include {' and '.join(required)}: {identifier}"
        )
    if kernel and not slug.endswith("-submit"):
        raise ValueError("finalized kernel identifier must end in '-submit'")


def prepare_selected(
    selection_path: Path,
    report_summary_path: Path,
    output_root: Path,
    dataset_id: str,
    kernel_id: str,
    *,
    require_report_score_above: float,
    report_score_tolerance: float,
) -> dict[str, object]:
    """Validate the full EXP-0007 selection/report chain before packaging it."""
    binding: FinalizationBinding = validate_selection_report_binding(
        selection_path,
        report_summary_path,
        competition_root=COMPETITION_ROOT,
    )
    baseline = float(require_report_score_above)
    tolerance = float(report_score_tolerance)
    if not math.isfinite(baseline):
        raise ValueError("report score baseline must be finite")
    if baseline != EXP7A_EPOCH5_REPORT_BASELINE:
        raise ValueError(
            "EXP-0007A finalization must use the pinned epoch-5 report baseline "
            f"{EXP7A_EPOCH5_REPORT_BASELINE}"
        )
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("report score tolerance must be finite and non-negative")
    gate = baseline + tolerance
    if not math.isfinite(gate):
        raise ValueError("effective report score gate must be finite")
    if binding.report_score <= gate:
        raise ValueError(
            f"selected report score {binding.report_score:.6f} does not exceed gate {gate:.6f}"
        )
    _validate_exp7_identifier(dataset_id, binding.completed_epoch, kernel=False)
    _validate_exp7_identifier(kernel_id, binding.completed_epoch, kernel=True)
    data_root = (COMPETITION_ROOT / "data").resolve(strict=True)
    if output_root.is_symlink():
        raise ValueError("finalized output root must not be a symlink")
    resolved_output = output_root.resolve()
    if not resolved_output.is_relative_to(data_root):
        raise ValueError("finalized output root must be inside competition data/")
    required_output_tokens = ("exp-0007a", f"epoch{binding.completed_epoch}")
    output_name = output_root.name.lower()
    if any(
        re.search(rf"(?:^|-){re.escape(token)}(?:-|$)", output_name) is None
        for token in required_output_tokens
    ):
        raise ValueError(
            "finalized output directory name must identify EXP-0007A and selected epoch"
        )
    return prepare(
        binding.experiment_config_path,
        binding.checkpoint_path,
        output_root,
        dataset_id,
        kernel_id,
        binding.inference_profile_path,
        selection_provenance=binding.provenance(
            report_score_baseline=baseline,
            report_score_tolerance=tolerance,
        ),
        expected_completed_epoch=binding.completed_epoch,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-id")
    parser.add_argument("--kernel-id")
    parser.add_argument("--inference-profile", type=Path)
    parser.add_argument("--selection-json", type=Path)
    parser.add_argument("--report-summary", type=Path)
    parser.add_argument("--require-report-score-above", type=float)
    parser.add_argument("--report-score-tolerance", type=float)
    args = parser.parse_args()
    if args.selection_json is not None:
        conflicting = [
            name
            for name, value in (
                ("--config", args.config),
                ("--checkpoint", args.checkpoint),
                ("--inference-profile", args.inference_profile),
            )
            if value is not None
        ]
        if conflicting:
            parser.error(
                "selection-driven packaging derives config/checkpoint/profile; remove "
                + ", ".join(conflicting)
            )
        required = {
            "--report-summary": args.report_summary,
            "--output-root": args.output_root,
            "--dataset-id": args.dataset_id,
            "--kernel-id": args.kernel_id,
            "--require-report-score-above": args.require_report_score_above,
            "--report-score-tolerance": args.report_score_tolerance,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            parser.error("selection-driven packaging requires " + ", ".join(missing))
        prepare_selected(
            args.selection_json.resolve(),
            args.report_summary.resolve(),
            args.output_root.resolve(),
            args.dataset_id,
            args.kernel_id,
            require_report_score_above=args.require_report_score_above,
            report_score_tolerance=args.report_score_tolerance,
        )
    else:
        if (
            args.report_summary is not None
            or args.require_report_score_above is not None
            or args.report_score_tolerance is not None
        ):
            parser.error("report gating requires --selection-json")
        config_path = (args.config or DEFAULT_CONFIG).resolve()
        with config_path.open("rb") as file:
            manual_config = tomllib.load(file)
        manual_contract = manual_config.get("backbone", {}).get("contract", "legacy")
        manual_model_api = manual_config.get("inference", {}).get(
            "model_api", manual_contract
        )
        if manual_contract == "corrected_v2" or manual_model_api == "corrected_v2":
            parser.error("corrected_v2 packaging requires --selection-json")
        if args.checkpoint is None:
            parser.error("legacy packaging requires --checkpoint")
        prepare(
            config_path,
            args.checkpoint.resolve(),
            (args.output_root or DEFAULT_OUTPUT).resolve(),
            args.dataset_id or DEFAULT_DATASET_ID,
            args.kernel_id or DEFAULT_KERNEL_ID,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
