#!/usr/bin/env python
"""Package an EXP-0009 temporal-graph checkpoint for an offline Kaggle Notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tomllib
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prepare_kaggle_submission import _notebook as _base_notebook
from prepare_kaggle_submission import _sha256

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = COMPETITION_ROOT.parents[1]
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0009-host-tgraph3-residual-30e.toml"
DEFAULT_TEMPORAL_GRAPH_SOURCE = COMPETITION_ROOT / "src/temporal_graph"
DEFAULT_INFERENCE_SCRIPT = COMPETITION_ROOT / "scripts/run_kaggle_inference.py"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"

BASE_BUNDLE_FILES = (
    "edge_predictor_best.pth",
    "config.json",
    "checkpoint-metadata.json",
    "ORGANIZER-LICENSE",
)
TRACKING_ARCHIVE_NAME = "tracking_cellmot_models.zip"
TRACKING_EXPANDED_NAME = "tracking_cellmot_models"
REQUIRED_TRACKING_MEMBERS = {
    "tracking_cellmot/__init__.py",
    "tracking_cellmot/models/__init__.py",
    "tracking_cellmot/models/simple_node_transformer.py",
    "tracking_cellmot/models/temporal_unet.py",
}
GRAPH_CHECKPOINT_NAME = "temporal_graph_checkpoint.pth"
GRAPH_ARCHIVE_NAME = "temporal_graph.zip"
EXPERIMENT_CONFIG_NAME = "temporal-graph-experiment.toml"
ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _resolve_competition_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else COMPETITION_ROOT / path


def _milestone(config: Mapping[str, Any], completed_epoch: int) -> dict[str, Any]:
    submission = config.get("submission")
    if not isinstance(submission, Mapping):
        raise ValueError("config is missing [submission]")
    milestones = submission.get("milestones")
    if not isinstance(milestones, list):
        raise ValueError("config is missing [[submission.milestones]]")
    matches = [
        dict(item)
        for item in milestones
        if isinstance(item, Mapping)
        and int(item.get("completed_epoch", -1)) == completed_epoch
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one submission milestone for completed epoch {completed_epoch}, "
            f"found {len(matches)}"
        )
    required = {"checkpoint", "dataset_id", "kernel_id", "postprocess_profile"}
    missing = sorted(required - matches[0].keys())
    if missing:
        raise ValueError(f"submission milestone is missing: {missing}")
    return matches[0]


def _load_graph_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by the BioHub environment
        raise RuntimeError(
            "PyTorch is required to inspect a temporal-graph checkpoint; "
            "run this script with the BioHub uv environment"
        ) from error

    raw = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError("temporal graph checkpoint wrapper must contain a mapping")
    payload = raw.get("temporal_graph", raw)
    if not isinstance(payload, dict):
        raise TypeError("temporal_graph checkpoint entry must contain a mapping")
    required = {
        "schema_version",
        "config",
        "state_dict",
        "base_checkpoint_sha256",
        "metadata",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"temporal graph checkpoint payload is missing: {missing}")
    if not isinstance(payload["config"], Mapping):
        raise TypeError("temporal graph checkpoint config must be a mapping")
    if not isinstance(payload["state_dict"], Mapping):
        raise TypeError("temporal graph checkpoint state_dict must be a mapping")
    if not isinstance(payload["metadata"], Mapping):
        raise TypeError("temporal graph checkpoint metadata must be a mapping")
    return raw, payload


def _checkpoint_completed_epochs(
    wrapper: Mapping[str, Any], payload: Mapping[str, Any]
) -> set[int]:
    """Collect unambiguous completed-epoch fields from a portable wrapper."""
    values: set[int] = set()
    containers: list[Mapping[str, Any]] = [wrapper, payload]
    for owner in (wrapper, payload):
        metadata = owner.get("metadata")
        if isinstance(metadata, Mapping):
            containers.append(metadata)
    for container in containers:
        for key in ("completed_epochs", "completed_epoch"):
            if key in container:
                raw_value = container[key]
                if isinstance(raw_value, bool):
                    raise TypeError(f"{key} must be an integer")
                values.add(int(raw_value))
    return values


def _validate_base_bundle(
    config: Mapping[str, Any], base_bundle: Path
) -> tuple[str, dict[str, Any]]:
    missing = [name for name in BASE_BUNDLE_FILES if not (base_bundle / name).is_file()]
    if missing:
        raise FileNotFoundError(f"base checkpoint bundle is missing: {missing}")
    if not (
        (base_bundle / TRACKING_ARCHIVE_NAME).is_file()
        or (base_bundle / TRACKING_EXPANDED_NAME).is_dir()
    ):
        raise FileNotFoundError(
            "base checkpoint bundle needs tracking_cellmot_models.zip or its "
            "expanded tracking_cellmot_models directory"
        )

    weights = base_bundle / "edge_predictor_best.pth"
    actual_sha256 = _sha256(weights)
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("config is missing [source]")
    expected_sha256 = source.get("base_checkpoint_sha256")
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "configured base checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, found {actual_sha256}"
        )

    metadata_path = base_bundle / "checkpoint-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("weights_sha256") != actual_sha256:
        raise ValueError("base checkpoint bytes do not match checkpoint-metadata.json")
    expected_base_epoch = int(source["base_checkpoint_completed_epochs"])
    if int(metadata.get("completed_epochs", -1)) != expected_base_epoch:
        raise ValueError(
            "base checkpoint completed epoch mismatch: "
            f"expected {expected_base_epoch}, found {metadata.get('completed_epochs')}"
        )
    return actual_sha256, metadata


def _validate_graph_checkpoint(
    graph_checkpoint: Path,
    *,
    expected_base_sha256: str,
    expected_completed_epoch: int,
    expected_experiment_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not graph_checkpoint.is_file():
        raise FileNotFoundError(f"temporal graph checkpoint is missing: {graph_checkpoint}")
    wrapper, payload = _load_graph_checkpoint(graph_checkpoint)
    if payload["base_checkpoint_sha256"] != expected_base_sha256:
        raise ValueError(
            "temporal graph/base checkpoint SHA-256 mismatch: "
            f"expected {expected_base_sha256}, found {payload['base_checkpoint_sha256']}"
        )

    completed_epochs = _checkpoint_completed_epochs(wrapper, payload)
    if not completed_epochs:
        raise ValueError(
            "temporal graph checkpoint must record completed_epochs in wrapper or metadata"
        )
    if completed_epochs != {expected_completed_epoch}:
        raise ValueError(
            "temporal graph checkpoint completed epoch mismatch: "
            f"expected {expected_completed_epoch}, found {sorted(completed_epochs)}"
        )

    recorded_experiment_ids: set[str] = set()
    for container in (wrapper, payload, payload["metadata"]):
        value = container.get("experiment_id")
        if value is not None:
            recorded_experiment_ids.add(str(value))
    if recorded_experiment_ids and recorded_experiment_ids != {expected_experiment_id}:
        raise ValueError(
            "temporal graph checkpoint experiment mismatch: "
            f"expected {expected_experiment_id}, found {sorted(recorded_experiment_ids)}"
        )
    return wrapper, payload


def _validate_archive_member(name: str) -> Path:
    relative = Path(name)
    if (
        not name
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in name
        or relative.as_posix() != name
    ):
        raise ValueError(f"unsafe archive member: {name!r}")
    return relative


def _expanded_files(source_dir: Path) -> dict[str, bytes]:
    if not source_dir.is_dir() or source_dir.is_symlink():
        raise FileNotFoundError(f"expanded package directory is missing: {source_dir}")
    descendants = tuple(source_dir.rglob("*"))
    symlinks = sorted(str(path) for path in descendants if path.is_symlink())
    if symlinks:
        raise ValueError(f"expanded package directory contains symlinks: {symlinks}")
    files = {
        path.relative_to(source_dir).as_posix(): path.read_bytes()
        for path in descendants
        if path.is_file()
    }
    if not files:
        raise FileNotFoundError(f"expanded package directory has no files: {source_dir}")
    return files


def _archive_files(source_archive: Path) -> dict[str, bytes]:
    if not source_archive.is_file() or source_archive.is_symlink():
        raise FileNotFoundError(f"package archive is missing: {source_archive}")
    files: dict[str, bytes] = {}
    with zipfile.ZipFile(source_archive) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            name = _validate_archive_member(info.filename).as_posix()
            if name in files:
                raise ValueError(f"duplicate archive member: {name}")
            unix_mode = (info.external_attr >> 16) & 0o170000
            if unix_mode == 0o120000:
                raise ValueError(f"archive member is a symlink: {name}")
            files[name] = archive.read(info)
    if not files:
        raise ValueError(f"package archive has no files: {source_archive}")
    return files


def _write_deterministic_zip(destination: Path, files: Mapping[str, bytes]) -> None:
    """Write stable archive bytes independent of source mtimes and permissions."""
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for name in sorted(files):
            normalized = _validate_archive_member(name).as_posix()
            info = zipfile.ZipInfo(normalized, date_time=ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, files[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def _package_tracking_models(base_bundle: Path, destination: Path) -> None:
    expanded = base_bundle / TRACKING_EXPANDED_NAME
    archive = base_bundle / TRACKING_ARCHIVE_NAME
    if expanded.is_dir():
        files = _expanded_files(expanded)
    elif archive.is_file():
        files = _archive_files(archive)
    else:
        raise FileNotFoundError(
            "base checkpoint bundle needs tracking_cellmot_models.zip or its "
            "expanded tracking_cellmot_models directory"
        )
    missing = sorted(REQUIRED_TRACKING_MEMBERS - files.keys())
    if missing:
        raise ValueError(f"tracking model package is missing required Python modules: {missing}")
    _write_deterministic_zip(destination, files)


def _archive_temporal_graph(source_dir: Path, destination: Path) -> None:
    sources = sorted(path for path in source_dir.rglob("*.py") if path.is_file())
    if not sources or not (source_dir / "__init__.py").is_file():
        raise FileNotFoundError(f"temporal graph Python package is incomplete: {source_dir}")
    files = {
        (Path("temporal_graph") / source.relative_to(source_dir)).as_posix(): source.read_bytes()
        for source in sources
    }
    _write_deterministic_zip(destination, files)


def _temporal_graph_notebook(
    dataset_id: str,
    title: str,
    postprocess_profile: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    """Extend the established offline Notebook source with graph verification."""
    notebook = _base_notebook(dataset_id, title, postprocess_profile)
    code_cell = notebook["cells"][1]
    source = list(code_cell["source"])
    import_end = source.index("import sys\n") + 1
    source[import_end:import_end] = [
        "import hashlib\n",
        "import importlib.util\n",
        "import json\n",
    ]

    bundle_end = source.index("bundle = weights[0].parent\n") + 1
    source[bundle_end:bundle_end] = [
        "manifest_path = bundle / 'manifest.json'\n",
        "assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == "
        f"{manifest_sha256!r}\n",
        "manifest = json.loads(manifest_path.read_text(encoding='utf-8'))\n",
        "inference_script = bundle / 'run_kaggle_inference.py'\n",
        "expected_script = manifest['files']['run_kaggle_inference.py']['sha256']\n",
        "assert hashlib.sha256(inference_script.read_bytes()).hexdigest() == expected_script\n",
        "verification_spec = importlib.util.spec_from_file_location(\n",
        "    'packaged_biohub_inference', inference_script\n",
        ")\n",
        "assert verification_spec is not None and verification_spec.loader is not None\n",
        "verification_module = importlib.util.module_from_spec(verification_spec)\n",
        "verification_spec.loader.exec_module(verification_module)\n",
        "verification_module._verify_bundle_manifest(bundle)\n",
        f"graph_checkpoint = bundle / {GRAPH_CHECKPOINT_NAME!r}\n",
    ]
    command_end = source.index("]\n")
    source.insert(
        command_end,
        "    '--temporal-graph-checkpoint', str(graph_checkpoint),\n",
    )
    code = "".join(source)
    compile(code, f"{title}.ipynb", "exec")
    code_cell["source"] = source
    notebook["cells"][0]["source"] = [
        f"# {title}\n",
        "\n",
        "Offline frozen-host inference with the EXP-0009 T_graph=3 residual head.",
    ]
    notebook["metadata"]["agentic_kaggle"].update(
        {
            "manifest_sha256": manifest_sha256,
            "temporal_graph_checkpoint": GRAPH_CHECKPOINT_NAME,
        }
    )
    return notebook


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _file_manifest(root: Path, paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted(paths):
        entry: dict[str, Any] = {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".zip":
            members = _archive_files(path)
            entry["members"] = {
                name: {
                    "bytes": len(contents),
                    "sha256": hashlib.sha256(contents).hexdigest(),
                }
                for name, contents in sorted(members.items())
            }
        result[path.relative_to(root).as_posix()] = entry
    return result


def prepare(
    config_path: Path,
    completed_epoch: int,
    output_root: Path | None = None,
    *,
    base_bundle_override: Path | None = None,
    graph_checkpoint_override: Path | None = None,
    temporal_graph_source: Path = DEFAULT_TEMPORAL_GRAPH_SOURCE,
    inference_script: Path = DEFAULT_INFERENCE_SCRIPT,
    dataset_id_override: str | None = None,
    kernel_id_override: str | None = None,
    dataset_title_override: str | None = None,
    kernel_title_override: str | None = None,
) -> dict[str, Any]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    experiment_id = str(config["experiment_id"])
    milestone = _milestone(config, completed_epoch)
    source = config["source"]
    output = config["output"]

    configured_base_weights = _resolve_competition_path(source["base_checkpoint_path"])
    base_bundle = (
        base_bundle_override.resolve()
        if base_bundle_override is not None
        else configured_base_weights.parent
    )
    graph_checkpoint = (
        graph_checkpoint_override.resolve()
        if graph_checkpoint_override is not None
        else _resolve_competition_path(Path(output["artifact_dir"]) / milestone["checkpoint"])
    )
    if output_root is None:
        output_root = COMPETITION_ROOT / (
            f"data/kaggle-submission-{experiment_id}-epoch{completed_epoch}"
        )
    output_root = output_root.resolve()

    base_sha256, base_metadata = _validate_base_bundle(config, base_bundle)
    _, graph_payload = _validate_graph_checkpoint(
        graph_checkpoint,
        expected_base_sha256=base_sha256,
        expected_completed_epoch=completed_epoch,
        expected_experiment_id=experiment_id,
    )
    if not inference_script.is_file():
        raise FileNotFoundError(f"integrated inference script is missing: {inference_script}")
    inference_source = inference_script.read_text(encoding="utf-8")
    if "--temporal-graph-checkpoint" not in inference_source:
        raise ValueError("inference script does not expose temporal-graph checkpoint wiring")
    if "def _verify_bundle_manifest(" not in inference_source:
        raise ValueError("inference script does not support content-addressed bundle verification")

    dataset_id = dataset_id_override or str(milestone["dataset_id"])
    kernel_id = kernel_id_override or str(milestone["kernel_id"])
    variant = str(milestone.get("variant", f"epoch-{completed_epoch}"))
    dataset_title = dataset_title_override or (
        f"Biohub {experiment_id} TGraph3 E{completed_epoch}"
    )
    kernel_title = kernel_title_override or (
        f"Biohub {experiment_id} TGraph3 E{completed_epoch} Submit"
    )
    postprocess_profile = str(milestone["postprocess_profile"])

    dataset_dir = output_root / "dataset"
    kernel_dir = output_root / "kernel"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernel_dir.mkdir(parents=True, exist_ok=True)

    copied: list[Path] = []
    for name in BASE_BUNDLE_FILES:
        destination = dataset_dir / name
        shutil.copy2(base_bundle / name, destination)
        copied.append(destination)
    tracking_archive = dataset_dir / TRACKING_ARCHIVE_NAME
    _package_tracking_models(base_bundle, tracking_archive)
    copied.append(tracking_archive)
    base_manifest = base_bundle / "manifest.json"
    if base_manifest.is_file():
        destination = dataset_dir / "base-bundle-manifest.json"
        shutil.copy2(base_manifest, destination)
        copied.append(destination)

    graph_destination = dataset_dir / GRAPH_CHECKPOINT_NAME
    shutil.copy2(graph_checkpoint, graph_destination)
    copied.append(graph_destination)
    inference_destination = dataset_dir / "run_kaggle_inference.py"
    shutil.copy2(inference_script, inference_destination)
    copied.append(inference_destination)
    experiment_config = dataset_dir / EXPERIMENT_CONFIG_NAME
    shutil.copy2(config_path, experiment_config)
    copied.append(experiment_config)
    graph_archive = dataset_dir / GRAPH_ARCHIVE_NAME
    _archive_temporal_graph(temporal_graph_source, graph_archive)
    copied.append(graph_archive)

    dataset_metadata_path = dataset_dir / "dataset-metadata.json"
    _write_json(
        dataset_metadata_path,
        {
            "title": dataset_title,
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        },
    )

    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "completed_epoch": completed_epoch,
        "variant": variant,
        "fold": int(config["data"]["fold"]),
        "base": {
            "experiment_id": source["base_experiment_id"],
            "completed_epochs": int(source["base_checkpoint_completed_epochs"]),
            "weights_sha256": base_sha256,
            "source_checkpoint_sha256": base_metadata.get("source_checkpoint_sha256"),
        },
        "temporal_graph": {
            "checkpoint": GRAPH_CHECKPOINT_NAME,
            "checkpoint_sha256": _sha256(graph_destination),
            "base_checkpoint_sha256": graph_payload["base_checkpoint_sha256"],
            "schema_version": int(graph_payload["schema_version"]),
            "config": dict(graph_payload["config"]),
        },
        "postprocess_profile": postprocess_profile,
        "files": _file_manifest(dataset_dir, copied),
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)

    notebook_name = f"{kernel_id.split('/', 1)[-1]}.ipynb"
    notebook_path = kernel_dir / notebook_name
    _write_json(
        notebook_path,
        _temporal_graph_notebook(
            dataset_id,
            kernel_title,
            postprocess_profile,
            manifest_sha256,
        ),
    )
    kernel_metadata_path = kernel_dir / "kernel-metadata.json"
    _write_json(
        kernel_metadata_path,
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
            "competition_sources": [COMPETITION_SLUG],
            "kernel_sources": [],
            "model_sources": [],
        },
    )

    result = {
        "experiment_id": experiment_id,
        "completed_epoch": completed_epoch,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "base_weights_sha256": base_sha256,
        "graph_checkpoint_sha256": _sha256(graph_destination),
        "manifest_sha256": manifest_sha256,
        "postprocess_profile": postprocess_profile,
    }
    _write_json(output_root / "bundle-manifest.json", result)
    print(json.dumps(result, indent=2))
    return result


def publish(result: Mapping[str, Any], *, dataset_version: bool = False) -> None:
    """Upload the private Dataset, then push the internet-disabled Notebook."""
    dataset_action = "version" if dataset_version else "create"
    dataset_command = [
        "uv",
        "run",
        "kaggle",
        "datasets",
        dataset_action,
        "-p",
        str(result["dataset_dir"]),
    ]
    if dataset_version:
        dataset_command.extend(
            ["-m", f"{result['experiment_id']} completed epoch {result['completed_epoch']}"]
        )
    subprocess.run(dataset_command, check=True, cwd=REPOSITORY_ROOT)
    subprocess.run(
        [
            "uv",
            "run",
            "kaggle",
            "kernels",
            "push",
            "-p",
            str(result["kernel_dir"]),
        ],
        check=True,
        cwd=REPOSITORY_ROOT,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--completed-epoch", type=int, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-bundle", type=Path, default=None)
    parser.add_argument("--graph-checkpoint", type=Path, default=None)
    parser.add_argument("--temporal-graph-source", type=Path, default=DEFAULT_TEMPORAL_GRAPH_SOURCE)
    parser.add_argument("--inference-script", type=Path, default=DEFAULT_INFERENCE_SCRIPT)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--kernel-id", default=None)
    parser.add_argument("--dataset-title", default=None)
    parser.add_argument("--kernel-title", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--generate-only",
        action="store_true",
        help="Only build and validate local Dataset/Notebook directories (the default).",
    )
    mode.add_argument(
        "--publish",
        action="store_true",
        help="Push Dataset and Notebook with Kaggle CLI.",
    )
    parser.add_argument(
        "--dataset-version",
        action="store_true",
        help="With --publish, create a new Dataset version instead of a new Dataset.",
    )
    args = parser.parse_args()
    if args.dataset_version and not args.publish:
        parser.error("--dataset-version requires --publish")

    result = prepare(
        args.config.resolve(),
        args.completed_epoch,
        args.output_root.resolve() if args.output_root is not None else None,
        base_bundle_override=(
            args.base_bundle.resolve() if args.base_bundle is not None else None
        ),
        graph_checkpoint_override=(
            args.graph_checkpoint.resolve() if args.graph_checkpoint is not None else None
        ),
        temporal_graph_source=args.temporal_graph_source.resolve(),
        inference_script=args.inference_script.resolve(),
        dataset_id_override=args.dataset_id,
        kernel_id_override=args.kernel_id,
        dataset_title_override=args.dataset_title,
        kernel_title_override=args.kernel_title,
    )
    if args.publish:
        publish(result, dataset_version=args.dataset_version)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
