#!/usr/bin/env python
"""Package the controlled EXP-0015 T_graph=4/T_graph=3 fallback submission."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from prepare_kaggle_submission import _notebook as _base_notebook
from prepare_temporal_graph_submission import (
    BASE_BUNDLE_FILES,
    COMPETITION_ROOT,
    COMPETITION_SLUG,
    DEFAULT_INFERENCE_SCRIPT,
    DEFAULT_TEMPORAL_GRAPH_SOURCE,
    GRAPH_ARCHIVE_NAME,
    REPOSITORY_ROOT,
    TRACKING_ARCHIVE_NAME,
    _archive_temporal_graph,
    _file_manifest,
    _package_tracking_models,
    _resolve_competition_path,
    _sha256,
    _validate_base_bundle,
    _validate_graph_checkpoint,
    _write_json,
)

DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0015-tgraph4-link-ensemble.toml"
DEFAULT_DATASET_ID = "suzukitaichi/biohub-exp-0015-tgraph4-link-5050"
DEFAULT_DATASET_TITLE = "Biohub EXP-0015 TGraph4 Link 5050"
DEFAULT_KERNEL_ID = "suzukitaichi/biohub-exp-0015-tgraph4-bounded-logit-5050-submit"
DEFAULT_KERNEL_TITLE = "Biohub EXP-0015 TGraph4 Bounded Logit 5050 Submit"

PRIMARY_MLP_CHECKPOINT_NAME = "temporal_graph_t4_mlp_checkpoint.pth"
PRIMARY_ATTENTION_CHECKPOINT_NAME = "temporal_graph_t4_attention_checkpoint.pth"
FALLBACK_MLP_CHECKPOINT_NAME = "temporal_graph_fallback_mlp_checkpoint.pth"
FALLBACK_ATTENTION_CHECKPOINT_NAME = "temporal_graph_fallback_attention_checkpoint.pth"
EXPERIMENT_CONFIG_NAME = "tgraph4-link-ensemble-experiment.toml"
TEMPORAL_LINK_MODE = "bounded_logit_5050"

HEAD_SPECS = {
    "mlp": {
        "architecture": "mlp",
        "window": "primary",
        "checkpoint_name": PRIMARY_MLP_CHECKPOINT_NAME,
    },
    "attention": {
        "architecture": "candidate_attention",
        "window": "primary",
        "checkpoint_name": PRIMARY_ATTENTION_CHECKPOINT_NAME,
    },
    "fallback_mlp": {
        "architecture": "mlp",
        "window": "fallback",
        "checkpoint_name": FALLBACK_MLP_CHECKPOINT_NAME,
    },
    "fallback_attention": {
        "architecture": "candidate_attention",
        "window": "fallback",
        "checkpoint_name": FALLBACK_ATTENTION_CHECKPOINT_NAME,
    },
}

# These fields determine which sources can be presented to either scorer. Model
# architecture, hidden width and graph window deliberately are not part of this
# cross-head contract.
CANDIDATE_CONTRACT_FIELDS = (
    "node_feature_dim",
    "top_k",
    "radius_um",
    "distance_scale_um",
    "middle_coord_atol",
    "image_window_size",
    "ownership",
)


def _mapping(owner: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = owner.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"config is missing [{key}]")
    return value


def _require_sha256(value: object, label: str) -> str:
    candidate = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return candidate


def _contract_default(name: str) -> object:
    defaults: dict[str, object] = {
        "middle_coord_atol": 1.0e-4,
        "image_window_size": 2,
        "graph_window_size": 3,
        "ownership": "right_transition",
        "architecture": "mlp",
    }
    return defaults.get(name)


def _contract_value(config: Mapping[str, Any], name: str) -> object:
    value = config.get(name, _contract_default(name))
    if value is None:
        raise ValueError(f"temporal graph checkpoint config is missing {name}")
    return value


def _checkpoint_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("temporal graph checkpoint metadata must be a mapping")
    return metadata


def _metadata_completed_epoch(metadata: Mapping[str, Any], label: str) -> int:
    values = {
        int(metadata[name])
        for name in ("completed_epoch", "completed_epochs")
        if name in metadata and not isinstance(metadata[name], bool)
    }
    if len(values) != 1:
        raise ValueError(f"{label} metadata must identify one completed epoch")
    return values.pop()


def _validate_head_checkpoint(
    *,
    label: str,
    path: Path,
    head_config: Mapping[str, Any],
    expected_architecture: str,
    expected_window_size: int,
    expected_image_window_size: int,
    expected_base_sha256: str,
    expected_cache_fingerprint: str,
    expected_source_sha256: str,
    attention_logit_bound: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_experiment = str(head_config["experiment_id"])
    expected_epoch = int(head_config["completed_epoch"])
    expected_checkpoint_sha256 = _require_sha256(
        head_config["checkpoint_sha256"], f"{label} checkpoint_sha256"
    )
    actual_checkpoint_sha256 = _sha256(path)
    if actual_checkpoint_sha256 != expected_checkpoint_sha256:
        raise ValueError(
            f"{label} checkpoint SHA-256 mismatch: expected "
            f"{expected_checkpoint_sha256}, found {actual_checkpoint_sha256}"
        )

    wrapper, payload = _validate_graph_checkpoint(
        path,
        expected_base_sha256=expected_base_sha256,
        expected_completed_epoch=expected_epoch,
        expected_experiment_id=expected_experiment,
    )
    metadata = _checkpoint_metadata(payload)
    if metadata.get("experiment_id") != expected_experiment:
        raise ValueError(f"{label} metadata experiment mismatch")
    if _metadata_completed_epoch(metadata, label) != expected_epoch:
        raise ValueError(f"{label} metadata completed epoch mismatch")

    recorded_experiments = {
        str(container["experiment_id"])
        for container in (wrapper, payload, metadata)
        if "experiment_id" in container
    }
    if recorded_experiments != {expected_experiment}:
        raise ValueError(f"{label} checkpoint experiment identity is ambiguous")
    if payload.get("base_checkpoint_sha256") != expected_base_sha256:
        raise ValueError(f"{label} frozen-host checkpoint mismatch")
    if metadata.get("cache_fingerprint") != expected_cache_fingerprint:
        raise ValueError(f"{label} cache fingerprint mismatch")
    if metadata.get("source_raw_checkpoint_sha256") != expected_source_sha256:
        raise ValueError(f"{label} raw source checkpoint mismatch")
    if metadata.get("ownership") != "right_transition":
        raise ValueError(f"{label} metadata ownership mismatch")
    if int(metadata.get("image_window_size", -1)) != expected_image_window_size:
        raise ValueError(f"{label} metadata image window mismatch")
    if int(metadata.get("graph_window_size", -1)) != expected_window_size:
        raise ValueError(f"{label} metadata graph window mismatch")

    graph_config = payload.get("config")
    if not isinstance(graph_config, Mapping):
        raise TypeError(f"{label} temporal graph config must be a mapping")
    architecture = str(_contract_value(graph_config, "architecture"))
    if architecture != expected_architecture:
        raise ValueError(
            f"{label} architecture mismatch: expected {expected_architecture}, found {architecture}"
        )
    if int(_contract_value(graph_config, "image_window_size")) != (expected_image_window_size):
        raise ValueError(f"{label} config image window mismatch")
    if int(_contract_value(graph_config, "graph_window_size")) != expected_window_size:
        raise ValueError(f"{label} config graph window mismatch")
    if _contract_value(graph_config, "ownership") != "right_transition":
        raise ValueError(f"{label} config ownership mismatch")
    if expected_architecture == "candidate_attention":
        recorded_bound = graph_config.get("residual_logit_bound")
        if recorded_bound is None or float(recorded_bound) != attention_logit_bound:
            raise ValueError(f"{label} bounded-Attention logit cap mismatch")
    return wrapper, dict(payload)


def _validate_candidate_contracts(payloads: Mapping[str, Mapping[str, Any]]) -> None:
    reference_name = "mlp"
    reference_config = payloads[reference_name]["config"]
    if not isinstance(reference_config, Mapping):
        raise TypeError("MLP temporal graph config must be a mapping")
    reference = {
        name: _contract_value(reference_config, name) for name in CANDIDATE_CONTRACT_FIELDS
    }
    for head_name, payload in payloads.items():
        graph_config = payload.get("config")
        if not isinstance(graph_config, Mapping):
            raise TypeError(f"{head_name} temporal graph config must be a mapping")
        mismatches = [
            name
            for name in CANDIDATE_CONTRACT_FIELDS
            if _contract_value(graph_config, name) != reference[name]
        ]
        if mismatches:
            raise ValueError(
                f"{head_name} uses a different candidate contract: " + ", ".join(mismatches)
            )


def _notebook(
    *,
    dataset_id: str,
    title: str,
    manifest_sha256: str,
    logit_bound: float,
    minimum_component_nodes: int,
) -> dict[str, Any]:
    notebook = _base_notebook(dataset_id, title, "public-applicable-v1")
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
        f"assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == {manifest_sha256!r}\n",
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
        f"primary_mlp = bundle / {PRIMARY_MLP_CHECKPOINT_NAME!r}\n",
        f"primary_attention = bundle / {PRIMARY_ATTENTION_CHECKPOINT_NAME!r}\n",
        f"fallback_mlp = bundle / {FALLBACK_MLP_CHECKPOINT_NAME!r}\n",
        f"fallback_attention = bundle / {FALLBACK_ATTENTION_CHECKPOINT_NAME!r}\n",
    ]
    command_start = source.index("command = [\n")
    command_end = source.index("]\n", command_start)
    source[command_end:command_end] = [
        "    '--temporal-graph-checkpoint', str(primary_mlp),\n",
        "    '--temporal-graph-attention-checkpoint', str(primary_attention),\n",
        "    '--temporal-graph-fallback-checkpoint', str(fallback_mlp),\n",
        "    '--temporal-graph-fallback-attention-checkpoint', str(fallback_attention),\n",
        f"    '--temporal-link-mode', {TEMPORAL_LINK_MODE!r},\n",
        f"    '--ensemble-logit-bound', {str(logit_bound)!r},\n",
        f"    '--minimum-component-nodes', {str(minimum_component_nodes)!r},\n",
    ]
    code = "".join(source)
    compile(code, f"{title}.ipynb", "exec")
    code_cell["source"] = source
    notebook["cells"][0]["source"] = [
        f"# {title}\n",
        "\n",
        "Frozen-host EXP-0015 T_graph=4 50:50 bounded-logit inference with "
        "T_graph=3 startup fallback.",
    ]
    notebook["metadata"]["agentic_kaggle"].update(
        {
            "experiment_id": "EXP-0015",
            "manifest_sha256": manifest_sha256,
            "temporal_link_mode": TEMPORAL_LINK_MODE,
            "primary_mlp_checkpoint": PRIMARY_MLP_CHECKPOINT_NAME,
            "primary_attention_checkpoint": PRIMARY_ATTENTION_CHECKPOINT_NAME,
            "fallback_mlp_checkpoint": FALLBACK_MLP_CHECKPOINT_NAME,
            "fallback_attention_checkpoint": FALLBACK_ATTENTION_CHECKPOINT_NAME,
            "minimum_component_nodes": minimum_component_nodes,
        }
    )
    return notebook


def _head_paths(heads: Mapping[str, Any], overrides: Mapping[str, Path | None]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in HEAD_SPECS:
        head = heads.get(name)
        if not isinstance(head, Mapping):
            raise ValueError(f"config is missing [heads.{name}]")
        override = overrides.get(name)
        paths[name] = (
            override.resolve()
            if override is not None
            else _resolve_competition_path(str(head["checkpoint_path"])).resolve()
        )
    return paths


def prepare(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
    *,
    base_bundle_override: Path | None = None,
    temporal_graph_checkpoint_override: Path | None = None,
    temporal_graph_attention_checkpoint_override: Path | None = None,
    temporal_graph_fallback_checkpoint_override: Path | None = None,
    temporal_graph_fallback_attention_checkpoint_override: Path | None = None,
    temporal_graph_source: Path = DEFAULT_TEMPORAL_GRAPH_SOURCE,
    inference_script: Path = DEFAULT_INFERENCE_SCRIPT,
    dataset_id_override: str | None = None,
    kernel_id_override: str | None = None,
    dataset_title_override: str | None = None,
    kernel_title_override: str | None = None,
) -> dict[str, Any]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    if str(config.get("experiment_id")) != "EXP-0015":
        raise ValueError("the T_graph=4 ensemble packager requires EXP-0015")

    source = _mapping(config, "source")
    data = _mapping(config, "data")
    fallback = _mapping(config, "fallback")
    heads = _mapping(config, "heads")
    ensemble = _mapping(config, "ensemble")
    inference = _mapping(config, "inference")
    submission = _mapping(config, "submission")
    output = _mapping(config, "output")

    image_window_size = int(data["image_window_size"])
    primary_window_size = int(data["graph_window_size"])
    fallback_window_size = int(fallback["graph_window_size"])
    if image_window_size != 2:
        raise ValueError("EXP-0015 requires the frozen T_image=2 host")
    if primary_window_size != 4:
        raise ValueError("EXP-0015 primary heads require graph_window_size=4")
    if fallback_window_size != 3:
        raise ValueError("EXP-0015 fallback heads require graph_window_size=3")
    if float(ensemble["mlp_weight"]) != 0.5 or float(ensemble["attention_weight"]) != 0.5:
        raise ValueError("EXP-0015 requires a 50:50 MLP/Attention ensemble")
    if ensemble.get("center_over_valid_candidates") is not True:
        raise ValueError("EXP-0015 requires centered valid-candidate logits")
    logit_bound = float(ensemble["logit_bound"])
    if logit_bound <= 0.0:
        raise ValueError("ensemble.logit_bound must be positive")
    minimum_component_nodes = int(inference["minimum_component_nodes"])
    if minimum_component_nodes != 7:
        raise ValueError("EXP-0015 requires minimum_component_nodes=7")

    expected_base_sha256 = _require_sha256(
        source["base_checkpoint_sha256"], "source.base_checkpoint_sha256"
    )
    expected_source_sha256 = _require_sha256(
        source["source_checkpoint_sha256"], "source.source_checkpoint_sha256"
    )
    primary_cache_fingerprint = _require_sha256(data["cache_fingerprint"], "data.cache_fingerprint")
    fallback_cache_fingerprint = _require_sha256(
        fallback["cache_fingerprint"], "fallback.cache_fingerprint"
    )
    primary_cache_manifest_sha256 = _require_sha256(
        data["cache_manifest_sha256"], "data.cache_manifest_sha256"
    )
    fallback_cache_manifest_sha256 = _require_sha256(
        fallback["cache_manifest_sha256"], "fallback.cache_manifest_sha256"
    )

    configured_base = _resolve_competition_path(str(source["base_checkpoint_path"]))
    base_bundle = (
        base_bundle_override.resolve()
        if base_bundle_override is not None
        else configured_base.parent.resolve()
    )
    head_paths = _head_paths(
        heads,
        {
            "mlp": temporal_graph_checkpoint_override,
            "attention": temporal_graph_attention_checkpoint_override,
            "fallback_mlp": temporal_graph_fallback_checkpoint_override,
            "fallback_attention": (temporal_graph_fallback_attention_checkpoint_override),
        },
    )
    if output_root is None:
        output_root = _resolve_competition_path(str(output["bundle_dir"]))
    output_root = output_root.resolve()

    base_sha256, base_metadata = _validate_base_bundle(config, base_bundle)
    if base_sha256 != expected_base_sha256:
        raise ValueError("validated base checkpoint does not match config")
    if base_metadata.get("source_checkpoint_sha256") != expected_source_sha256:
        raise ValueError("base bundle raw source checkpoint mismatch")

    payloads: dict[str, dict[str, Any]] = {}
    for name, spec in HEAD_SPECS.items():
        head_config = heads[name]
        if not isinstance(head_config, Mapping):
            raise TypeError(f"[heads.{name}] must be a mapping")
        expected_window = (
            primary_window_size if spec["window"] == "primary" else fallback_window_size
        )
        expected_cache = (
            primary_cache_fingerprint if spec["window"] == "primary" else fallback_cache_fingerprint
        )
        _, payload = _validate_head_checkpoint(
            label=name,
            path=head_paths[name],
            head_config=head_config,
            expected_architecture=str(spec["architecture"]),
            expected_window_size=expected_window,
            expected_image_window_size=image_window_size,
            expected_base_sha256=base_sha256,
            expected_cache_fingerprint=expected_cache,
            expected_source_sha256=expected_source_sha256,
            attention_logit_bound=logit_bound,
        )
        payloads[name] = payload
    _validate_candidate_contracts(payloads)

    if not inference_script.is_file():
        raise FileNotFoundError(f"integrated inference script is missing: {inference_script}")
    inference_source = inference_script.read_text(encoding="utf-8")
    required_wiring = (
        "--temporal-graph-checkpoint",
        "--temporal-graph-attention-checkpoint",
        "--temporal-graph-fallback-checkpoint",
        "--temporal-graph-fallback-attention-checkpoint",
        "--temporal-link-mode",
        "--minimum-component-nodes",
        "def _verify_bundle_manifest(",
    )
    missing_wiring = [item for item in required_wiring if item not in inference_source]
    if missing_wiring:
        raise ValueError(f"inference script is missing T_graph=4 wiring: {missing_wiring}")

    dataset_id = dataset_id_override or str(submission.get("dataset_id", DEFAULT_DATASET_ID))
    kernel_id = kernel_id_override or str(submission.get("kernel_id", DEFAULT_KERNEL_ID))
    dataset_title = dataset_title_override or str(
        submission.get("dataset_title", DEFAULT_DATASET_TITLE)
    )
    kernel_title = kernel_title_override or str(
        submission.get("kernel_title", DEFAULT_KERNEL_TITLE)
    )

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

    head_destinations: dict[str, Path] = {}
    for name, spec in HEAD_SPECS.items():
        destination = dataset_dir / str(spec["checkpoint_name"])
        shutil.copy2(head_paths[name], destination)
        copied.append(destination)
        head_destinations[name] = destination
    inference_destination = dataset_dir / "run_kaggle_inference.py"
    shutil.copy2(inference_script, inference_destination)
    copied.append(inference_destination)
    config_destination = dataset_dir / EXPERIMENT_CONFIG_NAME
    shutil.copy2(config_path, config_destination)
    copied.append(config_destination)
    graph_archive = dataset_dir / GRAPH_ARCHIVE_NAME
    _archive_temporal_graph(temporal_graph_source, graph_archive)
    copied.append(graph_archive)

    _write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "title": dataset_title,
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        },
    )
    manifest_heads: dict[str, Any] = {}
    for name, spec in HEAD_SPECS.items():
        head_config = heads[name]
        metadata = _checkpoint_metadata(payloads[name])
        manifest_heads[name] = {
            "checkpoint": spec["checkpoint_name"],
            "checkpoint_sha256": _sha256(head_destinations[name]),
            "experiment_id": head_config["experiment_id"],
            "completed_epoch": int(head_config["completed_epoch"]),
            "base_checkpoint_sha256": payloads[name]["base_checkpoint_sha256"],
            "cache_fingerprint": metadata["cache_fingerprint"],
            "source_raw_checkpoint_sha256": metadata["source_raw_checkpoint_sha256"],
            "config": dict(payloads[name]["config"]),
        }
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0015",
        "fold": int(data["fold"]),
        "base": {
            "experiment_id": source["base_experiment_id"],
            "completed_epochs": int(source["base_checkpoint_completed_epochs"]),
            "weights_sha256": base_sha256,
            "source_checkpoint_sha256": expected_source_sha256,
        },
        "temporal_contract": {
            "image_window_size": image_window_size,
            "primary_graph_window_size": primary_window_size,
            "fallback_graph_window_size": fallback_window_size,
            "primary_cache_fingerprint": primary_cache_fingerprint,
            "primary_cache_manifest_sha256": primary_cache_manifest_sha256,
            "fallback_cache_fingerprint": fallback_cache_fingerprint,
            "fallback_cache_manifest_sha256": fallback_cache_manifest_sha256,
            "ownership": "right_transition",
        },
        "temporal_graph_heads": manifest_heads,
        "ensemble": {
            **dict(ensemble),
            "mode": TEMPORAL_LINK_MODE,
        },
        "postprocess": {
            "profile": "public-applicable-v1",
            "minimum_component_nodes": minimum_component_nodes,
            "preserve_division_components": bool(inference["preserve_division_components"]),
        },
        "files": _file_manifest(dataset_dir, copied),
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)

    notebook_name = f"{kernel_id.split('/', 1)[-1]}.ipynb"
    _write_json(
        kernel_dir / notebook_name,
        _notebook(
            dataset_id=dataset_id,
            title=kernel_title,
            manifest_sha256=manifest_sha256,
            logit_bound=logit_bound,
            minimum_component_nodes=minimum_component_nodes,
        ),
    )
    _write_json(
        kernel_dir / "kernel-metadata.json",
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
        "experiment_id": "EXP-0015",
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "manifest_sha256": manifest_sha256,
        "inference_script_sha256": _sha256(inference_destination),
        "base_weights_sha256": base_sha256,
        "head_checkpoint_sha256": {
            name: _sha256(destination) for name, destination in head_destinations.items()
        },
        "temporal_link_mode": TEMPORAL_LINK_MODE,
        "minimum_component_nodes": minimum_component_nodes,
        "submission_message": submission.get(
            "submission_message",
            "EXP-0015 TGraph4 50-50 bounded-logit min-component-7",
        ),
    }
    _write_json(output_root / "bundle-manifest.json", result)
    print(json.dumps(result, indent=2))
    return result


def publish(result: Mapping[str, Any], *, dataset_version: bool = False) -> None:
    """Upload the private Dataset, then push its internet-disabled Notebook."""
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
        dataset_command.extend(["-m", "EXP-0015 TGraph4 bounded-logit 50:50"])
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
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-bundle", type=Path, default=None)
    parser.add_argument("--temporal-graph-checkpoint", type=Path, default=None)
    parser.add_argument("--temporal-graph-attention-checkpoint", type=Path, default=None)
    parser.add_argument("--temporal-graph-fallback-checkpoint", type=Path, default=None)
    parser.add_argument("--temporal-graph-fallback-attention-checkpoint", type=Path, default=None)
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
        help="Only generate and validate local artifacts (the default).",
    )
    mode.add_argument(
        "--publish",
        action="store_true",
        help="Push the Dataset and Notebook with Kaggle CLI.",
    )
    parser.add_argument(
        "--dataset-version",
        action="store_true",
        help="With --publish, create a new Dataset version.",
    )
    args = parser.parse_args()
    if args.dataset_version and not args.publish:
        parser.error("--dataset-version requires --publish")

    result = prepare(
        args.config.resolve(),
        args.output_root.resolve() if args.output_root is not None else None,
        base_bundle_override=(args.base_bundle.resolve() if args.base_bundle is not None else None),
        temporal_graph_checkpoint_override=(
            args.temporal_graph_checkpoint.resolve()
            if args.temporal_graph_checkpoint is not None
            else None
        ),
        temporal_graph_attention_checkpoint_override=(
            args.temporal_graph_attention_checkpoint.resolve()
            if args.temporal_graph_attention_checkpoint is not None
            else None
        ),
        temporal_graph_fallback_checkpoint_override=(
            args.temporal_graph_fallback_checkpoint.resolve()
            if args.temporal_graph_fallback_checkpoint is not None
            else None
        ),
        temporal_graph_fallback_attention_checkpoint_override=(
            args.temporal_graph_fallback_attention_checkpoint.resolve()
            if args.temporal_graph_fallback_attention_checkpoint is not None
            else None
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
