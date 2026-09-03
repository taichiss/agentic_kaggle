#!/usr/bin/env python
"""Package the controlled EXP-0016 T_graph=5/4/3 fallback submission."""

from __future__ import annotations

import argparse
import json
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
    _write_json,
)
from prepare_tgraph4_ensemble_submission import (
    _checkpoint_metadata,
    _mapping,
    _require_sha256,
    _validate_candidate_contracts,
    _validate_head_checkpoint,
)

DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0016-tgraph5-link-ensemble.toml"
DEFAULT_DATASET_ID = "suzukitaichi/biohub-exp-0016-tgraph5-link-5050"
DEFAULT_DATASET_TITLE = "Biohub EXP-0016 TGraph5 Link 5050"
DEFAULT_KERNEL_ID = "suzukitaichi/biohub-exp-0016-tgraph5-bounded-logit-5050-submit"
DEFAULT_KERNEL_TITLE = "Biohub EXP-0016 TGraph5 Bounded Logit 5050 Submit"

PRIMARY_MLP_CHECKPOINT_NAME = "temporal_graph_t5_mlp_checkpoint.pth"
PRIMARY_ATTENTION_CHECKPOINT_NAME = "temporal_graph_t5_attention_checkpoint.pth"
T4_FALLBACK_MLP_CHECKPOINT_NAME = "temporal_graph_t4_fallback_mlp_checkpoint.pth"
T4_FALLBACK_ATTENTION_CHECKPOINT_NAME = "temporal_graph_t4_fallback_attention_checkpoint.pth"
T3_FALLBACK_MLP_CHECKPOINT_NAME = "temporal_graph_t3_fallback_mlp_checkpoint.pth"
T3_FALLBACK_ATTENTION_CHECKPOINT_NAME = "temporal_graph_t3_fallback_attention_checkpoint.pth"
EXPERIMENT_CONFIG_NAME = "tgraph5-link-ensemble-experiment.toml"
TEMPORAL_LINK_MODE = "bounded_logit_5050"
ENSEMBLE_LOGIT_BOUND = 0.15
MINIMUM_COMPONENT_NODES = 7

HEAD_SPECS: dict[str, dict[str, str | int]] = {
    "mlp": {
        "architecture": "mlp",
        "tier": "primary",
        "window_size": 5,
        "checkpoint_name": PRIMARY_MLP_CHECKPOINT_NAME,
        "inference_flag": "--temporal-graph-checkpoint",
    },
    "attention": {
        "architecture": "candidate_attention",
        "tier": "primary",
        "window_size": 5,
        "checkpoint_name": PRIMARY_ATTENTION_CHECKPOINT_NAME,
        "inference_flag": "--temporal-graph-attention-checkpoint",
    },
    "t4_fallback_mlp": {
        "architecture": "mlp",
        "tier": "t4_fallback",
        "window_size": 4,
        "checkpoint_name": T4_FALLBACK_MLP_CHECKPOINT_NAME,
        "inference_flag": "--temporal-graph-t4-fallback-checkpoint",
    },
    "t4_fallback_attention": {
        "architecture": "candidate_attention",
        "tier": "t4_fallback",
        "window_size": 4,
        "checkpoint_name": T4_FALLBACK_ATTENTION_CHECKPOINT_NAME,
        "inference_flag": "--temporal-graph-t4-fallback-attention-checkpoint",
    },
    "t3_fallback_mlp": {
        "architecture": "mlp",
        "tier": "t3_fallback",
        "window_size": 3,
        "checkpoint_name": T3_FALLBACK_MLP_CHECKPOINT_NAME,
        "inference_flag": "--temporal-graph-fallback-checkpoint",
    },
    "t3_fallback_attention": {
        "architecture": "candidate_attention",
        "tier": "t3_fallback",
        "window_size": 3,
        "checkpoint_name": T3_FALLBACK_ATTENTION_CHECKPOINT_NAME,
        "inference_flag": "--temporal-graph-fallback-attention-checkpoint",
    },
}

TIER_SECTIONS = {
    "primary": "data",
    "t4_fallback": "fallback_t4",
    "t3_fallback": "fallback_t3",
}


def _head_paths(
    heads: Mapping[str, Any],
    overrides: Mapping[str, Path | None],
) -> dict[str, Path]:
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


def _tier_contracts(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for tier, section_name in TIER_SECTIONS.items():
        section = _mapping(config, section_name)
        expected_window = {"primary": 5, "t4_fallback": 4, "t3_fallback": 3}[tier]
        actual_window = int(section["graph_window_size"])
        if actual_window != expected_window:
            raise ValueError(f"EXP-0016 {tier} heads require graph_window_size={expected_window}")
        contracts[tier] = {
            "graph_window_size": actual_window,
            "cache_fingerprint": _require_sha256(
                section["cache_fingerprint"],
                f"{section_name}.cache_fingerprint",
            ),
            "cache_manifest_sha256": _require_sha256(
                section["cache_manifest_sha256"],
                f"{section_name}.cache_manifest_sha256",
            ),
        }
    return contracts


def _notebook(
    *,
    dataset_id: str,
    title: str,
    manifest_sha256: str,
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
    checkpoint_setup: list[str] = [
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
    ]
    for head_name, spec in HEAD_SPECS.items():
        variable = f"checkpoint_{head_name}"
        checkpoint_name = str(spec["checkpoint_name"])
        checkpoint_setup.extend(
            [
                f"{variable} = bundle / {checkpoint_name!r}\n",
                f"head_{head_name} = manifest['temporal_graph_heads'][{head_name!r}]\n",
                f"assert head_{head_name}['checkpoint'] == {checkpoint_name!r}\n",
                f"expected_{head_name} = manifest['files'][{checkpoint_name!r}]['sha256']\n",
                f"assert head_{head_name}['checkpoint_sha256'] == expected_{head_name}\n",
                f"actual_{head_name} = hashlib.sha256({variable}.read_bytes()).hexdigest()\n",
                f"assert actual_{head_name} == expected_{head_name}\n",
            ]
        )
    source[bundle_end:bundle_end] = checkpoint_setup

    command_start = source.index("command = [\n")
    command_end = source.index("]\n", command_start)
    command_arguments: list[str] = []
    for head_name, spec in HEAD_SPECS.items():
        command_arguments.append(
            f"    {str(spec['inference_flag'])!r}, str(checkpoint_{head_name}),\n"
        )
    command_arguments.extend(
        [
            f"    '--temporal-link-mode', {TEMPORAL_LINK_MODE!r},\n",
            f"    '--ensemble-logit-bound', {str(ENSEMBLE_LOGIT_BOUND)!r},\n",
            f"    '--minimum-component-nodes', {str(MINIMUM_COMPONENT_NODES)!r},\n",
        ]
    )
    source[command_end:command_end] = command_arguments
    code = "".join(source)
    compile(code, f"{title}.ipynb", "exec")
    code_cell["source"] = source
    notebook["cells"][0]["source"] = [
        f"# {title}\n",
        "\n",
        "Frozen-host EXP-0016 T_graph=5 bounded-logit 50:50 inference with "
        "T_graph=4, T_graph=3, then Host startup fallback.",
    ]
    notebook["metadata"]["agentic_kaggle"].update(
        {
            "experiment_id": "EXP-0016",
            "manifest_sha256": manifest_sha256,
            "temporal_link_mode": TEMPORAL_LINK_MODE,
            "ensemble_logit_bound": ENSEMBLE_LOGIT_BOUND,
            "minimum_component_nodes": MINIMUM_COMPONENT_NODES,
            "fallback_order": [5, 4, 3, "host"],
            "head_checkpoints": {
                name: spec["checkpoint_name"] for name, spec in HEAD_SPECS.items()
            },
        }
    )
    return notebook


def prepare(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
    *,
    base_bundle_override: Path | None = None,
    temporal_graph_checkpoint_override: Path | None = None,
    temporal_graph_attention_checkpoint_override: Path | None = None,
    temporal_graph_t4_fallback_checkpoint_override: Path | None = None,
    temporal_graph_t4_fallback_attention_checkpoint_override: Path | None = None,
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
    if str(config.get("experiment_id")) != "EXP-0016":
        raise ValueError("the T_graph=5 ensemble packager requires EXP-0016")

    source = _mapping(config, "source")
    data = _mapping(config, "data")
    heads = _mapping(config, "heads")
    ensemble = _mapping(config, "ensemble")
    inference = _mapping(config, "inference")
    controls = _mapping(config, "controls")
    submission = _mapping(config, "submission")
    output = _mapping(config, "output")
    tier_contracts = _tier_contracts(config)

    image_window_size = int(data["image_window_size"])
    if image_window_size != 2:
        raise ValueError("EXP-0016 requires the frozen T_image=2 host")
    if float(ensemble["mlp_weight"]) != 0.5 or float(ensemble["attention_weight"]) != 0.5:
        raise ValueError("EXP-0016 requires a 50:50 MLP/Attention ensemble")
    if ensemble.get("center_over_valid_candidates") is not True:
        raise ValueError("EXP-0016 requires centered valid-candidate logits")
    if float(ensemble["logit_bound"]) != ENSEMBLE_LOGIT_BOUND:
        raise ValueError(f"EXP-0016 requires ensemble.logit_bound={ENSEMBLE_LOGIT_BOUND}")
    if int(inference["minimum_component_nodes"]) != MINIMUM_COMPONENT_NODES:
        raise ValueError(f"EXP-0016 requires minimum_component_nodes={MINIMUM_COMPONENT_NODES}")
    if inference.get("base_postprocess_profile") != "public-applicable-v1":
        raise ValueError("EXP-0016 requires public-applicable-v1 post-processing")
    if inference.get("preserve_division_components") is not True:
        raise ValueError("EXP-0016 requires preserved division components")
    frozen_controls = (
        "freeze_image_model",
        "freeze_detection_head",
        "freeze_host_edge_scorer",
        "freeze_candidate_generation",
        "freeze_division_policy",
        "freeze_postprocess",
    )
    mutable_controls = [name for name in frozen_controls if controls.get(name) is not True]
    if mutable_controls:
        raise ValueError("EXP-0016 requires frozen Host controls: " + ", ".join(mutable_controls))
    if controls.get("effective_postprocess_profile") != ("public-applicable-v1-min-component-7"):
        raise ValueError("EXP-0016 postprocess control profile mismatch")

    expected_base_sha256 = _require_sha256(
        source["base_checkpoint_sha256"], "source.base_checkpoint_sha256"
    )
    expected_source_sha256 = _require_sha256(
        source["source_checkpoint_sha256"], "source.source_checkpoint_sha256"
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
            "t4_fallback_mlp": temporal_graph_t4_fallback_checkpoint_override,
            "t4_fallback_attention": (temporal_graph_t4_fallback_attention_checkpoint_override),
            "t3_fallback_mlp": temporal_graph_fallback_checkpoint_override,
            "t3_fallback_attention": (temporal_graph_fallback_attention_checkpoint_override),
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
        tier = str(spec["tier"])
        _, payload = _validate_head_checkpoint(
            label=name,
            path=head_paths[name],
            head_config=head_config,
            expected_architecture=str(spec["architecture"]),
            expected_window_size=int(spec["window_size"]),
            expected_image_window_size=image_window_size,
            expected_base_sha256=base_sha256,
            expected_cache_fingerprint=str(tier_contracts[tier]["cache_fingerprint"]),
            expected_source_sha256=expected_source_sha256,
            attention_logit_bound=ENSEMBLE_LOGIT_BOUND,
        )
        payloads[name] = payload
    _validate_candidate_contracts(payloads)

    if not inference_script.is_file():
        raise FileNotFoundError(f"integrated inference script is missing: {inference_script}")
    inference_source = inference_script.read_text(encoding="utf-8")
    required_wiring = (
        *(str(spec["inference_flag"]) for spec in HEAD_SPECS.values()),
        "--temporal-link-mode",
        "--ensemble-logit-bound",
        "--minimum-component-nodes",
        "def _verify_bundle_manifest(",
    )
    missing_wiring = [item for item in required_wiring if item not in inference_source]
    if missing_wiring:
        raise ValueError(f"inference script is missing T_graph=5 wiring: {missing_wiring}")

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
        "experiment_id": "EXP-0016",
        "fold": int(data["fold"]),
        "base": {
            "experiment_id": source["base_experiment_id"],
            "completed_epochs": int(source["base_checkpoint_completed_epochs"]),
            "weights_sha256": base_sha256,
            "source_checkpoint_sha256": expected_source_sha256,
        },
        "temporal_contract": {
            "image_window_size": image_window_size,
            "fallback_order": [5, 4, 3, "host"],
            "ownership": "right_transition",
            "tiers": tier_contracts,
        },
        "temporal_graph_heads": manifest_heads,
        "ensemble": {
            **dict(ensemble),
            "mode": TEMPORAL_LINK_MODE,
        },
        "postprocess": {
            "profile": "public-applicable-v1",
            "minimum_component_nodes": MINIMUM_COMPONENT_NODES,
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
        "experiment_id": "EXP-0016",
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
        "minimum_component_nodes": MINIMUM_COMPONENT_NODES,
        "fallback_order": [5, 4, 3, "host"],
        "submission_message": submission.get(
            "submission_message",
            "EXP-0016 TGraph5 50-50 bounded-logit min-component-7",
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
        dataset_command.extend(["-m", "EXP-0016 TGraph5 bounded-logit 50:50"])
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
    parser.add_argument("--temporal-graph-t4-fallback-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--temporal-graph-t4-fallback-attention-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument("--temporal-graph-fallback-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--temporal-graph-fallback-attention-checkpoint",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--temporal-graph-source",
        type=Path,
        default=DEFAULT_TEMPORAL_GRAPH_SOURCE,
    )
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
        temporal_graph_t4_fallback_checkpoint_override=(
            args.temporal_graph_t4_fallback_checkpoint.resolve()
            if args.temporal_graph_t4_fallback_checkpoint is not None
            else None
        ),
        temporal_graph_t4_fallback_attention_checkpoint_override=(
            args.temporal_graph_t4_fallback_attention_checkpoint.resolve()
            if args.temporal_graph_t4_fallback_attention_checkpoint is not None
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
