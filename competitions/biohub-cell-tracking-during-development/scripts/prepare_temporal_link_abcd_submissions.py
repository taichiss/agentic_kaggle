#!/usr/bin/env python
"""Package the EXP-0014 dual-head Dataset and four private Kaggle Notebooks."""

from __future__ import annotations

import argparse
import json
import shutil
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

DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0014-tgraph3-link-ensemble-abcd.toml"
MLP_CHECKPOINT_NAME = "temporal_graph_mlp_checkpoint.pth"
ATTENTION_CHECKPOINT_NAME = "temporal_graph_attention_checkpoint.pth"
EXPERIMENT_CONFIG_NAME = "temporal-link-abcd-experiment.toml"
ALLOWED_MODES = {
    "mlp",
    "bounded_attention",
    "bounded_logit_5050",
    "agreement_gate",
}
CANDIDATE_CONTRACT_FIELDS = (
    "node_feature_dim",
    "top_k",
    "radius_um",
    "distance_scale_um",
    "middle_coord_atol",
    "image_window_size",
    "graph_window_size",
    "ownership",
)


def _checkpoint_metadata(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise TypeError("temporal graph checkpoint metadata must be a mapping")
    return metadata


def _validate_pair_contract(
    config: Mapping[str, Any],
    mlp_payload: Mapping[str, Any],
    attention_payload: Mapping[str, Any],
) -> None:
    mlp_config = mlp_payload["config"]
    attention_config = attention_payload["config"]
    if not isinstance(mlp_config, Mapping) or not isinstance(attention_config, Mapping):
        raise TypeError("temporal graph checkpoint configs must be mappings")
    mlp_architecture = str(mlp_config.get("architecture", "mlp"))
    attention_architecture = str(attention_config.get("architecture", "mlp"))
    if mlp_architecture != "mlp":
        raise ValueError("configured MLP checkpoint does not use the MLP architecture")
    if attention_architecture != "candidate_attention":
        raise ValueError("configured Attention checkpoint is not candidate_attention")

    mismatches = [
        name
        for name in CANDIDATE_CONTRACT_FIELDS
        if mlp_config.get(name, _contract_default(name))
        != attention_config.get(name, _contract_default(name))
    ]
    if mismatches:
        raise ValueError(
            "temporal heads use different candidate contracts: "
            + ", ".join(mismatches)
        )

    data = config["data"]
    expected_cache = str(data["cache_fingerprint"])
    expected_source = str(config["source"]["source_checkpoint_sha256"])
    for label, payload in (("MLP", mlp_payload), ("Attention", attention_payload)):
        metadata = _checkpoint_metadata(payload)
        if metadata.get("cache_fingerprint") != expected_cache:
            raise ValueError(f"{label} cache fingerprint mismatch")
        if metadata.get("source_raw_checkpoint_sha256") != expected_source:
            raise ValueError(f"{label} source checkpoint mismatch")


def _contract_default(name: str) -> object:
    defaults: dict[str, object] = {
        "middle_coord_atol": 1.0e-4,
        "image_window_size": 2,
        "graph_window_size": 3,
        "ownership": "right_transition",
    }
    return defaults.get(name)


def _variants(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    submission = config.get("submission")
    if not isinstance(submission, Mapping):
        raise ValueError("config is missing [submission]")
    raw_variants = submission.get("variants")
    if not isinstance(raw_variants, list) or len(raw_variants) != 4:
        raise ValueError("EXP-0014 requires exactly four submission variants")
    variants = [dict(item) for item in raw_variants if isinstance(item, Mapping)]
    if len(variants) != 4:
        raise TypeError("submission variants must be mappings")
    modes = [str(item.get("mode", "")) for item in variants]
    if set(modes) != ALLOWED_MODES or len(set(modes)) != 4:
        raise ValueError("submission variants must cover each temporal-link mode once")
    required = {
        "arm",
        "variant",
        "mode",
        "kernel_id",
        "kernel_title",
        "submission_message",
    }
    for item in variants:
        missing = sorted(required - item.keys())
        if missing:
            raise ValueError(f"submission variant is missing: {missing}")
    return variants


def _notebook(
    *,
    dataset_id: str,
    title: str,
    mode: str,
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
        f"mlp_checkpoint = bundle / {MLP_CHECKPOINT_NAME!r}\n",
        f"attention_checkpoint = bundle / {ATTENTION_CHECKPOINT_NAME!r}\n",
    ]
    command_start = source.index("command = [\n")
    command_end = source.index("]\n", command_start)
    source[command_end:command_end] = [
        "    '--temporal-graph-checkpoint', str(mlp_checkpoint),\n",
        "    '--temporal-graph-attention-checkpoint', str(attention_checkpoint),\n",
        f"    '--temporal-link-mode', {mode!r},\n",
        f"    '--ensemble-logit-bound', {str(logit_bound)!r},\n",
        f"    '--minimum-component-nodes', {str(minimum_component_nodes)!r},\n",
    ]
    code = "".join(source)
    compile(code, f"{title}.ipynb", "exec")
    code_cell["source"] = source
    notebook["cells"][0]["source"] = [
        f"# {title}\n",
        "\n",
        "Frozen-host EXP-0014 temporal-link ABCD comparison.",
    ]
    notebook["metadata"]["agentic_kaggle"].update(
        {
            "experiment_id": "EXP-0014",
            "manifest_sha256": manifest_sha256,
            "temporal_link_mode": mode,
            "mlp_checkpoint": MLP_CHECKPOINT_NAME,
            "attention_checkpoint": ATTENTION_CHECKPOINT_NAME,
            "minimum_component_nodes": minimum_component_nodes,
        }
    )
    return notebook


def prepare(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
    *,
    base_bundle_override: Path | None = None,
    mlp_checkpoint_override: Path | None = None,
    attention_checkpoint_override: Path | None = None,
    temporal_graph_source: Path = DEFAULT_TEMPORAL_GRAPH_SOURCE,
    inference_script: Path = DEFAULT_INFERENCE_SCRIPT,
) -> dict[str, Any]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    experiment_id = str(config["experiment_id"])
    if experiment_id != "EXP-0014":
        raise ValueError("the ABCD packager requires experiment_id=EXP-0014")
    variants = _variants(config)
    source = config["source"]
    heads = config["heads"]
    inference = config["inference"]
    submission = config["submission"]

    configured_base = _resolve_competition_path(source["base_checkpoint_path"])
    base_bundle = (
        base_bundle_override.resolve()
        if base_bundle_override is not None
        else configured_base.parent
    )
    mlp_checkpoint = (
        mlp_checkpoint_override.resolve()
        if mlp_checkpoint_override is not None
        else _resolve_competition_path(heads["mlp"]["checkpoint_path"])
    )
    attention_checkpoint = (
        attention_checkpoint_override.resolve()
        if attention_checkpoint_override is not None
        else _resolve_competition_path(heads["attention"]["checkpoint_path"])
    )
    if output_root is None:
        output_root = _resolve_competition_path(config["output"]["bundle_dir"])
    output_root = output_root.resolve()

    base_sha256, base_metadata = _validate_base_bundle(config, base_bundle)
    _, mlp_payload = _validate_graph_checkpoint(
        mlp_checkpoint,
        expected_base_sha256=base_sha256,
        expected_completed_epoch=int(heads["mlp"]["completed_epoch"]),
        expected_experiment_id=str(heads["mlp"]["experiment_id"]),
    )
    _, attention_payload = _validate_graph_checkpoint(
        attention_checkpoint,
        expected_base_sha256=base_sha256,
        expected_completed_epoch=int(heads["attention"]["completed_epoch"]),
        expected_experiment_id=str(heads["attention"]["experiment_id"]),
    )
    for label, path, expected in (
        ("MLP", mlp_checkpoint, heads["mlp"]["checkpoint_sha256"]),
        ("Attention", attention_checkpoint, heads["attention"]["checkpoint_sha256"]),
    ):
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(
                f"{label} checkpoint SHA-256 mismatch: expected {expected}, found {actual}"
            )
    _validate_pair_contract(config, mlp_payload, attention_payload)
    if not inference_script.is_file():
        raise FileNotFoundError(f"integrated inference script is missing: {inference_script}")
    inference_source = inference_script.read_text(encoding="utf-8")
    required_wiring = (
        "--temporal-graph-attention-checkpoint",
        "--temporal-link-mode",
        "--minimum-component-nodes",
        "def _verify_bundle_manifest(",
    )
    missing_wiring = [value for value in required_wiring if value not in inference_source]
    if missing_wiring:
        raise ValueError(f"inference script is missing ABCD wiring: {missing_wiring}")

    dataset_dir = output_root / "dataset"
    kernels_dir = output_root / "kernels"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    kernels_dir.mkdir(parents=True, exist_ok=True)
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

    mlp_destination = dataset_dir / MLP_CHECKPOINT_NAME
    attention_destination = dataset_dir / ATTENTION_CHECKPOINT_NAME
    shutil.copy2(mlp_checkpoint, mlp_destination)
    shutil.copy2(attention_checkpoint, attention_destination)
    copied.extend((mlp_destination, attention_destination))
    inference_destination = dataset_dir / "run_kaggle_inference.py"
    shutil.copy2(inference_script, inference_destination)
    copied.append(inference_destination)
    config_destination = dataset_dir / EXPERIMENT_CONFIG_NAME
    shutil.copy2(config_path, config_destination)
    copied.append(config_destination)
    graph_archive = dataset_dir / GRAPH_ARCHIVE_NAME
    _archive_temporal_graph(temporal_graph_source, graph_archive)
    copied.append(graph_archive)

    dataset_id = str(submission["dataset_id"])
    _write_json(
        dataset_dir / "dataset-metadata.json",
        {
            "title": str(submission["dataset_title"]),
            "id": dataset_id,
            "licenses": [{"name": "other"}],
        },
    )
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "fold": int(config["data"]["fold"]),
        "base": {
            "experiment_id": source["base_experiment_id"],
            "completed_epochs": int(source["base_checkpoint_completed_epochs"]),
            "weights_sha256": base_sha256,
            "source_checkpoint_sha256": base_metadata.get(
                "source_checkpoint_sha256"
            ),
        },
        "temporal_graph_heads": {
            "mlp": {
                "checkpoint": MLP_CHECKPOINT_NAME,
                "checkpoint_sha256": _sha256(mlp_destination),
                "completed_epoch": int(heads["mlp"]["completed_epoch"]),
                "config": dict(mlp_payload["config"]),
            },
            "attention": {
                "checkpoint": ATTENTION_CHECKPOINT_NAME,
                "checkpoint_sha256": _sha256(attention_destination),
                "completed_epoch": int(heads["attention"]["completed_epoch"]),
                "config": dict(attention_payload["config"]),
            },
        },
        "ensemble": dict(config["ensemble"]),
        "postprocess": {
            "profile": config["controls"]["effective_postprocess_profile"],
            "minimum_component_nodes": int(inference["minimum_component_nodes"]),
            "preserve_division_components": bool(
                inference["preserve_division_components"]
            ),
        },
        "files": _file_manifest(dataset_dir, copied),
    }
    manifest_path = dataset_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)

    kernel_results: list[dict[str, Any]] = []
    for variant in variants:
        mode = str(variant["mode"])
        kernel_id = str(variant["kernel_id"])
        kernel_dir = kernels_dir / mode
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = f"{kernel_id.split('/', 1)[-1]}.ipynb"
        _write_json(
            kernel_dir / notebook_name,
            _notebook(
                dataset_id=dataset_id,
                title=str(variant["kernel_title"]),
                mode=mode,
                manifest_sha256=manifest_sha256,
                logit_bound=float(config["ensemble"]["logit_bound"]),
                minimum_component_nodes=int(inference["minimum_component_nodes"]),
            ),
        )
        _write_json(
            kernel_dir / "kernel-metadata.json",
            {
                "id": kernel_id,
                "title": str(variant["kernel_title"]),
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
        kernel_results.append(
            {
                "arm": variant["arm"],
                "variant": variant["variant"],
                "mode": mode,
                "kernel_id": kernel_id,
                "kernel_dir": str(kernel_dir),
                "submission_message": variant["submission_message"],
            }
        )

    result = {
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "manifest_sha256": manifest_sha256,
        "inference_script_sha256": _sha256(inference_destination),
        "base_weights_sha256": base_sha256,
        "mlp_checkpoint_sha256": _sha256(mlp_destination),
        "attention_checkpoint_sha256": _sha256(attention_destination),
        "postprocess_profile": config["controls"]["effective_postprocess_profile"],
        "minimum_component_nodes": int(inference["minimum_component_nodes"]),
        "kernels": kernel_results,
    }
    _write_json(output_root / "bundle-manifest.json", result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-bundle", type=Path, default=None)
    parser.add_argument("--mlp-checkpoint", type=Path, default=None)
    parser.add_argument("--attention-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--temporal-graph-source",
        type=Path,
        default=DEFAULT_TEMPORAL_GRAPH_SOURCE,
    )
    parser.add_argument(
        "--inference-script",
        type=Path,
        default=DEFAULT_INFERENCE_SCRIPT,
    )
    args = parser.parse_args()
    prepare(
        args.config.resolve(),
        args.output_root,
        base_bundle_override=args.base_bundle,
        mlp_checkpoint_override=args.mlp_checkpoint,
        attention_checkpoint_override=args.attention_checkpoint,
        temporal_graph_source=args.temporal_graph_source,
        inference_script=args.inference_script,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
