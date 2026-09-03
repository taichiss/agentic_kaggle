#!/usr/bin/env python
"""Package controlled T_graph=10/20 ensembles with a startup fallback stack."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
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

DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0017-tgraph10-link-ensemble.toml"
TEMPORAL_LINK_MODE = "bounded_logit_5050"
ENSEMBLE_LOGIT_BOUND = 0.15
MINIMUM_COMPONENT_NODES = 7
FALLBACK_STACK_NAME = "temporal_graph_fallback_stack.json"
EXPERIMENT_CONFIG_NAME = "long-tgraph-link-ensemble-experiment.toml"
LONG_CACHE_SCHEMA_VERSION = 4
LONG_FEATURE_SCHEMA = "tgraph-long-history-linear-aggregate-features-v1"

CONTROLLED_WINDOWS: dict[str, tuple[int, tuple[int, ...]]] = {
    "EXP-0017": (10, (5, 4, 3)),
    "EXP-0018": (20, (10, 5, 4, 3)),
}


def _head_name(tier: str, architecture: str) -> str:
    suffix = "attention" if architecture == "candidate_attention" else "mlp"
    return f"{tier}_{suffix}" if tier == "primary" else f"{tier}_fallback_{suffix}"


def _checkpoint_name(tier: str, window: int, architecture: str) -> str:
    suffix = "attention" if architecture == "candidate_attention" else "mlp"
    fallback = "_fallback" if tier != "primary" else ""
    return f"temporal_graph_t{window}{fallback}_{suffix}_checkpoint.pth"


def _tier_contracts(
    config: Mapping[str, Any],
    *,
    experiment_id: str,
) -> tuple[dict[str, Any], ...]:
    primary_window, expected_fallbacks = CONTROLLED_WINDOWS[experiment_id]
    data = _mapping(config, "data")
    if int(data["graph_window_size"]) != primary_window:
        raise ValueError(f"{experiment_id} requires graph_window_size={primary_window}")
    contracts: list[dict[str, Any]] = [
        {
            "tier": "primary",
            "graph_window_size": primary_window,
            "cache_fingerprint": _require_sha256(
                data["cache_fingerprint"], "data.cache_fingerprint"
            ),
            "cache_manifest_sha256": _require_sha256(
                data["cache_manifest_sha256"], "data.cache_manifest_sha256"
            ),
        }
    ]
    fallbacks = _mapping(config, "fallbacks")
    expected_names = {f"t{window}" for window in expected_fallbacks}
    if set(fallbacks) != expected_names:
        raise ValueError(
            f"{experiment_id} requires fallback tiers "
            + ", ".join(f"T_graph={window}" for window in expected_fallbacks)
        )
    for window in expected_fallbacks:
        tier = f"t{window}"
        section = _mapping(fallbacks, tier)
        if int(section["graph_window_size"]) != window:
            raise ValueError(f"fallbacks.{tier} requires graph_window_size={window}")
        contracts.append(
            {
                "tier": tier,
                "graph_window_size": window,
                "cache_fingerprint": _require_sha256(
                    section["cache_fingerprint"],
                    f"fallbacks.{tier}.cache_fingerprint",
                ),
                "cache_manifest_sha256": _require_sha256(
                    section["cache_manifest_sha256"],
                    f"fallbacks.{tier}.cache_manifest_sha256",
                ),
            }
        )
    return tuple(contracts)


def _head_specs(tiers: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for contract in tiers:
        tier = str(contract["tier"])
        window = int(contract["graph_window_size"])
        for architecture in ("mlp", "candidate_attention"):
            name = _head_name(tier, architecture)
            specs[name] = {
                "tier": tier,
                "architecture": architecture,
                "graph_window_size": window,
                "checkpoint_name": _checkpoint_name(tier, window, architecture),
            }
    return specs


def _head_paths(
    heads: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    overrides: Mapping[str, Path] | None,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in specs:
        head = heads.get(name)
        if not isinstance(head, Mapping):
            raise ValueError(f"config is missing [heads.{name}]")
        override = None if overrides is None else overrides.get(name)
        paths[name] = (
            override.resolve()
            if override is not None
            else _resolve_competition_path(str(head["checkpoint_path"])).resolve()
        )
    unknown = set(overrides or {}) - set(specs)
    if unknown:
        raise ValueError("unknown checkpoint overrides: " + ", ".join(sorted(unknown)))
    return paths


def _fallback_descriptor(
    tiers: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    fallbacks = []
    for contract in tiers[1:]:
        tier = str(contract["tier"])
        mlp = specs[_head_name(tier, "mlp")]
        attention = specs[_head_name(tier, "candidate_attention")]
        fallbacks.append(
            {
                "graph_window_size": int(contract["graph_window_size"]),
                "mlp_checkpoint": mlp["checkpoint_name"],
                "attention_checkpoint": attention["checkpoint_name"],
            }
        )
    return {"schema_version": 1, "fallbacks": fallbacks}


def _validate_tier_metadata(
    *,
    label: str,
    payload: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    """Bind long-window checkpoints to cache schema v4 and its fixed width."""
    metadata = _checkpoint_metadata(payload)
    configured_manifest = str(contract["cache_manifest_sha256"])
    recorded_manifest = metadata.get("cache_manifest_sha256")
    window = int(contract["graph_window_size"])
    if recorded_manifest is not None and recorded_manifest != configured_manifest:
        raise ValueError(f"{label} cache manifest SHA-256 mismatch")
    if window < 6:
        return
    if recorded_manifest != configured_manifest:
        raise ValueError(f"{label} long-window cache manifest is missing")
    if metadata.get("cache_schema_version") != LONG_CACHE_SCHEMA_VERSION:
        raise ValueError(f"{label} requires cache schema v{LONG_CACHE_SCHEMA_VERSION}")
    if metadata.get("feature_schema") != LONG_FEATURE_SCHEMA:
        raise ValueError(f"{label} long-window feature schema mismatch")
    graph_config = payload.get("config")
    if not isinstance(graph_config, Mapping):
        raise TypeError(f"{label} temporal graph config must be a mapping")
    expected_width = 3 * int(graph_config["node_feature_dim"]) + 26
    if metadata.get("feature_width") != expected_width:
        raise ValueError(
            f"{label} long-window feature width must be {expected_width}"
        )


def _notebook(
    *,
    dataset_id: str,
    title: str,
    experiment_id: str,
    manifest_sha256: str,
    tiers: Sequence[Mapping[str, Any]],
    specs: Mapping[str, Mapping[str, Any]],
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
    setup = [
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
    for name, spec in specs.items():
        checkpoint_name = str(spec["checkpoint_name"])
        setup.extend(
            [
                f"checkpoint_{name} = bundle / {checkpoint_name!r}\n",
                f"head_{name} = manifest['temporal_graph_heads'][{name!r}]\n",
                f"assert head_{name}['checkpoint'] == {checkpoint_name!r}\n",
                f"expected_{name} = manifest['files'][{checkpoint_name!r}]['sha256']\n",
                f"assert head_{name}['checkpoint_sha256'] == expected_{name}\n",
                "assert hashlib.sha256("
                f"checkpoint_{name}.read_bytes()).hexdigest() == expected_{name}\n",
            ]
        )
    setup.extend(
        [
            f"fallback_stack = bundle / {FALLBACK_STACK_NAME!r}\n",
            f"assert manifest['fallback_stack']['file'] == {FALLBACK_STACK_NAME!r}\n",
        ]
    )
    source[bundle_end:bundle_end] = setup

    command_start = source.index("command = [\n")
    command_end = source.index("]\n", command_start)
    primary_mlp = _head_name("primary", "mlp")
    primary_attention = _head_name("primary", "candidate_attention")
    source[command_end:command_end] = [
        f"    '--temporal-graph-checkpoint', str(checkpoint_{primary_mlp}),\n",
        f"    '--temporal-graph-attention-checkpoint', str(checkpoint_{primary_attention}),\n",
        "    '--temporal-graph-fallback-stack', str(fallback_stack),\n",
        f"    '--temporal-link-mode', {TEMPORAL_LINK_MODE!r},\n",
        f"    '--ensemble-logit-bound', {str(ENSEMBLE_LOGIT_BOUND)!r},\n",
        f"    '--minimum-component-nodes', {str(MINIMUM_COMPONENT_NODES)!r},\n",
    ]
    code = "".join(source)
    compile(code, f"{title}.ipynb", "exec")
    code_cell["source"] = source
    fallback_order = [int(tier["graph_window_size"]) for tier in tiers] + ["host"]
    notebook["cells"][0]["source"] = [
        f"# {title}\n",
        "\n",
        f"Frozen-host {experiment_id} bounded-logit 50:50 inference with ",
        f"startup fallback order {fallback_order}.",
    ]
    notebook["metadata"]["agentic_kaggle"].update(
        {
            "experiment_id": experiment_id,
            "manifest_sha256": manifest_sha256,
            "temporal_link_mode": TEMPORAL_LINK_MODE,
            "ensemble_logit_bound": ENSEMBLE_LOGIT_BOUND,
            "minimum_component_nodes": MINIMUM_COMPONENT_NODES,
            "fallback_order": fallback_order,
            "fallback_stack": FALLBACK_STACK_NAME,
            "head_checkpoints": {
                name: spec["checkpoint_name"] for name, spec in specs.items()
            },
        }
    )
    return notebook


def prepare(
    config_path: Path = DEFAULT_CONFIG,
    output_root: Path | None = None,
    *,
    base_bundle_override: Path | None = None,
    head_overrides: Mapping[str, Path] | None = None,
    temporal_graph_source: Path = DEFAULT_TEMPORAL_GRAPH_SOURCE,
    inference_script: Path = DEFAULT_INFERENCE_SCRIPT,
    dataset_id_override: str | None = None,
    kernel_id_override: str | None = None,
    dataset_title_override: str | None = None,
    kernel_title_override: str | None = None,
) -> dict[str, Any]:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    experiment_id = str(config.get("experiment_id"))
    if experiment_id not in CONTROLLED_WINDOWS:
        raise ValueError("long T_graph packager requires EXP-0017 or EXP-0018")

    source = _mapping(config, "source")
    data = _mapping(config, "data")
    heads = _mapping(config, "heads")
    ensemble = _mapping(config, "ensemble")
    inference = _mapping(config, "inference")
    controls = _mapping(config, "controls")
    submission = _mapping(config, "submission")
    output = _mapping(config, "output")
    tiers = _tier_contracts(config, experiment_id=experiment_id)
    specs = _head_specs(tiers)

    image_window_size = int(data["image_window_size"])
    if image_window_size != 2:
        raise ValueError(f"{experiment_id} requires the frozen T_image=2 host")
    if float(ensemble["mlp_weight"]) != 0.5 or float(ensemble["attention_weight"]) != 0.5:
        raise ValueError(f"{experiment_id} requires a 50:50 MLP/Attention ensemble")
    if ensemble.get("center_over_valid_candidates") is not True:
        raise ValueError(f"{experiment_id} requires centered valid-candidate logits")
    if float(ensemble["logit_bound"]) != ENSEMBLE_LOGIT_BOUND:
        raise ValueError(
            f"{experiment_id} requires ensemble.logit_bound={ENSEMBLE_LOGIT_BOUND}"
        )
    if int(inference["minimum_component_nodes"]) != MINIMUM_COMPONENT_NODES:
        raise ValueError(
            f"{experiment_id} requires minimum_component_nodes={MINIMUM_COMPONENT_NODES}"
        )
    if inference.get("base_postprocess_profile") != "public-applicable-v1":
        raise ValueError(f"{experiment_id} requires public-applicable-v1 post-processing")
    if inference.get("preserve_division_components") is not True:
        raise ValueError(f"{experiment_id} requires preserved division components")
    frozen_controls = (
        "freeze_image_model",
        "freeze_detection_head",
        "freeze_host_edge_scorer",
        "freeze_candidate_generation",
        "freeze_division_policy",
        "freeze_postprocess",
    )
    mutable_controls = [
        name for name in frozen_controls if controls.get(name) is not True
    ]
    if mutable_controls:
        raise ValueError(
            "long T_graph requires frozen Host controls: "
            + ", ".join(mutable_controls)
        )
    if (
        controls.get("effective_postprocess_profile")
        != "public-applicable-v1-min-component-7"
    ):
        raise ValueError("long T_graph postprocess control profile mismatch")

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
    paths = _head_paths(heads, specs, head_overrides)
    if output_root is None:
        output_root = _resolve_competition_path(str(output["bundle_dir"]))
    output_root = output_root.resolve()

    base_sha256, base_metadata = _validate_base_bundle(config, base_bundle)
    if base_sha256 != expected_base_sha256:
        raise ValueError("validated base checkpoint does not match config")
    if base_metadata.get("source_checkpoint_sha256") != expected_source_sha256:
        raise ValueError("base bundle raw source checkpoint mismatch")

    contracts_by_tier = {str(tier["tier"]): tier for tier in tiers}
    payloads: dict[str, dict[str, Any]] = {}
    for name, spec in specs.items():
        head_config = heads[name]
        if not isinstance(head_config, Mapping):
            raise TypeError(f"[heads.{name}] must be a mapping")
        contract = contracts_by_tier[str(spec["tier"])]
        _, payload = _validate_head_checkpoint(
            label=name,
            path=paths[name],
            head_config=head_config,
            expected_architecture=str(spec["architecture"]),
            expected_window_size=int(spec["graph_window_size"]),
            expected_image_window_size=image_window_size,
            expected_base_sha256=base_sha256,
            expected_cache_fingerprint=str(contract["cache_fingerprint"]),
            expected_source_sha256=expected_source_sha256,
            attention_logit_bound=ENSEMBLE_LOGIT_BOUND,
        )
        _validate_tier_metadata(
            label=name,
            payload=payload,
            contract=contract,
        )
        payloads[name] = payload
    candidate_payloads = {"mlp": payloads["primary_mlp"], **payloads}
    _validate_candidate_contracts(candidate_payloads)

    if not inference_script.is_file():
        raise FileNotFoundError(f"integrated inference script is missing: {inference_script}")
    inference_source = inference_script.read_text(encoding="utf-8")
    required_wiring = (
        "--temporal-graph-checkpoint",
        "--temporal-graph-attention-checkpoint",
        "--temporal-graph-fallback-stack",
        "--temporal-link-mode",
        "--ensemble-logit-bound",
        "--minimum-component-nodes",
        "history_pairs",
        "def _verify_bundle_manifest(",
    )
    missing_wiring = [item for item in required_wiring if item not in inference_source]
    if missing_wiring:
        raise ValueError(f"inference script is missing long T_graph wiring: {missing_wiring}")

    dataset_id = dataset_id_override or str(submission["dataset_id"])
    kernel_id = kernel_id_override or str(submission["kernel_id"])
    dataset_title = dataset_title_override or str(submission["dataset_title"])
    kernel_title = kernel_title_override or str(submission["kernel_title"])
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

    destinations: dict[str, Path] = {}
    for name, spec in specs.items():
        destination = dataset_dir / str(spec["checkpoint_name"])
        shutil.copy2(paths[name], destination)
        copied.append(destination)
        destinations[name] = destination
    inference_destination = dataset_dir / "run_kaggle_inference.py"
    shutil.copy2(inference_script, inference_destination)
    copied.append(inference_destination)
    config_destination = dataset_dir / EXPERIMENT_CONFIG_NAME
    shutil.copy2(config_path, config_destination)
    copied.append(config_destination)
    fallback_stack = dataset_dir / FALLBACK_STACK_NAME
    descriptor = _fallback_descriptor(tiers, specs)
    _write_json(fallback_stack, descriptor)
    copied.append(fallback_stack)
    graph_archive = dataset_dir / GRAPH_ARCHIVE_NAME
    _archive_temporal_graph(temporal_graph_source, graph_archive)
    copied.append(graph_archive)

    _write_json(
        dataset_dir / "dataset-metadata.json",
        {"title": dataset_title, "id": dataset_id, "licenses": [{"name": "other"}]},
    )
    manifest_heads: dict[str, Any] = {}
    for name, spec in specs.items():
        head_config = heads[name]
        metadata = _checkpoint_metadata(payloads[name])
        manifest_heads[name] = {
            "checkpoint": spec["checkpoint_name"],
            "checkpoint_sha256": _sha256(destinations[name]),
            "experiment_id": head_config["experiment_id"],
            "completed_epoch": int(head_config["completed_epoch"]),
            "base_checkpoint_sha256": payloads[name]["base_checkpoint_sha256"],
            "cache_fingerprint": metadata["cache_fingerprint"],
            "source_raw_checkpoint_sha256": metadata["source_raw_checkpoint_sha256"],
            "config": dict(payloads[name]["config"]),
        }
    fallback_order = [int(tier["graph_window_size"]) for tier in tiers] + ["host"]
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "fold": int(data["fold"]),
        "base": {
            "experiment_id": source["base_experiment_id"],
            "completed_epochs": int(source["base_checkpoint_completed_epochs"]),
            "weights_sha256": base_sha256,
            "source_checkpoint_sha256": expected_source_sha256,
        },
        "temporal_contract": {
            "image_window_size": image_window_size,
            "fallback_order": fallback_order,
            "ownership": "right_transition",
            "tiers": {
                str(tier["tier"]): {
                    key: value for key, value in tier.items() if key != "tier"
                }
                for tier in tiers
            },
        },
        "fallback_stack": {
            "file": FALLBACK_STACK_NAME,
            "sha256": _sha256(fallback_stack),
            **descriptor,
        },
        "temporal_graph_heads": manifest_heads,
        "ensemble": {**dict(ensemble), "mode": TEMPORAL_LINK_MODE},
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
            experiment_id=experiment_id,
            manifest_sha256=manifest_sha256,
            tiers=tiers,
            specs=specs,
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
        "experiment_id": experiment_id,
        "dataset_id": dataset_id,
        "dataset_dir": str(dataset_dir),
        "kernel_id": kernel_id,
        "kernel_dir": str(kernel_dir),
        "manifest_sha256": manifest_sha256,
        "inference_script_sha256": _sha256(inference_destination),
        "base_weights_sha256": base_sha256,
        "head_checkpoint_sha256": {
            name: _sha256(destination) for name, destination in destinations.items()
        },
        "temporal_link_mode": TEMPORAL_LINK_MODE,
        "minimum_component_nodes": MINIMUM_COMPONENT_NODES,
        "fallback_order": fallback_order,
        "submission_message": submission["submission_message"],
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
        dataset_command.extend(["-m", f"{result['experiment_id']} long TGraph ensemble"])
    subprocess.run(dataset_command, check=True, cwd=REPOSITORY_ROOT)
    subprocess.run(
        ["uv", "run", "kaggle", "kernels", "push", "-p", str(result["kernel_dir"])],
        check=True,
        cwd=REPOSITORY_ROOT,
    )


def _parse_head_overrides(raw_values: Sequence[str]) -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    for raw in raw_values:
        name, separator, path = raw.partition("=")
        if not separator or not name or not path or name in overrides:
            raise ValueError("--head-checkpoint requires unique NAME=PATH values")
        overrides[name] = Path(path).resolve()
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--base-bundle", type=Path, default=None)
    parser.add_argument(
        "--head-checkpoint",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override one configured head checkpoint; may be repeated.",
    )
    parser.add_argument("--temporal-graph-source", type=Path, default=DEFAULT_TEMPORAL_GRAPH_SOURCE)
    parser.add_argument("--inference-script", type=Path, default=DEFAULT_INFERENCE_SCRIPT)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--kernel-id", default=None)
    parser.add_argument("--dataset-title", default=None)
    parser.add_argument("--kernel-title", default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--generate-only", action="store_true")
    mode.add_argument("--publish", action="store_true")
    parser.add_argument("--dataset-version", action="store_true")
    args = parser.parse_args()
    if args.dataset_version and not args.publish:
        parser.error("--dataset-version requires --publish")
    try:
        head_overrides = _parse_head_overrides(args.head_checkpoint)
    except ValueError as error:
        parser.error(str(error))

    result = prepare(
        args.config.resolve(),
        args.output_root.resolve() if args.output_root is not None else None,
        base_bundle_override=(args.base_bundle.resolve() if args.base_bundle is not None else None),
        head_overrides=head_overrides,
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
