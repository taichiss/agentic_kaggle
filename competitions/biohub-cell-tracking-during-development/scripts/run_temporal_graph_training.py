#!/usr/bin/env python
"""Cache frozen host pairs and train compact temporal-graph residual heads.

The organizer image model always receives exactly two consecutive frames.  Two
adjacent frozen pairs are then combined into a three-frame graph window and the
compact residual head owns only the right transition.  Sparse annotations are
used only when one annotated parent is represented by exactly one detected
source; missing annotations never become negative or null labels.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0009-host-tgraph3-residual-30e.toml"
DEFAULT_INFERENCE_BUNDLE = (
    COMPETITION_ROOT
    / "data/kaggle-submission-EXP-0004-epoch30-postprocess-v1/dataset"
)
DEFAULT_VALIDATION_MANIFEST = COMPETITION_ROOT / "artifacts/EXP-0007/validation_split.json"

sys.path.insert(0, str(COMPETITION_ROOT / "src"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from temporal_graph import (  # noqa: E402
    FrozenPair,
    TemporalGraphConfig,
    TemporalGraphResidualHead,
    build_candidate_features,
    build_parent_candidates,
    candidate_feature_dim,
)
from torch.utils.data import DataLoader, Dataset  # noqa: E402

# Bump the corresponding value whenever feature order or feature math changes;
# cache fingerprints include this schema and must never alias two equations.
CACHE_SCHEMA_VERSION = 1  # Legacy T_graph=3 compact cache.
CACHE_SCHEMA_VERSIONS = {3: 1, 4: 2}
CACHE_FEATURE_SCHEMAS = {
    1: "tgraph3-candidate-features-v1",
    2: "tgraph4-acceleration-qbar-features-v1",
}
TRAINING_CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceBundle:
    """Validated flattened EXP-0004 epoch-30 host artifact."""

    organizer_repo: Path
    organizer_revision: str
    bundle_dir: Path
    weights_path: Path
    weights_sha256: str
    raw_checkpoint_sha256: str
    model_config: dict[str, Any]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ExtractedPair:
    """Frozen pair plus proposal-to-GT mappings needed by sparse supervision."""

    pair: FrozenPair
    source_coords_grid: torch.Tensor
    target_coords_grid: torch.Tensor
    source_matches: torch.Tensor
    target_matches: torch.Tensor


@dataclass(frozen=True)
class SparseExamples:
    """Compact target-major examples for masked parent cross-entropy."""

    features: torch.Tensor
    base_logits: torch.Tensor
    valid_mask: torch.Tensor
    labels: torch.Tensor
    supervised_parents: int
    candidate_parents: int


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        config = tomllib.load(file)
    if int(config.get("schema_version", 0)) != 1:
        raise ValueError("unsupported experiment config schema")
    return config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _cache_schema_version(graph_window_size: int) -> int:
    if isinstance(graph_window_size, bool) or not isinstance(graph_window_size, int):
        raise TypeError("graph_window_size must be an integer")
    try:
        return CACHE_SCHEMA_VERSIONS[graph_window_size]
    except KeyError as error:
        raise ValueError(
            "compact cache supports only graph_window_size 3 or 4"
        ) from error


def _cache_feature_schema(schema_version: int) -> str:
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise TypeError("cache schema_version must be an integer")
    try:
        return CACHE_FEATURE_SCHEMAS[schema_version]
    except KeyError as error:
        raise ValueError(f"unsupported cache schema: {schema_version}") from error


def _strict_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _resolve_competition_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else COMPETITION_ROOT / path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as file:
            os.fsync(file.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _source_value(source: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in source:
            return source[name]
    return None


def _validate_source_bundle(config: dict[str, Any]) -> SourceBundle:
    source = config["source"]
    repo_raw = _source_value(
        source, "organizer_repository_path", "repository_path"
    )
    revision_expected = _source_value(
        source, "organizer_revision", "revision"
    )
    if not repo_raw or not revision_expected:
        raise ValueError("source must declare organizer repository and revision")
    organizer_repo = _resolve_competition_path(repo_raw).resolve()
    revision = _git_revision(organizer_repo)
    if revision != revision_expected:
        raise ValueError(
            f"organizer revision mismatch: expected {revision_expected}, got {revision}"
        )

    bundle_raw = _source_value(source, "base_inference_bundle_path", "bundle_path")
    bundle_dir = (
        _resolve_competition_path(bundle_raw).resolve()
        if bundle_raw
        else DEFAULT_INFERENCE_BUNDLE.resolve()
    )
    weights_raw = _source_value(
        source,
        "base_inference_weights_path",
        "base_weights_path",
        "base_checkpoint_path",
    )
    weights_path = (
        _resolve_competition_path(weights_raw).resolve()
        if weights_raw
        else bundle_dir / "edge_predictor_best.pth"
    )
    metadata_path = bundle_dir / "checkpoint-metadata.json"
    model_config_path = bundle_dir / "config.json"
    for path in (weights_path, metadata_path, model_config_path):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"required frozen-host artifact is missing: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_config = json.loads(model_config_path.read_text(encoding="utf-8"))
    actual_weights_sha = _sha256(weights_path)
    expected_weights_sha = str(
        _source_value(
            source,
            "base_inference_weights_sha256",
            "weights_sha256",
            "base_checkpoint_sha256",
        )
        or metadata.get("weights_sha256", "")
    )
    if not expected_weights_sha or actual_weights_sha != expected_weights_sha:
        raise ValueError(
            "flattened host-weight SHA-256 mismatch: "
            f"expected {expected_weights_sha!r}, got {actual_weights_sha}"
        )
    if str(metadata.get("weights_sha256")) != actual_weights_sha:
        raise ValueError("checkpoint metadata does not identify the flattened host weights")

    raw_sha = str(
        _source_value(source, "source_checkpoint_sha256")
        or metadata.get("source_checkpoint_sha256", "")
    )
    if not raw_sha:
        raise ValueError("source.source_checkpoint_sha256 is required")
    if str(metadata.get("source_checkpoint_sha256")) != raw_sha:
        raise ValueError("checkpoint metadata does not identify the declared raw checkpoint")
    expected_epoch = int(source.get("base_checkpoint_completed_epochs", 30))
    if int(metadata.get("completed_epochs", -1)) != expected_epoch:
        raise ValueError("checkpoint metadata completed epoch mismatch")
    if int(model_config.get("window_size", -1)) != 2:
        raise ValueError("the frozen host artifact must use T_image=2")
    configured_downsample = tuple(int(value) for value in config["data"]["downsample"])
    artifact_downsample = tuple(int(value) for value in model_config["downsample"])
    if configured_downsample != artifact_downsample:
        raise ValueError(
            f"host/config downsample mismatch: {artifact_downsample} != "
            f"{configured_downsample}"
        )

    return SourceBundle(
        organizer_repo=organizer_repo,
        organizer_revision=revision,
        bundle_dir=bundle_dir,
        weights_path=weights_path,
        weights_sha256=actual_weights_sha,
        raw_checkpoint_sha256=raw_sha,
        model_config=model_config,
        metadata=metadata,
    )


def _graph_config(config: dict[str, Any], source: SourceBundle) -> TemporalGraphConfig:
    model = config["model"]
    candidates = model.get("candidates", {})
    feature_dim = int(source.model_config["unet_out_channels"])
    distance_scale = float(
        candidates.get("distance_scale_um", model.get("distance_scale_um", 10.0))
    )
    return TemporalGraphConfig(
        node_feature_dim=feature_dim,
        hidden_dim=int(model.get("hidden_dim", 64)),
        top_k=int(candidates.get("top_k_per_target", 8)),
        radius_um=float(candidates.get("radius_um", 15.0)),
        distance_scale_um=distance_scale,
        dropout=float(model.get("dropout", 0.1)),
        image_window_size=_strict_int(
            config["data"].get("image_window_size", 2),
            "data.image_window_size",
        ),
        graph_window_size=_strict_int(
            config["data"].get("graph_window_size", 3),
            "data.graph_window_size",
        ),
        architecture=str(model.get("architecture", "mlp")),
        attention_heads=int(model.get("attention_heads", 4)),
        residual_logit_bound=(
            float(model["residual_logit_bound"])
            if model.get("residual_logit_bound") is not None
            else None
        ),
    )


def _validation_selection(
    config: dict[str, Any], *, allow_incomplete_smoke: bool = False
) -> tuple[Path, str, list[str], str]:
    cache = config.get("cache", {})
    raw_path = cache.get(
        "validation_manifest",
        config["data"].get("validation_manifest", DEFAULT_VALIDATION_MANIFEST),
    )
    path = _resolve_competition_path(raw_path).resolve()
    if path.is_file() and not path.is_symlink():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = config["data"]
        train_dir = _resolve_competition_path(data["train_dir"]).resolve()
        try:
            folds, _ = _build_grouped_folds(
                train_dir, str(data["group_delimiter"])
            )
            selected = folds[int(data["fold"])]
            validation = sorted(str(name) for name in selected["test"])
        except ValueError:
            if not allow_incomplete_smoke:
                raise
            delimiter = str(data["group_delimiter"])
            expected_group = str(data["validation_group"])
            validation = sorted(
                path.name.removesuffix(".zarr")
                for path in train_dir.glob("*.zarr")
                if path.name.removesuffix(".zarr").split(delimiter, 1)[0]
                == expected_group
                and (
                    train_dir / f"{path.name.removesuffix('.zarr')}.geff"
                ).exists()
            )
            if not validation:
                raise ValueError(
                    "incomplete smoke has no paired dataset from validation_group"
                ) from None
        fallback_count = int(cache.get("validation_fallback_count", 35))
        generator = random.Random(int(config["seed"]))
        generator.shuffle(validation)
        payload = {
            "schema_version": 1,
            "seed": int(config["seed"]),
            "method": "seeded-shuffle-of-name-sorted-validation-group",
            "source_missing_manifest": str(path),
            "calibration": sorted(validation[:fallback_count]),
            "report": sorted(validation[fallback_count:]),
        }
        artifact_dir = _resolve_competition_path(config["output"]["artifact_dir"])
        path = (artifact_dir / "validation_split_fallback.json").resolve()
        _atomic_json(path, payload)
        print(
            f"Validation manifest missing; wrote deterministic fallback: {path}",
            flush=True,
        )
    subset = str(
        cache.get(
            "validation_subset",
            config["data"].get("validation_partition", "calibration"),
        )
    )
    if subset != "calibration":
        raise ValueError("EXP-0009 selection is restricted to the calibration subset")
    names = payload.get(subset)
    if not isinstance(names, list) or not names:
        raise ValueError(f"validation manifest has no {subset!r} dataset list")
    return path, subset, [str(name) for name in names], _sha256(path)


def _build_grouped_folds(
    train_dir: Path, delimiter: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stems = sorted(
        path.name.removesuffix(".zarr")
        for path in train_dir.glob("*.zarr")
        if (train_dir / f"{path.name.removesuffix('.zarr')}.geff").exists()
    )
    if not stems:
        raise FileNotFoundError(f"no paired Zarr/GEFF datasets found in {train_dir}")
    grouped: dict[str, list[str]] = {}
    for stem in stems:
        grouped.setdefault(stem.split(delimiter, 1)[0], []).append(stem)
    groups = sorted(grouped)
    if len(groups) < 2:
        raise ValueError("embryo-grouped folds require at least two groups")
    folds = []
    for index, validation_group in enumerate(groups):
        folds.append(
            {
                "split": index,
                "train_groups": [group for group in groups if group != validation_group],
                "test_groups": [validation_group],
                "train": [
                    name
                    for group in groups
                    if group != validation_group
                    for name in grouped[group]
                ],
                "test": grouped[validation_group],
            }
        )
    return folds, {group: len(names) for group, names in sorted(grouped.items())}


def _split_datasets(
    config: dict[str, Any],
    validation_names: list[str],
    *,
    allow_incomplete_smoke: bool = False,
) -> tuple[Path, dict[str, list[str]], dict[str, Any]]:
    data = config["data"]
    if data.get("split_strategy") != "embryo-prefix":
        raise ValueError("only the embryo-prefix split is supported")
    train_dir = _resolve_competition_path(data["train_dir"]).resolve()
    fold_index = int(data["fold"])
    try:
        folds, group_counts = _build_grouped_folds(
            train_dir, str(data["group_delimiter"])
        )
        selected = folds[fold_index]
    except ValueError:
        if not allow_incomplete_smoke:
            raise
        stems = sorted(
            path.name.removesuffix(".zarr")
            for path in train_dir.glob("*.zarr")
            if (train_dir / f"{path.name.removesuffix('.zarr')}.geff").exists()
        )
        expected_group = str(data["validation_group"])
        delimiter = str(data["group_delimiter"])
        validation = [
            name for name in stems if name.split(delimiter, 1)[0] == expected_group
        ]
        training = [name for name in stems if name not in validation]
        selected = {
            "train_groups": sorted(
                {name.split(delimiter, 1)[0] for name in training}
            ),
            "test_groups": [expected_group],
            "train": training,
            "test": validation,
        }
        group_counts = {
            group: sum(name.split(delimiter, 1)[0] == group for name in stems)
            for group in sorted({name.split(delimiter, 1)[0] for name in stems})
        }
    actual_group = selected["test_groups"][0]
    if actual_group != data.get("validation_group"):
        raise ValueError(
            f"validation group mismatch: expected {data.get('validation_group')}, "
            f"got {actual_group}"
        )
    missing = sorted(set(validation_names) - set(selected["test"]))
    if missing and not allow_incomplete_smoke:
        raise ValueError(f"validation manifest contains datasets outside fold 0: {missing[:3]}")
    available_validation = [
        name for name in validation_names if name in set(selected["test"])
    ]
    if not available_validation:
        raise ValueError("none of the selected validation datasets are available locally")
    split = {
        "train": list(selected["train"]),
        "validation": available_validation,
    }
    summary = {
        "fold": fold_index,
        "group_counts": group_counts,
        "train_groups": selected["train_groups"],
        "validation_groups": selected["test_groups"],
        "train_datasets": len(split["train"]),
        "validation_datasets": len(split["validation"]),
    }
    return train_dir, split, summary


def _cache_fingerprint(
    config: dict[str, Any],
    source: SourceBundle,
    validation_manifest_sha256: str,
    *,
    max_datasets: int | None,
    max_transitions: int | None,
) -> str:
    payload = {
        "schema_version": _cache_schema_version(
            _strict_int(
                config["data"].get("graph_window_size", 3),
                "data.graph_window_size",
            )
        ),
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "source_raw_sha256": source.raw_checkpoint_sha256,
        "source_weights_sha256": source.weights_sha256,
        "source_revision": source.organizer_revision,
        "validation_manifest_sha256": validation_manifest_sha256,
        "data": config["data"],
        "model": config["model"],
        "inference": config["inference"],
        "cache": config.get("cache", {}),
        "max_datasets": max_datasets,
        "max_transitions": max_transitions,
    }
    return _json_sha256(payload)


def _cache_variant_dir(
    artifact_dir: Path,
    config: dict[str, Any],
    max_datasets: int | None,
    max_transitions: int | None,
) -> Path:
    configured = config.get("cache", {}).get("directory")
    root = (
        _resolve_competition_path(configured)
        if configured
        else artifact_dir / "cache"
    )
    if max_datasets is None and max_transitions is None:
        return root
    datasets = "all" if max_datasets is None else str(max_datasets)
    transitions = "all" if max_transitions is None else str(max_transitions)
    return root.with_name(f"{root.name}-smoke-d{datasets}-t{transitions}")


def _load_host_runtime(source: SourceBundle, device: torch.device):
    host_src = source.organizer_repo / "src"
    host_scripts = source.organizer_repo / "scripts"
    sys.path[:0] = [str(host_src), str(host_scripts)]
    from tracking_cellmot.models import TemporalUNet3D
    from train_unet_transformer import (
        UNetNodeTransformer,
        _pos_embed_torch,
        detect_and_match,
        load_dataset_windows,
    )

    cfg = source.model_config
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=int(cfg["unet_out_channels"]),
        layers=[int(value) for value in cfg["unet_layers"]],
        gradient_checkpointing=False,
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=int(cfg["unet_out_channels"]),
        pos_feat_dim=32,
    )
    state = torch.load(source.weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False).to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("the frozen host model unexpectedly has trainable parameters")
    return model, load_dataset_windows, detect_and_match, _pos_embed_torch


def _autocast_context(config: dict[str, Any], device: torch.device):
    mode = str(
        config.get("cache", {}).get(
            "mixed_precision", config.get("runtime", {}).get("mixed_precision", "none")
        )
    )
    if device.type == "cuda" and mode == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if mode in {"none", "float32"} or device.type == "cpu":
        return contextlib.nullcontext()
    raise ValueError(f"unsupported cache mixed precision mode: {mode}")


def _proposal_threshold(config: dict[str, Any]) -> float:
    cache = config.get("cache", {})
    inference = config["inference"]
    if "detection_probability_threshold" in inference:
        deployment_probability = float(inference["detection_probability_threshold"])
    elif "detection_threshold" in inference:
        # The frozen 0.890 notebook names this CLI probability `det-threshold`.
        deployment_probability = float(inference["detection_threshold"])
    else:
        raise ValueError(
            "deployment detection probability is required; "
            "detection_logit_threshold is not the 0.890 inference contract"
        )
    cache_probability = float(
        cache.get("detection_probability_threshold", deployment_probability)
    )
    if not math.isclose(cache_probability, deployment_probability, abs_tol=0.0):
        raise ValueError("cache and deployment detection probabilities differ")
    if not 0.0 < deployment_probability < 1.0:
        raise ValueError("deployment detection probability must be in (0, 1)")
    return math.log(deployment_probability / (1.0 - deployment_probability))


def _cache_detection_tta(config: dict[str, Any]) -> bool:
    inference_tta = bool(config["inference"].get("detection_tta", True))
    cache_tta = bool(config.get("cache", {}).get("detection_tta", inference_tta))
    if cache_tta != inference_tta:
        raise ValueError("cache detection TTA must match deployment detection TTA")
    return cache_tta


def _position_features(
    coords: torch.Tensor,
    *,
    frame_index: int,
    image_shape: tuple[int, ...],
    position_function,
) -> torch.Tensor:
    time = torch.full(
        (*coords.shape[:2], 1),
        float(frame_index),
        device=coords.device,
        dtype=torch.float32,
    )
    return position_function(torch.cat([time, coords.float()], dim=-1), (2, *image_shape[1:]))


def _pad_matches(matches: torch.Tensor, proposal_count: int) -> torch.Tensor:
    """Align organizer matches with its at-least-one padded proposal axis."""
    if matches.ndim != 1:
        raise ValueError("organizer proposal matches must be one-dimensional")
    if proposal_count <= 0 or matches.numel() > proposal_count:
        raise ValueError("proposal match count is incompatible with the padded proposal axis")
    padded = torch.full(
        (1, proposal_count),
        -1,
        dtype=torch.long,
        device=matches.device,
    )
    if matches.numel():
        padded[0, : matches.numel()] = matches.long()
    return padded


def _to_cpu_pair(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    source_coords_grid: torch.Tensor,
    target_coords_grid: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    edge_logits: torch.Tensor,
    voxel_size: tuple[float, ...],
) -> FrozenPair:
    voxel = torch.tensor(voxel_size, dtype=torch.float32, device=source_coords_grid.device)
    return FrozenPair(
        source_features=source_features.detach().float().cpu(),
        target_features=target_features.detach().float().cpu(),
        source_coords_um=(source_coords_grid.detach().float() * voxel).cpu(),
        target_coords_um=(target_coords_grid.detach().float() * voxel).cpu(),
        source_mask=source_mask.detach().cpu(),
        target_mask=target_mask.detach().cpu(),
        edge_logits=edge_logits.detach().float().cpu(),
    )


@torch.inference_mode()
def _extract_pair(
    *,
    model,
    window,
    video_meta,
    zarr_array,
    config: dict[str, Any],
    device: torch.device,
    detect_and_match,
    position_function,
    reuse_source: ExtractedPair | None,
) -> ExtractedPair:
    downsample = tuple(int(value) for value in video_meta.downsample)
    dz, dy, dx = downsample
    raw = zarr_array[
        window.t_start : window.t_start + 2,
        ::dz,
        ::dy,
        ::dx,
    ].astype("float32")
    images = torch.from_numpy(raw)
    target_shape = tuple(int(value) for value in video_meta.image_shape[1:])
    if tuple(images.shape[1:]) != target_shape:
        images = F.interpolate(
            images[:, None], size=target_shape, mode="trilinear", align_corners=False
        )[:, 0]
    images = (
        (images - video_meta.q_low) / (video_meta.q_high - video_meta.q_low + 1e-6)
    ).clamp(0.0)
    images = images.unsqueeze(0).to(device)

    with _autocast_context(config, device):
        feature_maps, detection_logits = model.encode(images)
        if _cache_detection_tta(config):
            for dimensions in [(-1,), (-2,), (-2, -1)]:
                _, flipped_logits = model.encode(images.flip(dimensions))
                for index in range(2):
                    detection_logits[index] += flipped_logits[index].flip(dimensions)
            for index in range(2):
                detection_logits[index] /= 4

    voxel_size = tuple(float(value) for value in video_meta.voxel_size)
    threshold = _proposal_threshold(config)
    pool_kernel_um = float(config["inference"].get("pool_kernel_um", 5.0))
    max_match_distance = float(config.get("cache", {}).get("match_radius_um", 7.0))

    target_gt = window.coords[1].unsqueeze(0).to(device)
    target_gt_mask = torch.ones(
        target_gt.shape[:2], dtype=torch.bool, device=device
    )
    target_coords, target_position, target_mask, target_matches_list = detect_and_match(
        detection_logits[1],
        target_gt,
        target_gt_mask,
        tuple(int(value) for value in video_meta.image_shape),
        det_threshold=threshold,
        pool_kernel_um=pool_kernel_um,
        max_match_distance=max_match_distance,
        voxel_size=voxel_size,
        frame_index=1,
        window_size=2,
    )
    target_matches = _pad_matches(
        target_matches_list[0], target_coords.shape[1]
    )

    if reuse_source is None:
        source_gt = window.coords[0].unsqueeze(0).to(device)
        source_gt_mask = torch.ones(
            source_gt.shape[:2], dtype=torch.bool, device=device
        )
        source_coords, source_position, source_mask, source_matches_list = detect_and_match(
            detection_logits[0],
            source_gt,
            source_gt_mask,
            tuple(int(value) for value in video_meta.image_shape),
            det_threshold=threshold,
            pool_kernel_um=pool_kernel_um,
            max_match_distance=max_match_distance,
            voxel_size=voxel_size,
            frame_index=0,
            window_size=2,
        )
        source_matches = _pad_matches(
            source_matches_list[0], source_coords.shape[1]
        )
    else:
        source_coords = reuse_source.target_coords_grid.to(device)
        source_mask = reuse_source.pair.target_mask.to(device)
        source_matches = reuse_source.target_matches.to(device)
        source_position = _position_features(
            source_coords,
            frame_index=0,
            image_shape=tuple(int(value) for value in video_meta.image_shape),
            position_function=position_function,
        )

    source_features = model._index_features(feature_maps[:, 0], source_coords, source_mask)
    target_features = model._index_features(feature_maps[:, 1], target_coords, target_mask)
    if source_mask.any() and target_mask.any():
        downsample_tensor = torch.tensor(
            downsample, dtype=torch.float32, device=device
        )
        with _autocast_context(config, device):
            edge_logits = model.predict_edges(
                source_features,
                target_features,
                source_coords * downsample_tensor,
                target_coords * downsample_tensor,
                source_position,
                target_position,
                source_mask,
                target_mask,
            )
    else:
        edge_logits = torch.zeros(
            1,
            source_coords.shape[1],
            target_coords.shape[1],
            device=device,
            dtype=torch.float32,
        )

    pair = _to_cpu_pair(
        source_features,
        target_features,
        source_coords,
        target_coords,
        source_mask,
        target_mask,
        edge_logits,
        voxel_size,
    )
    result = ExtractedPair(
        pair=pair,
        source_coords_grid=source_coords.detach().float().cpu(),
        target_coords_grid=target_coords.detach().float().cpu(),
        source_matches=source_matches.detach().long().cpu(),
        target_matches=target_matches.detach().long().cpu(),
    )
    del images, feature_maps, detection_logits
    return result


def _sparse_parent_classes(
    gt_edges: torch.Tensor,
    source_matches: torch.Tensor,
    target_matches: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map sparse annotated parents to unique detected proposals."""
    if gt_edges.ndim != 3:
        raise ValueError("gt_edges must have shape (B,Gs,Gt)")
    batch, source_count = source_matches.shape
    target_count = target_matches.shape[1]
    classes = torch.full((batch, target_count), source_count, dtype=torch.long)
    supervised = torch.zeros((batch, target_count), dtype=torch.bool)
    for sample in range(batch):
        for target_index in range(target_count):
            if not bool(target_mask[sample, target_index]):
                continue
            gt_target = int(target_matches[sample, target_index])
            if gt_target < 0 or gt_target >= gt_edges.shape[2]:
                continue
            parents = torch.nonzero(gt_edges[sample, :, gt_target] > 0).flatten()
            if parents.numel() != 1:
                continue
            gt_parent = int(parents.item())
            if int((gt_edges[sample, gt_parent] > 0).sum()) != 1:
                # Keep the organizer/public-applicable division policy frozen.
                continue
            proposals = torch.nonzero(
                source_mask[sample]
                & (source_matches[sample] == gt_parent)
            ).flatten()
            if proposals.numel() == 1:
                classes[sample, target_index] = proposals.item()
                supervised[sample, target_index] = True
    return classes, supervised


def _build_sparse_candidate_examples(
    previous: ExtractedPair,
    current: ExtractedPair,
    gt_edges: torch.Tensor,
    graph_config: TemporalGraphConfig,
    *,
    prior: ExtractedPair | None = None,
) -> SparseExamples:
    candidates = build_parent_candidates(
        current.pair.source_coords_um,
        current.pair.target_coords_um,
        current.pair.source_mask,
        current.pair.target_mask,
        top_k=graph_config.top_k,
        radius_um=graph_config.radius_um,
    )
    feature_batch = build_candidate_features(
        previous.pair,
        current.pair,
        candidates,
        prior_pair=None if prior is None else prior.pair,
        graph_window_size=graph_config.graph_window_size,
        distance_scale_um=graph_config.distance_scale_um,
        middle_coord_atol=graph_config.middle_coord_atol,
    )
    classes, supervised = _sparse_parent_classes(
        gt_edges,
        current.source_matches,
        current.target_matches,
        current.pair.source_mask,
        current.pair.target_mask,
    )
    batch, target_count, top_k = candidates.source_index.shape
    gathered_base = torch.gather(
        current.pair.edge_logits.transpose(1, 2), 2, candidates.source_index
    )
    rows: list[torch.Tensor] = []
    bases: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    labels: list[int] = []
    total = int(supervised.sum())
    survived = 0
    for sample in range(batch):
        for target in range(target_count):
            if not bool(supervised[sample, target]):
                continue
            slots = torch.nonzero(
                candidates.valid_mask[sample, target]
                & (candidates.source_index[sample, target] == classes[sample, target])
            ).flatten()
            if slots.numel() != 1:
                continue
            survived += 1
            rows.append(feature_batch.features[sample, target])
            bases.append(gathered_base[sample, target])
            masks.append(candidates.valid_mask[sample, target])
            labels.append(int(slots.item()))

    feature_width = candidate_feature_dim(
        graph_config.node_feature_dim,
        graph_window_size=graph_config.graph_window_size,
    )
    if rows:
        features = torch.stack(rows).to(torch.float16)
        base_logits = torch.stack(bases).to(torch.float16)
        valid_mask = torch.stack(masks)
        label_tensor = torch.tensor(labels, dtype=torch.long)
    else:
        features = torch.empty(0, top_k, feature_width, dtype=torch.float16)
        base_logits = torch.empty(0, top_k, dtype=torch.float16)
        valid_mask = torch.empty(0, top_k, dtype=torch.bool)
        label_tensor = torch.empty(0, dtype=torch.long)
    return SparseExamples(
        features=features,
        base_logits=base_logits,
        valid_mask=valid_mask,
        labels=label_tensor,
        supervised_parents=total,
        candidate_parents=survived,
    )


def _cache_one_dataset(
    *,
    name: str,
    split: str,
    train_dir: Path,
    cache_file: Path,
    fingerprint: str,
    graph_config: TemporalGraphConfig,
    config: dict[str, Any],
    model,
    load_dataset_windows,
    detect_and_match,
    position_function,
    device: torch.device,
    max_transitions: int | None,
) -> dict[str, Any]:
    import zarr

    video_meta, windows = load_dataset_windows(
        train_dir / name,
        window_size=2,
        downsample=tuple(int(value) for value in config["data"]["downsample"]),
    )
    by_start = {int(window.t_start): window for window in windows}
    preceding_pairs = graph_config.graph_window_size - 2
    right_starts = [
        start
        for start in sorted(by_start)
        if all(start - offset in by_start for offset in range(1, preceding_pairs + 1))
    ]
    if max_transitions is not None:
        right_starts = right_starts[:max_transitions]

    zarr_array = zarr.open_group(str(video_meta.zarr_path), mode="r")["0"]
    examples: list[SparseExamples] = []
    retained: dict[int, ExtractedPair] = {}
    for number, current_start in enumerate(right_starts, 1):
        required_starts = list(
            range(current_start - preceding_pairs, current_start + 1)
        )
        for left_start, right_start in zip(
            required_starts, required_starts[1:], strict=False
        ):
            if not torch.allclose(
                by_start[left_start].coords[1], by_start[right_start].coords[0]
            ):
                raise ValueError(
                    f"middle-frame GT order drift for {name} at transition "
                    f"{right_start}"
                )
        for pair_start in required_starts:
            if pair_start in retained:
                continue
            retained[pair_start] = _extract_pair(
                model=model,
                window=by_start[pair_start],
                video_meta=video_meta,
                zarr_array=zarr_array,
                config=config,
                device=device,
                detect_and_match=detect_and_match,
                position_function=position_function,
                reuse_source=retained.get(pair_start - 1),
            )
        prior = retained.get(current_start - 2)
        previous = retained[current_start - 1]
        current = retained[current_start]
        current_window = by_start[current_start]
        examples.append(
            _build_sparse_candidate_examples(
                previous,
                current,
                current_window.targets[0].unsqueeze(0),
                graph_config,
                prior=prior,
            )
        )
        retained = {
            pair_start: pair
            for pair_start, pair in retained.items()
            if pair_start >= current_start - preceding_pairs + 1
        }
        if number % 10 == 0 or number == len(right_starts):
            print(
                f"  {split}/{name}: {number}/{len(right_starts)} right transitions",
                flush=True,
            )

    if examples:
        features = torch.cat([item.features for item in examples])
        base_logits = torch.cat([item.base_logits for item in examples])
        valid_mask = torch.cat([item.valid_mask for item in examples])
        labels = torch.cat([item.labels for item in examples])
    else:
        width = candidate_feature_dim(
            graph_config.node_feature_dim,
            graph_window_size=graph_config.graph_window_size,
        )
        features = torch.empty(0, graph_config.top_k, width, dtype=torch.float16)
        base_logits = torch.empty(0, graph_config.top_k, dtype=torch.float16)
        valid_mask = torch.empty(0, graph_config.top_k, dtype=torch.bool)
        labels = torch.empty(0, dtype=torch.long)
    total = sum(item.supervised_parents for item in examples)
    survived = sum(item.candidate_parents for item in examples)
    stats = {
        "dataset": name,
        "split": split,
        "right_transitions": len(right_starts),
        "examples": int(labels.numel()),
        "supervised_parents": total,
        "candidate_parents": survived,
        "candidate_recall": survived / total if total else None,
    }
    payload = {
        "schema_version": _cache_schema_version(graph_config.graph_window_size),
        "feature_schema": _cache_feature_schema(
            _cache_schema_version(graph_config.graph_window_size)
        ),
        "feature_width": int(features.shape[-1]),
        "fingerprint": fingerprint,
        "dataset": name,
        "split": split,
        "features": features,
        "base_logits": base_logits,
        "valid_mask": valid_mask,
        "labels": labels,
        "stats": stats,
    }
    _atomic_torch_save(payload, cache_file)
    return {**stats, "path": str(cache_file), "sha256": _sha256(cache_file)}


def build_cache(
    *,
    config: dict[str, Any],
    source: SourceBundle,
    graph_config: TemporalGraphConfig,
    artifact_dir: Path,
    validation_manifest_sha256: str,
    validation_names: list[str],
    max_datasets: int | None,
    max_transitions: int | None,
    wandb_run,
) -> tuple[Path, dict[str, Any]]:
    train_dir, split, split_summary = _split_datasets(
        config,
        validation_names,
        allow_incomplete_smoke=max_datasets is not None or max_transitions is not None,
    )
    fingerprint = _cache_fingerprint(
        config,
        source,
        validation_manifest_sha256,
        max_datasets=max_datasets,
        max_transitions=max_transitions,
    )
    cache_dir = _cache_variant_dir(
        artifact_dir, config, max_datasets, max_transitions
    ).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model, load_windows, detect_match, position_function = _load_host_runtime(
        source, device
    )
    started = time.monotonic()
    records: list[dict[str, Any]] = []
    cache_schema_version = _cache_schema_version(graph_config.graph_window_size)
    peak_allocated_bytes = 0
    peak_reserved_bytes = 0
    try:
        for split_name in ("train", "validation"):
            names = split[split_name]
            if max_datasets is not None:
                names = names[:max_datasets]
            for name in names:
                cache_file = cache_dir / split_name / f"{name}.pt"
                if cache_file.exists():
                    existing = torch.load(
                        cache_file, map_location="cpu", weights_only=False
                    )
                    if (
                        existing.get("schema_version") == cache_schema_version
                        and existing.get("fingerprint") == fingerprint
                    ):
                        stats = dict(existing["stats"])
                        record = {
                            **stats,
                            "path": str(cache_file),
                            "sha256": _sha256(cache_file),
                        }
                        records.append(record)
                        print(f"Reuse cache: {split_name}/{name}", flush=True)
                        continue
                    raise ValueError(f"incompatible cache file already exists: {cache_file}")
                record = _cache_one_dataset(
                    name=name,
                    split=split_name,
                    train_dir=train_dir,
                    cache_file=cache_file,
                    fingerprint=fingerprint,
                    graph_config=graph_config,
                    config=config,
                    model=model,
                    load_dataset_windows=load_windows,
                    detect_and_match=detect_match,
                    position_function=position_function,
                    device=device,
                    max_transitions=max_transitions,
                )
                records.append(record)
                wandb_run.log(
                    {
                        f"cache/{split_name}_datasets": sum(
                            item["split"] == split_name for item in records
                        ),
                        f"cache/{split_name}_examples": sum(
                            item["examples"]
                            for item in records
                            if item["split"] == split_name
                        ),
                    }
                )
    finally:
        if device.type == "cuda":
            peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    totals: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "validation"):
        selected = [item for item in records if item["split"] == split_name]
        total = sum(item["supervised_parents"] for item in selected)
        survived = sum(item["candidate_parents"] for item in selected)
        totals[split_name] = {
            "datasets": len(selected),
            "right_transitions": sum(item["right_transitions"] for item in selected),
            "examples": sum(item["examples"] for item in selected),
            "supervised_parents": total,
            "candidate_parents": survived,
            "candidate_recall": survived / total if total else None,
        }
    manifest = {
        "schema_version": cache_schema_version,
        "feature_schema": _cache_feature_schema(cache_schema_version),
        "feature_width": candidate_feature_dim(
            graph_config.node_feature_dim,
            graph_window_size=graph_config.graph_window_size,
        ),
        "fingerprint": fingerprint,
        "experiment_id": config["experiment_id"],
        "source_raw_checkpoint_sha256": source.raw_checkpoint_sha256,
        "source_weights_sha256": source.weights_sha256,
        "organizer_revision": source.organizer_revision,
        "validation_manifest_sha256": validation_manifest_sha256,
        "graph_config": graph_config.to_dict(),
        "split": split_summary,
        "limits": {
            "max_datasets": max_datasets,
            "max_transitions": max_transitions,
        },
        "totals": totals,
        "files": records,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "device": str(device),
        "peak_cuda_memory_allocated_bytes": peak_allocated_bytes,
        "peak_cuda_memory_reserved_bytes": peak_reserved_bytes,
    }
    _atomic_json(cache_dir / "manifest.json", manifest)
    for split_name, metrics in totals.items():
        wandb_run.log(
            {
                f"cache/{split_name}_examples_total": metrics["examples"],
                f"cache/{split_name}_candidate_recall": metrics["candidate_recall"],
            }
        )
    wandb_run.log(
        {
            "runtime/cache_peak_cuda_allocated_bytes": peak_allocated_bytes,
            "runtime/cache_peak_cuda_reserved_bytes": peak_reserved_bytes,
        }
    )
    wandb_run.summary["runtime/cache_peak_cuda_allocated_bytes"] = peak_allocated_bytes
    wandb_run.summary["runtime/cache_peak_cuda_reserved_bytes"] = peak_reserved_bytes
    return cache_dir, manifest


class CompactCacheDataset(Dataset):
    """In-memory compact rows; no frozen image tensor remains resident."""

    def __init__(
        self,
        paths: list[Path],
        fingerprint: str,
        schema_version: int = CACHE_SCHEMA_VERSION,
        *,
        top_k: int | None = None,
        feature_width: int | None = None,
    ) -> None:
        payloads = [
            torch.load(path, map_location="cpu", weights_only=False) for path in paths
        ]
        for path, payload in zip(paths, payloads, strict=True):
            if payload.get("schema_version") != schema_version:
                raise ValueError(f"cache schema mismatch: {path}")
            if payload.get("fingerprint") != fingerprint:
                raise ValueError(f"cache fingerprint mismatch: {path}")
            payload_feature_schema = payload.get("feature_schema")
            if payload_feature_schema is not None and payload_feature_schema != (
                _cache_feature_schema(schema_version)
            ):
                raise ValueError(f"cache feature-math revision mismatch: {path}")
            features = payload.get("features")
            base_logits = payload.get("base_logits")
            valid_mask = payload.get("valid_mask")
            labels = payload.get("labels")
            if not all(
                isinstance(value, torch.Tensor)
                for value in (features, base_logits, valid_mask, labels)
            ):
                raise TypeError(f"cache tensors are missing: {path}")
            if features.ndim != 3:
                raise ValueError(f"cache features must have shape (N,K,F): {path}")
            rows, candidates, width = features.shape
            if top_k is not None and candidates != top_k:
                raise ValueError(f"cache top-k mismatch: {path}")
            if feature_width is not None and width != feature_width:
                raise ValueError(f"cache feature width mismatch: {path}")
            payload_feature_width = payload.get("feature_width")
            if payload_feature_width is not None and int(payload_feature_width) != width:
                raise ValueError(f"cache recorded feature width mismatch: {path}")
            if base_logits.shape != (rows, candidates):
                raise ValueError(f"cache base-logit shape mismatch: {path}")
            if valid_mask.shape != (rows, candidates) or valid_mask.dtype != torch.bool:
                raise ValueError(f"cache valid-mask contract mismatch: {path}")
            if labels.shape != (rows,) or labels.dtype != torch.long:
                raise ValueError(f"cache label contract mismatch: {path}")
            if not features.is_floating_point() or not base_logits.is_floating_point():
                raise TypeError(f"cache features and logits must be floating point: {path}")
            if not torch.isfinite(features).all() or not torch.isfinite(base_logits).all():
                raise ValueError(f"cache contains non-finite values: {path}")
            if rows:
                if bool(((labels < 0) | (labels >= candidates)).any()):
                    raise ValueError(f"cache label is outside candidate range: {path}")
                if not bool(valid_mask.gather(1, labels.unsqueeze(1)).all()):
                    raise ValueError(f"cache label selects an invalid candidate: {path}")
        self.features = torch.cat([payload["features"] for payload in payloads])
        self.base_logits = torch.cat([payload["base_logits"] for payload in payloads])
        self.valid_mask = torch.cat([payload["valid_mask"] for payload in payloads])
        self.labels = torch.cat([payload["labels"] for payload in payloads])

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.base_logits[index],
            self.valid_mask[index],
            self.labels[index],
        )


def _cache_datasets(
    cache_dir: Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> tuple[CompactCacheDataset, CompactCacheDataset, dict[str, Any]]:
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"cache manifest is missing: {manifest_path}")
    if (
        expected_manifest_sha256 is not None
        and _sha256(manifest_path) != expected_manifest_sha256
    ):
        raise ValueError("shared cache manifest SHA-256 mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(manifest.get("schema_version", -1))
    if schema_version not in CACHE_SCHEMA_VERSIONS.values():
        raise ValueError(f"unsupported cache schema: {schema_version}")
    graph_config = manifest.get("graph_config")
    if not isinstance(graph_config, dict):
        raise ValueError("cache manifest is missing graph_config")
    graph_window_size = _strict_int(
        graph_config.get("graph_window_size"),
        "cache graph_window_size",
    )
    expected_schema = _cache_schema_version(graph_window_size)
    if schema_version != expected_schema:
        raise ValueError("cache schema does not match graph_window_size")
    top_k = _strict_int(graph_config.get("top_k"), "cache graph_config.top_k")
    node_feature_dim = _strict_int(
        graph_config.get("node_feature_dim"),
        "cache graph_config.node_feature_dim",
    )
    feature_width = candidate_feature_dim(
        node_feature_dim,
        graph_window_size=graph_window_size,
    )
    recorded_feature_schema = manifest.get("feature_schema")
    if recorded_feature_schema is not None and recorded_feature_schema != (
        _cache_feature_schema(schema_version)
    ):
        raise ValueError("cache feature schema mismatch")
    recorded_feature_width = manifest.get("feature_width")
    if recorded_feature_width is not None and recorded_feature_width != feature_width:
        raise ValueError("cache manifest feature width mismatch")
    fingerprint = str(manifest["fingerprint"])
    paths: dict[str, list[Path]] = {"train": [], "validation": []}
    for record in manifest["files"]:
        path = Path(record["path"])
        if not path.is_file() or _sha256(path) != record["sha256"]:
            raise ValueError(f"cache file integrity check failed: {path}")
        paths[record["split"]].append(path)
    train = CompactCacheDataset(
        paths["train"],
        fingerprint,
        schema_version,
        top_k=top_k,
        feature_width=feature_width,
    )
    validation = CompactCacheDataset(
        paths["validation"],
        fingerprint,
        schema_version,
        top_k=top_k,
        feature_width=feature_width,
    )
    if not len(train) or not len(validation):
        raise ValueError(
            f"cache has insufficient supervised examples: train={len(train)}, "
            f"validation={len(validation)}"
        )
    return train, validation, manifest


def _run_epoch(
    head: TemporalGraphResidualHead,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    gradient_clip_norm: float,
) -> dict[str, float | int]:
    training = optimizer is not None
    head.train(training)
    total_loss = 0.0
    total_base_correct = 0
    total_refined_correct = 0
    total_examples = 0
    delta_square_sum = 0.0
    delta_count = 0
    delta_abs_max = 0.0
    for features, base_logits, valid_mask, labels in loader:
        features = features.to(device, non_blocking=True).float().unsqueeze(1)
        base_logits = base_logits.to(device, non_blocking=True).float()
        valid_mask = valid_mask.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            residual = head.forward_candidate_features(
                features, valid_mask.unsqueeze(1)
            ).squeeze(1)
            refined = (base_logits + residual).masked_fill(~valid_mask, -torch.inf)
            loss = F.cross_entropy(refined, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), gradient_clip_norm)
                optimizer.step()
        batch_size = labels.numel()
        base = base_logits.masked_fill(~valid_mask, -torch.inf)
        total_loss += float(loss.detach()) * batch_size
        total_base_correct += int((base.argmax(dim=-1) == labels).sum())
        total_refined_correct += int((refined.argmax(dim=-1) == labels).sum())
        total_examples += batch_size
        valid_delta = residual.detach()[valid_mask]
        delta_square_sum += float(valid_delta.square().sum())
        delta_count += int(valid_delta.numel())
        if valid_delta.numel():
            delta_abs_max = max(delta_abs_max, float(valid_delta.abs().max()))
    if not total_examples:
        raise ValueError("empty epoch loader")
    return {
        "loss": total_loss / total_examples,
        "base_top1_accuracy": total_base_correct / total_examples,
        "refined_top1_accuracy": total_refined_correct / total_examples,
        "examples": total_examples,
        "delta_rms": math.sqrt(delta_square_sum / max(delta_count, 1)),
        "delta_abs_max": delta_abs_max,
    }


def _capture_rng(loader_generator: torch.Generator) -> dict[str, Any]:
    payload = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "loader": loader_generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def _restore_rng(payload: dict[str, Any], loader_generator: torch.Generator) -> None:
    random.setstate(payload["python"])
    torch.set_rng_state(payload["torch"])
    loader_generator.set_state(payload["loader"])
    if torch.cuda.is_available() and "cuda" in payload:
        torch.cuda.set_rng_state_all(payload["cuda"])


def _training_fingerprint(
    config: dict[str, Any],
    graph_config: TemporalGraphConfig,
    cache_fingerprint: str,
    cache_manifest_sha256: str,
) -> str:
    return _json_sha256(
        {
            "experiment_id": config["experiment_id"],
            "seed": config["seed"],
            "train": config["train"],
            "checkpoint": config["checkpoint"],
            "graph_config": graph_config.to_dict(),
            "cache_fingerprint": cache_fingerprint,
            "cache_manifest_sha256": cache_manifest_sha256,
        }
    )


def _validate_cache_training_contract(
    manifest: dict[str, Any],
    graph_config: TemporalGraphConfig,
    source: SourceBundle,
    config: dict[str, Any],
) -> tuple[int, str, int]:
    schema_version = int(manifest.get("schema_version", -1))
    expected_schema = _cache_schema_version(graph_config.graph_window_size)
    if schema_version != expected_schema:
        raise ValueError(
            f"cache schema mismatch for T_graph={graph_config.graph_window_size}: "
            f"expected {expected_schema}, found {schema_version}"
        )
    feature_schema = _cache_feature_schema(schema_version)
    recorded_feature_schema = manifest.get("feature_schema")
    if recorded_feature_schema is not None and recorded_feature_schema != feature_schema:
        raise ValueError("cache feature-math revision mismatch")
    feature_width = candidate_feature_dim(
        graph_config.node_feature_dim,
        graph_window_size=graph_config.graph_window_size,
    )
    recorded_width = manifest.get("feature_width")
    if recorded_width is not None and int(recorded_width) != feature_width:
        raise ValueError("cache feature width does not match the temporal head")
    if manifest.get("source_weights_sha256") != source.weights_sha256:
        raise ValueError("cache frozen-host weights do not match the training source")
    if manifest.get("source_raw_checkpoint_sha256") != source.raw_checkpoint_sha256:
        raise ValueError("cache raw source checkpoint does not match the training source")

    cached_graph = manifest.get("graph_config")
    if not isinstance(cached_graph, dict):
        raise ValueError("cache manifest is missing graph_config")
    candidate_fields = (
        "node_feature_dim",
        "top_k",
        "radius_um",
        "distance_scale_um",
        "middle_coord_atol",
        "image_window_size",
        "graph_window_size",
        "ownership",
    )
    expected_graph = graph_config.to_dict()
    mismatches = [
        name
        for name in candidate_fields
        if cached_graph.get(name) != expected_graph.get(name)
    ]
    if mismatches:
        raise ValueError(
            "cache and temporal head use different candidate contracts: "
            + ", ".join(mismatches)
        )
    source_experiment = config.get("cache", {}).get("source_experiment_id")
    if source_experiment is not None and manifest.get("experiment_id") != (
        source_experiment
    ):
        raise ValueError("shared cache source experiment mismatch")
    return schema_version, feature_schema, feature_width


def _training_checkpoint(
    *,
    config: dict[str, Any],
    head: TemporalGraphResidualHead,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    best_score: float,
    graph_source_sha256: str,
    raw_checkpoint_sha256: str,
    cache_fingerprint: str,
    cache_manifest_sha256: str,
    cache_schema_version: int,
    feature_schema: str,
    feature_width: int,
    training_fingerprint: str,
    history: list[dict[str, Any]],
    loader_generator: torch.Generator,
) -> dict[str, Any]:
    temporal_graph = head.checkpoint_payload(
        base_checkpoint_sha256=graph_source_sha256,
        metadata={
            "experiment_id": config["experiment_id"],
            "completed_epochs": completed_epochs,
            "cache_fingerprint": cache_fingerprint,
            "cache_manifest_sha256": cache_manifest_sha256,
            "cache_schema_version": cache_schema_version,
            "feature_schema": feature_schema,
            "feature_width": feature_width,
            "ownership": "right_transition",
            "image_window_size": 2,
            "graph_window_size": head.config.graph_window_size,
            "source_raw_checkpoint_sha256": raw_checkpoint_sha256,
        },
    )
    return {
        "checkpoint_schema_version": TRAINING_CHECKPOINT_SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "completed_epochs": completed_epochs,
        "best_score": best_score,
        "temporal_graph": temporal_graph,
        "optimizer_state_dict": optimizer.state_dict(),
        "cache_fingerprint": cache_fingerprint,
        "training_fingerprint": training_fingerprint,
        "rng_state": _capture_rng(loader_generator),
        "history": history,
    }


def _resume_candidates(artifact_dir: Path) -> list[Path]:
    periodic = sorted(
        artifact_dir.glob("checkpoint_epoch_*.pth"), reverse=True
    )
    return [artifact_dir / "last_checkpoint.pth", *periodic]


def _restore_training(
    *,
    artifact_dir: Path,
    config: dict[str, Any],
    head: TemporalGraphResidualHead,
    optimizer: torch.optim.Optimizer,
    source: SourceBundle,
    cache_fingerprint: str,
    training_fingerprint: str,
    loader_generator: torch.Generator,
    device: torch.device,
) -> tuple[int, float, list[dict[str, Any]]]:
    if not bool(config["checkpoint"].get("auto_resume", True)):
        return 0, -math.inf, []
    errors = []
    for path in _resume_candidates(artifact_dir):
        if not path.is_file():
            continue
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
            if payload.get("checkpoint_schema_version") != TRAINING_CHECKPOINT_SCHEMA_VERSION:
                raise ValueError("schema mismatch")
            if payload.get("experiment_id") != config["experiment_id"]:
                raise ValueError("experiment mismatch")
            if payload.get("cache_fingerprint") != cache_fingerprint:
                raise ValueError("cache fingerprint mismatch")
            if payload.get("training_fingerprint") != training_fingerprint:
                raise ValueError("training fingerprint mismatch")
            temporal = payload["temporal_graph"]
            if temporal["base_checkpoint_sha256"] != source.weights_sha256:
                raise ValueError("frozen host checkpoint mismatch")
            if temporal["config"] != head.config.to_dict():
                raise ValueError("graph model config mismatch")
            head.load_state_dict(temporal["state_dict"], strict=True)
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            _restore_rng(payload["rng_state"], loader_generator)
            completed = int(payload["completed_epochs"])
            history = list(payload["history"])
            if len(history) != completed:
                raise ValueError("checkpoint history length mismatch")
            print(f"Resume: {path} after {completed} completed epochs", flush=True)
            return completed, float(payload["best_score"]), history
        except Exception as error:  # noqa: BLE001 - try older atomic checkpoint
            errors.append(f"{path.name}: {error}")
    if errors:
        raise RuntimeError("no compatible resume checkpoint: " + "; ".join(errors))
    return 0, -math.inf, []


def train_head(
    *,
    config: dict[str, Any],
    source: SourceBundle,
    graph_config: TemporalGraphConfig,
    artifact_dir: Path,
    cache_dir: Path,
    wandb_run,
) -> dict[str, Any]:
    train_dataset, validation_dataset, cache_manifest = _cache_datasets(
        cache_dir,
        expected_manifest_sha256=config.get("cache", {}).get(
            "source_manifest_sha256"
        ),
    )
    cache_fingerprint = str(cache_manifest["fingerprint"])
    cache_manifest_sha256 = _sha256(cache_dir / "manifest.json")
    cache_schema_version, feature_schema, feature_width = (
        _validate_cache_training_contract(
            cache_manifest,
            graph_config,
            source,
            config,
        )
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = int(config["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    head = TemporalGraphResidualHead(graph_config).to(device)
    train_cfg = config["train"]
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    loader_generator = torch.Generator().manual_seed(seed)
    batch_size = int(train_cfg.get("batch_size", 2048))
    workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    fingerprint = _training_fingerprint(
        config,
        graph_config,
        cache_fingerprint,
        cache_manifest_sha256,
    )
    start_epoch, best_score, history = _restore_training(
        artifact_dir=artifact_dir,
        config=config,
        head=head,
        optimizer=optimizer,
        source=source,
        cache_fingerprint=cache_fingerprint,
        training_fingerprint=fingerprint,
        loader_generator=loader_generator,
        device=device,
    )
    history_path = artifact_dir / "epoch_history.jsonl"
    _atomic_write_text(
        history_path, "".join(json.dumps(item, sort_keys=True) + "\n" for item in history)
    )
    epochs = int(train_cfg["epochs"])
    checkpoint_every = int(config["checkpoint"].get("every_epochs", 5))
    if checkpoint_every <= 0:
        raise ValueError("checkpoint.every_epochs must be positive")
    gradient_clip = float(train_cfg.get("gradient_clip_norm", 1.0))
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        epoch_started = time.monotonic()
        train_metrics = _run_epoch(
            head, train_loader, device, optimizer, gradient_clip
        )
        with torch.inference_mode():
            validation_metrics = _run_epoch(
                head, validation_loader, device, None, gradient_clip
            )
        score = float(validation_metrics["refined_top1_accuracy"])
        is_best = score >= best_score
        best_score = max(best_score, score)
        record = {
            "epoch": epoch + 1,
            **{f"train/{key}": value for key, value in train_metrics.items()},
            **{f"validation/{key}": value for key, value in validation_metrics.items()},
            "validation/best_refined_top1_accuracy": best_score,
            "runtime/epoch_seconds": round(time.monotonic() - epoch_started, 3),
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, sort_keys=True) + "\n")
            file.flush()
            os.fsync(file.fileno())
        wandb_run.log(record)
        checkpoint = _training_checkpoint(
            config=config,
            head=head,
            optimizer=optimizer,
            completed_epochs=epoch + 1,
            best_score=best_score,
            graph_source_sha256=source.weights_sha256,
            raw_checkpoint_sha256=source.raw_checkpoint_sha256,
            cache_fingerprint=cache_fingerprint,
            cache_manifest_sha256=cache_manifest_sha256,
            cache_schema_version=cache_schema_version,
            feature_schema=feature_schema,
            feature_width=feature_width,
            training_fingerprint=fingerprint,
            history=history,
            loader_generator=loader_generator,
        )
        _atomic_torch_save(checkpoint, artifact_dir / "last_checkpoint.pth")
        if is_best and bool(config["checkpoint"].get("save_best", True)):
            _atomic_torch_save(checkpoint, artifact_dir / "best_model.pth")
        if (epoch + 1) % checkpoint_every == 0 or epoch + 1 == epochs:
            _atomic_torch_save(
                checkpoint, artifact_dir / f"checkpoint_epoch_{epoch + 1:04d}.pth"
            )
        print(
            f"Epoch {epoch + 1:03d}/{epochs}: "
            f"train_loss={train_metrics['loss']:.5f} "
            f"val_loss={validation_metrics['loss']:.5f} "
            f"base={validation_metrics['base_top1_accuracy']:.4f} "
            f"refined={score:.4f} best={best_score:.4f}",
            flush=True,
        )
    result = {
        "experiment_id": config["experiment_id"],
        "completed_epochs": epochs,
        "best_refined_top1_accuracy": best_score,
        "train_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "cache_fingerprint": cache_fingerprint,
        "cache_manifest_sha256": cache_manifest_sha256,
        "cache_schema_version": cache_schema_version,
        "feature_schema": feature_schema,
        "feature_width": feature_width,
        "source_raw_checkpoint_sha256": source.raw_checkpoint_sha256,
        "source_weights_sha256": source.weights_sha256,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "periodic_checkpoints": [
            {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for path in sorted(artifact_dir.glob("checkpoint_epoch_*.pth"))
        ],
    }
    _atomic_json(artifact_dir / "result.json", result)
    wandb_run.summary.update(result)
    return result


def _wandb_run_key(
    stage: str, max_datasets: int | None, max_transitions: int | None
) -> str:
    datasets = "all" if max_datasets is None else str(max_datasets)
    transitions = "all" if max_transitions is None else str(max_transitions)
    return f"{stage}-d{datasets}-t{transitions}"


def _generate_wandb_run_id() -> str:
    return uuid.uuid4().hex[:8]


def _init_wandb(
    config: dict[str, Any],
    artifact_dir: Path,
    stage: str,
    max_datasets: int | None,
    max_transitions: int | None,
):
    tracking = config.get("tracking", {})
    if tracking.get("provider") != "wandb" or tracking.get("mode") != "online":
        raise ValueError("EXP-0009 requires tracking.provider=wandb and mode=online")
    try:
        import wandb

        if not wandb.login(verify=True, timeout=30):
            raise RuntimeError("wandb.login returned false")
        run_key = _wandb_run_key(stage, max_datasets, max_transitions)
        run_id_path = artifact_dir / f"wandb-run-id-{run_key}.txt"
        if run_id_path.exists():
            run_id = run_id_path.read_text(encoding="utf-8").strip()
        else:
            run_id = _generate_wandb_run_id()
            _atomic_write_text(run_id_path, run_id + "\n")
        run = wandb.init(
            entity=tracking.get("entity"),
            project=str(tracking["project"]),
            id=run_id,
            name=f"{tracking.get('name', config['experiment_id'])}-{run_key}",
            group=tracking.get("group"),
            job_type=f"{tracking.get('job_type', 'train')}-{stage}",
            tags=list(tracking.get("tags", [])),
            notes=tracking.get("notes"),
            mode="online",
            resume="allow",
            dir=str(artifact_dir),
            config=config,
        )
    except Exception as error:  # noqa: BLE001 - turn every auth/init failure explicit
        raise RuntimeError(f"W&B online authentication/initialization failed: {error}") from error
    if run is None:
        raise RuntimeError("W&B online initialization returned no run")
    run.define_metric("epoch")
    for namespace in ("train", "validation"):
        run.define_metric(f"{namespace}/*", step_metric="epoch")
    return run


def run(
    config_path: Path,
    *,
    stage: str,
    max_datasets: int | None,
    max_transitions: int | None,
) -> dict[str, Any]:
    if max_datasets is not None and max_datasets <= 0:
        raise ValueError("--max-datasets must be positive")
    if max_transitions is not None and max_transitions <= 0:
        raise ValueError("--max-transitions must be positive")
    config = _load_config(config_path)
    _proposal_threshold(config)
    _cache_detection_tta(config)
    source = _validate_source_bundle(config)
    graph_config = _graph_config(config, source)
    validation_path, validation_subset, validation_names, validation_sha = (
        _validation_selection(
            config,
            allow_incomplete_smoke=(
                max_datasets is not None or max_transitions is not None
            ),
        )
    )
    artifact_dir = _resolve_competition_path(config["output"]["artifact_dir"]).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = _cache_variant_dir(
        artifact_dir, config, max_datasets, max_transitions
    ).resolve()
    wandb_run = _init_wandb(
        config,
        artifact_dir,
        stage,
        max_datasets,
        max_transitions,
    )
    try:
        wandb_run.summary["source/raw_checkpoint_sha256"] = source.raw_checkpoint_sha256
        wandb_run.summary["source/weights_sha256"] = source.weights_sha256
        wandb_run.summary["source/organizer_revision"] = source.organizer_revision
        wandb_run.summary["split/validation_manifest"] = str(validation_path)
        wandb_run.summary["split/validation_subset"] = validation_subset
        result: dict[str, Any] = {"stage": stage}
        if stage in {"cache", "all"}:
            cache_dir, cache_manifest = build_cache(
                config=config,
                source=source,
                graph_config=graph_config,
                artifact_dir=artifact_dir,
                validation_manifest_sha256=validation_sha,
                validation_names=validation_names,
                max_datasets=max_datasets,
                max_transitions=max_transitions,
                wandb_run=wandb_run,
            )
            result["cache"] = cache_manifest
        if stage in {"train", "all"}:
            result["training"] = train_head(
                config=config,
                source=source,
                graph_config=graph_config,
                artifact_dir=artifact_dir,
                cache_dir=cache_dir,
                wandb_run=wandb_run,
            )
        wandb_run.finish()
        return result
    except BaseException:
        wandb_run.finish(exit_code=1)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("cache", "train", "all"), default="all")
    parser.add_argument(
        "--max-datasets",
        type=int,
        default=None,
        help="Smoke limit applied independently to train and validation splits.",
    )
    parser.add_argument(
        "--max-transitions",
        type=int,
        default=None,
        help="Smoke limit for owned right transitions per selected dataset.",
    )
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        stage=args.stage,
        max_datasets=args.max_datasets,
        max_transitions=args.max_transitions,
    )
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
