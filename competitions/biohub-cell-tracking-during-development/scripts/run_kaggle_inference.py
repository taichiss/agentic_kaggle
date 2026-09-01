#!/usr/bin/env python
"""Run the trained host model on Kaggle test Zarrs and write submission.csv.

This submission-time entrypoint deliberately avoids GEFF/tracksdata dependencies. It
uses the organizer model classes bundled beside this file, reads the competition's
one-frame Zarr v3 chunks directly, and serializes the predicted graph to the required
long CSV schema.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import blosc2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

POS_EMBED_DIM = 8
EXP7A_EPOCH5_REPORT_BASELINE = 0.5688260117
CSV_COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)

TEMPORAL_GRAPH_ENV = "BIOHUB_TEMPORAL_GRAPH_CHECKPOINT"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_bundle_manifest(bundle_dir: Path) -> dict[str, object]:
    """Verify packaged bytes, including Kaggle-expanded ZIP members."""
    manifest_path = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported package manifest schema")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("package manifest has no files")
    for name, raw_entry in files.items():
        if Path(name).name != name or not isinstance(raw_entry, dict):
            raise ValueError(f"unsafe package manifest entry: {name!r}")
        expected_bytes = int(raw_entry["bytes"])
        expected_sha256 = str(raw_entry["sha256"])
        path = bundle_dir / name
        if path.is_file() and not path.is_symlink():
            if path.stat().st_size != expected_bytes or _sha256(path) != expected_sha256:
                raise ValueError(f"package file does not match manifest: {name}")
            continue
        members = raw_entry.get("members")
        expanded_root = bundle_dir / Path(name).stem
        if (
            not name.endswith(".zip")
            or not isinstance(members, dict)
            or not members
        ):
            raise FileNotFoundError(f"package file is missing: {path}")
        if not expanded_root.is_dir() or expanded_root.is_symlink():
            raise FileNotFoundError(f"expanded package archive is missing: {expanded_root}")
        expanded_resolved = expanded_root.resolve(strict=True)
        expected_members: set[str] = set()
        for member_name, raw_member in members.items():
            member_relative = Path(member_name)
            if (
                member_relative.is_absolute()
                or ".." in member_relative.parts
                or "\\" in member_name
                or not isinstance(raw_member, dict)
            ):
                raise ValueError(f"unsafe expanded archive member: {member_name!r}")
            expected_members.add(member_relative.as_posix())
            member = expanded_root / member_relative
            if member.is_symlink() or not member.is_file():
                raise FileNotFoundError(f"expanded archive member is missing: {member}")
            if not member.resolve(strict=True).is_relative_to(expanded_resolved):
                raise ValueError(f"expanded archive member escapes its root: {member}")
            if (
                member.stat().st_size != int(raw_member["bytes"])
                or _sha256(member) != raw_member["sha256"]
            ):
                raise ValueError(f"expanded archive member hash mismatch: {member_name}")
        descendants = tuple(expanded_root.rglob("*"))
        symlinks = sorted(str(item) for item in descendants if item.is_symlink())
        if symlinks:
            raise ValueError(f"expanded package archive contains symlinks: {symlinks}")
        actual_members = {
            item.relative_to(expanded_root).as_posix()
            for item in descendants
            if item.is_file()
        }
        if actual_members != expected_members:
            extra = sorted(actual_members - expected_members)
            missing_members = sorted(expected_members - actual_members)
            raise ValueError(
                "expanded package archive member set mismatch: "
                f"extra={extra}, missing={missing_members}"
            )

    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    if config.get("model_api") == "corrected_v2":
        required = {
            "config.json",
            "edge_predictor_best.pth",
            "inference_profile.json",
            "checkpoint-metadata.json",
            "selection-provenance.json",
            "run_kaggle_inference.py",
            "tracking_cellmot_models.zip",
            "backbone_ab.zip",
            "dynamic_network_architectures.zip",
        }
        missing = sorted(required - files.keys())
        if missing:
            raise ValueError(f"finalized corrected_v2 package is missing: {missing}")
        metadata = json.loads(
            (bundle_dir / "checkpoint-metadata.json").read_text(encoding="utf-8")
        )
        provenance = json.loads(
            (bundle_dir / "selection-provenance.json").read_text(encoding="utf-8")
        )
        provenance_hashes = {
            "checkpoint_sha256",
            "inference_profile_sha256",
            "experiment_config_sha256",
            "manifest_sha256",
            "selection_sha256",
            "report_summary_sha256",
            "report_config_sha256",
        }
        for key in provenance_hashes:
            value = provenance.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"finalized package provenance has invalid {key}")
        expected_epochs = list(range(5, 51, 5))
        if provenance.get("candidate_epochs") != expected_epochs:
            raise ValueError("finalized package does not cover the complete checkpoint sweep")
        completed_epoch = int(provenance.get("completed_epoch", 0))
        if (
            int(manifest.get("completed_epochs", 0)) != completed_epoch
            or int(metadata.get("completed_epochs", 0)) != completed_epoch
        ):
            raise ValueError("finalized package completed epoch mismatch")
        if (
            manifest.get("experiment_id") != "EXP-0007A"
            or metadata.get("experiment_id") != "EXP-0007A"
            or config.get("experiment_id") != "EXP-0007A"
        ):
            raise ValueError("finalized package experiment_id mismatch")
        if (
            metadata.get("source_checkpoint_sha256")
            != provenance.get("checkpoint_sha256")
        ):
            raise ValueError("finalized package source checkpoint mismatch")
        if (
            metadata.get("source_inference_profile_sha256")
            != provenance.get("inference_profile_sha256")
        ):
            raise ValueError("finalized package source inference profile mismatch")
        if (
            metadata.get("experiment_config_sha256")
            != provenance.get("experiment_config_sha256")
        ):
            raise ValueError("finalized package experiment config mismatch")
        if (
            metadata.get("validation_subset_manifest_sha256")
            != provenance.get("manifest_sha256")
        ):
            raise ValueError("finalized package validation split mismatch")
        score = float(provenance.get("report_score", float("nan")))
        baseline = float(provenance.get("report_score_baseline", float("nan")))
        tolerance = float(provenance.get("report_score_tolerance", float("nan")))
        gate = float(provenance.get("report_score_gate_exclusive", float("nan")))
        if (
            not np.isfinite(score)
            or not np.isfinite(baseline)
            or not np.isfinite(tolerance)
            or tolerance < 0.0
            or baseline != EXP7A_EPOCH5_REPORT_BASELINE
            or gate != baseline + tolerance
            or score <= gate
        ):
            raise ValueError("finalized package report score gate was not satisfied")
    return manifest


def _config_temporal_graph_checkpoint(config: dict[str, object]) -> str | None:
    """Return an explicitly configured graph checkpoint, without auto-enabling it."""
    inference = config.get("inference")
    temporal_graph = config.get("temporal_graph")
    raw_values: list[object] = [config.get("temporal_graph_checkpoint")]
    if isinstance(inference, dict):
        raw_values.append(inference.get("temporal_graph_checkpoint"))
    if isinstance(temporal_graph, dict):
        raw_values.append(temporal_graph.get("checkpoint"))
    for value in raw_values:
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise ValueError("configured temporal graph checkpoint must be a non-empty string")
        return value
    return None


def _resolve_temporal_graph_checkpoint(
    cli_path: Path | None,
    bundle_dir: Path,
    config: dict[str, object],
) -> Path | None:
    """Resolve CLI, environment, then bundle-config overrides in that order."""
    if cli_path is not None:
        return cli_path.expanduser()
    environment = os.environ.get(TEMPORAL_GRAPH_ENV) or os.environ.get(
        "TEMPORAL_GRAPH_CHECKPOINT"
    )
    if environment:
        return Path(environment).expanduser()
    configured = _config_temporal_graph_checkpoint(config)
    if configured is None:
        return None
    path = Path(configured)
    return path if path.is_absolute() else bundle_dir / path


def _base_checkpoint_identities(bundle_dir: Path) -> set[str]:
    """Return verified hashes that identify the frozen host checkpoint."""
    weights_path = bundle_dir / "edge_predictor_best.pth"
    weights_sha256 = _sha256(weights_path)
    identities = {weights_sha256}
    metadata_path = bundle_dir / "checkpoint-metadata.json"
    if not metadata_path.is_file():
        return identities

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    recorded_weights = metadata.get("weights_sha256")
    if recorded_weights is not None and recorded_weights != weights_sha256:
        raise ValueError("base checkpoint bytes do not match checkpoint-metadata.json")
    source_sha256 = metadata.get("source_checkpoint_sha256")
    if source_sha256 is not None:
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise ValueError("invalid source checkpoint SHA-256 in checkpoint metadata")
        identities.add(source_sha256)
    return identities


def _register_temporal_graph_import_paths(bundle_dir: Path) -> None:
    """Support repository sources and Kaggle's ZIP-expanded Dataset layout."""
    local_src = Path(__file__).resolve().parents[1] / "src"
    candidates = (
        local_src,
        bundle_dir,
        bundle_dir / "temporal_graph.zip",
        bundle_dir / "temporal_graph",
    )
    for candidate in candidates:
        if candidate.exists():
            sys.path.insert(0, str(candidate))


def _load_temporal_graph_head(
    checkpoint_path: Path,
    bundle_dir: Path,
    device: torch.device,
) -> Any:
    """Load a safe graph-head payload and bind it to the exact frozen host."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"temporal graph checkpoint is missing: {checkpoint_path}")
    _register_temporal_graph_import_paths(bundle_dir)
    from temporal_graph import TemporalGraphCheckpoint, TemporalGraphResidualHead

    raw_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(raw_payload, dict):
        raise TypeError("temporal graph checkpoint must contain a mapping")
    payload = raw_payload.get("temporal_graph", raw_payload)
    if not isinstance(payload, dict):
        raise TypeError("temporal_graph checkpoint entry must contain a mapping")
    checkpoint = TemporalGraphCheckpoint.from_payload(payload)
    base_identities = _base_checkpoint_identities(bundle_dir)
    if checkpoint.base_checkpoint_sha256 not in base_identities:
        actual = ", ".join(sorted(base_identities))
        raise ValueError(
            "temporal graph/base checkpoint SHA-256 mismatch: "
            f"expected {checkpoint.base_checkpoint_sha256}, found {actual}"
        )
    head = TemporalGraphResidualHead.from_checkpoint_payload(
        payload,
        map_location=device,
    )
    return head.eval()


def _temporal_graph_config(temporal_graph_head: Any) -> Any | None:
    """Return the effective config for a head or a two-head link ensemble."""
    config = getattr(temporal_graph_head, "config", None)
    if config is not None:
        return config
    mlp_head = getattr(temporal_graph_head, "mlp_head", None)
    return None if mlp_head is None else getattr(mlp_head, "config", None)


def _temporal_graph_window_size(temporal_graph_head: Any) -> int:
    """Read the graph window while retaining legacy T3 callable compatibility."""
    config = _temporal_graph_config(temporal_graph_head)
    if config is None:
        # Older tests and downstream wrappers supplied a callable without
        # exposing its T3 config. Keep that pre-T4 calling contract intact.
        return 3
    value = getattr(config, "graph_window_size", None)
    if isinstance(value, bool) or not isinstance(value, int) or value not in {3, 4}:
        raise ValueError("temporal graph config requires graph_window_size=3 or 4")
    return value


def _validate_temporal_graph_fallback_contract(
    temporal_graph_head: Any,
    temporal_graph_fallback_head: Any,
) -> None:
    """Fail closed unless a T4 primary and T3 fallback share candidate semantics."""
    primary = _temporal_graph_config(temporal_graph_head)
    fallback = _temporal_graph_config(temporal_graph_fallback_head)
    if primary is None or fallback is None:
        raise ValueError("primary and fallback temporal heads must expose their config")
    if getattr(primary, "graph_window_size", None) != 4:
        raise ValueError("a temporal graph fallback requires a T_graph=4 primary head")
    if getattr(fallback, "graph_window_size", None) != 3:
        raise ValueError("the temporal graph fallback must use T_graph=3")

    comparable_fields = (
        "node_feature_dim",
        "top_k",
        "radius_um",
        "distance_scale_um",
        "middle_coord_atol",
        "image_window_size",
        "ownership",
    )
    missing = [
        name
        for name in comparable_fields
        if not hasattr(primary, name) or not hasattr(fallback, name)
    ]
    if missing:
        raise ValueError(
            "temporal graph candidate contract is incomplete: " + ", ".join(missing)
        )
    mismatches = [
        name
        for name in comparable_fields
        if getattr(primary, name) != getattr(fallback, name)
    ]
    if mismatches:
        raise ValueError(
            "primary and fallback temporal heads use different candidate contracts: "
            + ", ".join(mismatches)
        )


def _owned_transition_logits(
    temporal_graph_head: Any | None,
    previous_pair: Any | None,
    current_pair: Any,
    *,
    prior_pair: Any | None = None,
    temporal_graph_fallback_head: Any | None = None,
) -> torch.Tensor:
    """Refine one owned transition when the configured history is complete."""
    if temporal_graph_head is None or previous_pair is None:
        return current_pair.edge_logits
    graph_window_size = _temporal_graph_window_size(temporal_graph_head)
    if graph_window_size == 3:
        return temporal_graph_head(previous_pair, current_pair).edge_logits
    if prior_pair is None:
        if temporal_graph_fallback_head is None:
            return current_pair.edge_logits
        return temporal_graph_fallback_head(previous_pair, current_pair).edge_logits
    return temporal_graph_head(
        previous_pair,
        current_pair,
        prior_pair=prior_pair,
    ).edge_logits


class UNetNodeTransformer(nn.Module):
    """Inference-compatible copy of the organizer's composed model."""

    def __init__(
        self,
        unet: nn.Module,
        unet_out_channels: int,
        pos_feat_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_blocks: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        from tracking_cellmot.models import SimpleNodeTransformer

        self.unet = unet
        self.unet_out_channels = unet_out_channels
        self.detect_head = nn.Conv3d(unet_out_channels, 1, kernel_size=1)
        self.transformer = SimpleNodeTransformer(
            feat_dim=unet_out_channels + pos_feat_dim,
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            n_blocks=n_blocks,
            dropout=dropout,
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        output = self.unet(images.unsqueeze(2))
        return output, [self.detect_head(output[:, index]) for index in range(output.shape[1])]

    def index_features(
        self,
        feature_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels = feature_maps.shape[:2]
        spatial = feature_maps.shape[2:]
        result = torch.zeros(
            batch,
            coords.shape[1],
            channels,
            device=feature_maps.device,
            dtype=feature_maps.dtype,
        )
        for sample in range(batch):
            count = int(mask[sample].sum().item())
            if count == 0:
                continue
            z = coords[sample, :count, 0].long().clamp(0, spatial[0] - 1)
            y = coords[sample, :count, 1].long().clamp(0, spatial[1] - 1)
            x = coords[sample, :count, 2].long().clamp(0, spatial[2] - 1)
            result[sample, :count] = feature_maps[sample, :, z, y, x].T
        return result

    def predict_edges(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor,
        source_coords: torch.Tensor,
        target_coords: torch.Tensor,
        source_position: torch.Tensor,
        target_position: torch.Tensor,
        source_mask: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        source = torch.cat([source_features, source_position], dim=-1)
        target = torch.cat([target_features, target_position], dim=-1)
        return self.transformer(
            source,
            target,
            source_coords,
            target_coords,
            source_mask,
            target_mask,
        )


def _position_features(coords: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    axes = [coords[:, axis] / max(size, 1) for axis, size in enumerate(image_shape)]
    frequencies = 2 ** np.arange(POS_EMBED_DIM // 2)
    parts = []
    for values in axes:
        angles = values[:, None] * frequencies * np.pi
        parts.extend([np.sin(angles), np.cos(angles)])
    return np.concatenate(parts, axis=1).astype(np.float32)


def _zarr_metadata(path: Path) -> dict[str, object]:
    root = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    array = json.loads((path / "0" / "zarr.json").read_text(encoding="utf-8"))
    attributes = root["attributes"]
    scale = attributes["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    quantiles = attributes["image_statistics"]["quantiles"]
    return {
        "shape": tuple(array["shape"]),
        "dtype": np.dtype(array["data_type"]),
        "scale": tuple(float(value) for value in scale[1:]),
        "q_low": float(quantiles["0.001"]),
        "q_high": float(quantiles["0.999"]),
    }


def _load_frame(
    dataset: Path,
    frame: int,
    spatial_shape: tuple[int, int, int],
    dtype: np.dtype,
    downsample: tuple[int, int, int],
) -> torch.Tensor:
    chunk = dataset / "0" / "c" / str(frame) / "0" / "0" / "0"
    raw = blosc2.decompress(chunk.read_bytes())
    image = np.frombuffer(raw, dtype=dtype).reshape(spatial_shape)
    dz, dy, dx = downsample
    return torch.from_numpy(image[::dz, ::dy, ::dx].astype(np.float32, copy=True))


def _pool_kernel(distance_um: float, voxel_size: tuple[float, ...]) -> tuple[int, ...]:
    result = []
    for size in voxel_size:
        kernel = max(1, round(distance_um / size))
        result.append(kernel if kernel % 2 == 1 else kernel + 1)
    return tuple(result)


def _detect(
    logits: torch.Tensor,
    frame: int,
    threshold: float,
    pool_kernel: tuple[int, ...],
) -> np.ndarray:
    volume = logits.unsqueeze(0)
    padding = tuple(size // 2 for size in pool_kernel)
    pooled = F.max_pool3d(volume, pool_kernel, stride=1, padding=padding)
    peaks = (volume == pooled) & (torch.sigmoid(volume) > threshold)
    indices = torch.nonzero(peaks[0, 0])
    if indices.shape[0] == 0:
        return np.empty((0, 4), dtype=np.int16)
    spatial = indices.cpu().numpy().astype(np.int16)
    time_column = np.full((len(spatial), 1), frame, dtype=np.int16)
    return np.concatenate([time_column, spatial], axis=1)


def _load_model(
    bundle_dir: Path,
    device: torch.device,
) -> tuple[Any, tuple[int, int, int], int, float, Any | None, dict[str, object]]:
    _verify_bundle_manifest(bundle_dir)
    sys.path.insert(0, str(bundle_dir))
    bundled_models = bundle_dir / "tracking_cellmot_models.zip"
    expanded_models = bundle_dir / "tracking_cellmot_models"
    # Prefer the content-addressed ZIP when both local representations exist.
    # Kaggle normally exposes only the expanded same-named directory.
    if bundled_models.is_file():
        sys.path.insert(0, str(bundled_models))
    elif expanded_models.is_dir():
        sys.path.insert(0, str(expanded_models))
    for archive_name in ("backbone_ab.zip", "dynamic_network_architectures.zip"):
        archive = bundle_dir / archive_name
        expanded = bundle_dir / archive.stem
        if archive.is_file():
            sys.path.insert(0, str(archive))
        elif expanded.is_dir():
            sys.path.insert(0, str(expanded))

    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    model_api = str(config.get("model_api", "legacy"))
    if model_api == "corrected_v2":
        from backbone_ab.backbones import build_joint_model
        from tracking_cellmot.models import SimpleNodeTransformer

        host_model_api = SimpleNamespace(
            SimpleNodeTransformer=SimpleNodeTransformer,
            _POS_EMBED_DIM=POS_EMBED_DIM,
        )
        model = build_joint_model(config["backbone"], host_model_api)
    elif config.get("backbone_type"):
        from backbone_ab.backbones import build_backbone

        unet = build_backbone(config["backbone"])
        output_channels = int(config["backbone"]["feature_dim"])
        model = UNetNodeTransformer(
            unet=unet,
            unet_out_channels=output_channels,
            pos_feat_dim=4 * POS_EMBED_DIM,
        )
    else:
        from tracking_cellmot.models import TemporalUNet3D

        unet = TemporalUNet3D(
            in_channels=1,
            out_channels=int(config["unet_out_channels"]),
            layers=[int(value) for value in config["unet_layers"]],
        )
        output_channels = int(config["unet_out_channels"])
        model = UNetNodeTransformer(
            unet=unet,
            unet_out_channels=output_channels,
            pos_feat_dim=4 * POS_EMBED_DIM,
        )
    profile = None
    if model_api == "corrected_v2":
        from backbone_ab.checkpointing import load_checkpoint, load_inference_profile

        checkpoint = load_checkpoint(
            bundle_dir / "edge_predictor_best.pth", map_location=device
        )
        profile = load_inference_profile(bundle_dir / "inference_profile.json")
        if checkpoint.sha256 != profile.checkpoint_sha256:
            raise ValueError("bundled checkpoint does not match inference profile")
        model.load_state_dict(checkpoint.state_dict)
    else:
        state = torch.load(
            bundle_dir / "edge_predictor_best.pth",
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
    model.to(device).eval()
    return (
        model,
        tuple(int(value) for value in config["downsample"]),
        int(config["window_size"]),
        float(config.get("pool_kernel_um", 5.0)),
        profile,
        config,
    )


@torch.inference_mode()
def predict_dataset(
    model: UNetNodeTransformer,
    dataset: Path,
    device: torch.device,
    downsample: tuple[int, int, int],
    window_size: int,
    det_threshold: float,
    edge_threshold: float,
    pool_kernel_um: float,
    detection_tta: bool,
    max_frames: int | None = None,
    temporal_graph_head: Any | None = None,
    temporal_graph_fallback_head: Any | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    if temporal_graph_fallback_head is not None and temporal_graph_head is None:
        raise ValueError("a temporal graph fallback requires a primary temporal head")
    if temporal_graph_head is not None and window_size != 2:
        raise ValueError("temporal graph inference requires the frozen host window_size=2")
    graph_window_size = 0
    if temporal_graph_head is not None:
        graph_window_size = _temporal_graph_window_size(temporal_graph_head)
    if temporal_graph_fallback_head is not None:
        _validate_temporal_graph_fallback_contract(
            temporal_graph_head,
            temporal_graph_fallback_head,
        )
    frozen_pair_type: Any | None = None
    if temporal_graph_head is not None:
        from temporal_graph import FrozenPair

        frozen_pair_type = FrozenPair

    metadata = _zarr_metadata(dataset)
    shape = metadata["shape"]
    frames = int(shape[0]) if max_frames is None else min(int(shape[0]), max_frames)
    if frames < window_size:
        raise ValueError(f"need at least {window_size} frames, got {frames}")
    spatial_shape = tuple(int(value) for value in shape[1:])
    target_shape = tuple(
        (size + stride - 1) // stride
        for size, stride in zip(spatial_shape, downsample, strict=True)
    )
    voxel_size = tuple(
        scale * stride
        for scale, stride in zip(metadata["scale"], downsample, strict=True)
    )
    pool_kernel = _pool_kernel(pool_kernel_um, voxel_size)
    downsample_array = np.asarray(downsample, dtype=np.float32)
    downsample_tensor = torch.from_numpy(downsample_array).to(device)
    voxel_size_tensor = (
        torch.tensor(voxel_size, dtype=torch.float32, device=device)
        if temporal_graph_head is not None
        else None
    )
    q_low = float(metadata["q_low"])
    q_high = float(metadata["q_high"])

    stride = max(window_size - 1, 1)
    starts = list(range(0, frames - window_size + 1, stride))
    if not starts or starts[-1] + window_size < frames:
        last = max(frames - window_size, 0)
        if not starts or starts[-1] != last:
            starts.append(last)

    seen_frames: set[int] = set()
    seen_pairs: set[tuple[int, int]] = set()
    coord_lists: list[np.ndarray] = []
    offsets: dict[int, tuple[int, int]] = {}
    node_count = 0
    edges: list[tuple[int, int, float]] = []
    pair_history: list[Any] = []
    previous_target_frame: int | None = None

    for window_number, start in enumerate(starts):
        frame_indices = list(range(start, start + window_size))
        images = torch.stack(
            [
                _load_frame(
                    dataset,
                    frame,
                    spatial_shape,
                    metadata["dtype"],
                    downsample,
                )
                for frame in frame_indices
            ]
        )
        images = ((images - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        images = images.unsqueeze(0).to(device)
        feature_maps, detection_logits = model.encode(images)

        if detection_tta:
            for dimensions in [(-1,), (-2,), (-2, -1)]:
                flipped = images.flip(dimensions)
                _, flipped_logits = model.encode(flipped)
                for index in range(window_size):
                    detection_logits[index] += flipped_logits[index].flip(dimensions)
            for index in range(window_size):
                detection_logits[index] /= 4

        for index, frame in enumerate(frame_indices):
            if frame in seen_frames:
                continue
            coords = _detect(
                detection_logits[index][0],
                frame,
                det_threshold,
                pool_kernel,
            )
            offsets[frame] = (node_count, node_count + len(coords))
            node_count += len(coords)
            coord_lists.append(coords)
            seen_frames.add(frame)

        stacked_coords = (
            np.concatenate(coord_lists)
            if coord_lists
            else np.empty((0, 4), dtype=np.int16)
        )
        for index in range(window_size - 1):
            source_frame, target_frame = frame_indices[index : index + 2]
            if (source_frame, target_frame) in seen_pairs:
                continue
            seen_pairs.add((source_frame, target_frame))
            source_start, source_end = offsets[source_frame]
            target_start, target_end = offsets[target_frame]
            if source_start == source_end or target_start == target_end:
                if temporal_graph_head is not None:
                    pair_history.clear()
                    previous_target_frame = None
                continue

            source = stacked_coords[source_start:source_end]
            target = stacked_coords[target_start:target_end]
            source_coords = (
                torch.from_numpy(source[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            )
            target_coords = (
                torch.from_numpy(target[:, 1:].astype(np.float32)).unsqueeze(0).to(device)
            )
            source_relative = source.copy()
            target_relative = target.copy()
            source_relative[:, 0] = index
            target_relative[:, 0] = index + 1
            position_shape = (window_size,) + target_shape
            source_position = torch.from_numpy(
                _position_features(source_relative, position_shape)
            ).unsqueeze(0).to(device)
            target_position = torch.from_numpy(
                _position_features(target_relative, position_shape)
            ).unsqueeze(0).to(device)
            source_mask = torch.ones(1, len(source), dtype=torch.bool, device=device)
            target_mask = torch.ones(1, len(target), dtype=torch.bool, device=device)
            source_features = model.index_features(
                feature_maps[:, index], source_coords, source_mask
            )
            target_features = model.index_features(
                feature_maps[:, index + 1], target_coords, target_mask
            )
            base_logits = model.predict_edges(
                source_features,
                target_features,
                source_coords * downsample_tensor,
                target_coords * downsample_tensor,
                source_position,
                target_position,
                source_mask,
                target_mask,
            )
            if temporal_graph_head is None:
                logits = base_logits[0]
            else:
                if frozen_pair_type is None or voxel_size_tensor is None:
                    raise RuntimeError("temporal graph runtime was not initialized")
                current_pair = frozen_pair_type(
                    source_features=source_features,
                    target_features=target_features,
                    source_coords_um=source_coords * voxel_size_tensor,
                    target_coords_um=target_coords * voxel_size_tensor,
                    source_mask=source_mask,
                    target_mask=target_mask,
                    edge_logits=base_logits,
                )
                if previous_target_frame != source_frame:
                    pair_history.clear()
                previous_pair = pair_history[-1] if pair_history else None
                prior_pair = pair_history[-2] if len(pair_history) >= 2 else None
                logits = _owned_transition_logits(
                    temporal_graph_head,
                    previous_pair,
                    current_pair,
                    prior_pair=prior_pair,
                    temporal_graph_fallback_head=temporal_graph_fallback_head,
                )[0]
                pair_history.append(current_pair.detached())
                history_limit = graph_window_size - 2
                if len(pair_history) > history_limit:
                    del pair_history[:-history_limit]
                previous_target_frame = target_frame
            probabilities = torch.softmax(logits, dim=0).cpu().numpy()
            candidates = sorted(
                (
                    (float(probabilities[i, j]), i, j)
                    for i in range(len(source))
                    for j in range(len(target))
                    if probabilities[i, j] > edge_threshold
                ),
                reverse=True,
            )
            children: dict[int, int] = {}
            parents: dict[int, int] = {}
            for _, source_index, target_index in candidates:
                if children.get(source_index, 0) >= 2 or parents.get(target_index, 0) >= 1:
                    continue
                edges.append(
                    (
                        source_start + source_index,
                        target_start + target_index,
                        float(probabilities[source_index, target_index]),
                    )
                )
                children[source_index] = children.get(source_index, 0) + 1
                parents[target_index] = parents.get(target_index, 0) + 1

        if (window_number + 1) % 25 == 0 or window_number + 1 == len(starts):
            print(
                f"  {dataset.stem}: windows {window_number + 1}/{len(starts)}, "
                f"nodes={node_count}, edges={len(edges)}",
                flush=True,
            )

    coords = (
        np.concatenate(coord_lists).astype(np.float32)
        if coord_lists
        else np.empty((0, 4), dtype=np.float32)
    )
    coords[:, 1:] *= downsample_array
    return coords.astype(np.int64), edges


def postprocess_prediction(
    coords: np.ndarray,
    raw_edges: list[tuple[int, int, float]],
    voxel_size_um: tuple[float, float, float],
    *,
    minimum_component_nodes: int = 6,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, int | float]]:
    """Apply the artifact-free topology repairs used by the public 0.926 harness.

    This intentionally excludes its second-seed ensemble, DeepCenter detector, synthetic
    gap nodes, and eight-view TTA. Those are additional models or inference changes rather
    than post-processing that can be applied to this checkpoint's frozen predictions.
    """
    if minimum_component_nodes <= 0:
        raise ValueError("minimum_component_nodes must be positive")
    if len(coords) == 0:
        return coords, [], {"raw_nodes": 0, "raw_edges": len(raw_edges)}

    scale = np.asarray(voxel_size_um, dtype=np.float64)
    positions_um = coords[:, 1:].astype(np.float64) * scale
    ids_by_time = {
        int(frame): np.flatnonzero(coords[:, 0] == frame).astype(np.int64).tolist()
        for frame in np.unique(coords[:, 0])
    }
    learned_prob = {
        (int(source), int(target)): float(np.clip(probability, 0.0, 1.0))
        for source, target, probability in raw_edges
    }
    learned_by_time: dict[int, list[tuple[int, int, float]]] = {}
    for (source, target), probability in learned_prob.items():
        source_time = int(coords[source, 0])
        if int(coords[target, 0]) == source_time + 1:
            learned_by_time.setdefault(source_time, []).append(
                (source, target, probability)
            )

    predecessor_position: dict[int, np.ndarray] = {}
    motion_edges: list[tuple[int, int, float]] = []
    tight_edges = 0
    relaxed_edges = 0

    def assign_pass(
        source_ids: list[int],
        target_ids: list[int],
        frame: int,
        gate_um: float,
    ) -> list[tuple[int, int, float]]:
        if not source_ids or not target_ids:
            return []
        source_pos = positions_um[source_ids]
        target_pos = positions_um[target_ids]
        predicted = source_pos.copy()
        for index, source_id in enumerate(source_ids):
            previous = predecessor_position.get(source_id)
            if previous is not None:
                predicted[index] += 0.5 * (source_pos[index] - previous)
        raw_distance = np.linalg.norm(
            source_pos[:, None, :] - target_pos[None, :, :], axis=2
        )
        motion_distance = np.linalg.norm(
            predicted[:, None, :] - target_pos[None, :, :], axis=2
        )
        probability = np.zeros_like(raw_distance)
        source_index = {node_id: index for index, node_id in enumerate(source_ids)}
        target_index = {node_id: index for index, node_id in enumerate(target_ids)}
        for source, target, value in learned_by_time.get(frame, []):
            row = source_index.get(source)
            column = target_index.get(target)
            if row is not None and column is not None:
                probability[row, column] = value
        big = gate_um * 1000.0 + 1.0
        cost = motion_distance + 0.05 * raw_distance - probability
        cost[raw_distance > gate_um] = big
        rows, columns = linear_sum_assignment(cost)
        return [
            (
                source_ids[int(row)],
                target_ids[int(column)],
                float(probability[row, column]),
            )
            for row, column in zip(rows, columns, strict=True)
            if cost[row, column] < big
        ]

    for frame in sorted(ids_by_time):
        source_ids = ids_by_time.get(frame, [])
        target_ids = ids_by_time.get(frame + 1, [])
        unmatched_sources = set(source_ids)
        unmatched_targets = set(target_ids)
        frame_edges: list[tuple[int, int, float]] = []
        for gate_um, is_tight in ((6.0, True), (10.0, False)):
            candidates = assign_pass(
                [node for node in source_ids if node in unmatched_sources],
                [node for node in target_ids if node in unmatched_targets],
                frame,
                gate_um,
            )
            for source, target, probability in candidates:
                if source not in unmatched_sources or target not in unmatched_targets:
                    continue
                unmatched_sources.remove(source)
                unmatched_targets.remove(target)
                frame_edges.append((source, target, probability))
                if is_tight:
                    tight_edges += 1
                else:
                    relaxed_edges += 1
        for source, target, probability in frame_edges:
            motion_edges.append((source, target, probability))
            predecessor_position[target] = positions_um[source]

    outgoing: dict[int, list[int]] = {}
    incoming: set[int] = set()
    for source, target, _ in motion_edges:
        outgoing.setdefault(source, []).append(target)
        incoming.add(target)
    global_division_cap = max(1, round(max(1, len(motion_edges)) * 0.00375))
    division_edges: list[tuple[int, int, float]] = []
    used_targets: set[int] = set()
    for frame in sorted(ids_by_time):
        child_ids = ids_by_time.get(frame + 1, [])
        source_ids = [
            node_id
            for node_id in ids_by_time.get(frame, [])
            if len(outgoing.get(node_id, [])) == 1 and node_id in incoming
        ]
        orphan_ids = [
            node_id
            for node_id in child_ids
            if node_id not in incoming and node_id not in used_targets
        ]
        if not source_ids or not orphan_ids:
            continue
        orphan_tree = cKDTree(positions_um[orphan_ids])
        frame_cap = max(1, round(len(source_ids) * 0.0076))
        proposals: list[tuple[float, int, int]] = []
        for source in source_ids:
            existing_child = outgoing[source][0]
            if np.linalg.norm(positions_um[source] - positions_um[existing_child]) > 10.0:
                continue
            _, nearest_index = orphan_tree.query(positions_um[existing_child])
            candidate = orphan_ids[int(nearest_index)]
            parent_distance = float(
                np.linalg.norm(positions_um[source] - positions_um[candidate])
            )
            sister_distance = float(
                np.linalg.norm(positions_um[existing_child] - positions_um[candidate])
            )
            if parent_distance > 8.0 or sister_distance > 11.0:
                continue
            first_successors = outgoing.get(existing_child, [])
            second_successors = outgoing.get(candidate, [])
            if len(first_successors) != 1 or len(second_successors) != 1:
                continue
            first_next = first_successors[0]
            second_next = second_successors[0]
            if (
                int(coords[first_next, 0]) != frame + 2
                or int(coords[second_next, 0]) != frame + 2
            ):
                continue
            next_sister_distance = float(
                np.linalg.norm(positions_um[first_next] - positions_um[second_next])
            )
            if next_sister_distance - sister_distance < 2.25:
                continue
            proposals.append(
                (parent_distance + 0.15 * sister_distance, source, candidate)
            )
        proposals.sort()
        added_this_frame = 0
        used_sources: set[int] = set()
        for _, source, candidate in proposals:
            if len(division_edges) >= global_division_cap or added_this_frame >= frame_cap:
                break
            if source in used_sources or candidate in used_targets or candidate in incoming:
                continue
            division_edges.append((source, candidate, 0.0))
            outgoing[source].append(candidate)
            incoming.add(candidate)
            used_sources.add(source)
            used_targets.add(candidate)
            added_this_frame += 1

    combined_edges = [*motion_edges, *division_edges]
    incident = {
        node_id
        for source, target, _ in combined_edges
        for node_id in (source, target)
    }

    parent = {node_id: node_id for node_id in incident}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[left_root] = right_root

    outdegree: dict[int, int] = {}
    for source, target, _ in combined_edges:
        union(source, target)
        outdegree[source] = outdegree.get(source, 0) + 1
    components: dict[int, list[int]] = {}
    for node_id in incident:
        components.setdefault(find(node_id), []).append(node_id)
    kept_ids: set[int] = set()
    for members in components.values():
        has_division = any(outdegree.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= minimum_component_nodes or has_division:
            kept_ids.update(members)
    if not kept_ids:
        kept_ids = incident
    kept_edges = [
        (source, target)
        for source, target, _ in combined_edges
        if source in kept_ids and target in kept_ids
    ]

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for source, target in kept_edges:
        predecessor.setdefault(target, []).append(source)
        successor.setdefault(source, []).append(target)
    smoothed = coords.astype(np.float64, copy=True)
    for node_id in sorted(kept_ids):
        if len(predecessor.get(node_id, [])) != 1 or len(successor.get(node_id, [])) != 1:
            continue
        neighborhood = [(0, node_id)]
        current = node_id
        for step in range(1, 3):
            previous = predecessor.get(current, [])
            if len(previous) != 1:
                break
            current = previous[0]
            neighborhood.append((-step, current))
        current = node_id
        for step in range(1, 3):
            following = successor.get(current, [])
            if len(following) != 1:
                break
            current = following[0]
            neighborhood.append((step, current))
        if len(neighborhood) < 3:
            continue
        times = np.asarray([time for time, _ in neighborhood], dtype=np.float64)
        points = np.asarray(
            [coords[member, 1:] for _, member in neighborhood], dtype=np.float64
        )
        fitted = np.asarray(
            [np.polyval(np.polyfit(times, points[:, axis], 1), 0.0) for axis in range(3)]
        )
        smoothed[node_id, 1:] = 0.2 * coords[node_id, 1:] + 0.8 * fitted

    ordered_ids = sorted(kept_ids)
    remap = {old: new for new, old in enumerate(ordered_ids)}
    output_coords = smoothed[ordered_ids]
    output_coords[:, 1:] = np.maximum(0, np.rint(output_coords[:, 1:]))
    output_edges = [(remap[source], remap[target]) for source, target in kept_edges]
    stats: dict[str, int | float] = {
        "raw_nodes": len(coords),
        "raw_edges": len(raw_edges),
        "motion_edges": len(motion_edges),
        "motion_tight_edges": tight_edges,
        "motion_relaxed_edges": relaxed_edges,
        "safe_divisions_added": len(division_edges),
        "minimum_component_nodes": minimum_component_nodes,
        "short_or_isolated_nodes_removed": len(coords) - len(output_coords),
        "nodes": len(output_coords),
        "edges": len(output_edges),
        "edge_node_ratio": round(len(output_edges) / max(len(output_coords), 1), 6),
    }
    return output_coords.astype(np.int64), output_edges, stats


def write_submission(
    predictions: list[tuple[str, np.ndarray, list[tuple[int, int] | tuple[int, int, float]]]],
    output_path: Path,
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_id = 0
    node_total = 0
    edge_total = 0
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(CSV_COLUMNS)
        for dataset, coords, edges in predictions:
            for node_id, (frame, z, y, x) in enumerate(coords):
                writer.writerow(
                    [row_id, dataset, "node", node_id, frame, z, y, x, -1, -1]
                )
                row_id += 1
                node_total += 1
            for edge in edges:
                source_id, target_id = int(edge[0]), int(edge[1])
                writer.writerow(
                    [row_id, dataset, "edge", -1, -1, -1, -1, -1, source_id, target_id]
                )
                row_id += 1
                edge_total += 1
    return {"rows": row_id, "nodes": node_total, "edges": edge_total}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--det-threshold", type=float)
    parser.add_argument("--edge-threshold", type=float)
    parser.add_argument("--pool-kernel-um", type=float, default=None)
    parser.add_argument("--no-detection-tta", action="store_true")
    parser.add_argument(
        "--temporal-graph-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional T_graph=3 residual checkpoint. Also configurable via "
            f"{TEMPORAL_GRAPH_ENV} or config.json."
        ),
    )
    parser.add_argument(
        "--temporal-graph-attention-checkpoint",
        type=Path,
        default=None,
        help="Second bounded-Attention checkpoint for controlled link combinations.",
    )
    parser.add_argument(
        "--temporal-graph-fallback-checkpoint",
        type=Path,
        default=None,
        help="Optional T_graph=3 MLP checkpoint for the second T_graph=4 transition.",
    )
    parser.add_argument(
        "--temporal-graph-fallback-attention-checkpoint",
        type=Path,
        default=None,
        help=(
            "Optional T_graph=3 bounded-Attention checkpoint paired with the "
            "fallback MLP."
        ),
    )
    parser.add_argument(
        "--temporal-link-mode",
        choices=(
            "single",
            "mlp",
            "bounded_attention",
            "bounded_logit_5050",
            "agreement_gate",
        ),
        default="single",
    )
    parser.add_argument("--ensemble-logit-bound", type=float, default=0.15)
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--postprocess-profile",
        choices=("none", "public-applicable-v1"),
    )
    parser.add_argument("--minimum-component-nodes", type=int, default=6)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU accelerator is required for submission inference")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    (
        model,
        downsample,
        window_size,
        trained_pool_kernel,
        profile,
        bundle_config,
    ) = _load_model(args.bundle_dir, device)
    bundle_inference = bundle_config.get("inference", {})
    if profile is not None:
        if any(
            value is not None
            for value in (
                args.det_threshold,
                args.edge_threshold,
                args.pool_kernel_um,
                args.postprocess_profile,
            )
        ) or args.no_detection_tta:
            raise ValueError("corrected_v2 inference settings are immutable; edit the profile")
        if profile.postprocess_profile != "none":
            raise ValueError("corrected_v2 Kaggle inference requires postprocess_profile=none")
        det_threshold = profile.detection_threshold
        edge_threshold = profile.edge_threshold
        pool_kernel = profile.pool_kernel_um
        detection_tta = profile.detection_tta
        postprocess_profile = profile.postprocess_profile
    else:
        det_threshold = args.det_threshold
        if det_threshold is None:
            det_threshold = bundle_inference.get("det_threshold")
        edge_threshold = args.edge_threshold
        if edge_threshold is None:
            edge_threshold = bundle_inference.get("edge_threshold")
        if det_threshold is None or edge_threshold is None:
            raise ValueError("legacy inference requires configured detection and edge thresholds")
        pool_kernel = (
            args.pool_kernel_um
            if args.pool_kernel_um is not None
            else trained_pool_kernel
        )
        detection_tta = bool(bundle_inference.get("det_tta", True))
        if args.no_detection_tta:
            detection_tta = False
        postprocess_profile = args.postprocess_profile or str(
            bundle_inference.get("postprocess_profile", "none")
        )

    temporal_graph_checkpoint = _resolve_temporal_graph_checkpoint(
        args.temporal_graph_checkpoint,
        args.bundle_dir,
        bundle_config,
    )
    temporal_graph_head = None
    temporal_graph_fallback_head = None
    if args.temporal_link_mode == "single":
        if args.temporal_graph_attention_checkpoint is not None:
            raise ValueError(
                "attention checkpoint requires an explicit comparative temporal-link mode"
            )
        if temporal_graph_checkpoint is not None:
            if profile is not None:
                raise ValueError(
                    "temporal graph residual is not part of the finalized corrected_v2 profile"
                )
            if window_size != 2:
                raise ValueError(
                    "temporal graph inference keeps T_image=2; bundle window_size must be 2"
                )
            temporal_graph_head = _load_temporal_graph_head(
                temporal_graph_checkpoint,
                args.bundle_dir,
                device,
            )
            print(
                f"temporal graph residual enabled: {temporal_graph_checkpoint}",
                flush=True,
            )
    else:
        if temporal_graph_checkpoint is None:
            raise ValueError("comparative temporal-link modes require the MLP checkpoint")
        attention_checkpoint = args.temporal_graph_attention_checkpoint
        if attention_checkpoint is None:
            raise ValueError(
                "comparative temporal-link modes require the Attention checkpoint"
            )
        if profile is not None:
            raise ValueError(
                "temporal graph residual is not part of the finalized corrected_v2 profile"
            )
        if window_size != 2:
            raise ValueError(
                "temporal graph inference keeps T_image=2; bundle window_size must be 2"
            )
        mlp_head = _load_temporal_graph_head(
            temporal_graph_checkpoint,
            args.bundle_dir,
            device,
        )
        attention_head = _load_temporal_graph_head(
            attention_checkpoint,
            args.bundle_dir,
            device,
        )
        from temporal_graph import TemporalGraphLinkEnsemble

        temporal_graph_head = TemporalGraphLinkEnsemble(
            mlp_head,
            attention_head,
            mode=args.temporal_link_mode,
            logit_bound=args.ensemble_logit_bound,
        ).eval()
        print(
            "temporal link comparison enabled: "
            f"mode={args.temporal_link_mode}, mlp={temporal_graph_checkpoint}, "
            f"attention={attention_checkpoint}",
            flush=True,
        )
    fallback_checkpoint = args.temporal_graph_fallback_checkpoint
    fallback_attention_checkpoint = args.temporal_graph_fallback_attention_checkpoint
    if (fallback_checkpoint is None) != (fallback_attention_checkpoint is None):
        raise ValueError(
            "T_graph=4 fallback requires both the MLP and Attention checkpoints"
        )
    if fallback_checkpoint is not None:
        if temporal_graph_head is None:
            raise ValueError("T_graph=4 fallback requires a primary temporal checkpoint")
        fallback_mlp_head = _load_temporal_graph_head(
            fallback_checkpoint.expanduser(),
            args.bundle_dir,
            device,
        )
        fallback_attention_head = _load_temporal_graph_head(
            fallback_attention_checkpoint.expanduser(),
            args.bundle_dir,
            device,
        )
        from temporal_graph import TemporalGraphLinkEnsemble

        temporal_graph_fallback_head = TemporalGraphLinkEnsemble(
            fallback_mlp_head,
            fallback_attention_head,
            mode="bounded_logit_5050",
            logit_bound=args.ensemble_logit_bound,
        ).eval()
        _validate_temporal_graph_fallback_contract(
            temporal_graph_head,
            temporal_graph_fallback_head,
        )
        print(
            "T_graph=3 bounded-logit fallback enabled: "
            f"mlp={fallback_checkpoint}, attention={fallback_attention_checkpoint}",
            flush=True,
        )
    if args.minimum_component_nodes <= 0:
        raise ValueError("minimum_component_nodes must be positive")
    datasets = sorted(args.test_dir.glob("*.zarr"))
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
    if not datasets:
        raise FileNotFoundError(f"no test Zarr datasets in {args.test_dir}")

    started = time.monotonic()
    predictions = []
    for dataset in datasets:
        if profile is not None:
            from backbone_ab.inference import predict_dataset_corrected

            coords, edges = predict_dataset_corrected(
                model,
                dataset,
                device,
                profile,
                max_frames=args.max_frames,
            )
        else:
            coords, edges = predict_dataset(
                model,
                dataset,
                device,
                downsample,
                window_size,
                float(det_threshold),
                float(edge_threshold),
                pool_kernel,
                detection_tta,
                args.max_frames,
                temporal_graph_head,
                temporal_graph_fallback_head,
            )
        if len(coords) == 0:
            raise RuntimeError(
                f"{dataset.stem} produced zero detections at threshold {det_threshold}"
            )
        if postprocess_profile == "public-applicable-v1":
            metadata = _zarr_metadata(dataset)
            coords, edges, postprocess_stats = postprocess_prediction(
                coords,
                edges,
                tuple(float(value) for value in metadata["scale"]),
                minimum_component_nodes=args.minimum_component_nodes,
            )
            if len(coords) == 0 or len(edges) == 0:
                raise RuntimeError(
                    f"{dataset.stem} post-processing produced an empty graph"
                )
            print(
                f"{dataset.stem} postprocess: "
                + json.dumps(postprocess_stats, sort_keys=True),
                flush=True,
            )
        predictions.append((dataset.stem, coords, edges))
        print(f"{dataset.stem}: {len(coords)} nodes, {len(edges)} edges", flush=True)

    counts = write_submission(predictions, args.output)
    counts["datasets"] = len(datasets)
    counts["seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(counts, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
