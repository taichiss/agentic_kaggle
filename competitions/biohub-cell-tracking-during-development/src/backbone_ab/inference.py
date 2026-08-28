"""Shared corrected-v2 inference primitives independent of organizer entrypoints."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .checkpointing import InferenceProfile
from .decoder import DecodeResult, GraphDecoder


@dataclass(frozen=True)
class PairPrediction:
    source_coords: Any
    target_coords: Any
    source_nodes: Any
    target_nodes: Any
    link_output: Any
    decoded: DecodeResult


def encoded_tensors(encoded: Any) -> tuple[Any, Any]:
    """Return stacked feature and detection tensors from either model API."""
    import torch

    if hasattr(encoded, "features") and hasattr(encoded, "detection_logits"):
        return encoded.features, encoded.detection_logits
    features, logits = encoded
    if isinstance(logits, (list, tuple)):
        logits = torch.stack(list(logits), dim=1)
    return features, logits


def with_detection_logits(encoded: Any, logits: Any) -> Any:
    """Replace logits without mutating a frozen EncodedWindow or legacy tuple."""
    if hasattr(encoded, "detection_logits"):
        return replace(encoded, detection_logits=logits)
    return encoded[0], [logits[:, index] for index in range(logits.shape[1])]


def encode_with_detection_tta(model: Any, images: Any, *, enabled: bool) -> Any:
    """Encode a window and apply the common X/Y four-view detection TTA."""
    encode = model.encode_window if hasattr(model, "encode_window") else model.encode
    encoded = encode(images)
    _, logits = encoded_tensors(encoded)
    if not enabled:
        return with_detection_logits(encoded, logits)
    logits_sum = logits.clone()
    for dims in ((-1,), (-2,), (-2, -1)):
        flipped = encode(images.flip(dims))
        _, flipped_logits = encoded_tensors(flipped)
        logits_sum += flipped_logits.flip(dims)
    return with_detection_logits(encoded, logits_sum / 4.0)


def pool_kernel_from_um(
    distance_um: float,
    voxel_size_um: Sequence[float],
) -> tuple[int, int, int]:
    kernel = []
    for spacing in voxel_size_um:
        size = max(1, round(distance_um / float(spacing)))
        kernel.append(size if size % 2 else size + 1)
    if len(kernel) != 3:
        raise ValueError("voxel_size_um must have three entries")
    return tuple(kernel)  # type: ignore[return-value]


def detect_window_nodes(
    detection_logits: Any,
    *,
    threshold: float,
    pool_kernel_um: float,
    voxel_size_um: Sequence[float],
    max_detections_per_frame: int,
) -> list[Any]:
    """Detect local maxima for a batch-one stacked detection-logit window."""
    import torch
    import torch.nn.functional as functional

    if detection_logits.ndim != 6 or detection_logits.shape[0] != 1:
        raise ValueError("detection_logits must have shape (1,T,1,Z,Y,X)")
    if max_detections_per_frame <= 0:
        raise ValueError("max_detections_per_frame must be positive")
    kernel = pool_kernel_from_um(pool_kernel_um, voxel_size_um)
    padding = tuple(value // 2 for value in kernel)
    coords = []
    for frame_logits in detection_logits[0]:
        batched = frame_logits.unsqueeze(0)
        pooled = functional.max_pool3d(batched, kernel, stride=1, padding=padding)
        peaks = (batched == pooled) & (batched.sigmoid() > threshold)
        indices = torch.nonzero(peaks[0, 0], as_tuple=False)
        if indices.shape[0] > max_detections_per_frame:
            scores = frame_logits[0, indices[:, 0], indices[:, 1], indices[:, 2]]
            keep = scores.topk(max_detections_per_frame, sorted=True).indices
            indices = indices[keep]
        coords.append(indices.float())
    return coords


def pad_window_nodes(coords_by_frame: Sequence[Any]) -> tuple[Any, Any]:
    """Pad one window's variable node sets to the model's (B,T,M,3) contract."""
    import torch

    if not coords_by_frame:
        raise ValueError("coords_by_frame must not be empty")
    maximum = max(1, max(int(coords.shape[0]) for coords in coords_by_frame))
    reference = coords_by_frame[0]
    coords = torch.zeros(
        1,
        len(coords_by_frame),
        maximum,
        3,
        dtype=reference.dtype,
        device=reference.device,
    )
    masks = torch.zeros(
        1,
        len(coords_by_frame),
        maximum,
        dtype=torch.bool,
        device=reference.device,
    )
    for frame, frame_coords in enumerate(coords_by_frame):
        count = int(frame_coords.shape[0])
        coords[0, frame, :count] = frame_coords
        masks[0, frame, :count] = True
    return coords, masks


def predict_corrected_pair(
    model: Any,
    encoded: Any,
    coords_by_frame: Sequence[Any],
    *,
    pair_index: int,
    image_shape: Sequence[int],
    voxel_size_um: Sequence[float],
    frame_indices: Sequence[int],
    delta_t: float,
    decoder: GraphDecoder,
    edge_threshold: float,
    source_offset: int = 0,
    target_offset: int = 0,
) -> PairPrediction:
    """Build corrected node features, link one pair, and decode it once."""
    coords, masks = pad_window_nodes(coords_by_frame)
    node_batches = model.build_nodes(
        encoded,
        coords,
        masks,
        tuple(int(value) for value in image_shape),
        tuple(float(value) for value in voxel_size_um),
        frame_indices=tuple(int(value) for value in frame_indices),
        delta_t=float(delta_t),
    )
    source_nodes = node_batches[pair_index]
    target_nodes = node_batches[pair_index + 1]
    link_output = model.link_pair(source_nodes, target_nodes)
    source_count = int(source_nodes.valid_mask[0].sum().item())
    target_count = int(target_nodes.valid_mask[0].sum().item())
    decoded = decoder.decode(
        link_output.edge_logits[0, :source_count, :target_count],
        edge_threshold=edge_threshold,
        null_parent_logits=link_output.null_parent_logits[0, :target_count],
        division_logits=link_output.division_logits[0, :source_count],
        source_coords_um=source_nodes.physical_coords_um[0, :source_count],
        target_coords_um=target_nodes.physical_coords_um[0, :target_count],
        source_offset=source_offset,
        target_offset=target_offset,
    )
    return PairPrediction(
        source_coords=coords_by_frame[pair_index],
        target_coords=coords_by_frame[pair_index + 1],
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        link_output=link_output,
        decoded=decoded,
    )


def _zarr_metadata(path: Path) -> dict[str, Any]:
    root = json.loads((path / "zarr.json").read_text(encoding="utf-8"))
    array = json.loads((path / "0" / "zarr.json").read_text(encoding="utf-8"))
    attributes = root["attributes"]
    transform = attributes["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]
    quantiles = attributes["image_statistics"]["quantiles"]
    return {
        "shape": tuple(array["shape"]),
        "dtype": np.dtype(array["data_type"]),
        "scale": tuple(float(value) for value in transform["scale"][1:]),
        "q_low": float(quantiles["0.001"]),
        "q_high": float(quantiles["0.999"]),
    }


def _load_zarr_frame(
    dataset: Path,
    frame: int,
    spatial_shape: tuple[int, int, int],
    dtype: np.dtype,
    downsample: tuple[int, int, int],
) -> Any:
    import blosc2
    import torch

    chunk = dataset / "0" / "c" / str(frame) / "0" / "0" / "0"
    raw = blosc2.decompress(chunk.read_bytes())
    image = np.frombuffer(raw, dtype=dtype).reshape(spatial_shape)
    z_step, y_step, x_step = downsample
    values = image[::z_step, ::y_step, ::x_step].astype(np.float32, copy=True)
    return torch.from_numpy(values)


def _window_starts(frames: int, window_size: int) -> list[int]:
    stride = max(window_size - 1, 1)
    starts = list(range(0, frames - window_size + 1, stride))
    if not starts or starts[-1] + window_size < frames:
        last = max(frames - window_size, 0)
        if not starts or starts[-1] != last:
            starts.append(last)
    return starts


def rescale_output_coordinates(
    coords: np.ndarray,
    downsample: Sequence[int],
) -> np.ndarray:
    """Map grid coordinates back to image voxels using nearest-integer rounding."""
    result = coords.astype(np.float64, copy=True)
    result[:, 1:] *= np.asarray(downsample, dtype=np.float64)
    return np.rint(result).astype(np.int64)


def predict_dataset_corrected(
    model: Any,
    dataset: Path,
    device: Any,
    profile: InferenceProfile,
    *,
    max_frames: int | None = None,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """Run the single corrected-v2 video path used locally and on Kaggle."""
    import torch

    if profile.model_api != "corrected_v2":
        raise ValueError("predict_dataset_corrected requires a corrected_v2 profile")
    metadata = _zarr_metadata(dataset)
    shape = metadata["shape"]
    frames = int(shape[0]) if max_frames is None else min(int(shape[0]), max_frames)
    if frames < profile.window_size:
        raise ValueError(f"need at least {profile.window_size} frames, got {frames}")
    spatial_shape = tuple(int(value) for value in shape[1:])
    target_shape = tuple(
        (size + stride - 1) // stride
        for size, stride in zip(spatial_shape, profile.downsample, strict=True)
    )
    voxel_size = tuple(
        scale * stride
        for scale, stride in zip(metadata["scale"], profile.downsample, strict=True)
    )
    q_low = float(metadata["q_low"])
    q_high = float(metadata["q_high"])
    decoder = GraphDecoder(profile.decoder, edge_activation=profile.edge_activation)

    starts = _window_starts(frames, profile.window_size)
    registered_coords: dict[int, np.ndarray] = {}
    offsets: dict[int, tuple[int, int]] = {}
    coord_lists: list[np.ndarray] = []
    seen_pairs: set[tuple[int, int]] = set()
    node_count = 0
    edges: list[tuple[int, int, float, float]] = []

    for window_number, start in enumerate(starts):
        frame_indices = list(range(start, start + profile.window_size))
        images = torch.stack(
            [
                _load_zarr_frame(
                    dataset,
                    frame,
                    spatial_shape,
                    metadata["dtype"],
                    profile.downsample,
                )
                for frame in frame_indices
            ]
        )
        images = ((images - q_low) / (q_high - q_low + 1e-6)).clamp(0.0)
        images = images.unsqueeze(0).to(device)
        with torch.inference_mode():
            encoded = encode_with_detection_tta(
                model, images, enabled=profile.detection_tta
            )
            _, detection_logits = encoded_tensors(encoded)
            detected = detect_window_nodes(
                detection_logits,
                threshold=profile.detection_threshold,
                pool_kernel_um=profile.pool_kernel_um,
                voxel_size_um=voxel_size,
                max_detections_per_frame=profile.max_detections_per_frame,
            )

            coords_by_frame = []
            for frame, current_coords in zip(frame_indices, detected, strict=True):
                if frame not in registered_coords:
                    spatial = current_coords.detach().cpu().numpy().astype(np.float32)
                    registered_coords[frame] = spatial
                    offsets[frame] = (node_count, node_count + len(spatial))
                    node_count += len(spatial)
                    time_column = np.full((len(spatial), 1), frame, dtype=np.float32)
                    coord_lists.append(np.concatenate([time_column, spatial], axis=1))
                coords_by_frame.append(
                    torch.from_numpy(registered_coords[frame]).to(device=device)
                )

            padded_coords, padded_masks = pad_window_nodes(coords_by_frame)
            node_batches = model.build_nodes(
                encoded,
                padded_coords,
                padded_masks,
                (profile.window_size, *target_shape),
                voxel_size,
                frame_indices=tuple(frame_indices),
                delta_t=1.0,
            )
            for pair_index in range(profile.window_size - 1):
                source_frame, target_frame = frame_indices[pair_index : pair_index + 2]
                pair_key = (source_frame, target_frame)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                source_start, source_end = offsets[source_frame]
                target_start, target_end = offsets[target_frame]
                source_count = source_end - source_start
                target_count = target_end - target_start
                if source_count == 0 or target_count == 0:
                    continue
                source_nodes = node_batches[pair_index]
                target_nodes = node_batches[pair_index + 1]
                link_output = model.link_pair(source_nodes, target_nodes)
                decoded = decoder.decode(
                    link_output.edge_logits[0, :source_count, :target_count],
                    edge_threshold=profile.edge_threshold,
                    null_parent_logits=link_output.null_parent_logits[0, :target_count],
                    division_logits=link_output.division_logits[0, :source_count],
                    source_coords_um=source_nodes.physical_coords_um[0, :source_count],
                    target_coords_um=target_nodes.physical_coords_um[0, :target_count],
                    source_offset=source_start,
                    target_offset=target_start,
                )
                edges.extend(
                    (
                        edge.source,
                        edge.target,
                        edge.probability,
                        edge.distance_um,
                    )
                    for edge in decoded.edges
                )

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
    return rescale_output_coordinates(coords, profile.downsample), edges
