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
import json
import sys
import time
from pathlib import Path

import blosc2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

POS_EMBED_DIM = 8
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
) -> tuple[UNetNodeTransformer, tuple[int, int, int], int, float]:
    sys.path.insert(0, str(bundle_dir))
    bundled_models = bundle_dir / "tracking_cellmot_models.zip"
    if bundled_models.exists():
        sys.path.insert(0, str(bundled_models))
    # Kaggle expands uploaded ZIP files into a same-named directory. Support
    # both the local ZIP bundle and its Dataset-mounted representation.
    expanded_models = bundle_dir / "tracking_cellmot_models"
    if expanded_models.is_dir():
        sys.path.insert(0, str(expanded_models))
    from tracking_cellmot.models import TemporalUNet3D

    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    unet = TemporalUNet3D(
        in_channels=1,
        out_channels=int(config["unet_out_channels"]),
        layers=[int(value) for value in config["unet_layers"]],
    )
    model = UNetNodeTransformer(
        unet=unet,
        unet_out_channels=int(config["unet_out_channels"]),
        pos_feat_dim=4 * POS_EMBED_DIM,
    )
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
) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
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
            logits = model.predict_edges(
                source_features,
                target_features,
                source_coords * downsample_tensor,
                target_coords * downsample_tensor,
                source_position,
                target_position,
                source_mask,
                target_mask,
            )[0]
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
) -> tuple[np.ndarray, list[tuple[int, int]], dict[str, int | float]]:
    """Apply the artifact-free topology repairs used by the public 0.926 harness.

    This intentionally excludes its second-seed ensemble, DeepCenter detector, synthetic
    gap nodes, and eight-view TTA. Those are additional models or inference changes rather
    than post-processing that can be applied to this checkpoint's frozen predictions.
    """
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
        if len(members) >= 6 or has_division:
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
    parser.add_argument("--det-threshold", type=float, default=0.99)
    parser.add_argument("--edge-threshold", type=float, default=0.5)
    parser.add_argument("--pool-kernel-um", type=float, default=None)
    parser.add_argument("--no-detection-tta", action="store_true")
    parser.add_argument("--max-datasets", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--postprocess-profile",
        choices=("none", "public-applicable-v1"),
        default="none",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("GPU accelerator is required for submission inference")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    model, downsample, window_size, trained_pool_kernel = _load_model(
        args.bundle_dir, device
    )
    pool_kernel = (
        args.pool_kernel_um
        if args.pool_kernel_um is not None
        else trained_pool_kernel
    )
    datasets = sorted(args.test_dir.glob("*.zarr"))
    if args.max_datasets is not None:
        datasets = datasets[: args.max_datasets]
    if not datasets:
        raise FileNotFoundError(f"no test Zarr datasets in {args.test_dir}")

    started = time.monotonic()
    predictions = []
    for dataset in datasets:
        coords, edges = predict_dataset(
            model,
            dataset,
            device,
            downsample,
            window_size,
            args.det_threshold,
            args.edge_threshold,
            pool_kernel,
            not args.no_detection_tta,
            args.max_frames,
        )
        if len(coords) == 0:
            raise RuntimeError(
                f"{dataset.stem} produced zero detections at threshold {args.det_threshold}"
            )
        if args.postprocess_profile == "public-applicable-v1":
            metadata = _zarr_metadata(dataset)
            coords, edges, postprocess_stats = postprocess_prediction(
                coords,
                edges,
                tuple(float(value) for value in metadata["scale"]),
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
