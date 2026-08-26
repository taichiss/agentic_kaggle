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
) -> tuple[np.ndarray, list[tuple[int, int]]]:
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
    edges: list[tuple[int, int]] = []

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
                    (source_start + source_index, target_start + target_index)
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


def write_submission(
    predictions: list[tuple[str, np.ndarray, list[tuple[int, int]]]],
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
            for source_id, target_id in edges:
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
        predictions.append((dataset.stem, coords, edges))
        print(f"{dataset.stem}: {len(coords)} nodes, {len(edges)} edges", flush=True)

    counts = write_submission(predictions, args.output)
    counts["datasets"] = len(datasets)
    counts["seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(counts, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
