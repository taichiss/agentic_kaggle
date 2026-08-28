#!/usr/bin/env python
"""Evaluate one backbone A/B checkpoint on the fixed organizer-metric screen."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
import tomllib
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = COMPETITION_ROOT / "src"
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0005b-competition-screen.toml"
sys.path.insert(0, str(SOURCE_ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as file:
        result = tomllib.load(file)
    if result.get("schema_version") != 1:
        raise ValueError(f"unsupported schema_version in {path}")
    return result


def _validate_subset_contract(
    config: dict[str, Any],
) -> tuple[Path | None, str | None]:
    """Validate split identity and enforce calibration/report inference roles."""
    data_cfg = config["data"]
    manifest_name = data_cfg.get("dataset_subset_manifest")
    subset_name = data_cfg.get("dataset_subset")
    if not manifest_name and not subset_name:
        return None, None
    if not manifest_name or subset_name not in {"calibration", "report"}:
        raise ValueError(
            "data.dataset_subset_manifest and calibration/report dataset_subset "
            "are required together"
        )
    expected_sha256 = data_cfg.get("dataset_subset_manifest_sha256")
    if not isinstance(expected_sha256, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_sha256
    ):
        raise ValueError("data.dataset_subset_manifest_sha256 must be lowercase SHA-256")
    manifest_path = COMPETITION_ROOT / manifest_name
    actual_sha256 = _sha256(manifest_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("dataset subset manifest SHA-256 mismatch")

    model_cfg = config.get("model", {})
    inference_cfg = config.get("inference", {})
    profile_configured = bool(model_cfg.get("inference_profile"))
    sweep_keys = {
        "detection_thresholds",
        "edge_thresholds",
        "null_parent_thresholds",
        "division_thresholds",
    }
    if subset_name == "calibration" and "inference_profile" in model_cfg:
        raise ValueError("calibration subset must sweep thresholds without an inference_profile")
    if subset_name == "report":
        if not profile_configured:
            raise ValueError("report subset requires a fixed model.inference_profile")
        configured_sweeps = sorted(sweep_keys & inference_cfg.keys())
        if configured_sweeps:
            raise ValueError(
                "report subset forbids threshold sweeps: "
                f"{configured_sweeps}"
            )
    return manifest_path, actual_sha256


def _filter_dataset_subset(
    config: dict[str, Any],
    specs: list[Any],
    video_data: list[Any],
) -> tuple[list[Any], list[Any]]:
    data_cfg = config["data"]
    manifest_path, _ = _validate_subset_contract(config)
    manifest_name = data_cfg.get("dataset_subset_manifest")
    subset_name = data_cfg.get("dataset_subset")
    if not manifest_name and not subset_name:
        return specs, video_data
    if not manifest_name or subset_name not in {"calibration", "report"}:
        raise ValueError(
            "data.dataset_subset_manifest and calibration/report dataset_subset "
            "are required together"
        )
    assert manifest_path is not None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(manifest[subset_name])
    filtered = [
        (spec, data)
        for spec, data in zip(specs, video_data, strict=True)
        if spec.dataset in selected
    ]
    if not filtered:
        raise RuntimeError(f"dataset subset {subset_name!r} produced an empty screen")
    return [item[0] for item in filtered], [item[1] for item in filtered]


def _build_graphs_for_thresholds(
    model,
    item: dict[str, Any],
    spec,
    inference_cfg: dict[str, Any],
    detection_threshold: float,
    decoder_thresholds: list[tuple[float, float, float]],
    device,
    encoded,
):
    import torch
    from predict_unet_transformer import build_graph
    from train_unet_transformer import detect_and_match

    gt_coords = item["coords"].unsqueeze(0).to(device)
    gt_masks = item["masks"].unsqueeze(0).to(device)
    target = item["targets"][0]
    image_shape = tuple(int(value) for value in item["image_shape"].tolist())
    voxel_size = tuple(float(value) for value in item["voxel_size"].tolist())
    downsample = item["downsample"].to(device)

    with torch.inference_mode():
        from backbone_ab.inference import encoded_tensors

        feature_maps, detection_logits_stacked = encoded_tensors(encoded)
        detection_logits = [
            detection_logits_stacked[:, index]
            for index in range(detection_logits_stacked.shape[1])
        ]

        corrected_v2 = inference_cfg.get("model_api", "legacy") == "corrected_v2"
        corrected_coords = None
        if corrected_v2:
            from backbone_ab.inference import detect_window_nodes

            corrected_coords = detect_window_nodes(
                detection_logits_stacked,
                threshold=detection_threshold,
                pool_kernel_um=float(inference_cfg["pool_kernel_um"]),
                voxel_size_um=voxel_size,
                max_detections_per_frame=int(
                    inference_cfg["max_detections_per_frame"]
                ),
            )
        else:
            raw_threshold = math.log(
                detection_threshold / (1.0 - detection_threshold)
            )
        detected = []
        for frame_index in range(2):
            if corrected_coords is not None:
                det_coords = corrected_coords[frame_index].unsqueeze(0)
                det_mask = torch.ones(
                    1,
                    det_coords.shape[1],
                    dtype=torch.bool,
                    device=device,
                )
                det_pos = None
            else:
                det_coords, det_pos, det_mask, _ = detect_and_match(
                    detection_logits[frame_index],
                    gt_coords[:, frame_index],
                    gt_masks[:, frame_index],
                    image_shape,
                    det_threshold=raw_threshold,
                    pool_kernel_um=float(inference_cfg["pool_kernel_um"]),
                    voxel_size=voxel_size,
                    frame_index=frame_index,
                    window_size=2,
                )
            features = None
            if not corrected_v2:
                features = model._index_features(
                    feature_maps[:, frame_index], det_coords, det_mask
                )
            detected.append((det_coords, det_pos, det_mask, features))

        if corrected_v2:
            from backbone_ab.inference import pad_window_nodes

            coords, masks = pad_window_nodes([frame[0][0, frame[2][0]] for frame in detected])
            node_batches = model.build_nodes(
                encoded,
                coords,
                masks,
                image_shape,
                voxel_size,
                frame_indices=(spec.t_start, spec.t_start + 1),
                delta_t=1.0,
            )
            link_output = model.link_pair(node_batches[0], node_batches[1])
        elif hasattr(model, "predict_edges_contextual"):
            voxel_size_tensor = torch.tensor(voxel_size, device=device)
            edge_logits, _ = model.predict_edges_contextual(
                detected[0][3],
                detected[1][3],
                detection_logits[0],
                detection_logits[1],
                detected[0][0],
                detected[1][0],
                detected[0][0] * voxel_size_tensor,
                detected[1][0] * voxel_size_tensor,
                detected[0][1],
                detected[1][1],
                detected[0][2],
                detected[1][2],
            )
            edge_logits = edge_logits[0]
        else:
            edge_logits = model.predict_edges(
                detected[0][3],
                detected[1][3],
                detected[0][0] * downsample,
                detected[1][0] * downsample,
                detected[0][1],
                detected[1][1],
                detected[0][2],
                detected[1][2],
            )[0]

    counts = [int(frame[2][0].sum().item()) for frame in detected]
    pred_coord_parts = []
    for frame_index, count in enumerate(counts):
        spatial = (detected[frame_index][0][0, :count] * downsample).cpu().numpy()
        times = np.full((count, 1), spec.t_start + frame_index, dtype=np.float32)
        pred_coord_parts.append(np.concatenate([times, spatial], axis=1))
    pred_coords = np.concatenate(pred_coord_parts)

    from backbone_ab.checkpointing import DecoderProfile
    from backbone_ab.decoder import GraphDecoder

    probability_decoder = GraphDecoder(
        DecoderProfile(
            max_parents_per_node=int(inference_cfg.get("max_parents_per_node", 1)),
            max_children_per_node=int(inference_cfg.get("max_children_per_node", 2)),
            null_parent_threshold=decoder_thresholds[0][1],
            division_threshold=decoder_thresholds[0][2],
        ),
        edge_activation=str(inference_cfg.get("edge_activation", "softmax")),
    )
    if corrected_v2:
        cached = probability_decoder.decode(
            link_output.edge_logits[0, : counts[0], : counts[1]],
            edge_threshold=decoder_thresholds[0][0],
            null_parent_logits=link_output.null_parent_logits[0, : counts[1]],
            division_logits=link_output.division_logits[0, : counts[0]],
        )
    else:
        cached = probability_decoder.decode(
            edge_logits[: counts[0], : counts[1]],
            edge_threshold=decoder_thresholds[0][0],
        )
    pred_graphs = {}
    for edge_threshold, null_threshold, division_threshold in decoder_thresholds:
        decoder = GraphDecoder(
            DecoderProfile(
                max_parents_per_node=int(inference_cfg.get("max_parents_per_node", 1)),
                max_children_per_node=int(inference_cfg.get("max_children_per_node", 2)),
                null_parent_threshold=null_threshold,
                division_threshold=division_threshold,
            ),
            edge_activation=str(inference_cfg.get("edge_activation", "softmax")),
        )
        decoded = decoder.decode_probabilities(
            cached.edge_probabilities,
            cached.null_parent_probabilities,
            cached.division_probabilities,
            edge_threshold=edge_threshold,
        )
        edges = [
            (edge.source, counts[0] + edge.target, edge.probability, edge.distance_um)
            for edge in decoded.edges
        ]
        key = (edge_threshold, null_threshold, division_threshold)
        pred_graphs[key] = build_graph(pred_coords, edges)

    gt_counts = [int(item["masks"][index].sum().item()) for index in range(2)]
    gt_coord_parts = []
    for frame_index, count in enumerate(gt_counts):
        spatial = (item["coords"][frame_index, :count] * item["downsample"]).numpy()
        times = np.full((count, 1), spec.t_start + frame_index, dtype=np.float32)
        gt_coord_parts.append(np.concatenate([times, spatial], axis=1))
    gt_coords_all = np.concatenate(gt_coord_parts)
    gt_edges = [
        (int(source), gt_counts[0] + int(target_index), 1.0, 0.0)
        for source, target_index in torch.nonzero(
            target[: gt_counts[0], : gt_counts[1]] > 0, as_tuple=False
        ).tolist()
    ]
    return pred_graphs, build_graph(gt_coords_all, gt_edges)


def run(config_path: Path) -> dict[str, Any]:
    logging.disable(logging.WARNING)
    warnings.filterwarnings("ignore", message="Predicted graph has no edges or no nodes.*")
    config_path = config_path.resolve()
    config = _load_toml(config_path)
    _, subset_manifest_sha256 = _validate_subset_contract(config)
    artifact_dir = COMPETITION_ROOT / config["output"]["artifact_dir"]
    artifact_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = artifact_dir / "screen_config.toml"
    if config_snapshot.exists() and config_snapshot.read_bytes() != config_path.read_bytes():
        raise FileExistsError(f"refusing to replace screen config snapshot: {config_snapshot}")
    shutil.copyfile(config_path, config_snapshot)
    source_cfg = config["source"]
    organizer = COMPETITION_ROOT / source_cfg["organizer_repository_path"]
    revision = subprocess.check_output(
        ["git", "-C", str(organizer), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != source_cfg["organizer_revision"]:
        raise ValueError(f"organizer revision mismatch: {revision}")
    sys.path[:0] = [str(organizer / "src"), str(organizer / "scripts"), str(Path(__file__).parent)]

    import torch
    import train_unet_transformer as host_training
    from backbone_ab.backbones import build_joint_model
    from backbone_ab.checkpointing import (
        InferenceProfile,
        load_checkpoint,
        load_inference_profile,
        sha256_file,
        write_inference_profile,
    )
    from backbone_ab.inference import encode_with_detection_tta
    from checkpoint_selection import annotate_checkpoint_selection, select_best_checkpoint
    from evaluate_host_checkpoints import _prepare_transition_screen
    from torch.utils.data import DataLoader
    from tracking_cellmot.metrics import evaluate, node_recall, per_sample_metrics, summarise
    from train_unet_transformer import FrameWindowDataset

    device = torch.device(config["inference"].get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA evaluation requested but unavailable")
    torch.set_num_threads(int(config["inference"].get("torch_threads", 1)))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    experiment_config_path = COMPETITION_ROOT / config["model"]["experiment_config"]
    experiment_config = _load_toml(experiment_config_path)
    checkpoint_path = COMPETITION_ROOT / config["model"]["checkpoint"]
    model = build_joint_model(experiment_config["backbone"], host_training).to(device)
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint.state_dict)
    model.eval()
    fixed_profile = None
    if config["model"].get("inference_profile"):
        fixed_profile = load_inference_profile(
            COMPETITION_ROOT / config["model"]["inference_profile"]
        )
        if fixed_profile.checkpoint_sha256 != checkpoint.sha256:
            raise ValueError("fixed inference profile checkpoint hash does not match")
        if fixed_profile.experiment_config_sha256 != sha256_file(experiment_config_path):
            raise ValueError("fixed inference profile experiment config hash does not match")

    specs, video_data = _prepare_transition_screen(config)
    specs, video_data = _filter_dataset_subset(config, specs, video_data)
    loader = DataLoader(
        FrameWindowDataset(video_data, augmentations=None),
        batch_size=None,
        shuffle=False,
        num_workers=0,
    )
    detection_thresholds = (
        [fixed_profile.detection_threshold]
        if fixed_profile is not None
        else [float(v) for v in config["inference"]["detection_thresholds"]]
    )
    edge_thresholds = (
        [fixed_profile.edge_threshold]
        if fixed_profile is not None
        else [float(v) for v in config["inference"]["edge_thresholds"]]
    )
    experiment_inference = experiment_config["inference"]
    model_api = (
        fixed_profile.model_api
        if fixed_profile is not None
        else str(
            config["inference"].get(
                "model_api", experiment_inference.get("model_api", "legacy")
            )
        )
    )
    configured_decoder = {
        **experiment_inference.get("decoder", {}),
        **config["inference"].get("decoder", {}),
    }
    if fixed_profile is not None:
        configured_decoder = fixed_profile.decoder.__dict__
    null_thresholds = (
        [fixed_profile.decoder.null_parent_threshold]
        if fixed_profile is not None
        else [
            float(value)
            for value in config["inference"].get(
                "null_parent_thresholds",
                [configured_decoder.get("null_parent_threshold", 1.0)],
            )
        ]
    )
    division_thresholds = (
        [fixed_profile.decoder.division_threshold]
        if fixed_profile is not None
        else [
            float(value)
            for value in config["inference"].get(
                "division_thresholds",
                [configured_decoder.get("division_threshold", 0.0)],
            )
        ]
    )
    decoder_thresholds = list(
        itertools.product(edge_thresholds, null_thresholds, division_thresholds)
    )
    inference_cfg = {
        **config["inference"],
        "model_api": model_api,
        "edge_activation": (
            fixed_profile.edge_activation
            if fixed_profile is not None
            else experiment_inference.get("edge_activation", "softmax")
        ),
        "pool_kernel_um": (
            fixed_profile.pool_kernel_um
            if fixed_profile is not None
            else config["inference"]["pool_kernel_um"]
        ),
        "max_parents_per_node": configured_decoder.get(
            "max_parents_per_node",
            config["inference"].get("max_parents_per_node", 1),
        ),
        "max_children_per_node": configured_decoder.get(
            "max_children_per_node",
            config["inference"].get("max_children_per_node", 2),
        ),
        "max_detections_per_frame": (
            fixed_profile.max_detections_per_frame
            if fixed_profile is not None
            else experiment_inference.get("max_detections_per_frame", 512)
        ),
    }
    rows_by_threshold: dict[
        tuple[float, float, float, float], list[dict[str, Any]]
    ] = defaultdict(list)
    started = time.monotonic()
    for index, (item, spec) in enumerate(zip(loader, specs, strict=True), start=1):
        images = item["imgs"].unsqueeze(0).to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            encoded = encode_with_detection_tta(
                model,
                images,
                enabled=(
                    fixed_profile.detection_tta
                    if fixed_profile is not None
                    else bool(config["inference"].get("detection_tta", False))
                ),
            )
        for detection_threshold in detection_thresholds:
            pred_graphs, gt_graph = _build_graphs_for_thresholds(
                model,
                item,
                spec,
                inference_cfg,
                detection_threshold,
                decoder_thresholds,
                device,
                encoded,
            )
            for decoder_key, pred_graph in pred_graphs.items():
                edge_threshold, null_threshold, division_threshold = decoder_key
                scale = tuple(
                    float(value)
                    for value in (item["voxel_size"] / item["downsample"]).tolist()
                )
                metric_result = evaluate(
                    pred_graph,
                    gt_graph,
                    scale=scale,
                    max_distance=float(config["evaluation"]["max_match_distance_um"]),
                )
                recall = (
                    node_recall(pred_graph, gt_graph)
                    if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0
                    else 0.0
                )
                estimated = spec.estimated_total_nodes * 2.0 / spec.total_frames
                row = per_sample_metrics(metric_result, estimated, recall)
                row.update(
                    {
                        "dataset": spec.dataset,
                        "t_start": spec.t_start,
                        "detection_threshold": detection_threshold,
                        "edge_threshold": edge_threshold,
                        "null_parent_threshold": null_threshold,
                        "division_threshold": division_threshold,
                    }
                )
                key = (
                    detection_threshold,
                    edge_threshold,
                    null_threshold,
                    division_threshold,
                )
                rows_by_threshold[key].append(row)
        if index % 10 == 0 or index == len(specs):
            print(f"{index}/{len(specs)} transitions", flush=True)

    records = []
    for threshold_key, rows in rows_by_threshold.items():
        detection_threshold, edge_threshold, null_threshold, division_threshold = threshold_key
        summary = summarise(rows)
        edge_tp = sum(int(row["edge_tp"]) for row in rows)
        edge_fp = sum(int(row["edge_fp"]) for row in rows)
        edge_fn = sum(int(row["edge_fn"]) for row in rows)
        records.append(
            {
                "detection_threshold": detection_threshold,
                "edge_threshold": edge_threshold,
                "null_parent_threshold": null_threshold,
                "division_threshold": division_threshold,
                "competition_score": summary["score"],
                "adjusted_edge_jaccard": summary["adj_edge_jaccard"],
                "edge_jaccard": summary["edge_jaccard"],
                "division_jaccard": summary["division_jaccard"],
                "node_recall": summary["node_recall"],
                "edge_precision": edge_tp / max(edge_tp + edge_fp, 1),
                "edge_recall": edge_tp / max(edge_tp + edge_fn, 1),
                "edge_tp": edge_tp,
                "edge_fp": edge_fp,
                "edge_fn": edge_fn,
                "division_tp": summary["division_tp"],
                "division_fp": summary["division_fp"],
                "division_fn": summary["division_fn"],
                "num_pred_nodes": sum(int(row["num_pred_nodes"]) for row in rows),
            }
        )
    records.sort(
        key=lambda row: (
            row["detection_threshold"],
            row["edge_threshold"],
            row["null_parent_threshold"],
            row["division_threshold"],
        )
    )
    ranking_records = [
        {
            **row,
            "completed_epoch": int(checkpoint.metadata.get("completed_epochs", 0)),
            "checkpoint_sha256": checkpoint.sha256,
        }
        for row in records
    ]
    annotated = annotate_checkpoint_selection(ranking_records)
    best_annotated = select_best_checkpoint(annotated)
    best = next(
        row
        for row in records
        if all(
            row[key] == best_annotated[key]
            for key in (
                "detection_threshold",
                "edge_threshold",
                "null_parent_threshold",
                "division_threshold",
            )
        )
    )
    reference_detection = config["inference"].get("reference_detection_threshold")
    reference_edge = config["inference"].get("reference_edge_threshold")
    if model_api == "legacy":
        reference_detection = 0.99 if reference_detection is None else reference_detection
        reference_edge = 0.5 if reference_edge is None else reference_edge
    fixed = next(
        (
            row
            for row in records
            if reference_detection is not None
            and reference_edge is not None
            and row["detection_threshold"] == float(reference_detection)
            and row["edge_threshold"] == float(reference_edge)
        ),
        None,
    )

    with (artifact_dir / "threshold_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    transition_rows = [row for rows in rows_by_threshold.values() for row in rows]
    with (artifact_dir / "per_transition_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(transition_rows[0]))
        writer.writeheader()
        writer.writerows(transition_rows)

    selected_config = json.loads(json.dumps(experiment_config))
    selected_config["inference"].update(
        {
            "model_api": model_api,
            "det_threshold": best["detection_threshold"],
            "edge_threshold": best["edge_threshold"],
            "det_tta": (
                fixed_profile.detection_tta
                if fixed_profile is not None
                else bool(config["inference"].get("detection_tta", False))
            ),
            "pool_kernel_um": float(inference_cfg["pool_kernel_um"]),
            "max_detections_per_frame": int(
                inference_cfg["max_detections_per_frame"]
            ),
        }
    )
    selected_config["inference"]["decoder"] = {
        "max_parents_per_node": int(inference_cfg["max_parents_per_node"]),
        "max_children_per_node": int(inference_cfg["max_children_per_node"]),
        "null_parent_threshold": best["null_parent_threshold"],
        "division_threshold": best["division_threshold"],
    }
    selected_profile = InferenceProfile.from_experiment_config(
        selected_config,
        checkpoint_sha256=checkpoint.sha256,
        experiment_config_sha256=sha256_file(experiment_config_path),
    )
    write_inference_profile(artifact_dir / "inference_profile.json", selected_profile)
    result = {
        "experiment_id": config["experiment_id"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint.sha256,
        "checkpoint_format": checkpoint.source_format,
        "completed_epochs": int(checkpoint.metadata.get("completed_epochs", 0)),
        "inference_profile_sha256": selected_profile.sha256,
        "screen": {
            "transitions": len(specs),
            "validation_datasets": len({spec.dataset for spec in specs}),
            "gt_edges": sum(spec.gt_edges for spec in specs),
            "gt_divisions": sum(spec.gt_divisions for spec in specs),
            "max_match_distance_um": config["evaluation"]["max_match_distance_um"],
            "detection_tta": (
                fixed_profile.detection_tta
                if fixed_profile is not None
                else bool(config["inference"]["detection_tta"])
            ),
            "threshold_selection": "fixed_profile" if fixed_profile else "screen_sweep",
            "dataset_subset": config["data"].get("dataset_subset", "all"),
            "dataset_subset_manifest": config["data"].get(
                "dataset_subset_manifest"
            ),
            "dataset_subset_manifest_sha256": subset_manifest_sha256,
            "absolute_lb_comparable": False,
        },
        "host_fixed_threshold_result": fixed,
        "best_threshold_result": best,
        "records": records,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "config_path": str(config_path),
        "screen_config_sha256": _sha256(config_snapshot),
    }
    (artifact_dir / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"fixed": fixed, "best": best}, indent=2), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
