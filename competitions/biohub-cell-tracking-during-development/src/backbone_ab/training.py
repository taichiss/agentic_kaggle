"""Shared joint detector/linker training used by both backbone variants."""

from __future__ import annotations

import contextlib
import time

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .losses import (
    build_division_states,
    build_parent_supervision,
    candidate_parent_counts,
    mask_non_candidate_parents,
    parent_accuracy,
    parent_classification_loss,
    three_state_division_loss,
)
from .proposals import (
    ProposalWindow,
    apply_source_dropout,
    ground_truth_proposals,
    mix_detector_proposals,
    predicted_ratio_for_epoch,
)
from .sparse_heatmap import sparse_heatmap_loss


def _autocast_context(device: torch.device, mixed_precision: str | None):
    if device.type != "cuda" or mixed_precision in (None, "none"):
        return contextlib.nullcontext()
    if mixed_precision == "bfloat16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported mixed precision mode: {mixed_precision}")


def _train_epoch_legacy(
    model,
    loader,
    optimizer,
    device: torch.device,
    host_training_module,
    heatmap_config: dict,
    det_loss_weight: float,
    pool_kernel_um: float,
    mixed_precision: str | None,
    division_loss_weight: float = 0.0,
    division_negative_weight: float = 0.1,
) -> tuple[float, float, float]:
    """Train one epoch while keeping the organizer transformer/edge loss unchanged."""
    model.train()
    total_edge_loss = 0.0
    total_detection_loss = 0.0
    total_division_loss = 0.0
    samples = 0
    started = time.monotonic()

    for batch in tqdm(loader, desc="  batches", leave=False):
        images = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        position_features = batch["pos_feats"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        downsample_scale = batch["downsample"][0].to(device)
        voxel_size = batch["voxel_size"][0].to(device)
        batch_size, window_size = images.shape[:2]

        with _autocast_context(device, mixed_precision):
            unet_output, detection_logits = model.encode(images)
            detection_loss = sparse_heatmap_loss(
                detection_logits,
                images,
                coords,
                masks,
                heatmap_config,
            )

            frame_nodes = []
            for frame_index in range(window_size):
                sampled_features = model._index_features(
                    unet_output[:, frame_index],
                    coords[:, frame_index],
                    masks[:, frame_index],
                )
                frame_nodes.append(
                    (
                        coords[:, frame_index],
                        position_features[:, frame_index],
                        masks[:, frame_index],
                        sampled_features,
                    )
                )

            edge_losses = []
            division_losses = []
            for frame_index in range(window_size - 1):
                source = frame_nodes[frame_index]
                target = frame_nodes[frame_index + 1]
                if hasattr(model, "predict_edges_contextual"):
                    edge_logits, division_logits = model.predict_edges_contextual(
                        source[3],
                        target[3],
                        detection_logits[frame_index],
                        detection_logits[frame_index + 1],
                        source[0],
                        target[0],
                        source[0] * voxel_size,
                        target[0] * voxel_size,
                        source[1],
                        target[1],
                        source[2],
                        target[2],
                    )
                    division_target = (
                        targets[:, frame_index].sum(dim=2) > 1
                    ).to(division_logits.dtype)
                    division_weights = torch.full_like(
                        division_target,
                        float(division_negative_weight),
                    )
                    division_weights[division_target > 0] = 1.0
                    division_weights *= source[2]
                    division_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        division_logits,
                        division_target,
                        weight=division_weights,
                        reduction="sum",
                    ) / division_weights.sum().clamp_min(1.0)
                    division_losses.append(division_loss)
                else:
                    edge_logits = model.predict_edges(
                        source[3],
                        target[3],
                        source[0] * downsample_scale,
                        target[0] * downsample_scale,
                        source[1],
                        target[1],
                        source[2],
                        target[2],
                    )
                edge_losses.append(
                    host_training_module.compute_batch_loss(
                        edge_logits,
                        targets[:, frame_index],
                        source[2],
                        target[2],
                    )
                )
            edge_loss = torch.stack(edge_losses).mean()
            division_loss = (
                torch.stack(division_losses).mean()
                if division_losses
                else edge_loss.new_zeros(())
            )
            loss = (
                edge_loss
                + det_loss_weight * detection_loss
                + division_loss_weight * division_loss
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_edge_loss += float(edge_loss.detach()) * batch_size
        total_detection_loss += float(detection_loss.detach()) * batch_size
        total_division_loss += float(division_loss.detach()) * batch_size
        samples += batch_size

    print(f"  epoch training time: {time.monotonic() - started:.1f}s", flush=True)
    denominator = max(samples, 1)
    return (
        total_edge_loss / denominator,
        total_detection_loss / denominator,
        total_division_loss / denominator,
    )


@torch.no_grad()
def _evaluate_predicted_nodes_legacy(
    model,
    loader,
    device: torch.device,
    host_training_module,
    pool_kernel_um: float,
    detection_probability_threshold: float,
) -> tuple[float, float, float]:
    """Evaluate the real detect-match-link path with a bounded probability threshold."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pairs = 0
    matched_gt = 0
    total_gt = 0
    threshold_logit = torch.logit(torch.tensor(detection_probability_threshold)).item()
    # The organizer helper accepts raw logits but compares against its fixed value 0.3.
    helper_shift = threshold_logit - 0.3

    for batch in loader:
        images = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shape = tuple(batch["image_shape"][0].tolist())
        voxel_size = tuple(batch["voxel_size"][0].tolist())
        voxel_size_tensor = batch["voxel_size"][0].to(device)
        downsample_scale = batch["downsample"][0].to(device)
        batch_size, window_size = images.shape[:2]
        unet_output, detection_logits = model.encode(images)
        frame_detections = []
        for frame_index in range(window_size):
            detected_coords, detected_pos, detected_mask, matches = (
                host_training_module.detect_and_match(
                    detection_logits[frame_index] - helper_shift,
                    coords[:, frame_index],
                    masks[:, frame_index],
                    image_shape,
                    voxel_size=voxel_size,
                    pool_kernel_um=pool_kernel_um,
                    frame_index=frame_index,
                    window_size=window_size,
                )
            )
            sampled_features = model._index_features(
                unet_output[:, frame_index],
                detected_coords,
                detected_mask,
            )
            frame_detections.append(
                (
                    detected_coords,
                    detected_pos,
                    detected_mask,
                    matches,
                    sampled_features,
                )
            )
            for sample in range(batch_size):
                total_gt += int(masks[sample, frame_index].sum().item())
                matched_gt += int((matches[sample] >= 0).sum().item())

        for frame_index in range(window_size - 1):
            source = frame_detections[frame_index]
            target = frame_detections[frame_index + 1]
            pair_target = host_training_module.build_matched_edge_targets(
                source[3],
                target[3],
                targets[:, frame_index],
                source[0].shape[1],
                target[0].shape[1],
            )
            if hasattr(model, "predict_edges_contextual"):
                edge_logits, _ = model.predict_edges_contextual(
                    source[4],
                    target[4],
                    detection_logits[frame_index],
                    detection_logits[frame_index + 1],
                    source[0],
                    target[0],
                    source[0] * voxel_size_tensor,
                    target[0] * voxel_size_tensor,
                    source[1],
                    target[1],
                    source[2],
                    target[2],
                )
            else:
                edge_logits = model.predict_edges(
                    source[4],
                    target[4],
                    source[0] * downsample_scale,
                    target[0] * downsample_scale,
                    source[1],
                    target[1],
                    source[2],
                    target[2],
                )
            for sample in range(batch_size):
                source_count = int(source[2][sample].sum().item())
                target_count = int(target[2][sample].sum().item())
                pair_loss, pair_correct, pair_total = host_training_module._evaluate_pair(
                    edge_logits[sample, :source_count, :target_count],
                    pair_target[sample, :source_count, :target_count],
                )
                total_loss += pair_loss
                correct += pair_correct
                total += pair_total
                pairs += 1
    return (
        total_loss / max(pairs, 1),
        correct / max(total, 1),
        matched_gt / max(total_gt, 1),
    )


def _detection_logits_list(encoded) -> list[torch.Tensor]:
    """Normalise corrected encoded-window logits to the legacy loss input."""
    logits = encoded.detection_logits
    if isinstance(logits, torch.Tensor):
        if logits.ndim != 6:
            raise ValueError("detection_logits must have shape (B,T,1,Z,Y,X)")
        return [logits[:, frame] for frame in range(logits.shape[1])]
    if isinstance(logits, (list, tuple)) and all(
        isinstance(item, torch.Tensor) for item in logits
    ):
        return list(logits)
    raise TypeError("encoded.detection_logits must be a tensor or tensor sequence")


@torch.no_grad()
def _detect_and_match_window(
    detection_logits: list[torch.Tensor],
    gt_coords: torch.Tensor,
    gt_masks: torch.Tensor,
    image_shapes: torch.Tensor,
    voxel_sizes: torch.Tensor,
    host_training_module,
    *,
    probability_threshold: float,
    pool_kernel_um: float,
    max_match_distance_um: float,
    max_proposals_per_frame: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-K local maxima before bounded one-to-one physical GT matching."""
    del host_training_module  # Corrected-v2 must not call the unbounded pinned helper.
    threshold_logit = torch.logit(
        torch.tensor(probability_threshold, dtype=torch.float32)
    ).item()
    batch, window_size = gt_coords.shape[:2]
    rows: list[list[tuple[torch.Tensor, torch.Tensor]]] = [
        [] for _ in range(batch)
    ]
    maximum = 1
    for sample in range(batch):
        voxel_size = tuple(float(value) for value in voxel_sizes[sample].tolist())
        for frame in range(window_size):
            logits = detection_logits[frame][sample : sample + 1].detach()
            pool_kernel = tuple(
                max(1, value if value % 2 else value + 1)
                for value in (
                    max(1, round(pool_kernel_um / spacing))
                    for spacing in voxel_size
                )
            )
            padding = tuple(value // 2 for value in pool_kernel)
            pooled = F.max_pool3d(
                logits, pool_kernel, stride=1, padding=padding
            )
            peak_mask = (logits == pooled) & (logits > threshold_logit)
            coords = torch.nonzero(peak_mask[0, 0], as_tuple=False).float()
            if coords.shape[0]:
                integer = coords.long()
                scores = logits[
                    0, 0, integer[:, 0], integer[:, 1], integer[:, 2]
                ]
                if (
                    max_proposals_per_frame is not None
                    and coords.shape[0] > max_proposals_per_frame
                ):
                    selected = scores.topk(max_proposals_per_frame).indices
                    coords = coords[selected]

            count = coords.shape[0]
            match = torch.full(
                (count,), -1, dtype=torch.long, device=gt_coords.device
            )
            gt_count = int(gt_masks[sample, frame].sum().item())
            if count and gt_count:
                gt = gt_coords[sample, frame, :gt_count]
                spacing = gt.new_tensor(voxel_size)
                distances = torch.cdist(coords * spacing, gt * spacing)
                minimum, nearest = distances.min(dim=1)
                gt_taken = torch.zeros(
                    gt_count, dtype=torch.bool, device=gt_coords.device
                )
                for detection_index in minimum.argsort():
                    if minimum[detection_index] > max_match_distance_um:
                        break
                    gt_index = nearest[detection_index]
                    if not gt_taken[gt_index]:
                        match[detection_index] = gt_index
                        gt_taken[gt_index] = True
            rows[sample].append((coords, match))
            maximum = max(maximum, count)

    device = gt_coords.device
    coords = gt_coords.new_zeros(batch, window_size, maximum, 3)
    masks = torch.zeros(batch, window_size, maximum, dtype=torch.bool, device=device)
    matches = torch.full(
        (batch, window_size, maximum), -1, dtype=torch.long, device=device
    )
    for sample, sample_rows in enumerate(rows):
        for frame, (frame_coords, frame_matches) in enumerate(sample_rows):
            count = frame_coords.shape[0]
            coords[sample, frame, :count] = frame_coords
            masks[sample, frame, :count] = True
            matches[sample, frame, :count] = frame_matches
    return coords, masks, matches


def _corrected_proposals(
    encoded,
    coords: torch.Tensor,
    masks: torch.Tensor,
    gt_edges: torch.Tensor,
    image_shapes: torch.Tensor,
    voxel_sizes: torch.Tensor,
    host_training_module,
    *,
    strategy: str,
    curriculum_config: dict,
    epoch_index: int,
    pool_kernel_um: float,
    generator: torch.Generator | None,
) -> tuple[ProposalWindow, float]:
    ratio = predicted_ratio_for_epoch(strategy, curriculum_config, epoch_index)
    if ratio > 0:
        logits = _detection_logits_list(encoded)
        detected_coords, detected_masks, detected_matches = _detect_and_match_window(
            logits,
            coords,
            masks,
            image_shapes,
            voxel_sizes,
            host_training_module,
            probability_threshold=float(
                curriculum_config.get("detection_threshold", 0.5)
            ),
            pool_kernel_um=pool_kernel_um,
            max_match_distance_um=float(
                curriculum_config.get("max_match_distance_um", 5.0)
            ),
            max_proposals_per_frame=int(
                curriculum_config.get("max_proposals_per_frame", 96)
            ),
        )
    else:
        exact = ground_truth_proposals(coords, masks)
        detected_coords = exact.coords
        detected_masks = exact.masks
        detected_matches = exact.gt_matches
    exact_gt = ground_truth_proposals(coords, masks)
    mixed = mix_detector_proposals(
            exact_gt.coords,
            exact_gt.masks,
            detected_coords,
            detected_masks,
            detected_matches,
            predicted_ratio=ratio,
            spatial_shape=tuple(int(value) for value in encoded.features.shape[-3:]),
            jitter_std_voxels=float(
                curriculum_config.get("jitter_std_voxels", 0.0)
            ),
            duplicate_probability=float(
                curriculum_config.get("duplicate_probability", 0.0)
            ),
            generator=generator,
    )
    return (
        apply_source_dropout(
            mixed,
            gt_edges,
            probability=float(
                curriculum_config.get("source_dropout_probability", 0.0)
            ),
            generator=generator,
        ),
        ratio,
    )


def _train_epoch_corrected_v2(
    model,
    loader,
    optimizer,
    device: torch.device,
    host_training_module,
    heatmap_config: dict,
    det_loss_weight: float,
    pool_kernel_um: float,
    mixed_precision: str | None,
    division_loss_weight: float,
    division_negative_weight: float,
    *,
    node_proposal_strategy: str,
    proposal_curriculum: dict,
    epoch_index: int,
    proposal_seed: int,
) -> tuple[float, float, float]:
    """Train the corrected parent/null and three-state division contract."""
    model.train()
    total_edge_loss = 0.0
    total_detection_loss = 0.0
    total_division_loss = 0.0
    samples = 0
    started = time.monotonic()
    proposal_generator = torch.Generator().manual_seed(
        int(proposal_seed) + 1_000_003 * int(epoch_index)
    )

    for batch in tqdm(loader, desc="  batches", leave=False):
        images = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shapes = batch["image_shape"].to(device, non_blocking=True)
        voxel_sizes = batch["voxel_size"].to(device, non_blocking=True)
        batch_size, window_size = images.shape[:2]

        with _autocast_context(device, mixed_precision):
            encoded = model.encode_window(images)
            logits = _detection_logits_list(encoded)
            detection_loss = sparse_heatmap_loss(
                logits,
                images,
                coords,
                masks,
                heatmap_config,
            )
            proposals, _ = _corrected_proposals(
                encoded,
                coords,
                masks,
                targets,
                image_shapes,
                voxel_sizes,
                host_training_module,
                strategy=node_proposal_strategy,
                curriculum_config=proposal_curriculum,
                epoch_index=epoch_index,
                pool_kernel_um=pool_kernel_um,
                generator=proposal_generator,
            )
            frame_nodes = model.build_nodes(
                encoded,
                proposals.coords,
                proposals.masks,
                image_shapes,
                voxel_sizes,
                frame_indices=None,
                delta_t=1.0,
            )
            if len(frame_nodes) != window_size:
                raise ValueError("build_nodes must return one NodeBatch per frame")

            edge_losses = []
            division_losses = []
            for frame in range(window_size - 1):
                output = model.link_pair(frame_nodes[frame], frame_nodes[frame + 1])
                supervision = build_parent_supervision(
                    targets[:, frame],
                    proposals.gt_matches[:, frame],
                    proposals.gt_matches[:, frame + 1],
                    proposals.masks[:, frame],
                    proposals.masks[:, frame + 1],
                    proposals.reliable_null[:, frame + 1],
                )
                supervision = mask_non_candidate_parents(
                    supervision, getattr(output, "candidate_mask", None)
                )
                edge_losses.append(
                    parent_classification_loss(output.parent_logits, supervision)
                )
                division_states = build_division_states(
                    targets[:, frame],
                    proposals.gt_matches[:, frame],
                    proposals.masks[:, frame],
                )
                division_losses.append(
                    three_state_division_loss(
                        output.division_logits,
                        division_states,
                        weak_negative_weight=division_negative_weight,
                    )
                )
            edge_loss = torch.stack(edge_losses).mean()
            division_loss = torch.stack(division_losses).mean()
            loss = (
                edge_loss
                + det_loss_weight * detection_loss
                + division_loss_weight * division_loss
            )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_edge_loss += float(edge_loss.detach()) * batch_size
        total_detection_loss += float(detection_loss.detach()) * batch_size
        total_division_loss += float(division_loss.detach()) * batch_size
        samples += batch_size

    print(f"  epoch training time: {time.monotonic() - started:.1f}s", flush=True)
    denominator = max(samples, 1)
    return (
        total_edge_loss / denominator,
        total_detection_loss / denominator,
        total_division_loss / denominator,
    )


@torch.no_grad()
def _evaluate_predicted_nodes_corrected_v2(
    model,
    loader,
    device: torch.device,
    host_training_module,
    pool_kernel_um: float,
    detection_probability_threshold: float,
    max_proposals_per_frame: int,
    metrics_out: dict[str, float | None] | None = None,
) -> tuple[float, float, float]:
    """Evaluate corrected detect/build/link path with sparse-safe parent labels."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    pairs = 0
    matched_gt = 0
    total_gt = 0
    candidate_survived = 0
    candidate_total = 0
    candidate_mask_observed = False

    for batch in loader:
        images = batch["imgs"].to(device, dtype=torch.float32, non_blocking=True)
        coords = batch["coords"].to(device, non_blocking=True)
        masks = batch["masks"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        image_shapes = batch["image_shape"].to(device, non_blocking=True)
        voxel_sizes = batch["voxel_size"].to(device, non_blocking=True)
        window_size = images.shape[1]
        encoded = model.encode_window(images)
        detected_coords, detected_masks, detected_matches = _detect_and_match_window(
            _detection_logits_list(encoded),
            coords,
            masks,
            image_shapes,
            voxel_sizes,
            host_training_module,
            probability_threshold=detection_probability_threshold,
            pool_kernel_um=pool_kernel_um,
            max_match_distance_um=5.0,
            max_proposals_per_frame=max_proposals_per_frame,
        )
        proposals = ProposalWindow(
            coords=detected_coords,
            masks=detected_masks,
            gt_matches=detected_matches,
            reliable_null=torch.zeros_like(detected_masks),
        )
        frame_nodes = model.build_nodes(
            encoded,
            proposals.coords,
            proposals.masks,
            image_shapes,
            voxel_sizes,
            frame_indices=None,
            delta_t=1.0,
        )

        matched_gt += int((detected_matches >= 0).sum().item())
        total_gt += int(masks.sum().item())
        for frame in range(window_size - 1):
            output = model.link_pair(frame_nodes[frame], frame_nodes[frame + 1])
            supervision = build_parent_supervision(
                targets[:, frame],
                proposals.gt_matches[:, frame],
                proposals.gt_matches[:, frame + 1],
                proposals.masks[:, frame],
                proposals.masks[:, frame + 1],
                proposals.reliable_null[:, frame + 1],
            )
            candidate_mask = getattr(output, "candidate_mask", None)
            if candidate_mask is not None:
                candidate_mask_observed = True
                survived, eligible = candidate_parent_counts(
                    supervision, candidate_mask
                )
                candidate_survived += survived
                candidate_total += eligible
            loss = parent_classification_loss(output.parent_logits, supervision)
            pair_correct, pair_total = parent_accuracy(
                output.parent_logits, supervision
            )
            total_loss += float(loss)
            correct += pair_correct
            total += pair_total
            pairs += 1
    if metrics_out is not None:
        metrics_out["candidate_recall"] = (
            candidate_survived / candidate_total
            if candidate_mask_observed and candidate_total
            else None
        )
    return (
        total_loss / max(pairs, 1),
        correct / max(total, 1),
        matched_gt / max(total_gt, 1),
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device: torch.device,
    host_training_module,
    heatmap_config: dict,
    det_loss_weight: float,
    pool_kernel_um: float,
    mixed_precision: str | None,
    division_loss_weight: float = 0.0,
    division_negative_weight: float = 0.1,
    *,
    contract: str = "legacy",
    node_proposal_strategy: str = "ground_truth",
    proposal_curriculum: dict | None = None,
    epoch_index: int = 0,
    proposal_seed: int = 0,
) -> tuple[float, float, float]:
    """Dispatch one epoch without changing the legacy experiment contract."""
    if contract == "legacy":
        return _train_epoch_legacy(
            model,
            loader,
            optimizer,
            device,
            host_training_module,
            heatmap_config,
            det_loss_weight,
            pool_kernel_um,
            mixed_precision,
            division_loss_weight,
            division_negative_weight,
        )
    if contract == "corrected_v2":
        return _train_epoch_corrected_v2(
            model,
            loader,
            optimizer,
            device,
            host_training_module,
            heatmap_config,
            det_loss_weight,
            pool_kernel_um,
            mixed_precision,
            division_loss_weight,
            division_negative_weight,
            node_proposal_strategy=node_proposal_strategy,
            proposal_curriculum=proposal_curriculum or {},
            epoch_index=epoch_index,
            proposal_seed=proposal_seed,
        )
    raise ValueError(f"unsupported training contract: {contract}")


def evaluate_predicted_nodes(
    model,
    loader,
    device: torch.device,
    host_training_module,
    pool_kernel_um: float,
    detection_probability_threshold: float,
    *,
    contract: str = "legacy",
    max_proposals_per_frame: int = 96,
    metrics_out: dict[str, float | None] | None = None,
) -> tuple[float, float, float]:
    """Dispatch validation through the matching model contract."""
    if contract == "legacy":
        if metrics_out is not None:
            metrics_out["candidate_recall"] = None
        return _evaluate_predicted_nodes_legacy(
            model,
            loader,
            device,
            host_training_module,
            pool_kernel_um,
            detection_probability_threshold,
        )
    if contract == "corrected_v2":
        return _evaluate_predicted_nodes_corrected_v2(
            model,
            loader,
            device,
            host_training_module,
            pool_kernel_um,
            detection_probability_threshold,
            max_proposals_per_frame,
            metrics_out,
        )
    raise ValueError(f"unsupported training contract: {contract}")
