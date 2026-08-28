"""Sparse-positive Gaussian center targets shared by both A/B backbones."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def build_sparse_targets(
    images: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Gaussian targets and PU-style voxel weights for `(B,T,Z,Y,X)` images."""
    if images.ndim != 5:
        raise ValueError(f"expected images (B,T,Z,Y,X), got {tuple(images.shape)}")
    targets = torch.zeros_like(images, dtype=torch.float32)
    impulses = torch.zeros_like(images, dtype=torch.float32)
    sigma = float(config["sigma"])
    radius = max(1, int(torch.ceil(torch.tensor(3 * sigma)).item()))
    spatial = images.shape[2:]

    for batch in range(images.shape[0]):
        for time in range(images.shape[1]):
            count = int(masks[batch, time].sum().item())
            centers = coords[batch, time, :count].round().long()
            if count == 0:
                continue
            valid = torch.ones(count, dtype=torch.bool, device=images.device)
            for axis, size in enumerate(spatial):
                valid &= (centers[:, axis] >= 0) & (centers[:, axis] < size)
            centers = centers[valid]
            impulses[batch, time, centers[:, 0], centers[:, 1], centers[:, 2]] = 1.0

    axis = torch.arange(-radius, radius + 1, device=images.device, dtype=torch.float32)
    zz, yy, xx = torch.meshgrid(axis, axis, axis, indexing="ij")
    kernel = torch.exp(-(zz**2 + yy**2 + xx**2) / (2 * sigma**2))[None, None]
    flattened = impulses.reshape(-1, 1, *spatial)
    targets = F.conv3d(flattened, kernel, padding=radius).reshape_as(images).clamp_max_(1.0)

    flat_images = images.float().reshape(images.shape[0] * images.shape[1], -1)
    cutoffs = torch.quantile(
        flat_images,
        float(config["background_quantile"]),
        dim=1,
    ).reshape(images.shape[0], images.shape[1], 1, 1, 1)
    weights = torch.full_like(images, float(config["unknown_weight"]), dtype=torch.float32)
    weights[images < cutoffs] = float(config["background_weight"])

    weights[targets > float(config["positive_threshold"])] = float(
        config["positive_weight"]
    )
    return targets, weights


def sparse_heatmap_loss(
    detection_logits: list[torch.Tensor],
    images: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    """Weighted BCE over positives, confident dark background, and low-weight unknowns."""
    logits = torch.stack([item[:, 0] for item in detection_logits], dim=1)
    targets, weights = build_sparse_targets(images, coords, masks, config)
    voxel_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return (voxel_loss * weights).sum() / weights.sum().clamp_min(1e-6)
