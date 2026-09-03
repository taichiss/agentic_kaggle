"""Coordinate-derived node features shared by training and inference."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def _samplewise_spatial_shape(
    image_shape: Sequence[int] | torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    shape = torch.as_tensor(image_shape, device=device, dtype=torch.float32)
    if shape.ndim == 1:
        if shape.numel() not in {3, 4}:
            raise ValueError("image_shape must contain (Z,Y,X) or (T,Z,Y,X)")
        return shape[-3:].view(1, 1, 1, 3).expand(batch_size, -1, -1, -1)
    if shape.ndim == 2 and shape.shape[0] == batch_size and shape.shape[1] in {3, 4}:
        return shape[:, -3:].view(batch_size, 1, 1, 3)
    raise ValueError("image_shape must be shared or have one row per batch sample")


def spatial_sinusoidal_embedding(
    coords: torch.Tensor,
    image_shape: Sequence[int] | torch.Tensor,
    embedding_dim: int = 8,
) -> torch.Tensor:
    """Embed only ``(z,y,x)`` after all coordinate transforms are complete."""
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (B,T,N,3)")
    if embedding_dim <= 0 or embedding_dim % 2:
        raise ValueError("embedding_dim must be a positive even integer")
    shape = _samplewise_spatial_shape(
        image_shape,
        batch_size=coords.shape[0],
        device=coords.device,
    )
    normalised = coords.to(torch.float32) / shape.clamp_min(1.0)
    frequencies = (
        2.0
        ** torch.arange(
            embedding_dim // 2,
            device=coords.device,
            dtype=torch.float32,
        )
    ) * torch.pi
    parts = []
    for axis in range(3):
        angles = normalised[..., axis].unsqueeze(-1) * frequencies
        parts.extend((angles.sin(), angles.cos()))
    return torch.cat(parts, dim=-1)


def physical_coordinates_um(
    coords: torch.Tensor,
    voxel_size: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Convert grid coordinates to micrometers using sample-wise voxel sizes."""
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (B,T,N,3)")
    dtype = coords.dtype if coords.is_floating_point() else torch.float32
    coordinates = coords.to(dtype)
    spacing = torch.as_tensor(voxel_size, device=coords.device, dtype=dtype)
    if spacing.ndim == 1 and spacing.numel() == 3:
        spacing = spacing.view(1, 1, 1, 3)
    elif spacing.ndim == 2 and spacing.shape == (coords.shape[0], 3):
        spacing = spacing.view(coords.shape[0], 1, 1, 3)
    else:
        raise ValueError("voxel_size must be length 3 or have shape (B,3)")
    return coordinates * spacing


def temporal_node_features(
    *,
    batch_size: int,
    frames: int,
    nodes: int,
    device: torch.device,
    dtype: torch.dtype,
    frame_indices: Sequence[int] | torch.Tensor | None,
    delta_t: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return relative frame role and elapsed time, shaped ``(B,T,N,1)``."""
    if frame_indices is None:
        indices = torch.arange(frames, device=device, dtype=torch.float32)
        indices = indices.view(1, frames).expand(batch_size, -1)
    else:
        indices = torch.as_tensor(frame_indices, device=device, dtype=torch.float32)
        if indices.ndim == 1 and indices.numel() == frames:
            indices = indices.view(1, frames).expand(batch_size, -1)
        elif indices.shape != (batch_size, frames):
            raise ValueError("frame_indices must have shape (T,) or (B,T)")
    relative = indices - indices[:, :1]
    span = relative.abs().amax(dim=1, keepdim=True).clamp_min(1.0)
    role = relative / span

    elapsed_input = torch.as_tensor(delta_t, device=device, dtype=torch.float32)
    if elapsed_input.ndim == 0:
        elapsed = relative * elapsed_input
    elif elapsed_input.ndim == 1 and elapsed_input.numel() == frames - 1:
        steps = elapsed_input.view(1, -1).expand(batch_size, -1)
        elapsed = torch.cat([torch.zeros_like(steps[:, :1]), steps.cumsum(dim=1)], dim=1)
    elif elapsed_input.ndim == 1 and elapsed_input.numel() == frames:
        elapsed = elapsed_input.view(1, frames).expand(batch_size, -1)
        elapsed = elapsed - elapsed[:, :1]
    elif elapsed_input.shape == (batch_size, frames):
        elapsed = elapsed_input - elapsed_input[:, :1]
    else:
        raise ValueError("delta_t must be scalar, (T-1,), (T,), or (B,T)")

    role = role.to(dtype).view(batch_size, frames, 1, 1).expand(-1, -1, nodes, -1)
    elapsed = elapsed.to(dtype).view(batch_size, frames, 1, 1).expand(-1, -1, nodes, -1)
    return role, elapsed
