"""Typed tensor contracts for the corrected detector/linker model."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class EncodedWindow:
    """Dense outputs for a complete temporal image window."""

    features: torch.Tensor
    detection_logits: torch.Tensor

    def __post_init__(self) -> None:
        if self.features.ndim != 6:
            raise ValueError("features must have shape (B,T,C,Z,Y,X)")
        if self.detection_logits.ndim != 6:
            raise ValueError("detection_logits must have shape (B,T,1,Z,Y,X)")
        if self.features.shape[:2] != self.detection_logits.shape[:2]:
            raise ValueError("features and detection_logits must share batch/time axes")
        if self.detection_logits.shape[2] != 1:
            raise ValueError("detection_logits must have one channel")
        if self.features.shape[-3:] != self.detection_logits.shape[-3:]:
            raise ValueError("features and detection_logits must share spatial shape")


@dataclass(frozen=True)
class NodeBatch:
    """All node-level inputs for one frame, generated from final coordinates."""

    appearance: torch.Tensor
    grid_coords: torch.Tensor
    physical_coords_um: torch.Tensor
    spatial_position: torch.Tensor
    valid_mask: torch.Tensor
    detection_probability: torch.Tensor
    division_probability: torch.Tensor
    division_logits: torch.Tensor
    frame_role: torch.Tensor
    delta_t: torch.Tensor

    def __post_init__(self) -> None:
        if self.grid_coords.ndim != 3 or self.grid_coords.shape[-1] != 3:
            raise ValueError("grid_coords must have shape (B,N,3)")
        batch, nodes = self.grid_coords.shape[:2]
        if self.valid_mask.shape != (batch, nodes):
            raise ValueError("valid_mask must have shape (B,N)")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if self.physical_coords_um.shape != self.grid_coords.shape:
            raise ValueError("physical_coords_um must match grid_coords")
        for name in (
            "appearance",
            "spatial_position",
            "detection_probability",
            "division_probability",
            "division_logits",
            "frame_role",
            "delta_t",
        ):
            value = getattr(self, name)
            if value.shape[:2] != (batch, nodes):
                raise ValueError(f"{name} must share the (B,N) node axes")
        for name in (
            "detection_probability",
            "division_probability",
            "division_logits",
            "frame_role",
            "delta_t",
        ):
            if getattr(self, name).shape[-1:] != (1,):
                raise ValueError(f"{name} must have one trailing feature dimension")


@dataclass(frozen=True)
class LinkOutput:
    """Structured logits for one source-to-target transition."""

    edge_logits: torch.Tensor
    null_parent_logits: torch.Tensor
    division_logits: torch.Tensor
    candidate_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if self.edge_logits.ndim != 3:
            raise ValueError("edge_logits must have shape (B,N_source,N_target)")
        batch, sources, targets = self.edge_logits.shape
        if self.null_parent_logits.shape != (batch, targets):
            raise ValueError("null_parent_logits must have shape (B,N_target)")
        if self.division_logits.shape != (batch, sources):
            raise ValueError("division_logits must have shape (B,N_source)")
        if self.candidate_mask is not None:
            if self.candidate_mask.shape != self.edge_logits.shape:
                raise ValueError("candidate_mask must match edge_logits")
            if self.candidate_mask.dtype != torch.bool:
                raise TypeError("candidate_mask must be boolean")

    @property
    def parent_logits(self) -> torch.Tensor:
        """Return target-major parent choices with the null choice last."""
        return torch.cat(
            [self.edge_logits.transpose(1, 2), self.null_parent_logits.unsqueeze(-1)],
            dim=-1,
        )
