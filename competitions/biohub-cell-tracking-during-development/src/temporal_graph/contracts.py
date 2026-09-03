"""Typed contracts for frozen-host temporal graph residual models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import torch


def _validate_nodes(
    name: str,
    features: torch.Tensor,
    coords_um: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if features.ndim != 3:
        raise ValueError(f"{name}_features must have shape (B,N,C)")
    if coords_um.ndim != 3 or coords_um.shape[-1] != 3:
        raise ValueError(f"{name}_coords_um must have shape (B,N,3)")
    if coords_um.shape[:2] != features.shape[:2]:
        raise ValueError(f"{name} features and coordinates must share (B,N)")
    if mask.shape != features.shape[:2]:
        raise ValueError(f"{name}_mask must have shape (B,N)")
    if mask.dtype != torch.bool:
        raise TypeError(f"{name}_mask must be boolean")
    if not features.is_floating_point() or not coords_um.is_floating_point():
        raise TypeError(f"{name} features and coordinates must be floating point")
    if features.device != coords_um.device or features.device != mask.device:
        raise ValueError(f"{name} tensors must be on the same device")


@dataclass(frozen=True)
class FrozenPair:
    """Frozen host outputs for one consecutive-frame transition.

    Edge logits use source-major layout ``(B, N_source, N_target)``. Features
    are allowed to retain pair-specific temporal context; only the coordinates
    and node order of the shared middle frame must agree between adjacent
    pairs.
    """

    source_features: torch.Tensor
    target_features: torch.Tensor
    source_coords_um: torch.Tensor
    target_coords_um: torch.Tensor
    source_mask: torch.Tensor
    target_mask: torch.Tensor
    edge_logits: torch.Tensor

    def __post_init__(self) -> None:
        _validate_nodes(
            "source", self.source_features, self.source_coords_um, self.source_mask
        )
        _validate_nodes(
            "target", self.target_features, self.target_coords_um, self.target_mask
        )
        batch = self.source_features.shape[0]
        if self.target_features.shape[0] != batch:
            raise ValueError("source and target nodes must share the batch axis")
        if self.source_features.shape[-1] != self.target_features.shape[-1]:
            raise ValueError("source and target features must share the channel dimension")
        expected = (batch, self.source_count, self.target_count)
        if self.edge_logits.shape != expected:
            raise ValueError(f"edge_logits must have shape {expected}")
        if not self.edge_logits.is_floating_point():
            raise TypeError("edge_logits must be floating point")
        if self.edge_logits.device != self.source_features.device:
            raise ValueError("all FrozenPair tensors must be on the same device")

    @property
    def batch_size(self) -> int:
        return self.source_features.shape[0]

    @property
    def source_count(self) -> int:
        return self.source_features.shape[1]

    @property
    def target_count(self) -> int:
        return self.target_features.shape[1]

    @property
    def feature_dim(self) -> int:
        return self.source_features.shape[-1]

    def detached(self) -> FrozenPair:
        """Return an explicitly detached view suitable for residual training."""
        return FrozenPair(
            source_features=self.source_features.detach(),
            target_features=self.target_features.detach(),
            source_coords_um=self.source_coords_um.detach(),
            target_coords_um=self.target_coords_um.detach(),
            source_mask=self.source_mask.detach(),
            target_mask=self.target_mask.detach(),
            edge_logits=self.edge_logits.detach(),
        )


@dataclass(frozen=True)
class RightTransitionTriplet:
    """Two frozen image pairs defining a three-frame graph window.

    For frames ``(t-1, t, t+1)``, ``previous_to_source`` is ``t-1 -> t`` and
    ``source_to_target`` is ``t -> t+1``. The residual model owns only the
    right transition, exposed by :attr:`owned_transition`.
    """

    previous_to_source: FrozenPair
    source_to_target: FrozenPair
    middle_coord_atol: float = 1.0e-4

    def __post_init__(self) -> None:
        previous = self.previous_to_source
        current = self.source_to_target
        if (
            isinstance(self.middle_coord_atol, bool)
            or not isinstance(self.middle_coord_atol, (int, float))
            or not math.isfinite(float(self.middle_coord_atol))
            or float(self.middle_coord_atol) < 0.0
        ):
            raise ValueError("middle_coord_atol must be finite and non-negative")
        if previous.batch_size != current.batch_size:
            raise ValueError("adjacent pairs must share the batch axis")
        if previous.target_features.device != current.source_features.device:
            raise ValueError("adjacent pairs must be on the same device")
        if previous.target_count != current.source_count:
            raise ValueError("the shared middle frame must have one common padded node axis")
        if previous.target_features.shape[-1] != current.source_features.shape[-1]:
            raise ValueError("middle-frame feature dimensions differ")
        if not torch.equal(previous.target_mask, current.source_mask):
            raise ValueError("middle-frame masks or node order differ")
        valid = previous.target_mask
        if valid.any() and (
            not torch.isfinite(previous.target_coords_um[valid]).all()
            or not torch.isfinite(current.source_coords_um[valid]).all()
        ):
            raise ValueError("middle-frame coordinates must be finite")
        if valid.any() and not torch.allclose(
            previous.target_coords_um[valid],
            current.source_coords_um[valid],
            rtol=0.0,
            atol=float(self.middle_coord_atol),
        ):
            raise ValueError("middle-frame coordinates or node order differ")

    @property
    def frame_count(self) -> int:
        return 3

    @property
    def owned_transition(self) -> FrozenPair:
        return self.source_to_target


@dataclass(frozen=True)
class ParentCandidates:
    """Target-major bounded source candidates for one owned transition."""

    source_index: torch.Tensor
    valid_mask: torch.Tensor
    distance_um: torch.Tensor
    source_count: int

    def __post_init__(self) -> None:
        if self.source_index.ndim != 3:
            raise ValueError("source_index must have shape (B,N_target,K)")
        if self.source_index.dtype != torch.long:
            raise TypeError("source_index must use torch.long")
        if self.valid_mask.shape != self.source_index.shape:
            raise ValueError("valid_mask must match source_index")
        if self.valid_mask.dtype != torch.bool:
            raise TypeError("valid_mask must be boolean")
        if self.distance_um.shape != self.source_index.shape:
            raise ValueError("distance_um must match source_index")
        if not self.distance_um.is_floating_point():
            raise TypeError("distance_um must be floating point")
        if not (
            self.source_index.device
            == self.valid_mask.device
            == self.distance_um.device
        ):
            raise ValueError("candidate tensors must be on the same device")
        if self.source_count < 0:
            raise ValueError("source_count must be non-negative")
        if self.valid_mask.any():
            selected = self.source_index[self.valid_mask]
            if int(selected.min()) < 0 or int(selected.max()) >= self.source_count:
                raise ValueError("a valid candidate source index is out of range")

    @property
    def batch_size(self) -> int:
        return self.source_index.shape[0]

    @property
    def target_count(self) -> int:
        return self.source_index.shape[1]

    @property
    def top_k(self) -> int:
        return self.source_index.shape[2]

    def dense_mask(self) -> torch.Tensor:
        """Materialize a source-major mask only for loss/debug boundaries."""
        result = torch.zeros(
            self.batch_size,
            self.source_count,
            self.target_count,
            dtype=torch.bool,
            device=self.source_index.device,
        )
        if not self.valid_mask.any():
            return result
        batch, target, slot = torch.nonzero(self.valid_mask, as_tuple=True)
        source = self.source_index[batch, target, slot]
        result[batch, source, target] = True
        return result


@dataclass(frozen=True)
class PreviousParentStatistics:
    """Sparse-safe parent summary for nodes in the shared middle frame."""

    expected_position_um: torch.Tensor
    entropy: torch.Tensor
    has_history: torch.Tensor

    def __post_init__(self) -> None:
        if self.expected_position_um.ndim != 3 or self.expected_position_um.shape[-1] != 3:
            raise ValueError("expected_position_um must have shape (B,N,3)")
        batch_nodes = self.expected_position_um.shape[:2]
        if self.entropy.shape != (*batch_nodes, 1):
            raise ValueError("entropy must have shape (B,N,1)")
        if self.has_history.shape != (*batch_nodes, 1):
            raise ValueError("has_history must have shape (B,N,1)")
        if self.has_history.dtype != torch.bool:
            raise TypeError("has_history must be boolean")


@dataclass(frozen=True)
class CandidateFeatureBatch:
    """O(B * N_target * K) residual features and their provenance."""

    features: torch.Tensor
    candidates: ParentCandidates
    previous_statistics: PreviousParentStatistics

    def __post_init__(self) -> None:
        if self.features.ndim != 4:
            raise ValueError("features must have shape (B,N_target,K,F)")
        if self.features.shape[:3] != self.candidates.source_index.shape:
            raise ValueError("feature candidate axes do not match ParentCandidates")
        if not self.features.is_floating_point():
            raise TypeError("candidate features must be floating point")

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.candidates.valid_mask


@dataclass(frozen=True)
class TemporalGraphOutput:
    """Right-transition residual result with compact diagnostics."""

    edge_logits: torch.Tensor
    candidate_residual: torch.Tensor
    candidate_features: CandidateFeatureBatch

    def __post_init__(self) -> None:
        candidates = self.candidate_features.candidates
        expected = (
            candidates.batch_size,
            candidates.source_count,
            candidates.target_count,
        )
        if self.edge_logits.shape != expected:
            raise ValueError(f"edge_logits must have shape {expected}")
        if self.candidate_residual.shape != candidates.source_index.shape:
            raise ValueError("candidate_residual must have shape (B,N_target,K)")


@dataclass(frozen=True)
class TemporalGraphConfig:
    """Pure-data model configuration safe to store in a checkpoint payload."""

    node_feature_dim: int
    hidden_dim: int = 64
    top_k: int = 32
    radius_um: float = 15.0
    distance_scale_um: float = 10.0
    dropout: float = 0.0
    middle_coord_atol: float = 1.0e-4
    image_window_size: int = 2
    graph_window_size: int = 3
    architecture: str = "mlp"
    attention_heads: int = 4
    residual_logit_bound: float | None = None
    ownership: str = "right_transition"
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("node_feature_dim", "hidden_dim", "top_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.radius_um <= 0 or self.distance_scale_um <= 0:
            raise ValueError("radius_um and distance_scale_um must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.middle_coord_atol < 0:
            raise ValueError("middle_coord_atol must be non-negative")
        for name in ("image_window_size", "graph_window_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.image_window_size != 2:
            raise ValueError("the frozen host contract requires image_window_size=2")
        if self.graph_window_size < 3:
            raise ValueError(
                "the temporal graph contract requires graph_window_size >= 3"
            )
        if self.architecture not in {"mlp", "candidate_attention"}:
            raise ValueError(
                "architecture must be 'mlp' or 'candidate_attention'"
            )
        if (
            isinstance(self.attention_heads, bool)
            or not isinstance(self.attention_heads, int)
            or self.attention_heads <= 0
        ):
            raise ValueError("attention_heads must be a positive integer")
        if self.architecture == "candidate_attention" and (
            self.hidden_dim % self.attention_heads
        ):
            raise ValueError(
                "hidden_dim must be divisible by attention_heads for candidate attention"
            )
        if self.residual_logit_bound is not None and (
            not isinstance(self.residual_logit_bound, (int, float))
            or isinstance(self.residual_logit_bound, bool)
            or not math.isfinite(float(self.residual_logit_bound))
            or float(self.residual_logit_bound) <= 0.0
        ):
            raise ValueError("residual_logit_bound must be a finite positive number")
        if self.ownership != "right_transition":
            raise ValueError("the temporal graph contract requires right_transition ownership")
        if self.schema_version != 1:
            raise ValueError("unsupported temporal-graph config schema")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Preserve byte-for-byte compatibility with schema-v1 MLP checkpoints
        # written before experimental architecture selection was introduced.
        if self.architecture == "mlp":
            payload.pop("architecture")
            payload.pop("attention_heads")
        if self.residual_logit_bound is None:
            payload.pop("residual_logit_bound")
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TemporalGraphConfig:
        return cls(**dict(payload))
