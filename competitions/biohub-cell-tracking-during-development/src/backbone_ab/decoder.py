"""One calibrated graph decoder shared by screen, local, and Kaggle inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .checkpointing import DecoderProfile


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.maximum(exponent.sum(axis=axis, keepdims=True), np.finfo(float).tiny)


@dataclass(frozen=True)
class DecodedEdge:
    source: int
    target: int
    probability: float
    distance_um: float


@dataclass(frozen=True)
class DecodeResult:
    edges: tuple[DecodedEdge, ...]
    edge_probabilities: np.ndarray
    null_parent_probabilities: np.ndarray
    division_probabilities: np.ndarray


class GraphDecoder:
    """Decode parent logits with explicit null-parent and division constraints."""

    def __init__(self, profile: DecoderProfile, *, edge_activation: str = "softmax") -> None:
        if edge_activation == "parent_softmax_with_null":
            edge_activation = "softmax"
        if edge_activation not in {"softmax", "sigmoid", "none"}:
            raise ValueError(f"unsupported edge activation: {edge_activation}")
        self.profile = profile
        self.edge_activation = edge_activation

    def probabilities(
        self,
        edge_logits: Any,
        null_parent_logits: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge = _to_numpy(edge_logits)
        if edge.ndim != 2:
            raise ValueError(f"edge_logits must have shape (source,target), got {edge.shape}")
        targets = edge.shape[1]
        if null_parent_logits is not None:
            null = _to_numpy(null_parent_logits).reshape(-1)
            if null.shape != (targets,):
                raise ValueError(
                    f"null_parent_logits must have shape ({targets},), got {null.shape}"
                )
        else:
            null = None

        if self.edge_activation == "softmax":
            if edge.shape[0] == 0:
                return np.empty_like(edge), np.ones(targets, dtype=np.float64)
            if null is None:
                return _softmax(edge, axis=0), np.zeros(targets, dtype=np.float64)
            parent_logits = np.concatenate([edge.T, null[:, None]], axis=1)
            parent_probability = _softmax(parent_logits, axis=1)
            return parent_probability[:, :-1].T, parent_probability[:, -1]
        if self.edge_activation == "sigmoid":
            return _sigmoid(edge), (
                _sigmoid(null) if null is not None else np.zeros(targets, dtype=np.float64)
            )
        return edge, null if null is not None else np.zeros(targets, dtype=np.float64)

    def decode(
        self,
        edge_logits: Any,
        *,
        edge_threshold: float,
        null_parent_logits: Any | None = None,
        division_logits: Any | None = None,
        source_coords_um: Any | None = None,
        target_coords_um: Any | None = None,
        source_offset: int = 0,
        target_offset: int = 0,
    ) -> DecodeResult:
        if not 0.0 <= edge_threshold <= 1.0:
            raise ValueError("edge_threshold must be in [0, 1]")
        edge_probability, null_probability = self.probabilities(
            edge_logits, null_parent_logits
        )
        source_count, target_count = edge_probability.shape
        if division_logits is None:
            division_probability = np.ones(source_count, dtype=np.float64)
        else:
            division = _to_numpy(division_logits).reshape(-1)
            if division.shape != (source_count,):
                raise ValueError(
                    f"division_logits must have shape ({source_count},), got {division.shape}"
                )
            division_probability = _sigmoid(division)

        return self.decode_probabilities(
            edge_probability,
            null_probability,
            division_probability,
            edge_threshold=edge_threshold,
            source_coords_um=source_coords_um,
            target_coords_um=target_coords_um,
            source_offset=source_offset,
            target_offset=target_offset,
        )

    def decode_probabilities(
        self,
        edge_probabilities: Any,
        null_parent_probabilities: Any,
        division_probabilities: Any,
        *,
        edge_threshold: float,
        source_coords_um: Any | None = None,
        target_coords_um: Any | None = None,
        source_offset: int = 0,
        target_offset: int = 0,
    ) -> DecodeResult:
        """Apply graph constraints to cached probabilities from one LinkOutput."""
        if not 0.0 <= edge_threshold <= 1.0:
            raise ValueError("edge_threshold must be in [0, 1]")
        edge_probability = _to_numpy(edge_probabilities)
        if edge_probability.ndim != 2:
            raise ValueError("edge_probabilities must have shape (source,target)")
        source_count, target_count = edge_probability.shape
        null_probability = _to_numpy(null_parent_probabilities).reshape(-1)
        division_probability = _to_numpy(division_probabilities).reshape(-1)
        if null_probability.shape != (target_count,):
            raise ValueError("null_parent_probabilities must have shape (target,)")
        if division_probability.shape != (source_count,):
            raise ValueError("division_probabilities must have shape (source,)")

        source_coords = None if source_coords_um is None else _to_numpy(source_coords_um)
        target_coords = None if target_coords_um is None else _to_numpy(target_coords_um)
        if source_coords is not None and source_coords.shape != (source_count, 3):
            raise ValueError("source_coords_um must have shape (source,3)")
        if target_coords is not None and target_coords.shape != (target_count, 3):
            raise ValueError("target_coords_um must have shape (target,3)")

        candidates = [
            (float(edge_probability[source, target]), source, target)
            for source, target in np.argwhere(edge_probability > edge_threshold)
            if null_probability[target] < self.profile.null_parent_threshold
        ]
        candidates.sort(reverse=True)
        source_children: dict[int, int] = {}
        target_parents: dict[int, int] = {}
        edges: list[DecodedEdge] = []
        for probability, source, target in candidates:
            max_children = (
                self.profile.max_children_per_node
                if division_probability[source] >= self.profile.division_threshold
                else 1
            )
            if source_children.get(source, 0) >= max_children:
                continue
            if target_parents.get(target, 0) >= self.profile.max_parents_per_node:
                continue
            distance = 0.0
            if source_coords is not None and target_coords is not None:
                distance = float(np.linalg.norm(source_coords[source] - target_coords[target]))
            edges.append(
                DecodedEdge(
                    source=source_offset + source,
                    target=target_offset + target,
                    probability=probability,
                    distance_um=distance,
                )
            )
            source_children[source] = source_children.get(source, 0) + 1
            target_parents[target] = target_parents.get(target, 0) + 1
        return DecodeResult(
            edges=tuple(edges),
            edge_probabilities=edge_probability,
            null_parent_probabilities=null_probability,
            division_probabilities=division_probability,
        )
