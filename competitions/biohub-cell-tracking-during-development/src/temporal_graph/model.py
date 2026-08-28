"""Zero-initialized residual scorer for the owned right transition."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

from .candidates import (
    build_candidate_features,
    build_parent_candidates,
    candidate_feature_dim,
    refine_logits,
)
from .checkpointing import TemporalGraphCheckpoint
from .contracts import (
    CandidateFeatureBatch,
    FrozenPair,
    ParentCandidates,
    RightTransitionTriplet,
    TemporalGraphConfig,
    TemporalGraphOutput,
)


class TemporalGraphResidualHead(nn.Module):
    """Refine frozen host logits with bounded three-frame graph context."""

    def __init__(
        self,
        config: TemporalGraphConfig | None = None,
        *,
        node_feature_dim: int | None = None,
        hidden_dim: int = 64,
        top_k: int = 32,
        radius_um: float = 15.0,
        distance_scale_um: float = 10.0,
        dropout: float = 0.0,
        middle_coord_atol: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if config is None:
            if node_feature_dim is None:
                raise ValueError("node_feature_dim is required when config is omitted")
            config = TemporalGraphConfig(
                node_feature_dim=node_feature_dim,
                hidden_dim=hidden_dim,
                top_k=top_k,
                radius_um=radius_um,
                distance_scale_um=distance_scale_um,
                dropout=dropout,
                middle_coord_atol=middle_coord_atol,
            )
        elif node_feature_dim is not None:
            raise ValueError("pass either config or node_feature_dim arguments, not both")
        self.config = config

        feature_dim = candidate_feature_dim(config.node_feature_dim)
        layers: list[nn.Module] = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.hidden_dim),
            nn.GELU(),
        ]
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        layers.extend(
            [
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.GELU(),
            ]
        )
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        output = nn.Linear(config.hidden_dim, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.residual_mlp = nn.Sequential(*layers)

    def forward_candidate_features(
        self,
        candidate_features: CandidateFeatureBatch | torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Score prebuilt candidate features without rebuilding graph context."""
        if isinstance(candidate_features, CandidateFeatureBatch):
            if valid_mask is not None:
                raise ValueError("valid_mask is already carried by CandidateFeatureBatch")
            valid_mask = candidate_features.valid_mask
            features = candidate_features.features
        else:
            features = candidate_features
            if valid_mask is None:
                raise ValueError("valid_mask is required for a raw feature tensor")
        if features.ndim != 4:
            raise ValueError("candidate_features must have shape (B,N_target,K,F)")
        if features.shape[-1] != candidate_feature_dim(self.config.node_feature_dim):
            raise ValueError("candidate feature width does not match the model config")
        if valid_mask.shape != features.shape[:3] or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be boolean with shape (B,N_target,K)")
        residual = self.residual_mlp(features.float()).squeeze(-1)
        return residual * valid_mask.to(residual.dtype)

    def forward(
        self,
        previous_pair: FrozenPair | RightTransitionTriplet,
        current_pair: FrozenPair | None = None,
        candidates: ParentCandidates | None = None,
    ) -> TemporalGraphOutput:
        """Refine only ``t -> t+1`` from frames ``(t-1, t, t+1)``."""
        if isinstance(previous_pair, RightTransitionTriplet):
            if current_pair is not None:
                raise ValueError("current_pair must be omitted when passing a triplet")
            triplet = previous_pair
            previous = triplet.previous_to_source
            current = triplet.source_to_target
        else:
            if current_pair is None:
                raise ValueError("current_pair is required")
            triplet = RightTransitionTriplet(
                previous_pair,
                current_pair,
                middle_coord_atol=self.config.middle_coord_atol,
            )
            previous = triplet.previous_to_source
            current = triplet.source_to_target
        if previous.feature_dim != self.config.node_feature_dim:
            raise ValueError("pair feature width does not match the model config")
        if candidates is None:
            candidates = build_parent_candidates(
                current.source_coords_um.detach(),
                current.target_coords_um.detach(),
                current.source_mask.detach(),
                current.target_mask.detach(),
                top_k=self.config.top_k,
                radius_um=self.config.radius_um,
            )
        features = build_candidate_features(
            previous,
            current,
            candidates,
            distance_scale_um=self.config.distance_scale_um,
            middle_coord_atol=self.config.middle_coord_atol,
        )
        residual = self.forward_candidate_features(features)
        logits = refine_logits(current.edge_logits, candidates, residual)
        return TemporalGraphOutput(logits, residual, features)

    def checkpoint_payload(
        self,
        *,
        base_checkpoint_sha256: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint = TemporalGraphCheckpoint(
            config=self.config,
            state_dict=self.state_dict(),
            base_checkpoint_sha256=base_checkpoint_sha256,
            metadata={} if metadata is None else metadata,
        )
        return checkpoint.to_payload()

    @classmethod
    def from_checkpoint_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        map_location: str | torch.device | None = None,
    ) -> TemporalGraphResidualHead:
        checkpoint = TemporalGraphCheckpoint.from_payload(payload)
        model = cls(checkpoint.config)
        state = dict(checkpoint.state_dict)
        if map_location is not None:
            state = {key: value.to(map_location) for key, value in state.items()}
            model.to(map_location)
        model.load_state_dict(state, strict=True)
        return model


TemporalGraphResidualScorer = TemporalGraphResidualHead
