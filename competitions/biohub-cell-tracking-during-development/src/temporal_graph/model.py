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


class CandidateAttentionResidual(nn.Module):
    """Jointly score the bounded parent candidates for each target node.

    The attention axis is the unordered candidate set, not time. Temporal
    evidence remains encoded in each candidate's three-frame features.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        attention_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.input_projection = nn.Linear(feature_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, targets, candidates, width = features.shape
        flat_features = features.reshape(batch * targets, candidates, width)
        flat_valid = valid_mask.reshape(batch * targets, candidates)
        padding_mask = ~flat_valid
        all_invalid = ~flat_valid.any(dim=1)
        if all_invalid.any():
            # MultiheadAttention cannot consume a row whose every key is
            # padded. The synthetic unmasked slot is zeroed again below.
            padding_mask = padding_mask.clone()
            padding_mask[all_invalid, 0] = False

        hidden = self.input_projection(self.input_norm(flat_features.float()))
        hidden = hidden * flat_valid.unsqueeze(-1).to(hidden.dtype)
        attended, _ = self.attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = self.attention_norm(hidden + attended)
        hidden = self.output_norm(hidden + self.feed_forward(hidden))
        residual = self.output(hidden).squeeze(-1)
        residual = residual * flat_valid.to(residual.dtype)
        return residual.reshape(batch, targets, candidates)


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
        if config.architecture == "mlp":
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
            self.residual_mlp: nn.Module | None = nn.Sequential(*layers)
            self.candidate_attention: nn.Module | None = None
        else:
            self.residual_mlp = None
            self.candidate_attention = CandidateAttentionResidual(
                feature_dim,
                config.hidden_dim,
                config.attention_heads,
                config.dropout,
            )

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
        if self.residual_mlp is not None:
            residual = self.residual_mlp(features.float()).squeeze(-1)
        else:
            if self.candidate_attention is None:
                raise RuntimeError("candidate-attention module is missing")
            residual = self.candidate_attention(features, valid_mask)
        if self.config.residual_logit_bound is not None:
            bound = float(self.config.residual_logit_bound)
            valid = valid_mask.to(residual.dtype)
            candidate_mean = (residual * valid).sum(dim=-1, keepdim=True) / valid.sum(
                dim=-1, keepdim=True
            ).clamp_min(1.0)
            centered = (residual - candidate_mean) * valid
            # A common offset cannot change parent selection, so remove it
            # before applying a smooth, non-compensable logit cap. The cap has
            # unit slope at zero and preserves the exact frozen-host start.
            residual = bound * torch.tanh(centered / bound)
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
