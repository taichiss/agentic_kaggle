"""Bounded parent candidates and O(NK) temporal graph features."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .contracts import (
    CandidateFeatureBatch,
    FrozenPair,
    ParentCandidates,
    PreviousParentStatistics,
    RightTransitionTriplet,
)


def build_parent_candidates(
    source_coords_um: torch.Tensor,
    target_coords_um: torch.Tensor,
    source_mask: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    top_k: int,
    radius_um: float,
) -> ParentCandidates:
    """Select the nearest valid sources inside a physical-radius gate.

    Output is always padded to exactly ``top_k`` slots and is target-major.
    Invalid slots use source index zero and distance zero; ``valid_mask`` is
    authoritative. The bounded output supports O(NK) downstream features.
    """
    if source_coords_um.ndim != 3 or source_coords_um.shape[-1] != 3:
        raise ValueError("source_coords_um must have shape (B,N_source,3)")
    if target_coords_um.ndim != 3 or target_coords_um.shape[-1] != 3:
        raise ValueError("target_coords_um must have shape (B,N_target,3)")
    if source_mask.shape != source_coords_um.shape[:2] or source_mask.dtype != torch.bool:
        raise ValueError("source_mask must be boolean with shape (B,N_source)")
    if target_mask.shape != target_coords_um.shape[:2] or target_mask.dtype != torch.bool:
        raise ValueError("target_mask must be boolean with shape (B,N_target)")
    if source_coords_um.shape[0] != target_coords_um.shape[0]:
        raise ValueError("source and target coordinates must share the batch axis")
    if source_coords_um.device != target_coords_um.device:
        raise ValueError("source and target coordinates must be on the same device")
    if (
        source_mask.device != source_coords_um.device
        or target_mask.device != source_coords_um.device
    ):
        raise ValueError("candidate inputs must be on the same device")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if radius_um <= 0:
        raise ValueError("radius_um must be positive")

    batch, source_count = source_coords_um.shape[:2]
    target_count = target_coords_um.shape[1]
    device = source_coords_um.device
    output_index = torch.zeros(batch, target_count, top_k, dtype=torch.long, device=device)
    output_mask = torch.zeros(batch, target_count, top_k, dtype=torch.bool, device=device)
    output_distance = torch.zeros(
        batch,
        target_count,
        top_k,
        dtype=torch.float32,
        device=device,
    )
    if batch == 0 or source_count == 0 or target_count == 0:
        return ParentCandidates(output_index, output_mask, output_distance, source_count)

    distances = torch.cdist(
        target_coords_um.detach().float(),
        source_coords_um.detach().float(),
    )
    eligible = target_mask.unsqueeze(-1) & source_mask.unsqueeze(1)
    eligible &= distances <= float(radius_um)
    bounded = distances.masked_fill(~eligible, torch.inf)
    selected_count = min(top_k, source_count)
    selected_distance, selected_index = torch.topk(
        bounded,
        k=selected_count,
        dim=-1,
        largest=False,
        sorted=True,
    )
    selected_valid = torch.isfinite(selected_distance)
    output_index[..., :selected_count] = torch.where(
        selected_valid, selected_index, torch.zeros_like(selected_index)
    )
    output_mask[..., :selected_count] = selected_valid
    output_distance[..., :selected_count] = torch.where(
        selected_valid, selected_distance, torch.zeros_like(selected_distance)
    )
    return ParentCandidates(output_index, output_mask, output_distance, source_count)


def expected_previous_parent_statistics(
    previous_pair: FrozenPair,
    *,
    eps: float = 1.0e-8,
) -> PreviousParentStatistics:
    """Return expected parent position and entropy for each middle-frame node.

    Parent probabilities are normalized across valid previous-frame sources.
    A node with no valid history falls back to its own position, with zero
    entropy and ``has_history=False``. No all-masked softmax is evaluated.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    batch, previous_count, middle_count = previous_pair.edge_logits.shape
    fallback = previous_pair.target_coords_um.detach().float()
    entropy = fallback.new_zeros(batch, middle_count, 1)
    has_history = torch.zeros(
        batch,
        middle_count,
        1,
        dtype=torch.bool,
        device=fallback.device,
    )
    if batch == 0 or previous_count == 0 or middle_count == 0:
        return PreviousParentStatistics(fallback, entropy, has_history)

    probabilities, has_parent = _normalized_parent_probabilities(
        previous_pair,
        eps=eps,
    )
    expected = torch.einsum(
        "bps,bpc->bsc",
        probabilities,
        previous_pair.source_coords_um.detach().float(),
    )
    expected = torch.where(has_parent.unsqueeze(-1), expected, fallback)
    entropy_values = -(
        probabilities * probabilities.clamp_min(eps).log()
    ).sum(dim=1)
    entropy = torch.where(
        has_parent, entropy_values, torch.zeros_like(entropy_values)
    ).unsqueeze(-1)
    has_history = has_parent.unsqueeze(-1)
    return PreviousParentStatistics(expected, entropy, has_history)


def _normalized_parent_probabilities(
    pair: FrozenPair,
    *,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return finite, mask-aware source probabilities for each target."""
    batch, source_count, target_count = pair.edge_logits.shape
    logits = pair.edge_logits.detach().float()
    has_parent = torch.zeros(
        batch,
        target_count,
        dtype=torch.bool,
        device=logits.device,
    )
    if batch == 0 or source_count == 0 or target_count == 0:
        return logits.new_zeros(batch, source_count, target_count), has_parent

    valid = pair.source_mask.unsqueeze(-1) & pair.target_mask.unsqueeze(1)
    valid &= torch.isfinite(logits)
    masked_logits = logits.masked_fill(~valid, -torch.inf)
    has_parent = valid.any(dim=1)
    maximum = masked_logits.amax(dim=1)
    maximum = torch.where(has_parent, maximum, torch.zeros_like(maximum))
    weights = torch.where(
        valid,
        torch.exp(masked_logits - maximum.unsqueeze(1)),
        torch.zeros_like(masked_logits),
    )
    probabilities = weights / weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return probabilities, has_parent


def _gather_source(values: torch.Tensor, candidates: ParentCandidates) -> torch.Tensor:
    """Gather ``(B,S,...)`` values into target-major ``(B,T,K,...)``."""
    batch, source_count = values.shape[:2]
    if batch != candidates.batch_size or source_count != candidates.source_count:
        raise ValueError("source values do not match ParentCandidates")
    trailing = values.shape[2:]
    output_shape = (
        candidates.batch_size,
        candidates.target_count,
        candidates.top_k,
        *trailing,
    )
    if source_count == 0:
        return values.new_zeros(output_shape)
    batch_index = torch.arange(batch, device=values.device)[:, None, None]
    return values[batch_index, candidates.source_index]


def _gather_edge_logits(
    edge_logits: torch.Tensor,
    candidates: ParentCandidates,
) -> torch.Tensor:
    if edge_logits.shape != (
        candidates.batch_size,
        candidates.source_count,
        candidates.target_count,
    ):
        raise ValueError("edge_logits do not match ParentCandidates")
    if candidates.source_count == 0:
        return edge_logits.new_zeros(candidates.source_index.shape)
    return torch.gather(edge_logits.transpose(1, 2), 2, candidates.source_index)


def candidate_feature_dim(
    node_feature_dim: int,
    graph_window_size: int = 3,
) -> int:
    """Return the stable feature width used by :func:`build_candidate_features`."""
    if isinstance(node_feature_dim, bool) or not isinstance(node_feature_dim, int):
        raise TypeError("node_feature_dim must be an integer")
    if node_feature_dim <= 0:
        raise ValueError("node_feature_dim must be positive")
    if isinstance(graph_window_size, bool) or not isinstance(graph_window_size, int):
        raise TypeError("graph_window_size must be an integer")
    if graph_window_size < 3:
        raise ValueError("graph_window_size must be at least 3")
    # T3/T4/T5 are immutable checkpoint contracts. Longer windows retain the
    # complete T5 prefix and add one fixed-width summary of the whole lineage,
    # so T10 and T20 do not grow the trainable head merely because they look
    # farther back in time.
    temporal_width = {3: 10, 4: 14, 5: 18}.get(graph_window_size, 26)
    return 3 * node_feature_dim + temporal_width


def _resolve_history_pairs(
    previous_pair: FrozenPair,
    *,
    history_pairs: Sequence[FrozenPair] | None,
    prior_pair: FrozenPair | None,
    oldest_pair: FrozenPair | None,
    graph_window_size: int,
    middle_coord_atol: float,
) -> tuple[FrozenPair, ...]:
    """Normalize legacy T4/T5 aliases into an oldest-to-newest history.

    A graph window of ``W`` owns the final transition, receives the immediately
    preceding transition separately as ``previous_pair``, and therefore needs
    exactly ``W - 3`` older pairs. Supplying less history would silently change
    the experiment, so incomplete histories fail closed. The legacy aliases
    remain byte-compatible call paths for existing T4/T5 submissions.
    """
    expected = graph_window_size - 3
    if history_pairs is not None:
        if prior_pair is not None or oldest_pair is not None:
            raise ValueError(
                "history_pairs cannot be combined with prior_pair or oldest_pair"
            )
        resolved = tuple(history_pairs)
    elif graph_window_size == 3:
        if prior_pair is not None:
            raise ValueError("prior_pair must be omitted when graph_window_size=3")
        if oldest_pair is not None:
            raise ValueError("oldest_pair must be omitted when graph_window_size=3")
        resolved = ()
    elif graph_window_size == 4:
        if prior_pair is None:
            raise ValueError("prior_pair is required when graph_window_size=4")
        if oldest_pair is not None:
            raise ValueError("oldest_pair must be omitted when graph_window_size=4")
        resolved = (prior_pair,)
    elif graph_window_size == 5:
        if prior_pair is None:
            raise ValueError("prior_pair is required when graph_window_size=5")
        if oldest_pair is None:
            raise ValueError("oldest_pair is required when graph_window_size=5")
        resolved = (oldest_pair, prior_pair)
    else:
        raise ValueError(
            "history_pairs is required for graph_window_size greater than 5"
        )

    if len(resolved) != expected:
        raise ValueError(
            "history_pairs must contain exactly "
            f"{expected} oldest-to-newest pairs for graph_window_size="
            f"{graph_window_size}; got {len(resolved)}"
        )
    if not all(isinstance(pair, FrozenPair) for pair in resolved):
        raise TypeError("history_pairs must contain only FrozenPair values")

    adjacent = (*resolved, previous_pair)
    for left, right in zip(adjacent, adjacent[1:], strict=False):
        RightTransitionTriplet(
            left,
            right,
            middle_coord_atol=middle_coord_atol,
        )
    return resolved


def _long_history_features(
    history_pairs: tuple[FrozenPair, ...],
    previous_pair: FrozenPair,
    current_pair: FrozenPair,
    candidates: ParentCandidates,
    target_coords: torch.Tensor,
    expected_previous: torch.Tensor,
    *,
    distance_scale_um: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Summarize every historical frame with path-aware linear motion.

    Dynamic programming propagates sufficient statistics over the frozen-host
    parent distributions without materializing dense all-path tensors. The
    resulting block is: long-horizon linear prediction residual (xyz), recent
    velocity minus the fitted long-horizon velocity (xyz), complete-path mass,
    and trajectory-fit RMSE. If no complete path reaches the deepest requested
    frame, all eight values are explicitly zero.
    """
    lineage_pairs = (*history_pairs, previous_pair)
    oldest = lineage_pairs[0]
    mass = oldest.source_mask.detach().float()
    oldest_coords = oldest.source_coords_um.detach().float()
    oldest_coords = torch.where(
        oldest.source_mask.unsqueeze(-1),
        oldest_coords,
        torch.zeros_like(oldest_coords),
    )
    summed_position = oldest_coords * mass.unsqueeze(-1)
    summed_time_position = torch.zeros_like(summed_position)
    summed_squared_norm = oldest_coords.square().sum(dim=-1) * mass

    for time_index, pair in enumerate(lineage_pairs, start=1):
        probabilities, _ = _normalized_parent_probabilities(pair, eps=1.0e-8)
        propagated_mass = torch.einsum("bst,bs->bt", probabilities, mass)
        propagated_position = torch.einsum(
            "bst,bsc->btc", probabilities, summed_position
        )
        propagated_time_position = torch.einsum(
            "bst,bsc->btc", probabilities, summed_time_position
        )
        propagated_squared_norm = torch.einsum(
            "bst,bs->bt", probabilities, summed_squared_norm
        )
        coordinates = pair.target_coords_um.detach().float()
        coordinates = torch.where(
            pair.target_mask.unsqueeze(-1),
            coordinates,
            torch.zeros_like(coordinates),
        )
        mass = propagated_mass
        summed_position = propagated_position + coordinates * mass.unsqueeze(-1)
        summed_time_position = (
            propagated_time_position
            + float(time_index) * coordinates * mass.unsqueeze(-1)
        )
        summed_squared_norm = (
            propagated_squared_norm
            + coordinates.square().sum(dim=-1) * mass
        )

    has_complete_path = mass > 1.0e-8
    normalizer = mass.unsqueeze(-1).clamp_min(1.0e-8)
    position_sum = summed_position / normalizer
    time_position_sum = summed_time_position / normalizer
    squared_norm_sum = summed_squared_norm / mass.clamp_min(1.0e-8)

    frame_count = float(len(lineage_pairs) + 1)
    time_sum = frame_count * (frame_count - 1.0) / 2.0
    time_square_sum = (
        frame_count * (frame_count - 1.0) * (2.0 * frame_count - 1.0) / 6.0
    )
    mean_time = time_sum / frame_count
    centered_time_square_sum = time_square_sum - frame_count * mean_time**2
    slope = (
        time_position_sum - mean_time * position_sum
    ) / centered_time_square_sum
    intercept = (position_sum - slope * time_sum) / frame_count
    prediction = intercept + slope * frame_count

    source_coords = current_pair.source_coords_um.detach().float()
    recent_velocity = source_coords - expected_previous
    velocity_delta = recent_velocity - slope

    fit_sse = (
        squared_norm_sum
        - 2.0 * (intercept * position_sum).sum(dim=-1)
        - 2.0 * (slope * time_position_sum).sum(dim=-1)
        + frame_count * intercept.square().sum(dim=-1)
        + 2.0 * time_sum * (intercept * slope).sum(dim=-1)
        + time_square_sum * slope.square().sum(dim=-1)
    )
    fit_rmse = torch.sqrt(
        fit_sse.clamp_min(0.0) / (frame_count * 3.0)
    ).unsqueeze(-1)

    prediction = _gather_source(prediction, candidates)
    velocity_delta = _gather_source(velocity_delta, candidates)
    path_mass = _gather_source(mass.unsqueeze(-1), candidates)
    fit_rmse = _gather_source(fit_rmse, candidates)
    available = _gather_source(has_complete_path.unsqueeze(-1), candidates)

    scale = float(distance_scale_um)
    prediction_residual = (target_coords - prediction) / scale
    velocity_delta = velocity_delta / scale
    fit_rmse = fit_rmse / scale
    prediction_residual = torch.where(
        available, prediction_residual, torch.zeros_like(prediction_residual)
    )
    velocity_delta = torch.where(
        available, velocity_delta, torch.zeros_like(velocity_delta)
    )
    path_mass = torch.where(available, path_mass, torch.zeros_like(path_mass))
    fit_rmse = torch.where(available, fit_rmse, torch.zeros_like(fit_rmse))
    return prediction_residual, velocity_delta, path_mass, fit_rmse


def build_candidate_features(
    previous_pair: FrozenPair,
    current_pair: FrozenPair,
    candidates: ParentCandidates,
    *,
    history_pairs: Sequence[FrozenPair] | None = None,
    prior_pair: FrozenPair | None = None,
    oldest_pair: FrozenPair | None = None,
    graph_window_size: int = 3,
    distance_scale_um: float = 10.0,
    middle_coord_atol: float = 1.0e-4,
) -> CandidateFeatureBatch:
    """Build O(N_target * K) features for the owned right transition.

    Feature order is: previous-view middle appearance, current-view source
    appearance, target appearance, normalized displacement (3), normalized
    constant-velocity residual (3), distance, frozen base logit, previous
    parent entropy, and a history-availability indicator. When ``prior_pair``
    supplies ``t-2 -> t-1``, the complete three-frame feature vector remains
    an unchanged prefix followed by normalized constant-acceleration residual
    (3) and probabilistically propagated second-history mass (1). For a
    five-frame window, ``oldest_pair`` supplies ``t-3 -> t-2`` and appends a
    constant-jerk residual (3) plus deepest-history path mass (1), while the
    complete T3 and T4 vectors remain unchanged prefixes. Longer windows use
    ``history_pairs`` in oldest-to-newest order. They keep the complete T5
    prefix and append eight fixed-width, path-aware long-history statistics.
    """
    if distance_scale_um <= 0:
        raise ValueError("distance_scale_um must be positive")
    candidate_feature_dim(current_pair.feature_dim, graph_window_size)
    resolved_history = _resolve_history_pairs(
        previous_pair,
        history_pairs=history_pairs,
        prior_pair=prior_pair,
        oldest_pair=oldest_pair,
        graph_window_size=graph_window_size,
        middle_coord_atol=middle_coord_atol,
    )
    prior_pair = resolved_history[-1] if resolved_history else None
    oldest_pair = resolved_history[-2] if len(resolved_history) >= 2 else None
    triplet = RightTransitionTriplet(
        previous_pair,
        current_pair,
        middle_coord_atol=middle_coord_atol,
    )
    current = triplet.owned_transition
    if candidates.batch_size != current.batch_size:
        raise ValueError("candidate and transition batch dimensions differ")
    if candidates.source_count != current.source_count:
        raise ValueError("candidate source axis differs from the owned transition")
    if candidates.target_count != current.target_count:
        raise ValueError("candidate target axis differs from the owned transition")

    history = expected_previous_parent_statistics(previous_pair)
    previous_view = _gather_source(
        previous_pair.target_features.detach().float(), candidates
    )
    current_source = _gather_source(current.source_features.detach().float(), candidates)
    source_coords = _gather_source(current.source_coords_um.detach().float(), candidates)
    expected_previous = _gather_source(history.expected_position_um, candidates)
    previous_entropy = _gather_source(history.entropy, candidates)
    has_history = _gather_source(history.has_history, candidates).float()

    target_view = current.target_features.detach().float().unsqueeze(2).expand(
        -1, -1, candidates.top_k, -1
    )
    target_coords = current.target_coords_um.detach().float().unsqueeze(2).expand(
        -1, -1, candidates.top_k, -1
    )
    displacement = (target_coords - source_coords) / float(distance_scale_um)
    velocity = source_coords - expected_previous
    predicted_target = source_coords + velocity
    velocity_residual = (target_coords - predicted_target) / float(distance_scale_um)
    distance = candidates.distance_um.unsqueeze(-1) / float(distance_scale_um)
    base_logit = _gather_edge_logits(current.edge_logits.detach().float(), candidates).unsqueeze(-1)

    feature_parts = [
        previous_view,
        current_source,
        target_view,
        displacement,
        velocity_residual,
        distance,
        base_logit,
        previous_entropy,
        has_history,
    ]
    if prior_pair is not None:
        prior_history = expected_previous_parent_statistics(prior_pair)
        parent_probabilities, _ = _normalized_parent_probabilities(
            previous_pair,
            eps=1.0e-8,
        )
        prior_available = prior_history.has_history.squeeze(-1).float()
        propagated_weights = parent_probabilities * prior_available.unsqueeze(-1)
        second_history_mass = propagated_weights.sum(dim=1)
        has_second_history = second_history_mass > 1.0e-8
        conditional_weights = propagated_weights / second_history_mass.unsqueeze(
            1
        ).clamp_min(1.0e-8)
        expected_first_parent = torch.einsum(
            "bst,bsc->btc",
            conditional_weights,
            previous_pair.source_coords_um.detach().float(),
        )
        expected_second_parent = torch.einsum(
            "bst,bsc->btc",
            conditional_weights,
            prior_history.expected_position_um.detach().float(),
        )

        previous_velocity = expected_first_parent - expected_second_parent
        current_velocity = (
            current.source_coords_um.detach().float() - expected_first_parent
        )
        acceleration = current_velocity - previous_velocity
        acceleration_prediction = (
            current.source_coords_um.detach().float() + current_velocity + acceleration
        )
        acceleration_prediction = _gather_source(acceleration_prediction, candidates)
        acceleration_residual = (
            target_coords - acceleration_prediction
        ) / float(distance_scale_um)
        gathered_has_second_history = _gather_source(
            has_second_history.unsqueeze(-1),
            candidates,
        )
        acceleration_residual = torch.where(
            gathered_has_second_history,
            acceleration_residual,
            torch.zeros_like(acceleration_residual),
        )
        second_history_mass = torch.where(
            has_second_history,
            second_history_mass,
            torch.zeros_like(second_history_mass),
        )
        gathered_second_history_mass = _gather_source(
            second_history_mass.unsqueeze(-1),
            candidates,
        )
        feature_parts.extend(
            [acceleration_residual, gathered_second_history_mass]
        )

        if oldest_pair is not None:
            oldest_history = expected_previous_parent_statistics(oldest_pair)
            prior_probabilities, _ = _normalized_parent_probabilities(
                prior_pair,
                eps=1.0e-8,
            )
            deepest_available = oldest_history.has_history.squeeze(-1).float()
            prior_deep_weights = (
                prior_probabilities * deepest_available.unsqueeze(-1)
            )
            deep_mass_per_second_parent = prior_deep_weights.sum(dim=1)
            deep_path_weights = (
                parent_probabilities * deep_mass_per_second_parent.unsqueeze(-1)
            )
            third_history_mass = deep_path_weights.sum(dim=1)
            has_third_history = third_history_mass > 1.0e-8
            normalizer = third_history_mass.unsqueeze(-1).clamp_min(1.0e-8)

            expected_second_parent = torch.einsum(
                "bst,bsc->btc",
                deep_path_weights,
                previous_pair.source_coords_um.detach().float(),
            ) / normalizer
            weighted_first_position = torch.einsum(
                "bst,bsc->btc",
                prior_deep_weights,
                prior_pair.source_coords_um.detach().float(),
            )
            expected_first_parent = torch.einsum(
                "bst,bsc->btc",
                parent_probabilities,
                weighted_first_position,
            ) / normalizer
            weighted_oldest_position = torch.einsum(
                "bst,bsc->btc",
                prior_deep_weights,
                oldest_history.expected_position_um.detach().float(),
            )
            expected_oldest_parent = torch.einsum(
                "bst,bsc->btc",
                parent_probabilities,
                weighted_oldest_position,
            ) / normalizer

            oldest_velocity = expected_first_parent - expected_oldest_parent
            previous_velocity = expected_second_parent - expected_first_parent
            current_velocity = (
                current.source_coords_um.detach().float() - expected_second_parent
            )
            previous_acceleration = previous_velocity - oldest_velocity
            current_acceleration = current_velocity - previous_velocity
            jerk = current_acceleration - previous_acceleration
            jerk_prediction = (
                current.source_coords_um.detach().float()
                + current_velocity
                + current_acceleration
                + jerk
            )
            jerk_prediction = _gather_source(jerk_prediction, candidates)
            jerk_residual = (
                target_coords - jerk_prediction
            ) / float(distance_scale_um)
            gathered_has_third_history = _gather_source(
                has_third_history.unsqueeze(-1),
                candidates,
            )
            jerk_residual = torch.where(
                gathered_has_third_history,
                jerk_residual,
                torch.zeros_like(jerk_residual),
            )
            third_history_mass = torch.where(
                has_third_history,
                third_history_mass,
                torch.zeros_like(third_history_mass),
            )
            gathered_third_history_mass = _gather_source(
                third_history_mass.unsqueeze(-1),
                candidates,
            )
            feature_parts.extend(
                [jerk_residual, gathered_third_history_mass]
            )

    if graph_window_size >= 6:
        feature_parts.extend(
            _long_history_features(
                resolved_history,
                previous_pair,
                current,
                candidates,
                target_coords,
                history.expected_position_um,
                distance_scale_um=distance_scale_um,
            )
        )

    features = torch.cat(feature_parts, dim=-1)
    expected_dim = candidate_feature_dim(current.feature_dim, graph_window_size)
    if features.shape[-1] != expected_dim:
        raise RuntimeError("candidate feature construction drifted from its public contract")
    features = features * candidates.valid_mask.unsqueeze(-1).to(features.dtype)
    return CandidateFeatureBatch(features, candidates, history)


def refine_logits(
    base_logits: torch.Tensor,
    candidates: ParentCandidates,
    candidate_residual: torch.Tensor,
) -> torch.Tensor:
    """Scatter residuals onto candidates while preserving every other logit.

    The base tensor is detached deliberately: this boundary guarantees the
    host model remains frozen even when callers pass tensors requiring grad.
    """
    expected = (
        candidates.batch_size,
        candidates.source_count,
        candidates.target_count,
    )
    if base_logits.shape != expected:
        raise ValueError(f"base_logits must have shape {expected}")
    if candidate_residual.shape != candidates.source_index.shape:
        raise ValueError("candidate_residual must have shape (B,N_target,K)")
    base = base_logits.detach()
    if not candidates.valid_mask.any():
        return base + candidate_residual.sum().to(base.dtype) * 0.0

    dense_delta = torch.zeros_like(base)
    batch, target, slot = torch.nonzero(candidates.valid_mask, as_tuple=True)
    source = candidates.source_index[batch, target, slot]
    values = candidate_residual[batch, target, slot].to(base.dtype)
    dense_delta.index_put_((batch, source, target), values, accumulate=True)
    return base + dense_delta
