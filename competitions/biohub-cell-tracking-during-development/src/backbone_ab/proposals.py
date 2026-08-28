"""Training-node proposal construction for corrected detector/linker training."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProposalWindow:
    """Padded node proposals and their sparse-GT correspondence.

    ``gt_matches`` maps each proposal to its GT-node index in the same frame.
    ``-1`` means that sparse GT cannot establish an identity. ``reliable_null``
    is reserved for targets whose annotated parent proposal was deliberately
    dropped; natural detector misses and injected duplicates remain unknown.
    """

    coords: torch.Tensor
    masks: torch.Tensor
    gt_matches: torch.Tensor
    reliable_null: torch.Tensor


def predicted_ratio_for_epoch(
    strategy: str,
    curriculum_config: dict | None,
    epoch_index: int,
) -> float:
    """Return the sample-level probability of using detector proposals."""
    if strategy == "ground_truth":
        return 0.0
    if strategy != "mixed_predicted":
        raise ValueError(f"unsupported node proposal strategy: {strategy}")
    config = curriculum_config or {}
    ratios = [float(value) for value in config.get("predicted_ratios", [])]
    if not ratios:
        raise ValueError("mixed_predicted requires proposal_curriculum.predicted_ratios")
    if epoch_index < 0:
        raise ValueError("epoch_index must be non-negative")
    return ratios[min(epoch_index, len(ratios) - 1)]


def ground_truth_proposals(
    coords: torch.Tensor,
    masks: torch.Tensor,
) -> ProposalWindow:
    """Create proposals whose identities are the post-augmentation GT nodes."""
    if coords.ndim != 4 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (B,T,M,3)")
    if masks.shape != coords.shape[:-1]:
        raise ValueError("masks must have shape (B,T,M)")
    batch, time = masks.shape[:2]
    maximum = max(1, int(masks.sum(dim=-1).max().item()))
    packed_coords = coords.new_zeros(batch, time, maximum, 3)
    packed_masks = torch.zeros(
        batch, time, maximum, dtype=torch.bool, device=coords.device
    )
    matches = torch.full(
        (batch, time, maximum), -1, dtype=torch.long, device=coords.device
    )
    for sample in range(batch):
        for frame in range(time):
            selected = torch.nonzero(masks[sample, frame], as_tuple=False).flatten()
            count = selected.numel()
            packed_coords[sample, frame, :count] = coords[sample, frame, selected]
            packed_masks[sample, frame, :count] = True
            matches[sample, frame, :count] = selected
    return ProposalWindow(
        coords=packed_coords.detach(),
        masks=packed_masks,
        gt_matches=matches,
        reliable_null=torch.zeros_like(packed_masks),
    )


def _random(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
    normal: bool = False,
) -> torch.Tensor:
    function = torch.randn if normal else torch.rand
    if generator is None:
        return function(shape, device=device)
    # A CPU generator is deliberately supported for deterministic unit tests
    # and produces only tiny proposal-control tensors.
    return function(shape, generator=generator).to(device)


def mix_detector_proposals(
    gt_coords: torch.Tensor,
    gt_masks: torch.Tensor,
    detected_coords: torch.Tensor,
    detected_masks: torch.Tensor,
    detected_matches: torch.Tensor,
    *,
    predicted_ratio: float,
    spatial_shape: tuple[int, int, int],
    jitter_std_voxels: float = 0.0,
    duplicate_probability: float = 0.0,
    generator: torch.Generator | None = None,
) -> ProposalWindow:
    """Mix exact GT and detached detector proposals at the sample level.

    Detector-selected samples expose the linker to misses, false positives,
    localization error, and optional synthetic duplicates. Natural unmatched
    peaks and duplicates stay unknown under sparse supervision; coherent null
    labels are added separately by :func:`apply_source_dropout`.
    """
    if gt_coords.ndim != 4 or detected_coords.ndim != 4:
        raise ValueError("proposal coordinates must have shape (B,T,M,3)")
    if gt_coords.shape[:2] != detected_coords.shape[:2]:
        raise ValueError("GT and detector proposals must share batch/time dimensions")
    if gt_masks.shape != gt_coords.shape[:-1]:
        raise ValueError("gt_masks shape does not match gt_coords")
    if detected_masks.shape != detected_coords.shape[:-1]:
        raise ValueError("detected_masks shape does not match detected_coords")
    if detected_matches.shape != detected_masks.shape:
        raise ValueError("detected_matches shape does not match detected_masks")
    if not 0.0 <= predicted_ratio <= 1.0:
        raise ValueError("predicted_ratio must be in [0, 1]")
    if jitter_std_voxels < 0:
        raise ValueError("jitter_std_voxels must be non-negative")
    if not 0.0 <= duplicate_probability <= 1.0:
        raise ValueError("duplicate_probability must be in [0, 1]")

    device = gt_coords.device
    batch, time = gt_coords.shape[:2]
    choose_detector = _random(
        (batch,), device=device, generator=generator
    ) < predicted_ratio
    rows: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = []
    maximum = 1

    for sample in range(batch):
        sample_rows = []
        for frame in range(time):
            if bool(choose_detector[sample]):
                count = int(detected_masks[sample, frame].sum().item())
                coords = detected_coords[sample, frame, :count].detach().clone()
                matches = detected_matches[sample, frame, :count].detach().clone().long()
                reliable_null = torch.zeros(count, dtype=torch.bool, device=device)
                if count and jitter_std_voxels:
                    coords += _random(
                        tuple(coords.shape),
                        device=device,
                        generator=generator,
                        normal=True,
                    ) * jitter_std_voxels
            else:
                count = int(gt_masks[sample, frame].sum().item())
                coords = gt_coords[sample, frame, :count].detach().clone()
                matches = torch.arange(count, device=device, dtype=torch.long)
                reliable_null = torch.zeros(count, dtype=torch.bool, device=device)

            # Inject exactly once after GT/detector selection. Originals in the
            # GT arms stay at exact coordinates. A duplicate is deliberately
            # unmatched but remains unknown; NMS/suppression and null-parent
            # supervision are different tasks.
            if count and duplicate_probability:
                duplicate = _random(
                    (count,), device=device, generator=generator
                ) < duplicate_probability
                duplicate &= matches >= 0
                if duplicate.any():
                    duplicate_coords = coords[duplicate].clone()
                    if jitter_std_voxels:
                        duplicate_coords += _random(
                            tuple(duplicate_coords.shape),
                            device=device,
                            generator=generator,
                            normal=True,
                        ) * jitter_std_voxels
                    duplicate_count = int(duplicate.sum().item())
                    coords = torch.cat([coords, duplicate_coords], dim=0)
                    matches = torch.cat(
                        [
                            matches,
                            torch.full(
                                (duplicate_count,),
                                -1,
                                device=device,
                                dtype=torch.long,
                            ),
                        ]
                    )
                    reliable_null = torch.cat(
                        [
                            reliable_null,
                            torch.zeros(
                                duplicate_count,
                                dtype=torch.bool,
                                device=device,
                            ),
                        ]
                    )

            if coords.numel():
                bounds = coords.new_tensor(spatial_shape).sub_(1)
                coords.clamp_(min=0)
                coords.copy_(torch.minimum(coords, bounds))
            maximum = max(maximum, coords.shape[0])
            sample_rows.append((coords, matches, reliable_null))
        rows.append(sample_rows)

    output_coords = gt_coords.new_zeros(batch, time, maximum, 3)
    output_masks = torch.zeros(batch, time, maximum, dtype=torch.bool, device=device)
    output_matches = torch.full(
        (batch, time, maximum), -1, dtype=torch.long, device=device
    )
    output_null = torch.zeros_like(output_masks)
    for sample, sample_rows in enumerate(rows):
        for frame, (coords, matches, reliable_null) in enumerate(sample_rows):
            count = coords.shape[0]
            output_coords[sample, frame, :count] = coords
            output_masks[sample, frame, :count] = True
            output_matches[sample, frame, :count] = matches
            output_null[sample, frame, :count] = reliable_null

    return ProposalWindow(
        coords=output_coords,
        masks=output_masks,
        gt_matches=output_matches,
        reliable_null=output_null,
    )


def apply_source_dropout(
    proposals: ProposalWindow,
    gt_edges: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator | None = None,
) -> ProposalWindow:
    """Create coherent null targets by deliberately hiding annotated parents.

    The current controlled contract uses T=2. A target becomes reliable-null
    only when it has exactly one annotated GT parent and the corresponding,
    previously present proposal was selected for intentional dropout. Missing
    detector parents are never retroactively treated as null supervision.
    """
    if not 0.0 <= probability <= 1.0:
        raise ValueError("source dropout probability must be in [0, 1]")
    if probability == 0:
        return proposals
    if proposals.coords.shape[1] != 2 or gt_edges.ndim != 4 or gt_edges.shape[1] != 1:
        raise ValueError("source dropout currently requires a T=2 proposal window")
    if gt_edges.shape[0] != proposals.coords.shape[0]:
        raise ValueError("proposals and GT edges must share a batch dimension")

    device = proposals.coords.device
    rows = []
    maximum = 1
    for sample in range(proposals.coords.shape[0]):
        source_valid = proposals.masks[sample, 0]
        target_valid = proposals.masks[sample, 1]
        source_coords = proposals.coords[sample, 0, source_valid].clone()
        source_matches = proposals.gt_matches[sample, 0, source_valid].clone()
        target_coords = proposals.coords[sample, 1, target_valid].clone()
        target_matches = proposals.gt_matches[sample, 1, target_valid].clone()
        target_null = proposals.reliable_null[sample, 1, target_valid].clone()

        eligible = source_matches >= 0
        drop = _random(
            (source_matches.shape[0],), device=device, generator=generator
        ) < probability
        drop &= eligible
        deliberately_dropped = set(source_matches[drop].tolist())
        keep = ~drop
        source_coords = source_coords[keep]
        source_matches = source_matches[keep]

        transition = gt_edges[sample, 0]
        for target_index, gt_target_tensor in enumerate(target_matches):
            gt_target = int(gt_target_tensor.item())
            if gt_target < 0 or gt_target >= transition.shape[1]:
                continue
            parents = torch.nonzero(
                transition[:, gt_target] > 0, as_tuple=False
            ).flatten()
            if parents.numel() == 1 and int(parents[0].item()) in deliberately_dropped:
                target_null[target_index] = True

        maximum = max(maximum, source_coords.shape[0], target_coords.shape[0])
        rows.append(
            (source_coords, source_matches, target_coords, target_matches, target_null)
        )

    coords = proposals.coords.new_zeros(proposals.coords.shape[0], 2, maximum, 3)
    masks = torch.zeros(
        proposals.coords.shape[0], 2, maximum, dtype=torch.bool, device=device
    )
    matches = torch.full(
        (proposals.coords.shape[0], 2, maximum),
        -1,
        dtype=torch.long,
        device=device,
    )
    reliable_null = torch.zeros_like(masks)
    for sample, row in enumerate(rows):
        source_coords, source_matches, target_coords, target_matches, target_null = row
        source_count = source_coords.shape[0]
        target_count = target_coords.shape[0]
        coords[sample, 0, :source_count] = source_coords
        masks[sample, 0, :source_count] = True
        matches[sample, 0, :source_count] = source_matches
        coords[sample, 1, :target_count] = target_coords
        masks[sample, 1, :target_count] = True
        matches[sample, 1, :target_count] = target_matches
        reliable_null[sample, 1, :target_count] = target_null
    return ProposalWindow(coords, masks, matches, reliable_null)
