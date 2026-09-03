"""Configuration contract for the controlled custom U-Net/nnU-Net comparison."""

from __future__ import annotations

import tomllib
from pathlib import Path

SUPPORTED_BACKBONES = {"custom_unet", "nnunet", "nnunet_temporal"}
SUPPORTED_CONTRACTS = {"legacy", "corrected_v2"}
SUPPORTED_TEMPORAL_FUSION = {"identity", "per_voxel_mha"}
SUPPORTED_NODE_PROPOSALS = {"ground_truth", "mixed_predicted"}
REQUIRED_SECTIONS = {
    "source",
    "data",
    "backbone",
    "sparse_heatmap",
    "train",
    "runtime",
    "checkpoint",
    "inference",
    "output",
}


def load_and_validate_config(path: Path) -> dict:
    """Load an A/B TOML file and fail early on contract drift."""
    with path.open("rb") as file:
        config = tomllib.load(file)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported backbone A/B config schema")
    missing = sorted(REQUIRED_SECTIONS - config.keys())
    if missing:
        raise ValueError(f"missing config sections: {missing}")

    backbone = config["backbone"]
    name = backbone.get("name")
    if name not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"unsupported backbone {name!r}; choose one of {sorted(SUPPORTED_BACKBONES)}"
        )
    if int(backbone.get("feature_dim", 0)) != 32:
        raise ValueError("the controlled A/B contract requires feature_dim=32")
    contract = backbone.get("contract", "legacy")
    if contract not in SUPPORTED_CONTRACTS:
        raise ValueError(
            f"unsupported backbone contract {contract!r}; "
            f"choose one of {sorted(SUPPORTED_CONTRACTS)}"
        )
    if contract == "corrected_v2" and name != "nnunet":
        raise ValueError("corrected_v2 currently requires backbone.name='nnunet'")
    if contract == "corrected_v2":
        common_head_seed = backbone.get("common_head_seed")
        if isinstance(common_head_seed, bool) or not isinstance(common_head_seed, int):
            raise ValueError("backbone.common_head_seed must be an integer")
        if float(backbone.get("link_candidate_radius_um", 0)) <= 0:
            raise ValueError("backbone.link_candidate_radius_um must be positive")
        link_candidate_top_k = backbone.get("link_candidate_top_k")
        if (
            isinstance(link_candidate_top_k, bool)
            or not isinstance(link_candidate_top_k, int)
            or link_candidate_top_k <= 0
        ):
            raise ValueError("backbone.link_candidate_top_k must be a positive integer")

    data = config["data"]
    if data.get("split_strategy") != "embryo-prefix":
        raise ValueError("the controlled A/B contract requires embryo-prefix splits")
    if int(data.get("window_size", 0)) != 2:
        raise ValueError("the initial controlled A/B contract requires window_size=2")
    subset_manifest = data.get("validation_subset_manifest")
    subset_name = data.get("validation_subset")
    if (subset_manifest is None) != (subset_name is None):
        raise ValueError(
            "data.validation_subset_manifest and data.validation_subset must be set together"
        )
    if subset_name is not None and subset_name not in {"calibration", "report"}:
        raise ValueError("data.validation_subset must be 'calibration' or 'report'")

    train = config["train"]
    declared_training_contract = train.get("training_contract", contract)
    if declared_training_contract != contract:
        raise ValueError("train.training_contract must match backbone.contract")
    if contract == "legacy":
        if train.get("edge_training_nodes") != "ground_truth":
            raise ValueError("the legacy A/B contract requires ground-truth node teacher forcing")
    else:
        proposal_strategy = train.get("node_proposal_strategy")
        if proposal_strategy not in SUPPORTED_NODE_PROPOSALS:
            raise ValueError(
                "train.node_proposal_strategy must be 'ground_truth' or "
                "'mixed_predicted' for corrected_v2"
            )
        curriculum = train.get("proposal_curriculum", {})
        if proposal_strategy == "mixed_predicted":
            ratios = curriculum.get("predicted_ratios", [])
            if not ratios:
                raise ValueError(
                    "mixed_predicted requires "
                    "train.proposal_curriculum.predicted_ratios"
                )
            if any(not 0 <= float(value) <= 1 for value in ratios):
                raise ValueError("proposal predicted ratios must be in [0, 1]")
        if float(curriculum.get("jitter_std_voxels", 0.0)) < 0:
            raise ValueError("proposal jitter_std_voxels must be non-negative")
        duplicate_probability = float(curriculum.get("duplicate_probability", 0.0))
        if not 0 <= duplicate_probability <= 1:
            raise ValueError("proposal duplicate_probability must be in [0, 1]")
        source_dropout_probability = float(
            curriculum.get("source_dropout_probability", 0.0)
        )
        if not 0 <= source_dropout_probability <= 1:
            raise ValueError("proposal source_dropout_probability must be in [0, 1]")
        proposal_threshold = float(
            curriculum.get("detection_threshold", train.get("validation_det_threshold", 0))
        )
        if not 0 < proposal_threshold < 1:
            raise ValueError("proposal detection_threshold must be in (0, 1)")
        if float(curriculum.get("max_match_distance_um", 5.0)) <= 0:
            raise ValueError("proposal max_match_distance_um must be positive")
        if int(curriculum.get("max_proposals_per_frame", 96)) <= 0:
            raise ValueError("proposal max_proposals_per_frame must be positive")
    threshold = float(train.get("validation_det_threshold", 0))
    if not 0 < threshold < 1:
        raise ValueError("train.validation_det_threshold must be in (0, 1)")
    validation_every = int(train.get("validation_every_epochs", 1))
    if validation_every <= 0:
        raise ValueError("train.validation_every_epochs must be positive")
    inference_contract = config["inference"].get("model_api", contract)
    if inference_contract != contract:
        raise ValueError("inference.model_api must match backbone.contract")

    heatmap = config["sparse_heatmap"]
    for key in ("sigma", "positive_threshold", "background_quantile"):
        if not 0 < float(heatmap[key]) <= 1:
            raise ValueError(f"sparse_heatmap.{key} must be in (0, 1]")
    for key in ("positive_weight", "background_weight", "unknown_weight"):
        if float(heatmap[key]) < 0:
            raise ValueError(f"sparse_heatmap.{key} must be non-negative")

    nnunet = backbone.get("nnunet", {})
    if name in {"nnunet", "nnunet_temporal"}:
        n_stages = int(nnunet.get("n_stages", 0))
        lengths = {
            "features_per_stage": n_stages,
            "kernel_sizes": n_stages,
            "strides": n_stages,
            "n_conv_per_stage": n_stages,
            "n_conv_per_stage_decoder": n_stages - 1,
        }
        for key, expected in lengths.items():
            if len(nnunet.get(key, [])) != expected:
                raise ValueError(f"backbone.nnunet.{key} must contain {expected} entries")
    if name == "nnunet_temporal":
        temporal = backbone.get("temporal", {})
        stages = [int(value) for value in temporal.get("stages", [])]
        if not stages:
            raise ValueError("backbone.temporal.stages must not be empty")
        if any(stage < 0 or stage >= int(nnunet["n_stages"]) for stage in stages):
            raise ValueError("backbone.temporal.stages contains an invalid encoder stage")
        heads = int(temporal.get("heads", 0))
        if heads <= 0:
            raise ValueError("backbone.temporal.heads must be positive")
        for stage in stages:
            if int(nnunet["features_per_stage"][stage]) % heads:
                raise ValueError("temporal stage channels must be divisible by heads")
    if contract == "corrected_v2":
        temporal_fusion = backbone.get("temporal_fusion", {})
        mode = temporal_fusion.get("mode")
        if mode not in SUPPORTED_TEMPORAL_FUSION:
            raise ValueError(
                "backbone.temporal_fusion.mode must be 'identity' or "
                "'per_voxel_mha'"
            )
        if mode == "per_voxel_mha":
            stages = [int(value) for value in temporal_fusion.get("stages", [])]
            if not stages:
                raise ValueError("temporal_fusion.stages must not be empty")
            if any(stage < 0 or stage >= int(nnunet["n_stages"]) for stage in stages):
                raise ValueError("temporal_fusion.stages contains an invalid encoder stage")
            heads = int(temporal_fusion.get("heads", 0))
            if heads <= 0:
                raise ValueError("temporal_fusion.heads must be positive")
            for stage in stages:
                if int(nnunet["features_per_stage"][stage]) % heads:
                    raise ValueError("temporal fusion stage channels must be divisible by heads")
    return config


def comparison_signature(config: dict) -> dict:
    """Return fields that must remain equal between the paired A/B configs."""
    return {
        "seed": config["seed"],
        "source": config["source"],
        "data": config["data"],
        "feature_dim": config["backbone"]["feature_dim"],
        "sparse_heatmap": config["sparse_heatmap"],
        "train": {
            key: value
            for key, value in config["train"].items()
            if key != "method"
        },
        "runtime": config["runtime"],
        "checkpoint": config["checkpoint"],
        "inference": config["inference"],
    }
