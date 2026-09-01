from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = COMPETITION_ROOT / "scripts/run_temporal_graph_training.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("run_temporal_graph_training", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _pair(module, source_coords, target_coords, logits):
    source = torch.tensor(source_coords, dtype=torch.float32).unsqueeze(0)
    target = torch.tensor(target_coords, dtype=torch.float32).unsqueeze(0)
    source_features = torch.arange(source.shape[1] * 2, dtype=torch.float32).reshape(
        1, source.shape[1], 2
    )
    target_features = torch.arange(target.shape[1] * 2, dtype=torch.float32).reshape(
        1, target.shape[1], 2
    )
    return module.FrozenPair(
        source_features=source_features,
        target_features=target_features,
        source_coords_um=source,
        target_coords_um=target,
        source_mask=torch.ones(source.shape[:2], dtype=torch.bool),
        target_mask=torch.ones(target.shape[:2], dtype=torch.bool),
        edge_logits=torch.tensor(logits, dtype=torch.float32).unsqueeze(0),
    )


def _triplet(module):
    previous_pair = _pair(
        module,
        [[0, 0, 0], [10, 0, 0]],
        [[1, 0, 0], [11, 0, 0]],
        [[4, 0], [0, 4]],
    )
    current_pair = _pair(
        module,
        [[1, 0, 0], [11, 0, 0]],
        [[2, 0, 0], [12, 0, 0]],
        [[3, 0], [0, 3]],
    )
    previous = module.ExtractedPair(
        pair=previous_pair,
        source_coords_grid=previous_pair.source_coords_um,
        target_coords_grid=previous_pair.target_coords_um,
        source_matches=torch.tensor([[0, 1]]),
        target_matches=torch.tensor([[0, 1]]),
    )
    current = module.ExtractedPair(
        pair=current_pair,
        source_coords_grid=current_pair.source_coords_um,
        target_coords_grid=current_pair.target_coords_um,
        source_matches=torch.tensor([[0, 1]]),
        target_matches=torch.tensor([[0, 1]]),
    )
    return previous, current


def _quadruplet(module):
    prior_pair = _pair(
        module,
        [[-1, 0, 0], [9, 0, 0]],
        [[0, 0, 0], [10, 0, 0]],
        [[4, 0], [0, 4]],
    )
    prior = module.ExtractedPair(
        pair=prior_pair,
        source_coords_grid=prior_pair.source_coords_um,
        target_coords_grid=prior_pair.target_coords_um,
        source_matches=torch.tensor([[0, 1]]),
        target_matches=torch.tensor([[0, 1]]),
    )
    previous, current = _triplet(module)
    return prior, previous, current


def test_sparse_candidate_examples_mask_unknown_targets_and_count_recall():
    module = _load_script()
    previous, current = _triplet(module)
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2, hidden_dim=4, top_k=2, radius_um=5.0
    )
    gt_edges = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    examples = module._build_sparse_candidate_examples(
        previous, current, gt_edges, graph_config
    )

    assert examples.supervised_parents == 1
    assert examples.candidate_parents == 1
    assert examples.features.shape == (1, 2, module.candidate_feature_dim(2))
    assert examples.labels.tolist() == [0]
    assert examples.valid_mask.tolist() == [[True, False]]


def test_sparse_candidate_examples_report_parent_dropped_by_radius():
    module = _load_script()
    previous, current = _triplet(module)
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2, hidden_dim=4, top_k=2, radius_um=0.5
    )
    gt_edges = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    examples = module._build_sparse_candidate_examples(
        previous, current, gt_edges, graph_config
    )

    assert examples.supervised_parents == 1
    assert examples.candidate_parents == 0
    assert examples.features.shape[0] == 0


def test_tgraph4_sparse_examples_use_cache_schema_v2_and_wider_features():
    module = _load_script()
    prior, previous, current = _quadruplet(module)
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2,
        hidden_dim=4,
        top_k=2,
        radius_um=5.0,
        graph_window_size=4,
    )
    gt_edges = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])

    examples = module._build_sparse_candidate_examples(
        previous,
        current,
        gt_edges,
        graph_config,
        prior=prior,
    )

    assert module._cache_schema_version(3) == 1
    assert module._cache_schema_version(4) == 2
    with pytest.raises(ValueError, match="only graph_window_size 3 or 4"):
        module._cache_schema_version(5)
    with pytest.raises(TypeError, match="must be an integer"):
        module._cache_schema_version(4.0)
    assert examples.features.shape == (
        1,
        2,
        module.candidate_feature_dim(2, graph_window_size=4),
    )
    assert examples.features.shape[-1] == module.candidate_feature_dim(2) + 4


def test_zero_detection_matches_are_padded_and_sparse_cache_stays_empty():
    module = _load_script()
    padded = module._pad_matches(torch.empty(0, dtype=torch.long), 1)
    assert padded.tolist() == [[-1]]

    empty_pair = module.FrozenPair(
        source_features=torch.zeros(1, 1, 2),
        target_features=torch.zeros(1, 1, 2),
        source_coords_um=torch.zeros(1, 1, 3),
        target_coords_um=torch.zeros(1, 1, 3),
        source_mask=torch.zeros(1, 1, dtype=torch.bool),
        target_mask=torch.zeros(1, 1, dtype=torch.bool),
        edge_logits=torch.zeros(1, 1, 1),
    )
    extracted = module.ExtractedPair(
        pair=empty_pair,
        source_coords_grid=torch.zeros(1, 1, 3),
        target_coords_grid=torch.zeros(1, 1, 3),
        source_matches=padded,
        target_matches=padded,
    )
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2, hidden_dim=4, top_k=2, radius_um=5.0
    )

    examples = module._build_sparse_candidate_examples(
        extracted,
        extracted,
        torch.ones(1, 1, 1),
        graph_config,
    )

    assert examples.supervised_parents == 0
    assert examples.candidate_parents == 0
    assert examples.features.shape == (0, 2, module.candidate_feature_dim(2))


def test_training_checkpoint_nests_portable_head_with_flattened_host_sha():
    module = _load_script()
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2, hidden_dim=4, top_k=2, radius_um=5.0
    )
    head = module.TemporalGraphResidualHead(graph_config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    generator = torch.Generator().manual_seed(17)
    config = {"experiment_id": "EXP-TEST"}

    payload = module._training_checkpoint(
        config=config,
        head=head,
        optimizer=optimizer,
        completed_epochs=5,
        best_score=0.75,
        graph_source_sha256="flattened-sha",
        raw_checkpoint_sha256="raw-wrapper-sha",
        cache_fingerprint="cache-sha",
        cache_manifest_sha256="manifest-sha",
        cache_schema_version=1,
        feature_schema="tgraph3-candidate-features-v1",
        feature_width=module.candidate_feature_dim(2),
        training_fingerprint="training-sha",
        history=[{}] * 5,
        loader_generator=generator,
    )

    temporal = payload["temporal_graph"]
    assert temporal["base_checkpoint_sha256"] == "flattened-sha"
    assert temporal["metadata"]["completed_epochs"] == 5
    assert temporal["metadata"]["source_raw_checkpoint_sha256"] == "raw-wrapper-sha"
    assert temporal["metadata"]["cache_manifest_sha256"] == "manifest-sha"
    assert temporal["metadata"]["feature_width"] == module.candidate_feature_dim(2)
    assert all(tensor.device.type == "cpu" for tensor in temporal["state_dict"].values())


def test_training_checkpoint_records_tgraph4_contract():
    module = _load_script()
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2,
        hidden_dim=4,
        top_k=2,
        radius_um=5.0,
        graph_window_size=4,
    )
    head = module.TemporalGraphResidualHead(graph_config)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-4)
    payload = module._training_checkpoint(
        config={"experiment_id": "EXP-T4"},
        head=head,
        optimizer=optimizer,
        completed_epochs=5,
        best_score=0.75,
        graph_source_sha256="flattened-sha",
        raw_checkpoint_sha256="raw-wrapper-sha",
        cache_fingerprint="cache-sha",
        cache_manifest_sha256="manifest-sha",
        cache_schema_version=2,
        feature_schema="tgraph4-acceleration-qbar-features-v1",
        feature_width=module.candidate_feature_dim(2, graph_window_size=4),
        training_fingerprint="training-sha",
        history=[{}] * 5,
        loader_generator=torch.Generator().manual_seed(17),
    )

    assert payload["temporal_graph"]["metadata"]["graph_window_size"] == 4
    assert payload["temporal_graph"]["config"]["graph_window_size"] == 4
    assert payload["temporal_graph"]["metadata"]["cache_schema_version"] == 2


def test_shared_cache_contract_rejects_different_host_or_candidate_geometry():
    module = _load_script()
    graph_config = module.TemporalGraphConfig(
        node_feature_dim=2,
        hidden_dim=4,
        top_k=2,
        radius_um=5.0,
        graph_window_size=4,
    )
    source = SimpleNamespace(
        weights_sha256="host-sha",
        raw_checkpoint_sha256="raw-sha",
    )
    manifest = {
        "schema_version": 2,
        "feature_schema": "tgraph4-acceleration-qbar-features-v1",
        "feature_width": module.candidate_feature_dim(2, graph_window_size=4),
        "experiment_id": "EXP-CACHE",
        "source_weights_sha256": "host-sha",
        "source_raw_checkpoint_sha256": "raw-sha",
        "graph_config": graph_config.to_dict(),
    }
    config = {"cache": {"source_experiment_id": "EXP-CACHE"}}

    assert module._validate_cache_training_contract(
        manifest,
        graph_config,
        source,
        config,
    ) == (
        2,
        "tgraph4-acceleration-qbar-features-v1",
        module.candidate_feature_dim(2, graph_window_size=4),
    )

    wrong_host = dict(manifest, source_weights_sha256="other")
    with pytest.raises(ValueError, match="frozen-host weights"):
        module._validate_cache_training_contract(
            wrong_host,
            graph_config,
            source,
            config,
        )
    wrong_graph = {**manifest, "graph_config": {**manifest["graph_config"], "top_k": 3}}
    with pytest.raises(ValueError, match="different candidate contracts"):
        module._validate_cache_training_contract(
            wrong_graph,
            graph_config,
            source,
            config,
        )


def test_real_organizer_detect_and_match_signature_accepts_cache_call_contract():
    host_repo = (
        COMPETITION_ROOT
        / "data/external/royerlab-kaggle-cell-tracking-competition"
    )
    sys.path[:0] = [str(host_repo / "src"), str(host_repo / "scripts")]
    from train_unet_transformer import detect_and_match

    signature = inspect.signature(detect_and_match)
    assert list(signature.parameters)[:4] == [
        "det_logits",
        "gt_coords",
        "mask",
        "image_shape",
    ]
    signature.bind(
        object(),
        object(),
        object(),
        (2, 64, 64, 64),
        det_threshold=4.5,
        pool_kernel_um=5.0,
        max_match_distance=7.0,
        voxel_size=(1.625, 1.625, 1.625),
        frame_index=0,
        window_size=2,
    )


def test_proposal_threshold_matches_deployment_probability_semantics():
    module = _load_script()

    expected = torch.logit(torch.tensor(0.99)).item()
    legacy = module._proposal_threshold(
        {"inference": {"detection_threshold": 0.99}}
    )
    explicit = module._proposal_threshold(
        {
            "cache": {"detection_probability_threshold": 0.99},
            "inference": {"detection_probability_threshold": 0.99},
        }
    )
    assert legacy == pytest.approx(expected)
    assert explicit == pytest.approx(expected)
    with pytest.raises(ValueError, match="not the 0.890 inference contract"):
        module._proposal_threshold(
            {"inference": {"detection_logit_threshold": 0.99}}
        )


def test_cache_detection_tta_must_equal_deployment():
    module = _load_script()

    assert module._cache_detection_tta(
        {
            "cache": {"detection_tta": True},
            "inference": {"detection_tta": True},
        }
    )


def test_shared_cache_manifest_requires_pinned_sha256(tmp_path):
    module = _load_script()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "manifest.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        module._cache_datasets(
            cache_dir,
            expected_manifest_sha256="0" * 64,
        )
    with pytest.raises(ValueError, match="must match deployment"):
        module._cache_detection_tta(
            {
                "cache": {"detection_tta": False},
                "inference": {"detection_tta": True},
            }
        )


def test_wandb_run_key_separates_smoke_from_full_run():
    module = _load_script()

    assert module._wandb_run_key("cache", 1, 2) == "cache-d1-t2"
    assert module._wandb_run_key("all", None, None) == "all-dall-tall"


def test_wandb_run_id_does_not_depend_on_wandb_util_api():
    module = _load_script()

    first = module._generate_wandb_run_id()
    second = module._generate_wandb_run_id()
    assert len(first) == 8
    assert int(first, 16) >= 0
    assert first != second


def test_missing_validation_manifest_allows_single_group_only_for_smoke(tmp_path):
    module = _load_script()
    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "44b6_example.zarr").mkdir()
    (train_dir / "44b6_example.geff").touch()
    config = {
        "seed": 17,
        "data": {
            "train_dir": str(train_dir),
            "group_delimiter": "_",
            "fold": 0,
            "validation_group": "44b6",
        },
        "cache": {"validation_manifest": str(tmp_path / "missing.json")},
        "output": {"artifact_dir": str(tmp_path / "artifacts")},
    }

    with pytest.raises(ValueError, match="at least two groups"):
        module._validation_selection(config, allow_incomplete_smoke=False)
    _, subset, names, _ = module._validation_selection(
        config, allow_incomplete_smoke=True
    )

    assert subset == "calibration"
    assert names == ["44b6_example"]
    assert (tmp_path / "artifacts/validation_split_fallback.json").is_file()
