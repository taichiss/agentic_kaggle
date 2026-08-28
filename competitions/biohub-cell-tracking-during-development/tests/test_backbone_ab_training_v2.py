"""Corrected-v2 proposal, supervision, and training contract tests."""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

COMPETITION_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = COMPETITION_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from backbone_ab import training as training_module  # noqa: E402
from backbone_ab.contracts import EncodedWindow, LinkOutput  # noqa: E402
from backbone_ab.dataset import DeterministicAugmentationDataset  # noqa: E402
from backbone_ab.losses import (  # noqa: E402
    DIVISION,
    UNKNOWN_DIVISION,
    WEAK_NON_DIVISION,
    ParentSupervision,
    build_division_states,
    build_parent_supervision,
    candidate_parent_counts,
    mask_non_candidate_parents,
    parent_classification_loss,
    three_state_division_loss,
)
from backbone_ab.proposals import (  # noqa: E402
    apply_source_dropout,
    ground_truth_proposals,
    mix_detector_proposals,
    predicted_ratio_for_epoch,
)
from backbone_ab.training import (  # noqa: E402
    _detect_and_match_window,
    evaluate_predicted_nodes,
    train_epoch,
)


def _load_training_script():
    module_path = COMPETITION_ROOT / "scripts/run_backbone_ab_training.py"
    spec = importlib.util.spec_from_file_location("backbone_training_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_curriculum_uses_last_ratio_after_configured_epochs() -> None:
    config = {"predicted_ratios": [0.0, 0.25, 0.5, 0.75, 0.75]}

    assert predicted_ratio_for_epoch("ground_truth", {}, 4) == 0.0
    assert predicted_ratio_for_epoch("mixed_predicted", config, 0) == 0.0
    assert predicted_ratio_for_epoch("mixed_predicted", config, 3) == 0.75
    assert predicted_ratio_for_epoch("mixed_predicted", config, 49) == 0.75


def test_parent_supervision_masks_sparse_unknown_and_missing_parent() -> None:
    gt_edges = torch.zeros(1, 3, 4)
    gt_edges[0, 0, 0] = 1
    gt_edges[0, 1, 1] = 1
    source_matches = torch.tensor([[1, 0, -1]])
    source_mask = torch.tensor([[True, True, True]])
    target_matches = torch.tensor([[0, 1, 2, -1]])
    target_mask = torch.tensor([[True, True, True, True]])
    reliable_null = torch.tensor([[False, False, False, True]])

    supervision = build_parent_supervision(
        gt_edges,
        source_matches,
        target_matches,
        source_mask,
        target_mask,
        reliable_null,
    )

    # Reordered proposal indices are preserved, a zero-parent sparse label is
    # unknown, and only the synthetic duplicate receives the null class.
    assert supervision.classes.tolist() == [[1, 0, 3, 3]]
    assert supervision.mask.tolist() == [[True, True, False, True]]

    missing_parent = build_parent_supervision(
        gt_edges,
        torch.tensor([[0]]),
        torch.tensor([[1]]),
        torch.tensor([[True]]),
        torch.tensor([[True]]),
        torch.tensor([[False]]),
    )
    assert not missing_parent.mask.any()


def test_division_loss_has_three_states_and_separate_normalisation() -> None:
    gt_edges = torch.zeros(1, 3, 3)
    gt_edges[0, 0, 0] = 1
    gt_edges[0, 1, 1:] = 1
    states = build_division_states(
        gt_edges,
        torch.tensor([[0, 1, 2, -1]]),
        torch.tensor([[True, True, True, True]]),
    )

    assert states.tolist() == [
        [WEAK_NON_DIVISION, DIVISION, UNKNOWN_DIVISION, UNKNOWN_DIVISION]
    ]
    loss = three_state_division_loss(
        torch.zeros_like(states, dtype=torch.float32),
        states,
        weak_negative_weight=0.1,
    )
    assert float(loss) == pytest.approx(1.1 * torch.log(torch.tensor(2.0)).item())


def test_parent_loss_uses_target_major_parent_classes() -> None:
    logits = torch.tensor(
        [[[0.0, 4.0, -1.0], [-1.0, 0.0, 4.0]]],
        requires_grad=True,
    )
    supervision = ParentSupervision(
        classes=torch.tensor([[1, 2]]),
        mask=torch.tensor([[True, True]]),
    )

    loss = parent_classification_loss(logits, supervision)
    loss.backward()

    assert float(loss.detach()) < 0.05
    assert logits.grad is not None


def test_parent_outside_candidate_graph_is_masked_but_null_remains() -> None:
    supervision = ParentSupervision(
        classes=torch.tensor([[0, 1, 2]]),
        mask=torch.tensor([[True, True, True]]),
    )
    candidate_mask = torch.tensor(
        [[[False, False, False], [False, True, False]]]
    )

    masked = mask_non_candidate_parents(supervision, candidate_mask)

    # Parent 0 is unavailable, parent 1 is available, and class 2 is null.
    assert masked.mask.tolist() == [[False, True, True]]
    assert candidate_parent_counts(supervision, candidate_mask) == (1, 2)


def test_mixed_proposals_detach_predictions_and_keep_duplicates_unknown() -> None:
    gt_coords = torch.tensor([[[[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]]])
    gt_masks = torch.tensor([[[True, True]]])
    detected = gt_coords.clone().requires_grad_()
    detected_masks = gt_masks.clone()
    detected_matches = torch.tensor([[[0, 1]]])

    mixed = mix_detector_proposals(
        gt_coords,
        gt_masks,
        detected,
        detected_masks,
        detected_matches,
        predicted_ratio=1.0,
        spatial_shape=(4, 4, 4),
        duplicate_probability=1.0,
        generator=torch.Generator().manual_seed(7),
    )

    assert mixed.coords.shape == (1, 1, 4, 3)
    assert mixed.gt_matches.tolist() == [[[0, 1, -1, -1]]]
    assert not mixed.reliable_null.any()
    assert not mixed.coords.requires_grad

    exact = ground_truth_proposals(gt_coords, gt_masks)
    assert exact.gt_matches.tolist() == [[[0, 1]]]
    assert not exact.reliable_null.any()

    gt_with_null = mix_detector_proposals(
        gt_coords,
        gt_masks,
        detected.detach(),
        detected_masks,
        detected_matches,
        predicted_ratio=0.0,
        spatial_shape=(4, 4, 4),
        jitter_std_voxels=0.5,
        duplicate_probability=1.0,
        generator=torch.Generator().manual_seed(7),
    )
    assert gt_with_null.gt_matches.tolist() == [[[0, 1, -1, -1]]]
    assert not gt_with_null.reliable_null.any()


def test_only_intentionally_dropped_parent_creates_reliable_null() -> None:
    proposals = ground_truth_proposals(
        torch.tensor([[[[1.0, 1.0, 1.0]], [[2.0, 1.0, 1.0]]]]),
        torch.tensor([[[True], [True]]]),
    )
    gt_edges = torch.ones(1, 1, 1, 1)

    dropped = apply_source_dropout(
        proposals,
        gt_edges,
        probability=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert not dropped.masks[0, 0].any()
    assert dropped.reliable_null[0, 1, 0]

    # An already missing detector parent is sparse-unknown, not synthetic null.
    naturally_missing = type(proposals)(
        coords=proposals.coords,
        masks=torch.tensor([[[False], [True]]]),
        gt_matches=torch.tensor([[[-1], [0]]]),
        reliable_null=torch.zeros_like(proposals.masks),
    )
    unchanged = apply_source_dropout(
        naturally_missing,
        gt_edges,
        probability=1.0,
        generator=torch.Generator().manual_seed(3),
    )
    assert not unchanged.reliable_null.any()


def test_detection_caps_plateau_peaks_before_pinned_host_matching() -> None:
    class _PinnedHostMustNotRun:
        @staticmethod
        def detect_and_match(*args, **kwargs):
            raise AssertionError("unbounded host matcher was called")

    detected, masks, matches = _detect_and_match_window(
        [torch.ones(1, 1, 5, 5, 5)],
        torch.tensor([[[[2.0, 2.0, 2.0]]]]),
        torch.tensor([[[True]]]),
        torch.tensor([[1, 5, 5, 5]]),
        torch.tensor([[1.0, 1.0, 1.0]]),
        _PinnedHostMustNotRun(),
        probability_threshold=0.1,
        pool_kernel_um=3.0,
        max_match_distance_um=5.0,
        max_proposals_per_frame=7,
    )

    assert detected.shape == (1, 1, 7, 3)
    assert int(masks.sum()) == 7
    assert int((matches >= 0).sum()) == 1


class _TinyCorrectedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))
        self.build_nodes_called = False

    def encode_window(self, images: torch.Tensor) -> EncodedWindow:
        batch, time, *spatial = images.shape
        features = self.scale.expand(batch, time, 2, *spatial)
        logits = self.scale.expand(batch, time, 1, *spatial)
        return EncodedWindow(features, logits)

    def build_nodes(
        self,
        encoded,
        coords,
        masks,
        image_shape,
        voxel_size,
        frame_indices=None,
        delta_t=1.0,
    ):
        del encoded, image_shape, voxel_size, frame_indices, delta_t
        self.build_nodes_called = True
        return tuple(
            SimpleNamespace(valid_mask=masks[:, frame], coords=coords[:, frame])
            for frame in range(coords.shape[1])
        )

    def link_pair(self, source, target) -> LinkOutput:
        batch, sources = source.valid_mask.shape
        targets = target.valid_mask.shape[1]
        return LinkOutput(
            edge_logits=self.scale.expand(batch, sources, targets),
            null_parent_logits=self.scale.expand(batch, targets),
            division_logits=self.scale.expand(batch, sources),
        )


class _CandidateMetricModel(_TinyCorrectedModel):
    def encode_window(self, images: torch.Tensor) -> EncodedWindow:
        batch, time, *spatial = images.shape
        features = torch.zeros(batch, time, 2, *spatial)
        logits = torch.full((batch, time, 1, *spatial), -10.0)
        logits[:, :, 0, 1, 1, 1] = 10.0
        return EncodedWindow(features, logits)

    def link_pair(self, source, target) -> LinkOutput:
        batch, sources = source.valid_mask.shape
        targets = target.valid_mask.shape[1]
        candidate_mask = torch.zeros(
            batch, sources, targets, dtype=torch.bool
        )
        return LinkOutput(
            edge_logits=torch.full((batch, sources, targets), -1e4),
            null_parent_logits=torch.zeros(batch, targets),
            division_logits=torch.zeros(batch, sources),
            candidate_mask=candidate_mask,
        )


def test_corrected_training_builds_nodes_after_batch_augmentation() -> None:
    model = _TinyCorrectedModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    batch = {
        "imgs": torch.zeros(1, 2, 4, 4, 4),
        "coords": torch.tensor(
            [[[[1.25, 1.0, 1.0], [0.0, 0.0, 0.0]], [[1.5, 1.0, 1.0], [0, 0, 0]]]]
        ),
        "masks": torch.tensor([[[True, False], [True, False]]]),
        "targets": torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]]),
        "image_shape": torch.tensor([[2, 4, 4, 4]]),
        "voxel_size": torch.tensor([[1.0, 2.0, 3.0]]),
    }
    heatmap = {
        "sigma": 1.0,
        "positive_threshold": 0.05,
        "background_quantile": 0.4,
        "positive_weight": 1.0,
        "background_weight": 1.0,
        "unknown_weight": 0.1,
    }

    losses = train_epoch(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        SimpleNamespace(),
        heatmap,
        1.0,
        5.0,
        None,
        division_loss_weight=0.1,
        contract="corrected_v2",
        node_proposal_strategy="ground_truth",
        proposal_curriculum={
            "duplicate_probability": 1.0,
            "jitter_std_voxels": 0.5,
            "source_dropout_probability": 1.0,
        },
        epoch_index=0,
    )

    assert model.build_nodes_called
    assert all(torch.isfinite(torch.tensor(losses)))


def test_corrected_validation_reports_candidate_recall_without_changing_tuple() -> None:
    batch = {
        "imgs": torch.zeros(1, 2, 4, 4, 4),
        "coords": torch.tensor([[[[1.0, 1.0, 1.0]], [[1.0, 1.0, 1.0]]]]),
        "masks": torch.tensor([[[True], [True]]]),
        "targets": torch.ones(1, 1, 1, 1),
        "image_shape": torch.tensor([[2, 4, 4, 4]]),
        "voxel_size": torch.tensor([[1.0, 1.0, 1.0]]),
    }
    metrics: dict[str, float | None] = {}

    result = evaluate_predicted_nodes(
        _CandidateMetricModel(),
        [batch],
        torch.device("cpu"),
        SimpleNamespace(),
        3.0,
        0.5,
        contract="corrected_v2",
        metrics_out=metrics,
    )

    assert len(result) == 3
    assert metrics == {"candidate_recall": 0.0}
    assert result[1] == 0.0


def test_legacy_validation_dispatch_keeps_three_tuple_and_none_metric(
    monkeypatch,
) -> None:
    expected = (1.0, 0.5, 0.25)

    def fake_legacy(*args):
        assert len(args) == 6
        return expected

    monkeypatch.setattr(
        training_module, "_evaluate_predicted_nodes_legacy", fake_legacy
    )
    metrics: dict[str, float | None] = {}

    actual = training_module.evaluate_predicted_nodes(
        object(),
        [],
        torch.device("cpu"),
        object(),
        5.0,
        0.5,
        contract="legacy",
        metrics_out=metrics,
    )

    assert actual == expected
    assert metrics == {"candidate_recall": None}


def test_resume_fingerprint_allows_only_epoch_extension() -> None:
    module = _load_training_script()
    first = {"train": {"epochs": 5, "learning_rate": 1e-4}, "seed": 7}
    extended = {"train": {"epochs": 50, "learning_rate": 1e-4}, "seed": 7}
    changed = {"train": {"epochs": 50, "learning_rate": 2e-4}, "seed": 7}

    assert module._resume_fingerprint(first) == module._resume_fingerprint(extended)
    assert module._resume_fingerprint(first) != module._resume_fingerprint(changed)


def test_atomic_checkpoint_save_replaces_complete_file(tmp_path: Path) -> None:
    module = _load_training_script()
    destination = tmp_path / "last_checkpoint.pth"
    destination.write_bytes(b"old-complete-checkpoint")

    class _ByteTorch:
        @staticmethod
        def save(payload, stream) -> None:
            stream.write(payload)

    module._atomic_torch_save(_ByteTorch, b"new-complete-checkpoint", destination)

    assert destination.read_bytes() == b"new-complete-checkpoint"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_atomic_checkpoint_is_loadable_by_torch(tmp_path: Path) -> None:
    module = _load_training_script()
    destination = tmp_path / "best_model.pth"
    payload = {"model_state_dict": {"weight": torch.arange(3)}}

    module._atomic_torch_save(torch, payload, destination)

    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["model_state_dict"]["weight"], torch.arange(3))


@pytest.mark.parametrize("failure_stage", ["save", "replace"])
def test_atomic_checkpoint_failure_preserves_existing_and_cleans_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_stage: str,
) -> None:
    module = _load_training_script()
    destination = tmp_path / "checkpoint_epoch_0005.pth"
    original = b"known-good-checkpoint"
    destination.write_bytes(original)

    class _FailingTorch:
        @staticmethod
        def save(payload, stream) -> None:
            del payload
            stream.write(b"partial-new-data")
            if failure_stage == "save":
                raise RuntimeError("simulated save failure")

    if failure_stage == "replace":
        monkeypatch.setattr(
            module.os,
            "replace",
            lambda *_: (_ for _ in ()).throw(OSError("simulated replace failure")),
        )

    with pytest.raises((OSError, RuntimeError), match="simulated"):
        module._atomic_torch_save(_FailingTorch, object(), destination)

    assert destination.read_bytes() == original
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_resume_falls_back_from_corrupt_incompatible_and_mismatched_candidates(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_training_script()
    last = tmp_path / "last_checkpoint.pth"
    newest = tmp_path / "checkpoint_epoch_0015.pth"
    incompatible = tmp_path / "checkpoint_epoch_0010.pth"
    fallback = tmp_path / "checkpoint_epoch_0005.pth"
    last.write_bytes(b"corrupt-but-retained")
    newest.write_bytes(b"valid-container-wrong-config")
    incompatible.write_bytes(b"metadata-valid-model-incompatible")
    fallback.write_bytes(b"valid-container-correct-config")

    def metadata(completed_epochs: int, fingerprint: str) -> dict:
        return {
            "experiment_id": "EXP-TEST",
            "resume_fingerprint": fingerprint,
            "validation_subset_manifest_sha256": "manifest-hash",
            "optimizer_state_dict": {"state": {}},
            "completed_epochs": completed_epochs,
            "best_score": 0.5,
            "rng_state": {"python": object()},
        }

    def fake_loader(path: Path, *, map_location: str):
        assert map_location == "cpu"
        if path == last:
            raise EOFError("truncated archive")
        if path == newest:
            return SimpleNamespace(
                metadata=metadata(15, "different-config"),
                state_dict={"weight": object()},
            )
        if path == incompatible:
            return SimpleNamespace(
                metadata=metadata(10, "expected-config"),
                state_dict={"wrong.weight": object()},
            )
        return SimpleNamespace(
            metadata=metadata(5, "expected-config"),
            state_dict={"weight": object()},
        )

    def restore_checkpoint(loaded) -> None:
        if "wrong.weight" in loaded.state_dict:
            raise RuntimeError("simulated model state mismatch")

    selected = module._select_resume_checkpoint(
        tmp_path,
        experiment_id="EXP-TEST",
        contract="corrected_v2",
        resume_fingerprint="expected-config",
        validation_manifest_hash="manifest-hash",
        target_epochs=50,
        checkpoint_loader=fake_loader,
        restore_checkpoint=restore_checkpoint,
    )

    assert selected is not None and selected[0] == fallback
    assert last.read_bytes() == b"corrupt-but-retained"
    assert newest.exists() and incompatible.exists() and fallback.exists()
    errors = capsys.readouterr().err
    assert "last_checkpoint.pth" in errors
    assert "checkpoint_epoch_0015.pth" in errors
    assert "checkpoint_epoch_0010.pth" in errors


def test_resume_selects_newer_periodic_over_stale_valid_last(tmp_path: Path) -> None:
    module = _load_training_script()
    last = tmp_path / "last_checkpoint.pth"
    newer = tmp_path / "checkpoint_epoch_0010.pth"
    last.touch()
    newer.touch()

    def fake_loader(path: Path, *, map_location: str):
        del map_location
        completed_epochs = 5 if path == last else 10
        return SimpleNamespace(
            metadata={
                "experiment_id": "EXP-TEST",
                "resume_fingerprint": "fingerprint",
                "validation_subset_manifest_sha256": None,
                "optimizer_state_dict": {"state": {}},
                "completed_epochs": completed_epochs,
                "best_score": 0.5,
                "rng_state": {"python": object()},
            },
            state_dict={"weight": completed_epochs},
        )

    selected = module._select_resume_checkpoint(
        tmp_path,
        experiment_id="EXP-TEST",
        contract="corrected_v2",
        resume_fingerprint="fingerprint",
        validation_manifest_hash=None,
        target_epochs=50,
        checkpoint_loader=fake_loader,
    )

    assert selected is not None and selected[0] == newer


def test_resume_reapplies_selection_after_partial_candidate_restore_failure(
    tmp_path: Path,
) -> None:
    module = _load_training_script()
    last = tmp_path / "last_checkpoint.pth"
    newer = tmp_path / "checkpoint_epoch_0010.pth"
    last.touch()
    newer.touch()

    def fake_loader(path: Path, *, map_location: str):
        del map_location
        completed_epochs = 5 if path == last else 10
        value = "last" if path == last else "newer"
        return SimpleNamespace(
            metadata={
                "experiment_id": "EXP-TEST",
                "resume_fingerprint": "fingerprint",
                "validation_subset_manifest_sha256": None,
                "optimizer_state_dict": {"state": {}},
                "completed_epochs": completed_epochs,
                "best_score": 0.5,
                "rng_state": {"python": object()},
            },
            state_dict={"value": value},
        )

    runtime_state = {"model": None, "optimizer": None}
    restore_calls: list[str] = []

    def restore_checkpoint(loaded) -> None:
        value = loaded.state_dict["value"]
        runtime_state["model"] = value
        restore_calls.append(value)
        if value == "newer":
            raise RuntimeError("simulated partial restore failure")
        runtime_state["optimizer"] = value

    selected = module._select_resume_checkpoint(
        tmp_path,
        experiment_id="EXP-TEST",
        contract="corrected_v2",
        resume_fingerprint="fingerprint",
        validation_manifest_hash=None,
        target_epochs=50,
        checkpoint_loader=fake_loader,
        restore_checkpoint=restore_checkpoint,
    )

    assert selected is not None and selected[0] == last
    assert runtime_state == {"model": "last", "optimizer": "last"}
    assert restore_calls == ["last", "newer", "last"]


def test_resume_rejects_checkpoint_beyond_requested_target(tmp_path: Path) -> None:
    module = _load_training_script()
    last = tmp_path / "last_checkpoint.pth"
    last.touch()

    def fake_loader(path: Path, *, map_location: str):
        del path, map_location
        return SimpleNamespace(
            metadata={
                "experiment_id": "EXP-TEST",
                "resume_fingerprint": "fingerprint",
                "validation_subset_manifest_sha256": None,
                "optimizer_state_dict": {"state": {}},
                "completed_epochs": 50,
                "best_score": 0.5,
                "rng_state": {"python": object()},
            },
            state_dict={"weight": object()},
        )

    with pytest.raises(RuntimeError, match="exceeds target 5"):
        module._select_resume_checkpoint(
            tmp_path,
            experiment_id="EXP-TEST",
            contract="corrected_v2",
            resume_fingerprint="fingerprint",
            validation_manifest_hash=None,
            target_epochs=5,
            checkpoint_loader=fake_loader,
        )

    assert last.exists()


def test_resume_refuses_contraction_instead_of_falling_back_to_older_periodic(
    tmp_path: Path,
) -> None:
    module = _load_training_script()
    last = tmp_path / "last_checkpoint.pth"
    periodic = tmp_path / "checkpoint_epoch_0005.pth"
    history = tmp_path / "epoch_history.jsonl"
    last.touch()
    periodic.touch()
    original_history = b'{"epoch": 50}\n'
    history.write_bytes(original_history)

    def fake_loader(path: Path, *, map_location: str):
        del map_location
        completed_epochs = 50 if path == last else 5
        return SimpleNamespace(
            metadata={
                "experiment_id": "EXP-TEST",
                "resume_fingerprint": "fingerprint",
                "validation_subset_manifest_sha256": None,
                "optimizer_state_dict": {"state": {}},
                "completed_epochs": completed_epochs,
                "best_score": 0.5,
                "rng_state": {"python": object()},
            },
            state_dict={"weight": completed_epochs},
        )

    with pytest.raises(RuntimeError, match="refusing to contract.*50.*target 5"):
        module._select_resume_checkpoint(
            tmp_path,
            experiment_id="EXP-TEST",
            contract="corrected_v2",
            resume_fingerprint="fingerprint",
            validation_manifest_hash=None,
            target_epochs=5,
            checkpoint_loader=fake_loader,
        )

    assert last.exists() and periodic.exists()
    assert history.read_bytes() == original_history


def test_resume_failure_preserves_existing_run_provenance(tmp_path: Path) -> None:
    module = _load_training_script()
    artifact_dir = tmp_path / "artifacts" / "EXP-TEST"
    artifact_dir.mkdir(parents=True)
    saved_config = artifact_dir / "experiment.toml"
    saved_splits = artifact_dir / "dataset_splits.json"
    saved_config.write_bytes(b"epochs = 50\n")
    saved_splits.write_bytes(b'[{"fold": "original"}]\n')
    config_path = tmp_path / "contracting.toml"
    config_path.write_bytes(b"epochs = 5\n")

    def reject_contraction():
        raise RuntimeError("refusing to contract an existing training run")

    with pytest.raises(RuntimeError, match="refusing to contract"):
        module._select_resume_then_persist_provenance(
            artifact_dir,
            config_path,
            [{"fold": "replacement"}],
            resume_selector=reject_contraction,
        )

    assert saved_config.read_bytes() == b"epochs = 50\n"
    assert saved_splits.read_bytes() == b'[{"fold": "original"}]\n'


def test_validation_checkpointing_uses_only_manifest_subset(tmp_path: Path) -> None:
    module = _load_training_script()
    manifest = tmp_path / "validation_split.json"
    manifest.write_text(
        '{"schema_version":1,"calibration":["d1"],"report":["d2"]}\n',
        encoding="utf-8",
    )
    config = {
        "data": {
            "validation_subset_manifest": "validation_split.json",
            "validation_subset": "calibration",
        }
    }

    selected, metadata = module._resolve_validation_subset(
        config,
        {"test": ["d1", "d2"]},
        competition_root=tmp_path,
    )

    assert selected == ["d1"]
    assert metadata is not None
    assert metadata["subset"] == "calibration"
    assert len(metadata["manifest_sha256"]) == 64

    manifest.write_text(
        '{"schema_version":1,"calibration":["outside"],"report":["d2"]}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside the selected fold"):
        module._resolve_validation_subset(
            config,
            {"test": ["d1", "d2"]},
            competition_root=tmp_path,
        )


def test_augmentation_is_deterministic_by_seed_epoch_and_item() -> None:
    class _BaseDataset(torch.utils.data.Dataset):
        def __len__(self) -> int:
            return 2

        def __getitem__(self, index: int):
            return {
                "imgs": torch.zeros(2, 2, 2, 2),
                "coords": torch.zeros(2, 1, 3),
                "masks": torch.ones(2, 1, dtype=torch.bool),
                "index": index,
            }

    def random_marker(images, coords, masks, *, rng):
        images = images + float(rng.uniform())
        coords = coords.clone()
        coords[..., 0] = float(rng.uniform())
        return images, coords, masks

    left = DeterministicAugmentationDataset(
        _BaseDataset(), [random_marker], seed=20260827
    )
    right = DeterministicAugmentationDataset(
        _BaseDataset(), [random_marker], seed=20260827
    )

    left.set_epoch(3)
    right.set_epoch(3)
    assert torch.equal(left[1]["imgs"], right[1]["imgs"])
    assert torch.equal(left[1]["coords"], right[1]["coords"])

    right.set_epoch(4)
    assert not torch.equal(left[1]["imgs"], right[1]["imgs"])
    with pytest.raises(ValueError, match="non-negative"):
        right.set_epoch(-1)


def test_rng_checkpoint_restores_python_numpy_torch_and_loader() -> None:
    module = _load_training_script()
    random.seed(13)
    np.random.seed(13)
    torch.manual_seed(13)
    loader_generator = torch.Generator().manual_seed(13)
    state = module._capture_rng_state(torch, np, loader_generator)
    expected = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=loader_generator)),
    )

    module._restore_rng_state(state, torch, np, loader_generator)
    actual = (
        random.random(),
        float(np.random.random()),
        float(torch.rand(())),
        float(torch.rand((), generator=loader_generator)),
    )

    assert actual == expected
