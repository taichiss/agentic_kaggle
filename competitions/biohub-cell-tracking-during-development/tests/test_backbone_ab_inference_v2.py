from __future__ import annotations

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

COMPETITION_ROOT = Path(__file__).parents[1]
SOURCE_ROOT = COMPETITION_ROOT / "src"
sys.path.insert(0, str(SOURCE_ROOT))

from backbone_ab import inference as inference_v2  # noqa: E402
from backbone_ab.checkpointing import (  # noqa: E402
    DecoderProfile,
    InferenceProfile,
    load_inference_profile,
    normalise_checkpoint_payload,
    write_inference_profile,
)
from backbone_ab.contracts import EncodedWindow, LinkOutput, NodeBatch  # noqa: E402
from backbone_ab.decoder import GraphDecoder  # noqa: E402
from backbone_ab.inference import (  # noqa: E402
    detect_window_nodes,
    predict_dataset_corrected,
    rescale_output_coordinates,
)


def _load_inference_script():
    module_path = COMPETITION_ROOT / "scripts/run_backbone_ab_inference.py"
    spec = importlib.util.spec_from_file_location("backbone_inference_script", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _profile() -> InferenceProfile:
    return InferenceProfile(
        experiment_id="EXP-TEST",
        model_api="corrected_v2",
        checkpoint_sha256="a" * 64,
        experiment_config_sha256="b" * 64,
        source_revision="revision",
        downsample=(1, 4, 4),
        window_size=2,
        detection_threshold=0.03,
        detection_tta=True,
        pool_kernel_um=5.0,
        edge_activation="softmax",
        edge_threshold=0.15,
        max_detections_per_frame=512,
        decoder=DecoderProfile(
            max_parents_per_node=1,
            max_children_per_node=2,
            null_parent_threshold=0.5,
            division_threshold=0.7,
        ),
    )


def test_inference_profile_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "inference_profile.json"
    profile = _profile()

    write_inference_profile(path, profile)
    write_inference_profile(path, profile)

    assert load_inference_profile(path) == profile
    changed = InferenceProfile.from_dict(
        {**profile.to_dict(), "detection_threshold": 0.04}
    )
    with pytest.raises(FileExistsError, match="immutable"):
        write_inference_profile(path, changed)


def test_inference_profile_rejects_tampered_hash(tmp_path: Path) -> None:
    path = tmp_path / "inference_profile.json"
    write_inference_profile(path, _profile())
    contents = path.read_text(encoding="utf-8").replace(
        '"profile_sha256": "', '"profile_sha256": "0'
    )
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="hash mismatch"):
        load_inference_profile(path)


def test_checkpoint_payload_supports_raw_and_wrapped_state_dicts() -> None:
    raw = {"layer.weight": object()}
    wrapped = {
        "model_state_dict": raw,
        "completed_epochs": 5,
        "optimizer_state_dict": {"state": {}},
    }

    raw_result = normalise_checkpoint_payload(raw)
    wrapped_result = normalise_checkpoint_payload(wrapped)

    assert raw_result.state_dict is raw
    assert raw_result.source_format == "raw_state_dict"
    assert wrapped_result.state_dict is raw
    assert wrapped_result.metadata["completed_epochs"] == 5
    assert wrapped_result.source_format == "wrapped:model_state_dict"


def test_inference_converts_only_new_isolated_run_outputs(tmp_path: Path) -> None:
    module = _load_inference_script()
    workspace = tmp_path / "workspace"
    artifact_dir = workspace / "artifacts" / "EXP-TEST"
    legacy_dir = artifact_dir / "predicted"
    legacy_dir.mkdir(parents=True)
    stale = legacy_dir / "stale.geff"
    stale.mkdir()
    unrelated = artifact_dir / "do-not-delete.txt"
    unrelated.write_text("keep", encoding="utf-8")

    run_dir = module._create_prediction_run_dir(
        artifact_dir,
        allowed_root=workspace,
    )
    fresh = run_dir / "fresh.geff"
    fresh.mkdir()
    submission = workspace / "submissions" / "submission.csv"
    converted_from: list[Path] = []

    def fake_geffs_to_csv(in_dir: Path, csv_path: Path) -> None:
        converted_from.append(Path(in_dir))
        csv_path.parent.mkdir(parents=True)
        csv_path.write_text("current-run-only\n", encoding="utf-8")

    module._convert_current_prediction_run(
        fake_geffs_to_csv,
        run_dir,
        submission,
        ["fresh"],
    )

    assert converted_from == [run_dir]
    assert submission.read_text(encoding="utf-8") == "current-run-only\n"
    assert stale.is_dir()
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert not (run_dir / "stale.geff").exists()
    second_run = module._create_prediction_run_dir(
        artifact_dir,
        allowed_root=workspace,
    )
    assert second_run != run_dir


def test_prediction_run_rejects_run_root_symlink_escape(tmp_path: Path) -> None:
    module = _load_inference_script()
    workspace = tmp_path / "workspace"
    artifact_dir = workspace / "artifacts" / "EXP-TEST"
    artifact_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (artifact_dir / "predicted-runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        module._create_prediction_run_dir(
            artifact_dir,
            allowed_root=workspace,
        )

    assert not list(outside.iterdir())


def test_prediction_output_validation_rejects_geff_symlink(
    tmp_path: Path,
) -> None:
    module = _load_inference_script()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    outside = tmp_path / "outside.geff"
    outside.mkdir()
    (run_dir / "sample.geff").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="contains a symlink"):
        module._validate_prediction_outputs(run_dir, ["sample"])


def test_prediction_output_validation_rejects_stale_extra_geff(
    tmp_path: Path,
) -> None:
    module = _load_inference_script()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "current.geff").mkdir()
    (run_dir / "stale.geff").mkdir()

    with pytest.raises(RuntimeError, match=r"extra=\['stale'\]"):
        module._validate_prediction_outputs(run_dir, ["current"])


def test_decoder_applies_null_parent_and_calibrated_division_gate() -> None:
    decoder = GraphDecoder(
        DecoderProfile(
            max_parents_per_node=1,
            max_children_per_node=2,
            null_parent_threshold=0.5,
            division_threshold=0.8,
        ),
        edge_activation="none",
    )
    result = decoder.decode(
        np.asarray([[0.95, 0.90, 0.80], [0.70, 0.60, 0.55]]),
        edge_threshold=0.5,
        null_parent_logits=np.asarray([0.1, 0.1, 0.9]),
        division_logits=np.asarray([2.0, -2.0]),
        source_coords_um=np.asarray([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
        target_coords_um=np.asarray(
            [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        ),
        source_offset=10,
        target_offset=20,
    )

    assert [(edge.source, edge.target) for edge in result.edges] == [(10, 20), (10, 21)]
    assert [edge.distance_um for edge in result.edges] == [1.0, 2.0]
    assert result.null_parent_probabilities.tolist() == [0.1, 0.1, 0.9]


def test_decoder_softmax_normalises_parent_and_null_together() -> None:
    decoder = GraphDecoder(
        DecoderProfile(1, 2, null_parent_threshold=0.9, division_threshold=0.5),
        edge_activation="softmax",
    )
    edge_probability, null_probability = decoder.probabilities(
        np.asarray([[2.0], [1.0]]),
        np.asarray([0.0]),
    )

    assert np.isclose(edge_probability[:, 0].sum() + null_probability[0], 1.0)
    assert edge_probability[0, 0] > edge_probability[1, 0] > null_probability[0]


def test_output_coordinate_rescaling_rounds_instead_of_truncating() -> None:
    coords = np.asarray([[2.0, 1.5, 2.49, 2.51]], dtype=np.float32)

    result = rescale_output_coordinates(coords, (1, 1, 1))

    assert result.tolist() == [[2, 2, 2, 3]]


def test_detection_cap_keeps_highest_logit_peaks() -> None:
    logits = torch.arange(7, dtype=torch.float32).reshape(1, 1, 1, 1, 1, 7)

    detected = detect_window_nodes(
        logits,
        threshold=0.1,
        pool_kernel_um=0.5,
        voxel_size_um=(1.0, 1.0, 1.0),
        max_detections_per_frame=2,
    )

    assert detected[0].tolist() == [[0.0, 0.0, 6.0], [0.0, 0.0, 5.0]]


class _FakeCorrectedModel:
    def encode_window(self, images: torch.Tensor) -> EncodedWindow:
        batch, frames, z_size, y_size, x_size = images.shape
        features = torch.zeros(
            batch, frames, 2, z_size, y_size, x_size, device=images.device
        )
        logits = torch.full(
            (batch, frames, 1, z_size, y_size, x_size),
            -10.0,
            device=images.device,
        )
        logits[:, :, :, 1, 1, 1] = 10.0
        return EncodedWindow(features, logits)

    def build_nodes(
        self,
        encoded: EncodedWindow,
        coords: torch.Tensor,
        masks: torch.Tensor,
        image_shape: tuple[int, ...],
        voxel_size: tuple[float, ...],
        **_: object,
    ) -> tuple[NodeBatch, ...]:
        del image_shape
        result = []
        spacing = torch.tensor(voxel_size, device=coords.device)
        for frame in range(coords.shape[1]):
            grid = coords[:, frame]
            valid = masks[:, frame]
            one = torch.ones(*grid.shape[:2], 1, device=coords.device)
            result.append(
                NodeBatch(
                    appearance=torch.zeros(*grid.shape[:2], 2, device=coords.device),
                    grid_coords=grid,
                    physical_coords_um=grid * spacing,
                    spatial_position=torch.zeros(
                        *grid.shape[:2], 3, device=coords.device
                    ),
                    valid_mask=valid,
                    detection_probability=one,
                    division_probability=one,
                    division_logits=torch.full_like(one, 5.0),
                    frame_role=one * frame,
                    delta_t=one * frame,
                )
            )
        return tuple(result)

    def link_pair(self, source: NodeBatch, target: NodeBatch) -> LinkOutput:
        batch, sources = source.valid_mask.shape
        targets = target.valid_mask.shape[1]
        return LinkOutput(
            edge_logits=torch.full((batch, sources, targets), 5.0),
            null_parent_logits=torch.full((batch, targets), -5.0),
            division_logits=torch.full((batch, sources), 5.0),
        )


def test_corrected_dataset_path_uses_profile_and_common_decoder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        inference_v2,
        "_zarr_metadata",
        lambda _: {
            "shape": (2, 3, 3, 3),
            "dtype": np.dtype("uint16"),
            "scale": (1.0, 1.0, 1.0),
            "q_low": 0.0,
            "q_high": 1.0,
        },
    )
    monkeypatch.setattr(
        inference_v2,
        "_load_zarr_frame",
        lambda *args, **kwargs: torch.zeros(3, 3, 3),
    )
    profile = replace(
        _profile(),
        detection_tta=False,
        pool_kernel_um=1.0,
        edge_threshold=0.1,
        decoder=DecoderProfile(1, 2, 0.5, 0.5),
    )

    coords, edges = predict_dataset_corrected(
        _FakeCorrectedModel(), tmp_path / "sample.zarr", torch.device("cpu"), profile
    )

    assert coords.tolist() == [[0, 1, 4, 4], [1, 1, 4, 4]]
    assert [(edge[0], edge[1]) for edge in edges] == [(0, 1)]


def _load_screen_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    module_path = Path(__file__).parents[1] / "scripts/evaluate_backbone_ab_competition_screen.py"
    spec = importlib.util.spec_from_file_location("biohub_screen_v2", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "COMPETITION_ROOT", tmp_path)
    return module


def test_screen_filters_whole_datasets_by_calibration_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_screen_module(monkeypatch, tmp_path)
    manifest = tmp_path / "validation_split.json"
    manifest.write_text(
        '{"calibration": ["a", "c"], "report": ["b"]}\n', encoding="utf-8"
    )
    specs = [SimpleNamespace(dataset=name) for name in ("a", "b", "c")]
    video_data = [1, 2, 3]

    selected_specs, selected_data = module._filter_dataset_subset(
        {
            "data": {
                "dataset_subset_manifest": manifest.name,
                "dataset_subset_manifest_sha256": module._sha256(manifest),
                "dataset_subset": "calibration",
            },
            "model": {},
            "inference": {},
        },
        specs,
        video_data,
    )

    assert [item.dataset for item in selected_specs] == ["a", "c"]
    assert selected_data == [1, 3]


def test_screen_rejects_report_without_fixed_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_screen_module(monkeypatch, tmp_path)
    manifest = tmp_path / "split.json"
    manifest.write_text('{"calibration": ["a"], "report": ["b"]}\n', encoding="utf-8")
    config = {
        "data": {
            "dataset_subset_manifest": manifest.name,
            "dataset_subset_manifest_sha256": module._sha256(manifest),
            "dataset_subset": "report",
        },
        "model": {},
        "inference": {},
    }

    with pytest.raises(ValueError, match="fixed model.inference_profile"):
        module._validate_subset_contract(config)


def test_screen_rejects_calibration_profile_reuse_and_report_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_screen_module(monkeypatch, tmp_path)
    manifest = tmp_path / "split.json"
    manifest.write_text('{"calibration": ["a"], "report": ["b"]}\n', encoding="utf-8")
    data = {
        "dataset_subset_manifest": manifest.name,
        "dataset_subset_manifest_sha256": module._sha256(manifest),
    }
    calibration = {
        "data": {**data, "dataset_subset": "calibration"},
        "model": {"inference_profile": "profile.json"},
        "inference": {},
    }
    report = {
        "data": {**data, "dataset_subset": "report"},
        "model": {"inference_profile": "profile.json"},
        "inference": {"edge_thresholds": [0.1, 0.2]},
    }

    with pytest.raises(ValueError, match="calibration subset"):
        module._validate_subset_contract(calibration)
    with pytest.raises(ValueError, match="forbids threshold sweeps"):
        module._validate_subset_contract(report)


def test_screen_rejects_changed_split_manifest_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_screen_module(monkeypatch, tmp_path)
    manifest = tmp_path / "split.json"
    manifest.write_text('{"calibration": ["a"], "report": ["b"]}\n', encoding="utf-8")
    pinned = module._sha256(manifest)
    manifest.write_text('{"calibration": ["b"], "report": ["a"]}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        module._validate_subset_contract(
            {
                "data": {
                    "dataset_subset_manifest": manifest.name,
                    "dataset_subset_manifest_sha256": pinned,
                    "dataset_subset": "calibration",
                },
                "model": {},
                "inference": {},
            }
        )
