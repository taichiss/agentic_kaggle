"""Fail-closed integrity checks for finalized corrected_v2 Kaggle bundles."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts/run_kaggle_inference.py"
SPEC = importlib.util.spec_from_file_location("run_kaggle_inference_bundle", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
runtime = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _manifest_entry(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _install_bundle(root: Path) -> tuple[Path, dict, dict]:
    root.mkdir()
    checkpoint_sha = "a" * 64
    profile_sha = "b" * 64
    experiment_config_sha = "c" * 64
    split_sha = "d" * 64
    config = {"experiment_id": "EXP-0007A", "model_api": "corrected_v2"}
    metadata = {
        "experiment_id": "EXP-0007A",
        "completed_epochs": 30,
        "source_checkpoint_sha256": checkpoint_sha,
        "source_inference_profile_sha256": profile_sha,
        "experiment_config_sha256": experiment_config_sha,
        "validation_subset_manifest_sha256": split_sha,
    }
    provenance = {
        "candidate_epochs": list(range(5, 51, 5)),
        "completed_epoch": 30,
        "checkpoint_sha256": checkpoint_sha,
        "inference_profile_sha256": profile_sha,
        "experiment_config_sha256": experiment_config_sha,
        "manifest_sha256": split_sha,
        "selection_sha256": "e" * 64,
        "report_summary_sha256": "f" * 64,
        "report_config_sha256": "1" * 64,
        "report_score": 0.7,
        "report_score_baseline": runtime.EXP7A_EPOCH5_REPORT_BASELINE,
        "report_score_tolerance": 0.0,
        "report_score_gate_exclusive": runtime.EXP7A_EPOCH5_REPORT_BASELINE,
    }
    _write_json(root / "config.json", config)
    _write_json(root / "checkpoint-metadata.json", metadata)
    _write_json(root / "selection-provenance.json", provenance)
    for name in (
        "edge_predictor_best.pth",
        "inference_profile.json",
        "run_kaggle_inference.py",
        "tracking_cellmot_models.zip",
        "backbone_ab.zip",
        "dynamic_network_architectures.zip",
    ):
        (root / name).write_bytes(name.encode("utf-8"))
    names = sorted(path.name for path in root.iterdir())
    manifest = {
        "schema_version": 1,
        "experiment_id": "EXP-0007A",
        "completed_epochs": 30,
        "files": {name: _manifest_entry(root / name) for name in names},
    }
    _write_json(root / "manifest.json", manifest)
    return root, manifest, provenance


def _expand_archives(bundle: Path, manifest: dict) -> None:
    for name in (
        "tracking_cellmot_models.zip",
        "backbone_ab.zip",
        "dynamic_network_architectures.zip",
    ):
        archive = bundle / name
        member_name = f"{Path(name).stem}/module.py"
        member_bytes = f"expanded-{name}".encode()
        expanded_member = bundle / Path(name).stem / member_name
        expanded_member.parent.mkdir(parents=True)
        expanded_member.write_bytes(member_bytes)
        archive.unlink()
        manifest["files"][name]["members"] = {
            member_name: {
                "bytes": len(member_bytes),
                "sha256": hashlib.sha256(member_bytes).hexdigest(),
            }
        }
    _write_json(bundle / "manifest.json", manifest)


def test_finalized_bundle_manifest_accepts_one_complete_bound_package(
    tmp_path: Path,
) -> None:
    bundle, _, _ = _install_bundle(tmp_path / "bundle")

    result = runtime._verify_bundle_manifest(bundle)

    assert result["completed_epochs"] == 30


def test_finalized_bundle_manifest_rejects_tampered_runtime_bytes(
    tmp_path: Path,
) -> None:
    bundle, _, _ = _install_bundle(tmp_path / "bundle")
    (bundle / "run_kaggle_inference.py").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="does not match manifest"):
        runtime._verify_bundle_manifest(bundle)


def test_finalized_bundle_manifest_rejects_an_incomplete_epoch_sweep(
    tmp_path: Path,
) -> None:
    bundle, manifest, provenance = _install_bundle(tmp_path / "bundle")
    provenance["candidate_epochs"] = [5, 10, 15, 20, 25, 30]
    _write_json(bundle / "selection-provenance.json", provenance)
    manifest["files"]["selection-provenance.json"] = _manifest_entry(
        bundle / "selection-provenance.json"
    )
    _write_json(bundle / "manifest.json", manifest)

    with pytest.raises(ValueError, match="complete checkpoint sweep"):
        runtime._verify_bundle_manifest(bundle)


def test_finalized_bundle_manifest_rejects_split_provenance_drift(
    tmp_path: Path,
) -> None:
    bundle, manifest, provenance = _install_bundle(tmp_path / "bundle")
    provenance["manifest_sha256"] = "e" * 64
    _write_json(bundle / "selection-provenance.json", provenance)
    manifest["files"]["selection-provenance.json"] = _manifest_entry(
        bundle / "selection-provenance.json"
    )
    _write_json(bundle / "manifest.json", manifest)

    with pytest.raises(ValueError, match="validation split mismatch"):
        runtime._verify_bundle_manifest(bundle)


def test_finalized_bundle_manifest_rejects_inconsistent_gate_tolerance(
    tmp_path: Path,
) -> None:
    bundle, manifest, provenance = _install_bundle(tmp_path / "bundle")
    provenance["report_score_tolerance"] = 0.01
    _write_json(bundle / "selection-provenance.json", provenance)
    manifest["files"]["selection-provenance.json"] = _manifest_entry(
        bundle / "selection-provenance.json"
    )
    _write_json(bundle / "manifest.json", manifest)

    with pytest.raises(ValueError, match="score gate was not satisfied"):
        runtime._verify_bundle_manifest(bundle)


def test_finalized_bundle_manifest_accepts_exact_kaggle_expanded_archives(
    tmp_path: Path,
) -> None:
    bundle, manifest, _ = _install_bundle(tmp_path / "bundle")
    _expand_archives(bundle, manifest)

    result = runtime._verify_bundle_manifest(bundle)

    assert result["completed_epochs"] == 30


def test_finalized_bundle_manifest_rejects_tampered_expanded_member(
    tmp_path: Path,
) -> None:
    bundle, manifest, _ = _install_bundle(tmp_path / "bundle")
    _expand_archives(bundle, manifest)
    member = bundle / "backbone_ab/backbone_ab/module.py"
    member.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="expanded archive member hash mismatch"):
        runtime._verify_bundle_manifest(bundle)


def test_finalized_bundle_manifest_rejects_untracked_expanded_member(
    tmp_path: Path,
) -> None:
    bundle, manifest, _ = _install_bundle(tmp_path / "bundle")
    _expand_archives(bundle, manifest)
    shadow = bundle / "backbone_ab/backbone_ab.py"
    shadow.write_text("raise RuntimeError('shadowed')\n", encoding="utf-8")

    with pytest.raises(ValueError, match="member set mismatch"):
        runtime._verify_bundle_manifest(bundle)
