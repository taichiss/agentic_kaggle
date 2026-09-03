"""Checkpoint and immutable inference-profile contracts for backbone experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(value: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible mapping using a stable representation."""
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Normalised view of a raw state dict or a wrapped training checkpoint."""

    state_dict: Mapping[str, Any]
    metadata: Mapping[str, Any]
    source_format: str
    sha256: str


def _looks_like_state_dict(value: object) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    return all(isinstance(key, str) for key in value) and not any(
        key in value for key in ("model_state_dict", "state_dict", "model")
    )


def normalise_checkpoint_payload(payload: object, *, digest: str = "") -> LoadedCheckpoint:
    """Extract model weights from supported raw and wrapped checkpoint layouts."""
    if _looks_like_state_dict(payload):
        return LoadedCheckpoint(payload, {}, "raw_state_dict", digest)  # type: ignore[arg-type]
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a state dict or a mapping")

    state_key = next(
        (key for key in ("model_state_dict", "state_dict", "model") if key in payload),
        None,
    )
    if state_key is None or not isinstance(payload[state_key], Mapping):
        raise ValueError("checkpoint does not contain model_state_dict, state_dict, or model")
    metadata = {key: value for key, value in payload.items() if key != state_key}
    return LoadedCheckpoint(
        payload[state_key],
        metadata,
        f"wrapped:{state_key}",
        digest,
    )


def load_checkpoint(path: Path, *, map_location: str | object = "cpu") -> LoadedCheckpoint:
    """Load and normalise a checkpoint while recording its content digest."""
    import torch

    payload = torch.load(path, map_location=map_location, weights_only=False)
    return normalise_checkpoint_payload(payload, digest=sha256_file(path))


def load_model_checkpoint(
    model: object,
    path: Path,
    *,
    map_location: str | object = "cpu",
    strict: bool = True,
) -> LoadedCheckpoint:
    """Load either checkpoint format into a model and return normalised metadata."""
    loaded = load_checkpoint(path, map_location=map_location)
    model.load_state_dict(loaded.state_dict, strict=strict)  # type: ignore[attr-defined]
    return loaded


@dataclass(frozen=True)
class DecoderProfile:
    """Calibrated graph constraints applied to every inference entrypoint."""

    max_parents_per_node: int
    max_children_per_node: int
    null_parent_threshold: float
    division_threshold: float

    def __post_init__(self) -> None:
        if self.max_parents_per_node <= 0 or self.max_children_per_node <= 0:
            raise ValueError("decoder degree limits must be positive")
        for name in ("null_parent_threshold", "division_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"decoder.{name} must be in [0, 1]")


@dataclass(frozen=True)
class InferenceProfile:
    """Portable, content-addressed inference settings bound to one checkpoint."""

    experiment_id: str
    model_api: str
    checkpoint_sha256: str
    experiment_config_sha256: str
    source_revision: str
    downsample: tuple[int, int, int]
    window_size: int
    detection_threshold: float
    detection_tta: bool
    pool_kernel_um: float
    edge_activation: str
    edge_threshold: float
    max_detections_per_frame: int
    decoder: DecoderProfile
    postprocess_profile: str = "none"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported inference profile schema")
        if self.model_api not in {"legacy", "corrected_v2"}:
            raise ValueError("model_api must be 'legacy' or 'corrected_v2'")
        if len(self.downsample) != 3 or any(value <= 0 for value in self.downsample):
            raise ValueError("downsample must contain three positive integers")
        if self.window_size < 2:
            raise ValueError("window_size must be at least two")
        for name in ("detection_threshold", "edge_threshold"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.pool_kernel_um <= 0:
            raise ValueError("pool_kernel_um must be positive")
        if self.max_detections_per_frame <= 0:
            raise ValueError("max_detections_per_frame must be positive")
        if self.edge_activation not in {
            "softmax",
            "parent_softmax_with_null",
            "sigmoid",
            "none",
        }:
            raise ValueError("unsupported edge activation")
        for name in ("checkpoint_sha256", "experiment_config_sha256"):
            digest = getattr(self, name)
            invalid_character = any(
                character not in "0123456789abcdef" for character in digest
            )
            if len(digest) != 64 or invalid_character:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_json_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> InferenceProfile:
        data = dict(value)
        data["decoder"] = DecoderProfile(**data["decoder"])
        data["downsample"] = tuple(int(item) for item in data["downsample"])
        return cls(**data)

    @classmethod
    def from_experiment_config(
        cls,
        config: Mapping[str, Any],
        *,
        checkpoint_sha256: str,
        experiment_config_sha256: str,
    ) -> InferenceProfile:
        inference = config["inference"]
        model_api = str(inference.get("model_api", "legacy"))
        decoder = inference.get("decoder", {})
        if model_api == "corrected_v2":
            if bool(inference.get("use_ilp", False)):
                raise ValueError("corrected_v2 requires use_ilp=false for portable inference")
            required = {
                "max_parents_per_node",
                "max_children_per_node",
                "null_parent_threshold",
                "division_threshold",
            }
            missing = sorted(required - decoder.keys())
            if missing:
                raise ValueError(f"corrected_v2 decoder config is missing: {missing}")
            if "max_detections_per_frame" not in inference:
                raise ValueError(
                    "corrected_v2 inference requires max_detections_per_frame"
                )
        else:
            decoder = {
                "max_parents_per_node": decoder.get("max_parents_per_node", 1),
                "max_children_per_node": decoder.get("max_children_per_node", 2),
                "null_parent_threshold": decoder.get("null_parent_threshold", 1.0),
                "division_threshold": decoder.get("division_threshold", 0.0),
            }
        return cls(
            experiment_id=str(config["experiment_id"]),
            model_api=model_api,
            checkpoint_sha256=checkpoint_sha256,
            experiment_config_sha256=experiment_config_sha256,
            source_revision=str(config["source"]["organizer_revision"]),
            downsample=tuple(int(value) for value in config["train"]["downsample"]),
            window_size=int(config["data"]["window_size"]),
            detection_threshold=float(inference["det_threshold"]),
            detection_tta=bool(inference.get("det_tta", inference.get("detection_tta", False))),
            pool_kernel_um=float(inference["pool_kernel_um"]),
            edge_activation=str(inference.get("edge_activation", "softmax")),
            edge_threshold=float(inference["edge_threshold"]),
            max_detections_per_frame=int(
                inference.get("max_detections_per_frame", 512)
            ),
            decoder=DecoderProfile(**decoder),
            postprocess_profile=str(inference.get("postprocess_profile", "none")),
        )


def write_inference_profile(path: Path, profile: InferenceProfile) -> None:
    """Write a profile once; reruns may only reproduce identical content."""
    payload = {"profile": profile.to_dict(), "profile_sha256": profile.sha256}
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"refusing to replace immutable inference profile: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")


def load_inference_profile(path: Path) -> InferenceProfile:
    """Load an inference profile and verify its embedded content hash."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = InferenceProfile.from_dict(payload["profile"])
    if payload.get("profile_sha256") != profile.sha256:
        raise ValueError(f"inference profile hash mismatch: {path}")
    return profile
