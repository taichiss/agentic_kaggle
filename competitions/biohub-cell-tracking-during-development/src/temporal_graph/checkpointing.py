"""Portable residual-head checkpoint payloads."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from .contracts import TemporalGraphConfig

CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TemporalGraphCheckpoint:
    """Checkpoint content whose serialized representation is a plain mapping."""

    config: TemporalGraphConfig
    state_dict: Mapping[str, torch.Tensor]
    base_checkpoint_sha256: str
    metadata: Mapping[str, Any]
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError("unsupported temporal-graph checkpoint schema")
        if not isinstance(self.base_checkpoint_sha256, str):
            raise TypeError("base_checkpoint_sha256 must be a string")
        if not all(isinstance(key, str) for key in self.state_dict):
            raise TypeError("state_dict keys must be strings")
        if not all(torch.is_tensor(value) for value in self.state_dict.values()):
            raise TypeError("state_dict values must be tensors")

    def to_payload(self) -> dict[str, Any]:
        """Return only built-in containers and CPU tensors for ``torch.save``."""
        return {
            "schema_version": self.schema_version,
            "config": self.config.to_dict(),
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self.state_dict.items()
            },
            "base_checkpoint_sha256": self.base_checkpoint_sha256,
            "metadata": deepcopy(dict(self.metadata)),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TemporalGraphCheckpoint:
        required = {
            "schema_version",
            "config",
            "state_dict",
            "base_checkpoint_sha256",
            "metadata",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"temporal-graph checkpoint is missing: {missing}")
        return cls(
            schema_version=int(payload["schema_version"]),
            config=TemporalGraphConfig.from_dict(payload["config"]),
            state_dict=dict(payload["state_dict"]),
            base_checkpoint_sha256=str(payload["base_checkpoint_sha256"]),
            metadata=deepcopy(dict(payload["metadata"])),
        )
