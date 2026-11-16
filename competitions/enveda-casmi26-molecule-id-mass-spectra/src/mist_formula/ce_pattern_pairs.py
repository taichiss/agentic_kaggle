"""Exact multi-CE-pattern partner selection on top of the frozen strict M1 cache."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from train_mix_same_ce_strict import Cache as StrictCache


class PatternCache(StrictCache):
    """Keep strict kind-0/1 selection and permit matching, valid kind-2 patterns."""

    def __init__(self, root: Path, precise_mzs: Path, source_metadata: Path,
                 pattern_metadata: Path):
        super().__init__(root, precise_mzs, source_metadata)
        self.pattern_id = np.load(pattern_metadata / "pattern_id.npy", mmap_mode="r")
        self.raw_row_index = np.load(pattern_metadata / "raw_row_index.npy", mmap_mode="r")
        self.pattern_manifest = json.loads((pattern_metadata / "manifest.json").read_text())
        for name, field in (
            ("pattern_id.npy", "pattern_id_sha256"),
            ("raw_row_index.npy", "raw_row_index_sha256"),
        ):
            digest = hashlib.sha256((pattern_metadata / name).read_bytes()).hexdigest()
            if digest != self.pattern_manifest[field]:
                raise AssertionError(f"CE-pattern sidecar hash mismatch: {name}")
        if len(self.pattern_id) != len(self.spec) or len(self.raw_row_index) != len(self.spec):
            raise AssertionError("CE-pattern sidecar does not align with frozen cache")
        if np.any(self.raw_row_index < 0):
            raise AssertionError("CE-pattern raw-row alignment has gaps")
        if np.any((self.pattern_id >= 0) & (self.ce_kind != 2)):
            raise AssertionError("Only kind-2 spectra may have multi-CE signatures")
        if self.pattern_manifest["kind2_eligible_rows"] != int((self.pattern_id >= 0).sum()):
            raise AssertionError("CE-pattern sidecar and manifest disagree")
        self.same_mol_pattern = defaultdict(list)
        for row in np.flatnonzero(self.spec.split.eq("train")):
            if self.ce_kind[row] != 2 or self.pattern_id[row] < 0:
                continue
            if self.instrument_id[row] < 0:
                continue
            self.same_mol_pattern[self.pattern_key(int(row))].append(int(row))

    def pattern_key(self, row: int) -> tuple[int, int, int, int, int, int]:
        return (int(self.mol_idx[row]), int(self.adduct[row]),
                int(self.ingest_id[row]), int(self.instrument_id[row]),
                int(self.meta[row, 3]), int(self.pattern_id[row]))

    def partner(self, row: int, rng: np.random.Generator) -> int:
        if self.ce_kind[row] <= 1:
            return super().partner(row, rng)
        if self.ce_kind[row] != 2 or self.pattern_id[row] < 0:
            return -1
        if self.spec.split.iloc[row] != "train" or self.instrument_id[row] < 0:
            return -1
        candidates = self.same_mol_pattern[self.pattern_key(row)]
        if len(candidates) < 2:
            return -1
        precursor = float(self.meta[row, 0])
        for _ in range(min(12, 2 * len(candidates))):
            other = int(candidates[int(rng.integers(len(candidates)))])
            if other == row:
                continue
            if abs(float(self.meta[other, 0]) - precursor) > max(precursor * 20e-6, 1e-7):
                continue
            if self.ce_kind[other] != 2 or self.pattern_key(other) != self.pattern_key(row):
                raise AssertionError("Multi-CE partner violates exact-pattern invariants")
            return other
        return -1
