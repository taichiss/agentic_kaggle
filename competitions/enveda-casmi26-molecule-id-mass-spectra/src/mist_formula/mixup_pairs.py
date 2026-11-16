"""Train-only partner selection and pre-feature MS/MS peak-list mixing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from model import ADDUCTS, spectrum_features


class PartnerPool:
    """Pair spectra under a fixed formula/instrument/adduct/CE policy."""

    def __init__(
        self,
        data,
        metadata_dir: str | Path,
        *,
        mode: str = "cross",
        ce_tol: float = 10.0,
        precursor_ppm: float = 20.0,
    ) -> None:
        if mode not in {"cross", "same"}:
            raise ValueError("mode must be 'cross' or 'same'")
        self.data = data
        self.mode = mode
        self.ce_tol = float(ce_tol)
        self.precursor_ppm = float(precursor_ppm)
        meta = Path(metadata_dir)
        self.ingest_id = np.load(meta / "ingest_id.npy", mmap_mode="r")
        self.instrument_id = np.load(meta / "instrument_id.npy", mmap_mode="r")
        self.ce_ev = np.load(meta / "ce_ev.npy", mmap_mode="r")
        self.ce_kind = np.load(meta / "ce_kind.npy", mmap_mode="r")
        self.categories = json.loads((meta / "categories.json").read_text())
        n = len(data.spec)
        if any(len(x) != n for x in (self.ingest_id, self.instrument_id, self.ce_ev, self.ce_kind)):
            raise ValueError("Mixup metadata is not aligned to training cache")
        train = np.asarray(data.spec.split.eq("train"))
        source_ok = (self.ingest_id >= 0) & (self.instrument_id >= 0)
        ce_ok = self.ce_kind <= 1
        rows = np.flatnonzero(train & source_ok & ce_ok).astype(np.int32)
        if mode == "cross":
            formula_id = pd.factorize(data.mol.formula, sort=False)[0]
            identity = formula_id[data.mol_idx[rows]]
        else:
            identity = data.mol_idx[rows]
        keys = np.stack(
            [
                identity,
                data.adduct[rows],
                self.ingest_id[rows],
                self.instrument_id[rows],
                self.ce_kind[rows],
                np.rint(data.meta[rows, 3]).astype(np.int32),
            ],
            axis=1,
        ).astype(np.int32)
        _unique, group_id = np.unique(keys, axis=0, return_inverse=True)
        self.group_id = np.full(n, -1, dtype=np.int32)
        self.group_id[rows] = group_id.astype(np.int32)
        sort_ce = np.where(self.ce_kind[rows] == 1, self.ce_ev[rows], 0)
        order = np.lexsort((sort_ce, group_id))
        self.rows = rows[order]
        self.sorted_ce = sort_ce[order]
        counts = np.bincount(group_id, minlength=len(_unique))
        self.offsets = np.concatenate(([0], np.cumsum(counts)))
        self.stats = {
            "mode": mode,
            "train_rows": int(train.sum()),
            "indexed_rows": len(rows),
            "groups": len(_unique),
            "attempted": 0,
            "paired": 0,
        }

    def sample_partner(self, anchor_cache_row: int, rng: np.random.Generator) -> int | None:
        """Return one valid training row, or None without relaxing the policy."""
        row = int(anchor_cache_row)
        self.stats["attempted"] += 1
        group = int(self.group_id[row])
        if group < 0:
            return None
        lo, hi = int(self.offsets[group]), int(self.offsets[group + 1])
        if hi - lo < 2:
            return None
        if self.ce_kind[row] == 1:
            level = float(self.ce_ev[row])
            view = self.sorted_ce[lo:hi]
            lo += int(np.searchsorted(view, level - self.ce_tol, side="left"))
            hi = int(self.offsets[group]) + int(
                np.searchsorted(view, level + self.ce_tol, side="right")
            )
        candidates = self.rows[lo:hi]
        if len(candidates) < 2:
            return None
        mol = int(self.data.mol_idx[row])
        candidate_mols = self.data.mol_idx[candidates]
        keep = candidate_mols != mol if self.mode == "cross" else candidates != row
        precursor = float(self.data.meta[row, 0])
        delta_ppm = np.abs(self.data.meta[candidates, 0] - precursor) / max(precursor, 1e-8) * 1e6
        valid = candidates[keep & (delta_ppm <= self.precursor_ppm)]
        if not len(valid):
            return None
        partner = int(rng.choice(valid))
        if self.data.spec.split.iloc[partner] != "train":
            raise AssertionError("Validation spectrum selected as mixup partner")
        if self.mode == "cross":
            assert self.data.mol_idx[partner] != mol
            assert (
                self.data.mol.formula.iloc[self.data.mol_idx[partner]]
                == (self.data.mol.formula.iloc[mol])
            )
        assert self.data.meta[partner, 3] == self.data.meta[row, 3]
        self.stats["paired"] += 1
        return partner


def merge_peaks(
    mz_a: np.ndarray,
    intensity_a: np.ndarray,
    mz_b: np.ndarray,
    intensity_b: np.ndarray,
    fraction_a: float,
    *,
    ppm: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge measured m/z within tolerance, sum linear relative intensities."""
    if not 0 < fraction_a < 1:
        raise ValueError("fraction_a must lie strictly between 0 and 1")
    mz = np.concatenate([mz_a, mz_b]).astype(np.float64)
    intensity = np.concatenate(
        [
            fraction_a * np.asarray(intensity_a, np.float64),
            (1 - fraction_a) * np.asarray(intensity_b, np.float64),
        ]
    )
    valid = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz, intensity = mz[valid], intensity[valid]
    order = np.argsort(mz, kind="stable")
    mz, intensity = mz[order], intensity[order]
    if not len(mz):
        raise ValueError("Mixed spectrum contains no peaks")
    out_mz, out_it = [], []
    start = 0
    for index in range(1, len(mz) + 1):
        if index < len(mz) and mz[index] - mz[index - 1] <= mz[index - 1] * ppm * 1e-6:
            continue
        block = intensity[start:index]
        strongest = start + int(np.argmax(block))
        out_mz.append(float(mz[strongest]))
        out_it.append(float(block.sum()))
        start = index
    return np.asarray(out_mz, np.float32), np.asarray(out_it, np.float32)


def mixed_batch(
    data,
    catalog,
    anchor_rows: np.ndarray,
    partner_rows: np.ndarray,
    fractions_a: np.ndarray,
    device: str,
) -> list[torch.Tensor]:
    """Return Cache.batch-compatible tensors after redoing peak and formula features."""
    anchors = np.asarray(anchor_rows, dtype=np.int64)
    partners = np.asarray(partner_rows, dtype=np.int64)
    fractions = np.asarray(fractions_a, dtype=np.float32)
    if anchors.shape != partners.shape or anchors.shape != fractions.shape:
        raise ValueError("Mixup arrays must have the same shape")
    batch = data.batch(anchors, catalog, device)
    for pos, (anchor, partner, fraction) in enumerate(
        zip(anchors, partners, fractions, strict=True)
    ):
        if partner < 0:
            continue
        n_a, n_b = int(data.length[anchor]), int(data.length[partner])
        mz, it = merge_peaks(
            np.asarray(data.mz[anchor, :n_a]),
            np.asarray(data.x[anchor, :n_a, 1], np.float32) ** 2,
            np.asarray(data.mz[partner, :n_b]),
            np.asarray(data.x[partner, :n_b, 1], np.float32) ** 2,
            float(fraction),
        )
        precursor = float(data.meta[anchor, 0] * 1000)
        feature, length = spectrum_features(mz, it, precursor, max_peaks=96)
        selected = np.argsort(-it, kind="stable")[:96]
        selected = selected[np.argsort(mz[selected], kind="stable")]
        precise_mz = mz[selected]
        if not getattr(data, "precise_raw_features", False):
            feature = feature.astype("<f2").astype(np.float32)
        adduct = ADDUCTS[data.adduct[anchor] - 1] if data.adduct[anchor] else ""
        assignments = catalog.assign(
            precise_mz,
            precursor,
            data.formulas[data.mol_idx[anchor]],
            adduct=adduct,
            polarity=float(data.meta[anchor, 3]),
        )
        batch[0][pos] = torch.as_tensor(feature, device=device)
        batch[1][pos] = length
        for target, value in zip((5, 6, 7, 8), assignments, strict=True):
            batch[target][pos] = torch.as_tensor(value, device=device)
    return batch
