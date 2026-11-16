"""Candidate-formula conditioned spectrum-to-fingerprint Transformer.

The model uses several plausible CHNO subformula assignments for each observed
fragment ion and neutral loss. Formula masses are recomputed from isotope masses;
the supplied CSV is used only as a list of compositions. The catalog is embedded
in each checkpoint, so inference needs only this module and the checkpoint.
"""

from __future__ import annotations

import csv
import functools
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

EXACT_MASS = np.array([12.0, 1.00782503223, 14.00307400443, 15.99491461957])
PROTON = 1.007276466621
ELECTRON = 0.000548579909
ADDUCTS = (
    "[M+H]+", "[M-H]-", "[M+Na]+", "[M+K]+", "[M+NH4]+",
    "[M+Cl]-", "[M+FA-H]-", "[M+H-H2O]+", "[M+2H]2+", "[M-2H]2-",
)
# Channel 0 is padding; 1-6 denote fragment-ion hypotheses, 7 neutral loss.
ION_SHIFTS = {
    1: PROTON,
    2: -PROTON,
    3: -ELECTRON,
    4: ELECTRON,
    5: 22.989218,
    6: 38.963158,
}


def read_formula_catalog(path: str | Path) -> tuple[np.ndarray, str]:
    """Read only compositions, validating and discarding tabulated mass values."""
    path = Path(path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = np.asarray([[int(row[x]) for x in "CHNO"] for row in rows], dtype=np.int16)
    if counts.ndim != 2 or counts.shape[1] != 4 or np.any(counts < 0):
        raise ValueError("Expected a nonnegative CHNO composition table")
    counts = np.unique(counts, axis=0)
    counts = counts[np.any(counts, axis=1)]
    masses = counts @ EXACT_MASS
    return counts[np.argsort(masses)], hashlib.sha256(path.read_bytes()).hexdigest()


def formula_counts(formula: str) -> tuple[int, int, int, int]:
    """Extract CHNO counts; other elements remain available to raw m/z tokens."""
    import re

    result = {x: 0 for x in "CHNO"}
    for symbol, number in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if symbol in result:
            result[symbol] += int(number or 1)
    return tuple(result[x] for x in "CHNO")


def spectrum_features(
    mzs: object,
    intensities: object,
    precursor_mz: float,
    *,
    max_peaks: int = 96,
) -> tuple[np.ndarray, int]:
    """Match the existing Saner cache's stable top-intensity peak preprocessing."""
    mz = np.asarray(mzs, dtype=np.float32).reshape(-1)
    it = np.asarray(intensities, dtype=np.float32).reshape(-1)
    if len(mz) != len(it):
        raise ValueError("Peak and intensity arrays have different lengths")
    keep = np.isfinite(mz) & np.isfinite(it) & (mz > 0) & (it > 0)
    mz, it = mz[keep], it[keep]
    if not len(mz):
        raise ValueError("No positive peaks")
    idx = np.argsort(-it, kind="stable")[:max_peaks]
    idx = idx[np.argsort(mz[idx], kind="stable")]
    mz, it = mz[idx], it[idx]
    it = it / it.max()
    result = np.zeros((max_peaks, 4), dtype=np.float32)
    result[: len(mz)] = np.stack(
        [mz / 1000, np.sqrt(it), np.log1p(100 * it) / np.log(101),
         (float(precursor_mz) - mz) / 1000], axis=1,
    )
    return result, len(mz)


def spectrum_metadata(row: dict) -> tuple[np.ndarray, int]:
    adduct = str(row.get("adduct") or "")
    mode = str(row.get("ionization_mode") or "").lower()
    if mode in ("positive", "pos", "1", "+"):
        polarity = 1.0
    elif mode in ("negative", "neg", "-1", "-"):
        polarity = -1.0
    elif adduct.endswith("+"):
        polarity = 1.0
    elif adduct.endswith("-"):
        polarity = -1.0
    else:
        polarity = 0.0
    ce = row.get("collision_energy_ev")
    try:
        # The shared cache computes CE from float32 before reducing; preserving
        # that order also avoids one-ULP alignment failures on values like 77.1.
        values = np.asarray(ce if ce is not None else [], dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
    except (TypeError, ValueError):
        values = np.empty(0)
    meta = np.asarray([
        float(row["precursor_mz"]) / 1000,
        min(float(values.mean()), 500) / 100 if len(values) else 0,
        float(bool(len(values))), polarity,
    ], dtype=np.float32)
    return meta, ADDUCTS.index(adduct) + 1 if adduct in ADDUCTS else 0


class FormulaCatalog:
    def __init__(self, counts: np.ndarray, *, max_peaks: int = 96, ppm: float = 20.0, k: int = 3):
        self.counts = np.asarray(counts, dtype=np.int16)
        masses = self.counts @ EXACT_MASS
        if not np.all(np.diff(masses) >= 0):
            order = np.argsort(masses)
            self.counts, masses = self.counts[order], masses[order]
        self.masses = masses
        self.max_peaks = max_peaks
        self.ppm = ppm
        self.k = k
        self._allowed_cache = {}

    def _allowed(self, parent: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
        if parent in self._allowed_cache:
            return self._allowed_cache[parent]
        mask = np.all(self.counts <= np.asarray(parent), axis=1)
        if len(self._allowed_cache) >= 2048:
            self._allowed_cache.clear()
        result = self.counts[mask], self.masses[mask]
        self._allowed_cache[parent] = result
        return result

    def assign(
        self, peak_mz: np.ndarray, precursor_mz: float, parent: tuple[int, int, int, int],
        *, adduct: str = "", polarity: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return K candidate subformulas per ion and neutral loss, never a forced label."""
        counts, masses = self._allowed(tuple(parent))
        n = len(peak_mz)
        ion_values = np.zeros((self.max_peaks, self.k, 5), dtype=np.float32)
        ion_channels = np.zeros((self.max_peaks, self.k), dtype=np.int64)
        loss_values = np.zeros_like(ion_values)
        loss_channels = np.zeros_like(ion_channels)
        if not len(masses) or n == 0:
            return ion_values, ion_channels, loss_values, loss_channels
        channels = ([1, 3, 5, 6] if polarity >= 0 else [2, 4])
        if adduct != "[M+Na]+":
            channels = [c for c in channels if c != 5]
        if adduct != "[M+K]+":
            channels = [c for c in channels if c != 6]
        self._assign_many(peak_mz, counts, masses, channels, ion_values, ion_channels)
        # Subtracting m/z values yields a neutral loss only when precursor and
        # fragment retain the same single charge. Do not invent such labels for
        # doubly charged precursors.
        if adduct not in ("[M+2H]2+", "[M-2H]2-"):
            losses = float(precursor_mz) - peak_mz
            self._assign_many(losses, counts, masses, [7], loss_values, loss_channels)
        return ion_values, ion_channels, loss_values, loss_channels

    def _assign_many(self, targets, counts, masses, channels, values, channel_out):
        n = min(len(targets), self.max_peaks)
        targets = np.asarray(targets[:n], dtype=np.float64)
        options = []
        for channel in channels:
            offset = 0.0 if channel == 7 else ION_SHIFTS[channel]
            neutral = targets - offset
            insertion = np.searchsorted(masses, neutral)
            for delta in (-2, -1, 0, 1, 2):
                indices = np.clip(insertion + delta, 0, len(masses) - 1)
                error = (masses[indices] - neutral) / np.maximum(targets, 1.0) * 1e6
                valid = (targets > 0) & (np.abs(error) <= self.ppm)
                options.append((indices, error, valid, channel))
        if not options:
            return
        all_err = np.stack([np.where(o[2], np.abs(o[1]), np.inf) for o in options], axis=1)
        for row in range(n):
            chosen = set()
            slot = 0
            for option in np.argsort(all_err[row], kind="stable"):
                if not np.isfinite(all_err[row, option]):
                    break
                indices, errors, _, channel = options[int(option)]
                key = (int(indices[row]), channel)
                if key in chosen:
                    continue
                chosen.add(key)
                values[row, slot, :4] = counts[indices[row]] / np.array([20, 40, 6, 6])
                values[row, slot, 4] = np.clip(errors[row] / self.ppm, -1, 1)
                channel_out[row, slot] = channel
                slot += 1
                if slot == self.k:
                    break


class FormulaTransformer(nn.Module):
    def __init__(self, *, width=128, layers=2, heads=4, fp_bits=2048, dropout=0.1,
                 formula_features=True):
        super().__init__()
        self.formula_features = formula_features
        self.config = dict(
            width=width, layers=layers, heads=heads, fp_bits=fp_bits, dropout=dropout,
        )
        if not formula_features:
            self.config["formula_features"] = False
        self.register_buffer("freq", 2.0 ** torch.arange(10).float())
        self.peak = nn.Linear(44, width)
        self.formula = nn.Linear(5, width, bias=False)
        self.channel = nn.Embedding(8, width, padding_idx=0)
        self.parent = nn.Linear(4, width)
        self.meta = nn.Linear(4, width)
        self.adduct = nn.Embedding(len(ADDUCTS) + 1, width)
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        encoder = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.fp_head = nn.Linear(width, fp_bits)

    def forward(self, x, lengths, meta, adduct, parent, ion_values, ion_channels,
                loss_values, loss_channels):
        phase = x[:, :, [0, 3]].unsqueeze(-1) * self.freq * math.pi
        peak = self.peak(torch.cat([x, phase.sin().flatten(2), phase.cos().flatten(2)], -1))
        for values, channels in ((ion_values, ion_channels), (loss_values, loss_channels)):
            if not self.formula_features:
                continue
            encoded = self.formula(values) + self.channel(channels)
            valid = (channels > 0).unsqueeze(-1)
            peak = peak + (encoded * valid).sum(2) / valid.sum(2).clamp_min(1)
        cls = (self.cls.expand(len(x), -1, -1) + self.meta(meta)[:, None]
               + self.adduct(adduct)[:, None] + self.parent(parent)[:, None])
        tokens = torch.cat([cls, peak], dim=1)
        padding = torch.arange(x.shape[1] + 1, device=x.device)[None] > lengths[:, None]
        hidden = self.norm(self.transformer(tokens, src_key_padding_mask=padding)[:, 0])
        return self.fp_head(hidden)


@functools.lru_cache(maxsize=200_000)
def candidate_fingerprint(
    smiles: str, fp_bits=2048,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Use the same stereo-free tautomer-canonical Morgan definition as training."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    Chem.RemoveStereochemistry(mol)
    mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    bits = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits)
    fp = bits.GetFingerprintAsNumPy(mol).astype(np.float32)
    return fp, formula_counts(rdMolDescriptors.CalcMolFormula(mol))


@functools.lru_cache(maxsize=4)
def _load_checkpoint(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = FormulaTransformer(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    catalog = FormulaCatalog(
        checkpoint["formula_counts"], max_peaks=checkpoint["max_peaks"],
        ppm=checkpoint["ppm"], k=checkpoint["candidate_formulas"],
    )
    return model, catalog


def score_candidates(
    spectra: list[dict], smiles: list[str], checkpoint_path: str | Path, device: str = "cpu",
) -> np.ndarray:
    """Score ordered candidate SMILES for spectra of one molecule (higher is better)."""
    if not spectra:
        raise ValueError("At least one query spectrum is required")
    model, catalog = _load_checkpoint(str(Path(checkpoint_path).resolve()), device)
    parsed = []
    for text in smiles:
        try:
            parsed.append(candidate_fingerprint(text, model.config["fp_bits"]))
        except (ValueError, RuntimeError):
            parsed.append(None)
    scores = np.full(len(smiles), -np.inf, dtype=np.float32)
    groups = {}
    for i, result in enumerate(parsed):
        if result is not None:
            groups.setdefault(result[1], []).append(i)
    with torch.inference_mode():
        for counts, indices in groups.items():
            arrays = [[] for _ in range(9)]
            for spectrum in spectra:
                x, length = spectrum_features(
                    spectrum["ms2_mzs"], spectrum["ms2_normalized_intensities"],
                    spectrum["precursor_mz"], max_peaks=catalog.max_peaks,
                )
                meta, adduct = spectrum_metadata(spectrum)
                n = length
                assignments = catalog.assign(
                    x[:n, 0] * 1000, spectrum["precursor_mz"], counts,
                    adduct=str(spectrum.get("adduct") or ""), polarity=float(meta[3]),
                )
                row = (x, np.int64(length), meta, np.int64(adduct),
                       np.asarray(counts, np.float32) / np.asarray([20, 40, 6, 6], np.float32),
                       *assignments)
                for output, value in zip(arrays, row, strict=True):
                    output.append(value)
            batch = [torch.as_tensor(np.stack(values), device=device) for values in arrays]
            logits = model(*batch).mean(0).sigmoid().cpu().numpy()
            candidate_fp = np.stack([parsed[i][0] for i in indices])
            intersection = candidate_fp @ logits
            union = candidate_fp.sum(1) + logits.sum() - intersection
            scores[indices] = intersection / np.maximum(union, 1e-8)
    return scores
