"""M1 retrain with ambiguous CE-kind rows excluded from spectrum pairing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from mixup_pairs import merge_peaks
from model import (
    ADDUCTS,
    FormulaCatalog,
    FormulaTransformer,
    formula_counts,
    read_formula_catalog,
    spectrum_features,
)
from torch.nn import functional as F


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


class Cache:
    def __init__(self, root: Path, precise_mzs: Path, source_metadata: Path):
        self.mol = pd.read_parquet(root / "molecules.parquet")
        self.spec = pd.read_parquet(root / "spectra.parquet")
        n = len(self.spec)
        self.x = np.memmap(root / "peaks.bin", mode="r", dtype="<f2", shape=(n, 96, 4))
        self.mz = np.memmap(precise_mzs, mode="r", dtype="<f4", shape=(n, 96))
        self.meta = np.memmap(root / "metadata.bin", mode="r", dtype="<f4", shape=(n, 4))
        self.ingest_id = np.load(source_metadata / "ingest_id.npy", mmap_mode="r")
        self.instrument_id = np.load(source_metadata / "instrument_id.npy", mmap_mode="r")
        self.ce_kind = np.load(source_metadata / "ce_kind.npy", mmap_mode="r")
        self.ce_ev = np.load(source_metadata / "ce_ev.npy", mmap_mode="r")
        if any(len(array) != n for array in (
            self.ingest_id, self.instrument_id, self.ce_kind, self.ce_ev
        )):
            raise AssertionError("Source metadata is not aligned to the spectrum cache")
        known_ce = self.meta[:, 2] > 0.5
        valid_ce = self.ce_kind <= 1
        if not np.array_equal(known_ce[valid_ce], self.ce_kind[valid_ce] == 1):
            raise AssertionError("ce_kind 0/1 disagrees with the original known-CE flag")
        self.ingest_categories = json.loads(
            (source_metadata / "categories.json").read_text()
        )["ingest_lib"]
        self.fp = np.load(root / "fingerprints.npy", mmap_mode="r")
        self.length = self.spec.length.to_numpy(dtype=np.int64)
        self.adduct = self.spec.adduct_id.to_numpy(dtype=np.int64)
        self.mol_idx = self.spec.mol_idx.to_numpy(dtype=np.int64)
        self.formulas = [formula_counts(f) for f in self.mol.formula]
        self.parent = np.asarray(self.formulas, dtype=np.float32) / [20, 40, 6, 6]
        self.train_mols = np.flatnonzero(self.mol.split.eq("train"))
        self.val_mols = np.flatnonzero(self.mol.split.eq("validation"))
        if set(self.train_mols) & set(self.val_mols):
            raise AssertionError("Train/validation molecule overlap")
        self.train_spectra = defaultdict(list)
        for row in np.flatnonzero(self.spec.split.eq("train")):
            self.train_spectra[int(self.mol_idx[row])].append(int(row))
        self.same_formula = defaultdict(list)
        for mol in self.train_mols:
            self.same_formula[self.mol.formula.iloc[mol]].append(int(mol))
        self.same_mol_adduct_source_instrument_kind_polarity = defaultdict(list)
        for row in np.flatnonzero(self.spec.split.eq("train")):
            if self.instrument_id[row] < 0 or self.ce_kind[row] > 1:
                continue
            key = (int(self.mol_idx[row]), int(self.adduct[row]),
                   int(self.ingest_id[row]), int(self.instrument_id[row]),
                   int(self.ce_kind[row]),
                   int(self.meta[row, 3]))
            self.same_mol_adduct_source_instrument_kind_polarity[key].append(int(row))

    def partner(self, row: int, rng: np.random.Generator) -> int:
        if self.instrument_id[row] < 0 or self.ce_kind[row] > 1:
            return -1
        key = (int(self.mol_idx[row]), int(self.adduct[row]),
               int(self.ingest_id[row]), int(self.instrument_id[row]),
               int(self.ce_kind[row]),
               int(self.meta[row, 3]))
        candidates = self.same_mol_adduct_source_instrument_kind_polarity[key]
        if len(candidates) < 2:
            return -1
        main_ce_known = self.ce_kind[row] == 1
        main_ce = float(self.ce_ev[row])
        if main_ce_known and not np.isfinite(main_ce):
            raise AssertionError("Single CE sidecar value must be finite")
        main_mz = self.meta[row, 0]
        for _ in range(min(12, len(candidates) * 2)):
            other = int(candidates[int(rng.integers(len(candidates)))])
            if other == row:
                continue
            if bool(self.meta[other, 2] > 0.5) != bool(main_ce_known):
                continue
            if self.ce_kind[other] != self.ce_kind[row]:
                raise AssertionError("Accepted CE kinds 0 and 1 must match")
            if main_ce_known and (not np.isfinite(self.ce_ev[other])
                                  or abs(float(self.ce_ev[other]) - main_ce) > 10):
                continue
            if abs(self.meta[other, 0] - main_mz) > max(main_mz * 20e-6, 1e-7):
                continue
            return other
        return -1

    def mixed_spectrum(self, row: int, other: int, alpha: float):
        n1, n2 = int(self.length[row]), int(self.length[other])
        mz, intensity = merge_peaks(
            self.mz[row, :n1],
            np.asarray(self.x[row, :n1, 1], dtype=np.float32) ** 2,
            self.mz[other, :n2],
            np.asarray(self.x[other, :n2, 1], dtype=np.float32) ** 2,
            alpha, ppm=20.0,
        )
        feature, n = spectrum_features(
            mz, intensity, float(self.meta[row, 0] * 1000), max_peaks=96
        )
        # Rebuild the exact selected m/z used for formula assignment.
        selected = np.argsort(-intensity, kind="stable")[:96]
        selected = selected[np.argsort(mz[selected], kind="stable")]
        mz96 = np.zeros(96, dtype=np.float32)
        mz96[:n] = mz[selected]
        return feature.astype(np.float16).astype(np.float32), mz96, n

    def batch(self, rows: np.ndarray, catalog: FormulaCatalog, device: str,
              partners=None, alphas=None):
        features = np.asarray(self.x[rows], dtype=np.float32).copy()
        lengths = np.asarray(self.length[rows], dtype=np.int64).copy()
        mixed_mz = {}
        if partners is not None:
            for i, partner in enumerate(partners):
                if partner >= 0:
                    feature, mz, n = self.mixed_spectrum(int(rows[i]), int(partner),
                                                          float(alphas[i]))
                    features[i], lengths[i], mixed_mz[i] = feature, n, mz
        assignments = [[], [], [], []]
        for i, row in enumerate(rows):
            row = int(row)
            n = int(lengths[i])
            adduct = ADDUCTS[self.adduct[row] - 1] if self.adduct[row] else ""
            result = catalog.assign(
                mixed_mz[i][:n] if i in mixed_mz else self.mz[row, :n],
                float(self.meta[row, 0] * 1000),
                self.formulas[self.mol_idx[row]], adduct=adduct,
                polarity=float(self.meta[row, 3]),
            )
            for dest, feature in zip(assignments, result, strict=True):
                dest.append(feature)
        arrays = (
            features, lengths,
            np.asarray(self.meta[rows], dtype=np.float32), self.adduct[rows],
            np.asarray(self.parent[self.mol_idx[rows]], dtype=np.float32),
            *(np.stack(part) for part in assignments),
        )
        return [torch.as_tensor(a, device=device) for a in arrays]


def tanimoto(query: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    intersection = (query * fp).sum(-1)
    return intersection / (query.sum(-1) + fp.sum(-1) - intersection).clamp_min(1e-8)


@torch.inference_mode()
def evaluate(
    model: FormulaTransformer, data: Cache, catalog: FormulaCatalog,
    panel: pd.DataFrame, device: str, limit: int,
) -> dict:
    model.eval()
    reciprocal = []
    ranks = []
    query_mols = []
    for item in panel.itertuples(index=False):
        mol = int(item.mol_idx)
        if data.mol.split.iloc[mol] != "validation":
            raise AssertionError("Panel query is not held out")
        candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
        target = int(item.target_index)
        if candidates[target] != mol:
            raise AssertionError("Panel target does not match query molecule")
        rows = np.asarray(item.query_cache_rows, dtype=np.int64)
        # The panel uses same-formula candidates, so one formula-conditioned
        # spectrum encoding scores every candidate in the group.
        parent = {data.formulas[int(i)] for i in candidates}
        if len(parent) != 1:
            raise AssertionError("Expected same-formula candidate panel")
        probs = model(*data.batch(rows, catalog, device)).mean(0).sigmoid()
        fp = torch.as_tensor(np.asarray(data.fp[candidates], dtype=np.float32), device=device)
        scores = tanimoto(probs[None], fp).cpu().numpy()
        rank = int(np.count_nonzero(scores >= scores[target]))
        # >= includes the target itself and yields pessimistic tie handling.
        ranks.append(rank)
        query_mols.append(mol)
        reciprocal.append(1 / rank if rank <= 25 else 0.0)
        if len(ranks) >= limit:
            break
    if not ranks:
        raise AssertionError("No validation queries")
    return {"queries": len(ranks), "mrr25": float(np.mean(reciprocal)),
            "top1": float(np.mean(np.asarray(ranks) == 1)),
            "top25": float(np.mean(np.asarray(ranks) <= 25)),
            "query_mols": query_mols, "ranks": ranks}


def save_checkpoint(model, catalog, config, path: Path, step: int):
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": model.config,
        "formula_counts": catalog.counts,
        "max_peaks": catalog.max_peaks,
        "ppm": catalog.ppm,
        "candidate_formulas": catalog.k,
        "step": step,
        "seed": config.seed,
    }
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def run(config):
    output = config.output
    output.mkdir(parents=True, exist_ok=True)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(4)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)
    mix_rng = np.random.default_rng(config.seed + 300_007)
    data = Cache(config.cache, config.precise_mzs, config.source_metadata)
    counts, formula_sha = read_formula_catalog(config.formula_csv)
    catalog = FormulaCatalog(counts, max_peaks=96, ppm=config.ppm, k=3)
    panel = pd.read_parquet(config.panel)
    if not panel.mol_idx.isin(data.val_mols).all():
        raise AssertionError("Common panel must contain validation molecules only")
    model = FormulaTransformer(width=config.width, layers=config.layers).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=config.device == "cuda")
    # Count fingerprint positives on training molecules only.
    bit_counts = np.zeros(2048, dtype=np.float64)
    for start in range(0, len(data.train_mols), 4096):
        bit_counts += np.asarray(data.fp[data.train_mols[start:start+4096]]).sum(0)
    pos_weight = torch.as_tensor(
        np.clip((len(data.train_mols) - bit_counts) / (bit_counts + 1), 1, 20),
        device=config.device, dtype=torch.float32,
    )
    history = []
    milestone_validation = {}
    mix_attempted = 0
    mix_applied = 0
    attempts_by_source = defaultdict(int)
    applied_by_source = defaultdict(int)
    eligible_spectra = set()
    eligible_molecules = set()
    start_time = time.monotonic()
    for step in range(1, config.steps + 1):
        model.train()
        ids = rng.choice(data.train_mols, size=config.batch_size, replace=False)
        rows = np.asarray([rng.choice(data.train_spectra[int(i)]) for i in ids], np.int64)
        negatives = []
        for mol_id in ids:
            same = data.same_formula[data.mol.formula.iloc[mol_id]]
            if len(same) > 1:
                other = int(rng.choice(same))
                while other == mol_id:
                    other = int(rng.choice(same))
            else:
                other = int(rng.choice(data.train_mols))
            negatives.append(other)
        truth = torch.as_tensor(np.asarray(data.fp[ids], dtype=np.float32), device=config.device)
        decoys = torch.as_tensor(np.asarray(data.fp[negatives], dtype=np.float32),
                                 device=config.device)
        partners = np.full(len(rows), -1, dtype=np.int64)
        alphas = np.ones(len(rows), dtype=np.float32)
        masks = mix_rng.random(len(rows)) < config.mix_probability
        mix_attempted += int(masks.sum())
        for i in np.flatnonzero(masks):
            source = data.ingest_categories[int(data.ingest_id[int(rows[i])])]
            attempts_by_source[source] += 1
            partners[i] = data.partner(int(rows[i]), mix_rng)
            if partners[i] >= 0:
                alphas[i] = mix_rng.uniform(config.mix_alpha_min, config.mix_alpha_max)
                mix_applied += 1
                applied_by_source[source] += 1
                eligible_spectra.add(int(rows[i]))
                eligible_molecules.add(int(data.mol_idx[int(rows[i])]))
        batch = data.batch(rows, catalog, config.device, partners, alphas)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=config.device == "cuda"):
            logits = model(*batch)
            bce = F.binary_cross_entropy_with_logits(logits.float(), truth, pos_weight=pos_weight)
            probabilities = logits.float().sigmoid()
            positive_score = tanimoto(probabilities, truth)
            negative_score = tanimoto(probabilities, decoys)
            ranking = F.softplus(10 * (negative_score - positive_score)).mean()
            loss = bce + config.rank_weight * ranking
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step == 1 or step % config.log_every == 0 or step == config.steps:
            event = {"step": step, "loss": float(loss.detach()),
                     "bce": float(bce.detach()), "ranking": float(ranking.detach()),
                     "seconds": round(time.monotonic()-start_time, 1),
                     "mix_attempted": mix_attempted, "mix_applied": mix_applied}
            print(json.dumps(event), flush=True)
            history.append(event)
            (output / "history.json").write_text(json.dumps(history, indent=2))
        if step % config.save_every == 0 or step == config.steps:
            save_checkpoint(model, catalog, config, output / "last.pt", step)
        if step in config.milestone_steps:
            checkpoint = output / f"step-{step}.pt"
            save_checkpoint(model, catalog, config, checkpoint, step)
            metrics = evaluate(model, data, catalog, panel, config.device, config.eval_queries)
            milestone_validation[str(step)] = {
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": digest(checkpoint),
                "validation": metrics,
            }
            (output / f"validation-step-{step}.json").write_text(
                json.dumps(milestone_validation[str(step)], indent=2)
            )
            print(json.dumps({"milestone": step, **milestone_validation[str(step)]}), flush=True)
    evaluation = milestone_validation[str(config.steps)]["validation"] if (
        config.steps in config.milestone_steps
    ) else evaluate(model, data, catalog, panel, config.device, config.eval_queries)
    save_checkpoint(model, catalog, config, output / "final.pt", config.steps)
    manifest = {
        "model": "candidate-formula-conditioned spectrum-to-Morgan-fingerprint Transformer",
        "formula_assignment": (
            "up to 3 CHNO subformula candidates within ppm tolerance per peak and neutral loss"
        ),
        "ion_channels": (
            "protonated, deprotonated, radical cation/anion; Na/K when precursor adduct allows"
        ),
        "non_chno": (
            "raw m/z, intensity, neutral-loss Fourier features remain active; "
            "unmatched formulas are zero-masked"
        ),
        "formula_mass": "recomputed from monoisotopic atomic masses, not CSV mass column",
        "data": str(config.cache), "precise_mzs": str(config.precise_mzs),
        "source_metadata": str(config.source_metadata),
        "raw_train_sha256": digest(config.raw_train),
        "precise_mzs_alignment": (
            "all accepted cache rows were matched in source order by peak features, "
            "metadata (float32 tolerance 1e-6), adduct, length, and "
            "molecular InChIKey14 or molecular formula"
        ),
        "aligned_spectra": len(data.spec),
        "cache_spectra_sha256": digest(config.cache / "spectra.parquet"),
        "cache_molecules_sha256": digest(config.cache / "molecules.parquet"),
        "precise_mzs_sha256": digest(config.precise_mzs),
        "ingest_id_sha256": digest(config.source_metadata / "ingest_id.npy"),
        "instrument_id_sha256": digest(config.source_metadata / "instrument_id.npy"),
        "ce_kind_sha256": digest(config.source_metadata / "ce_kind.npy"),
        "ce_ev_sha256": digest(config.source_metadata / "ce_ev.npy"),
        "shared_peak_merge_source_sha256": digest(Path(__file__).with_name("mixup_pairs.py")),
        "formula_csv_sha256": formula_sha, "formula_rows": len(counts),
        "panel_sha256": digest(config.panel), "split": "existing Saner molecule-disjoint split",
        "seed": config.seed, "steps": config.steps, "batch_size": config.batch_size,
        "learning_rate": config.lr, "validation": evaluation,
        "mixup": {"kind": (
                      "same molecule, adduct, polarity, ingest_lib, known instrument; "
                      "matched CE and precursor; exclude ce_kind > 1 on both spectra"
                  ),
                  "ce_kind_policy": "only 0 (missing) or 1 (single numeric); exclude 2 (multiple/invalid)",
                  "ce_comparison": "sidecar ce_ev within 10 eV for kind 1; kind 0 matches kind 0",
                  "pair_index_key": "molecule, adduct, ingest_lib, instrument, ce_kind, polarity",
                  "probability": config.mix_probability,
                  "alpha_range": [config.mix_alpha_min, config.mix_alpha_max],
                  "attempted": mix_attempted, "applied": mix_applied,
                  "attempts_by_source": dict(attempts_by_source),
                  "applied_by_source": dict(applied_by_source),
                  "observed_eligible_spectra": len(eligible_spectra),
                  "observed_eligible_molecules": len(eligible_molecules),
                  "max_partner_search_draws": 12,
                  "intensity_reconstruction": (
                      "squared float16-quantized sqrt intensity from selected top96 peaks"
                  ),
                  "peak_merge_policy": "shared mixup_pairs.merge_peaks: strongest peak m/z within each 20ppm cluster",
                  "merged_mz_ppm": 20},
        "milestones": milestone_validation,
        "checkpoint_sha256": digest(output / "final.pt"),
        "trainer_source_sha256": digest(Path(__file__)),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps({"validation": evaluation, "checkpoint": str(output / "final.pt")}),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--formula-csv", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--mix-probability", type=float, default=0.25)
    parser.add_argument("--mix-alpha-min", type=float, default=0.25)
    parser.add_argument("--mix-alpha-max", type=float, default=0.75)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ppm", type=float, default=20.0)
    parser.add_argument("--eval-queries", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--milestone-steps", nargs="*", type=int, default=[])
    args = parser.parse_args()
    if args.batch_size < 2 or args.steps < 1 or args.eval_queries < 1:
        parser.error("batch-size >= 2, steps >= 1 and eval-queries >= 1 required")
    if len(set(args.milestone_steps)) != len(args.milestone_steps) or any(
        step < 1 or step > args.steps for step in args.milestone_steps
    ):
        parser.error("milestone steps must be unique and within 1..steps")
    run(args)


if __name__ == "__main__":
    main()
