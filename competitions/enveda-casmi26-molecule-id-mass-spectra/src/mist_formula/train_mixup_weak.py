"""Train and evaluate the formula-conditioned Transformer on the shared split."""

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
from mixup_pairs import PartnerPool, mixed_batch
from model import ADDUCTS, FormulaCatalog, FormulaTransformer, formula_counts, read_formula_catalog
from torch.nn import functional as F


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


class Cache:
    def __init__(self, root: Path, precise_mzs: Path):
        self.precise_raw_features = False
        self.mol = pd.read_parquet(root / "molecules.parquet")
        self.spec = pd.read_parquet(root / "spectra.parquet")
        n = len(self.spec)
        self.x = np.memmap(root / "peaks.bin", mode="r", dtype="<f2", shape=(n, 96, 4))
        self.mz = np.memmap(precise_mzs, mode="r", dtype="<f4", shape=(n, 96))
        self.meta = np.memmap(root / "metadata.bin", mode="r", dtype="<f4", shape=(n, 4))
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

    def batch(self, rows: np.ndarray, catalog: FormulaCatalog, device: str):
        assignments = [[], [], [], []]
        for row in rows:
            row = int(row)
            n = int(self.length[row])
            adduct = ADDUCTS[self.adduct[row] - 1] if self.adduct[row] else ""
            result = catalog.assign(
                self.mz[row, :n], float(self.meta[row, 0] * 1000),
                self.formulas[self.mol_idx[row]], adduct=adduct,
                polarity=float(self.meta[row, 3]),
            )
            for dest, feature in zip(assignments, result, strict=True):
                dest.append(feature)
        arrays = (
            np.asarray(self.x[rows], dtype=np.float32), self.length[rows],
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
        "mixup": {"mode": "cross", "fraction": config.mixup_fraction,
                  "anchor_min": config.anchor_min, "anchor_max": config.anchor_max,
                  "partner_seed": config.seed + 200_003,
                  "metadata_sha256": digest(config.metadata_dir / "manifest.json")},
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
    mix_rng = np.random.default_rng(config.seed + 200_003)
    data = Cache(config.cache, config.precise_mzs)
    partner_pool = PartnerPool(data, config.metadata_dir, mode="cross")
    source_metadata = json.loads((config.metadata_dir / "manifest.json").read_text())
    query_hash = hashlib.sha256()
    mix_hash = hashlib.sha256()
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
    start_time = time.monotonic()
    for step in range(1, config.steps + 1):
        model.train()
        ids = rng.choice(data.train_mols, size=config.batch_size, replace=False)
        rows = np.asarray([rng.choice(data.train_spectra[int(i)]) for i in ids], np.int64)
        query_hash.update(ids.tobytes())
        query_hash.update(rows.tobytes())
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
        fractions = mix_rng.uniform(config.anchor_min, config.anchor_max, size=len(rows))
        flagged = mix_rng.random(len(rows)) < config.mixup_fraction
        for pos in np.flatnonzero(flagged):
            partner = partner_pool.sample_partner(int(rows[pos]), mix_rng)
            if partner is not None:
                partners[pos] = partner
        mix_hash.update(partners.tobytes())
        mix_hash.update(fractions.astype("<f4").tobytes())
        batch = mixed_batch(data, catalog, rows, partners, fractions, config.device)
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
                     "seconds": round(time.monotonic()-start_time, 1)}
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
    total_queries = config.steps * config.batch_size
    manifest = {
        "model": "candidate-formula-conditioned spectrum-to-Morgan-fingerprint Transformer",
        "mixup": {"mode": "cross", "requested_fraction": config.mixup_fraction,
                  "anchor_min": config.anchor_min, "anchor_max": config.anchor_max,
                  "partner_seed": config.seed + 200_003,
                  "partner_pool": partner_pool.stats,
                  "attempted_fraction_of_queries": partner_pool.stats["attempted"] / total_queries,
                  "effective_fraction_of_queries": partner_pool.stats["paired"] / total_queries,
                  "success_fraction_of_attempts": (
                      partner_pool.stats["paired"] / partner_pool.stats["attempted"]
                      if partner_pool.stats["attempted"] else 0.0
                  ),
                  "partner_stream_sha256": mix_hash.hexdigest(),
                  "metadata_manifest_sha256": digest(config.metadata_dir / "manifest.json"),
                  "peak_source": "frozen top96 precise m/z and recovered linear relative intensity",
                  "peak_merge_ppm": 20, "ce_tol_ev": 10, "precursor_tol_ppm": 20},
        "source_metadata_coverage": {
            "aligned_cache_rows": source_metadata["rows"],
            "raw_rows_seen": source_metadata["raw_rows_seen"],
            "known_instrument_rows": source_metadata["known_instrument_rows"],
            "single_ce_rows": source_metadata["single_ce_rows"],
            "missing_ce_rows": source_metadata["missing_ce_rows"],
            "unknown_instrument_excluded": source_metadata["excluded_unknown_instrument_rows"],
            "libraries": source_metadata["libraries"],
        },
        "query_stream_sha256": query_hash.hexdigest(),
        "training_script_sha256": digest(Path(__file__)),
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
        "formula_csv_sha256": formula_sha, "formula_rows": len(counts),
        "panel_sha256": digest(config.panel), "split": "existing Saner molecule-disjoint split",
        "seed": config.seed, "steps": config.steps, "batch_size": config.batch_size,
        "learning_rate": config.lr, "validation": evaluation,
        "milestones": milestone_validation,
        "checkpoint_sha256": digest(output / "final.pt"),
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
    parser.add_argument("--metadata-dir", type=Path, required=True)
    parser.add_argument("--formula-csv", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--rank-weight", type=float, default=0.5)
    parser.add_argument("--mixup-fraction", type=float, default=0.25)
    parser.add_argument("--anchor-min", type=float, default=0.85)
    parser.add_argument("--anchor-max", type=float, default=0.95)
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
    if not (0 <= args.mixup_fraction <= 1 and 0 < args.anchor_min <= args.anchor_max < 1):
        parser.error("mixup fraction and anchor interval are invalid")
    if len(set(args.milestone_steps)) != len(args.milestone_steps) or any(
        step < 1 or step > args.steps for step in args.milestone_steps
    ):
        parser.error("milestone steps must be unique and within 1..steps")
    run(args)


if __name__ == "__main__":
    main()
