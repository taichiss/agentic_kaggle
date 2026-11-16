"""Compare fixed, label-free collision-energy pooling rules for frozen A."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bcf_rerank import rerank
from evaluate_mixup_parent import rank as parent_rank
from frozen import a_0479_model as frozen
from train_mix_soft import Cache

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "data/derived"
CONDITIONS = ("equal", "ce_balanced", "ce_mid40")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ce_label(values: object, adduct: str) -> tuple[str, str, float]:
    """Keep missing, composite, and measured single energies distinct."""
    try:
        ce = np.asarray(values if values is not None else [], dtype=np.float32).reshape(-1)
        ce = ce[np.isfinite(ce)]
    except (TypeError, ValueError):
        ce = np.empty(0, dtype=np.float32)
    if not len(ce):
        return adduct, "missing", 0.0
    if len(ce) > 1:
        return adduct, "composite", 0.0
    return adduct, "single", round(float(ce[0]), 1)


def cache_label(
    row: int, data: Cache, kind: np.ndarray, energy: np.ndarray
) -> tuple[str, str, float]:
    adduct = int(data.adduct[row])
    k = int(kind[row])
    if k == 0:
        return str(adduct), "missing", 0.0
    if k == 2:
        return str(adduct), "composite", 0.0
    if k != 1 or not np.isfinite(energy[row]):
        raise ValueError(f"Unrecognized collision-energy metadata at cache row {row}")
    return str(adduct), "single", round(float(energy[row]), 1)


def pool(logits: torch.Tensor, labels: list[tuple[str, str, float]], mode: str) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[0] != len(labels):
        raise ValueError("Logit and CE-label shapes differ")
    if mode == "equal":
        return logits.mean(0)
    groups: dict[tuple[str, str, float], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    views = [logits[indices].mean(0) for indices in groups.values()]
    weights = [
        2.0 if mode == "ce_mid40" and label[1] == "single" and abs(label[2] - 40) <= 5 else 1.0
        for label in groups
    ]
    if mode not in CONDITIONS:
        raise ValueError(mode)
    return torch.stack([weight * view for weight, view in zip(weights, views, strict=True)]).sum(
        0
    ) / sum(weights)


def rank_scores(scores: np.ndarray, target: int) -> int:
    return int(np.count_nonzero(scores >= scores[target]))


def score_parent(
    spectra: list[dict],
    names: list[str],
    checkpoint: Path,
    device: str,
    mode: str,
) -> np.ndarray:
    model, catalog = frozen._load_checkpoint(str(checkpoint.resolve()), device)
    parsed = []
    for text in names:
        try:
            parsed.append(frozen.candidate_fingerprint(text, model.config["fp_bits"]))
        except (ValueError, RuntimeError):
            parsed.append(None)
    scores = np.full(len(names), -np.inf, dtype=np.float32)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index, item in enumerate(parsed):
        if item is not None:
            groups[item[1]].append(index)
    labels = [
        ce_label(row.get("collision_energy_ev"), str(row.get("adduct") or "")) for row in spectra
    ]
    with torch.inference_mode():
        for counts, indices in groups.items():
            arrays: list[list[np.ndarray | np.int64]] = [[] for _ in range(9)]
            for spectrum in spectra:
                x, length = frozen.spectrum_features(
                    spectrum["ms2_mzs"],
                    spectrum["ms2_normalized_intensities"],
                    spectrum["precursor_mz"],
                    max_peaks=catalog.max_peaks,
                )
                meta, adduct = frozen.spectrum_metadata(spectrum)
                assignments = catalog.assign(
                    x[:length, 0] * 1000,
                    spectrum["precursor_mz"],
                    counts,
                    adduct=str(spectrum.get("adduct") or ""),
                    polarity=float(meta[3]),
                )
                row = (
                    x,
                    np.int64(length),
                    meta,
                    np.int64(adduct),
                    np.asarray(counts, np.float32) / np.asarray([20, 40, 6, 6], np.float32),
                    *assignments,
                )
                for output, value in zip(arrays, row, strict=True):
                    output.append(value)
            batch = [torch.as_tensor(np.stack(values), device=device) for values in arrays]
            probs = pool(model(*batch), labels, mode).sigmoid().cpu().numpy()
            fp = np.stack([parsed[index][0] for index in indices])
            intersection = fp @ probs
            union = fp.sum(1) + probs.sum() - intersection
            scores[indices] = intersection / np.maximum(union, 1e-8)
    return scores


def panel1000(
    args: argparse.Namespace, model: torch.nn.Module, catalog: frozen.FormulaCatalog
) -> list[dict]:
    data = Cache(args.cache, args.precise_mzs)
    panel = pd.read_parquet(args.panel)
    if len(panel) != 1000 or len(set(panel.mol_idx)) != 1000:
        raise ValueError("Fixed 1000-molecule panel changed")
    kind = np.load(args.ce_metadata / "ce_kind.npy", mmap_mode="r")
    energy = np.load(args.ce_metadata / "ce_ev.npy", mmap_mode="r")
    if len(kind) != len(data.spec) or len(energy) != len(data.spec):
        raise ValueError("CE sidecar/cache row alignment changed")
    prior = {
        int(row["mol_idx"]): row
        for row in map(json.loads, args.panel_prior.read_text().splitlines())
    }
    records = []
    with torch.inference_mode():
        for item in panel.itertuples(index=False):
            mol = int(item.mol_idx)
            if data.mol.split.iloc[mol] != "validation":
                raise ValueError("Panel contains a training molecule")
            rows = np.asarray(item.query_cache_rows, dtype=np.int64)
            candidates = np.asarray(item.candidate_mol_indices, dtype=np.int64)
            target = int(item.target_index)
            if candidates[target] != mol or len(candidates) != 25:
                raise ValueError("Fixed panel candidates changed")
            labels = [cache_label(int(row), data, kind, energy) for row in rows]
            logits = model(*data.batch(rows, catalog, args.device))
            fp = torch.as_tensor(
                np.asarray(data.fp[candidates], dtype=np.float32), device=args.device
            )
            rec = {"mol_idx": mol, "ce_labels": labels, "n_ce_groups": len(set(labels))}
            for condition in CONDITIONS:
                probs = pool(logits, labels, condition).sigmoid()
                intersection = (probs[None] * fp).sum(-1)
                scores = intersection / (probs.sum() + fp.sum(-1) - intersection).clamp_min(1e-8)
                rec[condition + "_rank"] = rank_scores(scores.cpu().numpy(), target)
            if rec["equal_rank"] != prior[mol]["A_rank"]:
                raise ValueError(f"A panel rank parity failed at {mol}")
            records.append(rec)
    return records


def parent80(args: argparse.Namespace) -> list[dict]:
    control = pd.read_csv(args.parent_control, dtype=str, keep_default_na=False)
    spectra = pd.read_parquet(args.parent_spectra)
    groups = {
        str(mid): frame.to_dict("records")
        for mid, frame in spectra.groupby("molecule_id", sort=False)
    }
    truth = pd.read_csv(args.parent_labels, dtype=str).set_index("molecule_id").metric_key.to_dict()
    prior = {
        row["molecule_id"]: row
        for row in map(json.loads, args.parent_prior.read_text().splitlines())
    }
    if (
        len(control) != 80
        or set(control.molecule_id) != set(groups)
        or set(control.molecule_id) != set(truth)
    ):
        raise ValueError("Saved 80-molecule parent panel changed")
    records = []
    for item in control.itertuples(index=False):
        names = item.smiles.split(";")
        spectra_i = groups[item.molecule_id]
        labels = [
            ce_label(row.get("collision_energy_ev"), str(row.get("adduct") or ""))
            for row in spectra_i
        ]
        rec = {
            "molecule_id": item.molecule_id,
            "ce_labels": labels,
            "n_ce_groups": len(set(labels)),
        }
        for condition in CONDITIONS:
            scores = score_parent(spectra_i, names, args.checkpoint, args.device, condition)
            if condition == "equal":
                reference = frozen.score_candidates(spectra_i, names, args.checkpoint, args.device)
                if not np.allclose(scores, reference, rtol=0, atol=1e-7):
                    raise ValueError(f"Frozen A score parity failed at {item.molecule_id}")
            order, _ = rerank(names, scores, weight=0.25)
            ranked = [names[index] for index in order]
            if ranked[0] != names[0] or sorted(ranked) != sorted(names):
                raise ValueError("Candidate set or rank one changed")
            rec[condition + "_rank"] = parent_rank(ranked, truth[item.molecule_id])
        if rec["equal_rank"] != prior[item.molecule_id]["A_rank"]:
            raise ValueError(f"A parent rank parity failed at {item.molecule_id}")
        records.append(rec)
    return records


def summarize(rows: list[dict], seed: int) -> dict:
    values = {
        name: np.asarray([1 / row[name + "_rank"] if row[name + "_rank"] else 0 for row in rows])
        for name in CONDITIONS
    }
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(rows), size=(5000, len(rows)))
    result = {"molecules": len(rows), "conditions": {}}
    for name in CONDITIONS:
        ranks = [row[name + "_rank"] for row in rows]
        delta = values[name] - values["equal"]
        ci = np.quantile(delta[samples].mean(axis=1), [0.025, 0.975])
        result["conditions"][name] = {
            "mrr25": float(values[name].mean()),
            "top1": sum(rank == 1 for rank in ranks) / len(rows),
            "paired_vs_equal": {
                "delta_mrr25": float(delta.mean()),
                "bootstrap_95pct": [float(value) for value in ci],
                "improved": int(np.count_nonzero(delta > 0)),
                "worsened": int(np.count_nonzero(delta < 0)),
                "unchanged": int(np.count_nonzero(delta == 0)),
            },
        }
    result["ce_group_counts"] = {
        str(count): sum(row["n_ce_groups"] == count for row in rows)
        for count in sorted({row["n_ce_groups"] for row in rows})
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=DERIVED / "mist-duration-20261107/dataset-12k/final.pt"
    )
    parser.add_argument(
        "--cache", type=Path, default=DERIVED / "saner-multipositive/cache-parallel"
    )
    parser.add_argument(
        "--precise-mzs", type=Path, default=DERIVED / "mist-formula-20261107/precise-mzs.bin"
    )
    parser.add_argument(
        "--panel", type=Path, default=DERIVED / "model-comparison-20261107/common-panel.parquet"
    )
    parser.add_argument(
        "--ce-metadata", type=Path, default=DERIVED / "mist-mixup-20261116/source-metadata"
    )
    parser.add_argument(
        "--panel-prior",
        type=Path,
        default=DERIVED / "mist-bcf-20261110/panel-evaluation/per-molecule.jsonl",
    )
    parser.add_argument(
        "--parent-control", type=Path, default=DERIVED / "exp0015-cv/submission-control.csv"
    )
    parser.add_argument(
        "--parent-spectra",
        type=Path,
        default=DERIVED / "candidate-rescue-corrected-v2/test.parquet",
    )
    parser.add_argument(
        "--parent-labels",
        type=Path,
        default=DERIVED / "candidate-rescue-corrected-v2/labels-local-only.csv",
    )
    parser.add_argument(
        "--parent-prior",
        type=Path,
        default=DERIVED / "mist-bcf-20261110/parent-proxy-evaluation/per-molecule.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=DERIVED / "mist-mixup-20261116/ce-aggregation-local"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, catalog = frozen._load_checkpoint(str(args.checkpoint.resolve()), args.device)
    p1000 = panel1000(args, model, catalog)
    p80 = parent80(args)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "panel1000.jsonl").write_text("".join(json.dumps(row) + "\n" for row in p1000))
    (args.output / "parent80.jsonl").write_text("".join(json.dumps(row) + "\n" for row in p80))
    report = {
        "policy": {
            "equal": "Frozen A: equal spectrum logits, then sigmoid",
            "ce_balanced": "Equal CE/adduct-group logits after within-group average",
            "ce_mid40": "As ce_balanced, but a measured single 40±5 eV group has weight 2",
        },
        "panel1000": summarize(p1000, args.seed),
        "parent80": summarize(p80, args.seed),
        "input_sha256": {
            name: sha(getattr(args, name))
            for name in (
                "checkpoint",
                "precise_mzs",
                "panel",
                "panel_prior",
                "parent_control",
                "parent_spectra",
                "parent_labels",
                "parent_prior",
            )
        },
        "cache_sha256": {
            name: sha(args.cache / name)
            for name in (
                "molecules.parquet",
                "spectra.parquet",
                "peaks.bin",
                "metadata.bin",
                "fingerprints.npy",
            )
        },
        "sidecar_sha256": {
            name: sha(args.ce_metadata / name)
            for name in ("manifest.json", "ce_kind.npy", "ce_ev.npy")
        },
        "frozen_source_sha256": sha(Path(frozen.__file__)),
        "limitations": (
            "Fixed1000 is an artificial same-formula panel; parent80 has only 54 "
            "truth-present rows and is not unbiased. Neither estimates Public LB."
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"panel1000": report["panel1000"], "parent80": report["parent80"]}), flush=True
    )


if __name__ == "__main__":
    main()
