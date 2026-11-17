"""Audit measured Enveda-180 20/40/60 eV views without reading test labels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "data/derived/saner-multipositive/cache-parallel"
DEFAULT_METADATA = ROOT / "data/derived/mist-mixup-20261116/source-metadata"
DEFAULT_PATTERNS = ROOT / "data/derived/mist-mixup-20261116/ce-pattern-signatures"
DEFAULT_OUTPUT = ROOT / "data/derived/ce-views-20261117/data-audit.json"
ENERGIES = (20, 40, 60)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_composite_patterns(path: Path) -> list[int]:
    patterns = json.loads(path.read_text())
    expected_ev = [float(value).hex() for value in ENERGIES]
    original_options = {("20", "40", "60"), ("-60", "-40", "-20")}
    ids = [
        int(item["id"])
        for item in patterns
        if str(item["unit"]).lower() == "ev"
        and tuple(item["original_values_sorted"]) in original_options
        and item["ev_values_sorted_hex"] == expected_ev
    ]
    if len(ids) != 2:
        raise ValueError(f"Expected two exact Enveda 20/40/60 eV patterns, found {ids}")
    return sorted(ids)


def run(cache: Path, metadata: Path, patterns: Path, output: Path) -> dict:
    mol = pd.read_parquet(cache / "molecules.parquet", columns=["key", "split"])
    spec = pd.read_parquet(cache / "spectra.parquet", columns=["mol_idx", "adduct_id", "split"])
    n = len(spec)
    if mol.key.duplicated().any():
        raise ValueError("Structure key duplicated across molecule cache")
    mol_idx = spec.mol_idx.to_numpy(dtype=np.int64)
    if (mol_idx < 0).any() or (mol_idx >= len(mol)).any():
        raise ValueError("Spectrum molecule index out of bounds")
    if not np.array_equal(spec.split.to_numpy(), mol.split.to_numpy()[mol_idx]):
        raise ValueError("Spectrum and molecule split disagree")

    categories = json.loads((metadata / "categories.json").read_text())
    source_id = categories["ingest_lib"].index("enveda-180")
    instrument_id = categories["instrument_type"].index("timsTOF")
    ingest = np.load(metadata / "ingest_id.npy", mmap_mode="r")
    instrument = np.load(metadata / "instrument_id.npy", mmap_mode="r")
    kind = np.load(metadata / "ce_kind.npy", mmap_mode="r")
    ce_ev = np.load(metadata / "ce_ev.npy", mmap_mode="r")
    pattern_ids = np.load(patterns / "pattern_id.npy", mmap_mode="r")
    for array in (ingest, instrument, kind, ce_ev, pattern_ids):
        if len(array) != n:
            raise ValueError("CE sidecar and spectrum cache lengths differ")
    exact_ids = exact_composite_patterns(patterns / "patterns.json")

    enveda = ingest == source_id
    if np.any(instrument[enveda] != instrument_id):
        raise ValueError("Enveda-180 includes non-timsTOF or unknown instrument rows")
    adduct = spec.adduct_id.to_numpy(dtype=np.int16)
    selected = np.flatnonzero(enveda & (adduct > 0))
    spectrum_meta = np.memmap(cache / "metadata.bin", mode="r", dtype="<f4", shape=(n, 4))[selected]
    polarity = spectrum_meta[:, 3]
    if not np.isin(polarity, (-1.0, 1.0)).all():
        raise ValueError("Selected Enveda spectra have unknown polarity")

    rows = spec.iloc[selected].copy()
    rows["polarity"] = polarity.astype(np.int8)
    rows["precursor_mz"] = spectrum_meta[:, 0] * 1000
    rows["kind"] = kind[selected]
    rows["ce_ev"] = ce_ev[selected]
    rows["pattern_id"] = pattern_ids[selected]
    for energy in ENERGIES:
        rows[f"e{energy}"] = rows.kind.eq(1) & rows.ce_ev.eq(float(energy))
    rows["composite"] = rows.pattern_id.isin(exact_ids)
    flags = [f"e{energy}" for energy in ENERGIES] + ["composite"]
    group_cols = ["split", "mol_idx", "adduct_id", "polarity"]
    groups = rows.groupby(group_cols, sort=False)
    grouped = groups[flags].max()
    grouped["triple"] = grouped[flags[:3]].all(axis=1)
    grouped["complete"] = grouped.triple & grouped.composite
    precursor = groups.precursor_mz.agg(["min", "max", "median"])
    span_ppm = ((precursor["max"] - precursor["min"]) / precursor["median"] * 1e6)[grouped.complete]
    if not np.isfinite(span_ppm).all() or (span_ppm > 20).any():
        raise ValueError("Complete CE-view groups contain invalid or >20 ppm precursor spans")

    split_report = {}
    for split in ("train", "validation"):
        groups = grouped.loc[split]
        complete = groups[groups.complete].reset_index()
        split_span = span_ppm.loc[split]
        split_report[split] = {
            "source_rows_known_adduct": int(rows.split.eq(split).sum()),
            "molecule_adduct_polarity_groups": len(groups),
            "all_three_single_ce_groups": int(groups.triple.sum()),
            "complete_groups_with_exact_composite": len(complete),
            "complete_unique_molecule_keys": int(complete.mol_idx.nunique()),
            "complete_precursor_span_max_ppm": float(split_span.max()),
            "complete_precursor_span_p99_ppm": float(split_span.quantile(0.99)),
            "complete_by_polarity": {
                str(key): int(value)
                for key, value in complete.polarity.value_counts().sort_index().items()
            },
            "complete_by_adduct_id": {
                str(key): int(value)
                for key, value in complete.adduct_id.value_counts().sort_index().items()
            },
        }

    report = {
        "definition": (
            "Enveda-180 timsTOF, known adduct and polarity, same molecule/adduct/polarity, "
            "exact 20/40/60 eV singles and exact original 3-CE composite"
        ),
        "split_source": "Existing Saner molecule-disjoint cache split; no new split created",
        "cache_molecules": len(mol),
        "cache_spectra": n,
        "enveda_rows_all_adducts": int(enveda.sum()),
        "enveda_rows_known_adduct": len(rows),
        "exact_composite_pattern_ids": exact_ids,
        "split": split_report,
        "sha256": {
            "cache_molecules": sha256(cache / "molecules.parquet"),
            "cache_spectra": sha256(cache / "spectra.parquet"),
            "source_categories": sha256(metadata / "categories.json"),
            "source_ce_ev": sha256(metadata / "ce_ev.npy"),
            "source_ce_kind": sha256(metadata / "ce_kind.npy"),
            "source_ingest_id": sha256(metadata / "ingest_id.npy"),
            "source_instrument_id": sha256(metadata / "instrument_id.npy"),
            "ce_pattern_ids": sha256(patterns / "pattern_id.npy"),
            "ce_pattern_definitions": sha256(patterns / "patterns.json"),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--patterns", type=Path, default=DEFAULT_PATTERNS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.cache, args.metadata, args.patterns, args.output)
    print(json.dumps({"output": str(args.output), "split": report["split"]}))


if __name__ == "__main__":
    main()
