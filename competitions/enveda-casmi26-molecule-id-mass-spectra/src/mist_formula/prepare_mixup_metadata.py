"""Align original library/instrument/CE metadata to the frozen 96-peak cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from model import spectrum_features, spectrum_metadata


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run(raw: Path, cache: Path, precise: Path, output: Path) -> None:
    spec = pq.read_table(
        cache / "spectra.parquet", columns=["key", "length", "adduct_id", "mol_idx"]
    )
    mol = pq.read_table(cache / "molecules.parquet", columns=["formula"])
    keys = spec["key"].to_pylist()
    lengths = spec["length"].to_numpy()
    adducts = spec["adduct_id"].to_numpy()
    mol_idx = spec["mol_idx"].to_numpy()
    formulas = mol["formula"].to_pylist()
    n = len(keys)
    peaks = np.memmap(cache / "peaks.bin", mode="r", dtype="<f2", shape=(n, 96, 4))
    metadata = np.memmap(cache / "metadata.bin", mode="r", dtype="<f4", shape=(n, 4))
    mzs = np.memmap(precise, mode="r", dtype="<f4", shape=(n, 96))
    ingest_ids = np.full(n, -1, dtype=np.int32)
    instrument_ids = np.full(n, -1, dtype=np.int32)
    ce_ev = np.full(n, np.nan, dtype=np.float32)
    ce_kind = np.full(n, 2, dtype=np.int8)  # 0 missing; 1 single finite; 2 multi/invalid
    libs, instruments = {}, {}

    def encode(value: str | None, mapping: dict[str, int]) -> int:
        if not value:
            return -1
        if value not in mapping:
            mapping[value] = len(mapping)
        if value.strip().lower() in {"nan", "n/a", "na", "none", "unknown"}:
            return -1
        return mapping[value]

    output.mkdir(parents=True, exist_ok=True)
    reader = pq.ParquetFile(raw)
    columns = [
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "adduct",
        "ionization_mode",
        "collision_energy_ev",
        "molecular_formula",
        "inchikey14",
        "ingest_lib",
        "instrument_type",
    ]
    matched = raw_seen = excluded_unknown_instrument_rows = 0
    start = time.monotonic()
    for batch in reader.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            raw_seen += 1
            if matched >= n:
                break
            if (
                row["inchikey14"] != keys[matched]
                and row["molecular_formula"] != formulas[mol_idx[matched]]
            ):
                continue
            try:
                feature, length = spectrum_features(
                    row["ms2_mzs"], row["ms2_normalized_intensities"], row["precursor_mz"]
                )
                meta, adduct = spectrum_metadata(row)
            except (ValueError, TypeError, OverflowError):
                continue
            if length != lengths[matched] or adduct != adducts[matched]:
                continue
            if not np.array_equal(feature.astype("<f2"), peaks[matched]):
                continue
            if not np.array_equal((feature[:, 0] * 1000).astype("<f4"), mzs[matched]):
                continue
            if not np.allclose(meta, metadata[matched], rtol=0, atol=1e-6):
                continue
            ingest_ids[matched] = encode(row["ingest_lib"], libs)
            instrument_ids[matched] = encode(row["instrument_type"], instruments)
            if str(row["instrument_type"]).strip().lower() in {
                "nan", "n/a", "na", "none", "unknown"
            }:
                excluded_unknown_instrument_rows += 1
            ce = row["collision_energy_ev"]
            if ce is None or len(ce) == 0:
                ce_kind[matched] = 0
            elif len(ce) == 1 and ce[0] is not None and np.isfinite(ce[0]):
                ce_kind[matched] = 1
                ce_ev[matched] = ce[0]
            matched += 1
            if matched % 200000 == 0:
                print(
                    "aligned",
                    matched,
                    "of",
                    n,
                    "raw",
                    raw_seen,
                    "seconds",
                    round(time.monotonic() - start),
                    flush=True,
                )
        if matched >= n:
            break
    if matched != n:
        raise RuntimeError(f"Aligned only {matched}/{n} cache rows; no output saved")
    np.save(output / "ingest_id.npy", ingest_ids)
    np.save(output / "instrument_id.npy", instrument_ids)
    np.save(output / "ce_ev.npy", ce_ev)
    np.save(output / "ce_kind.npy", ce_kind)
    categories = {
        "ingest_lib": sorted(libs, key=libs.get),
        "instrument_type": sorted(instruments, key=instruments.get),
    }
    (output / "categories.json").write_text(json.dumps(categories, indent=2))
    manifest = {
        "raw_sha256": sha(raw),
        "cache_spectra_sha256": sha(cache / "spectra.parquet"),
        "precise_mzs_sha256": sha(precise),
        "rows": n,
        "raw_rows_seen": raw_seen,
        "known_instrument_rows": int((instrument_ids >= 0).sum()),
        "excluded_unknown_instrument_rows": excluded_unknown_instrument_rows,
        "unknown_instrument_tokens": ["nan", "N/A"],
        "single_ce_rows": int((ce_kind == 1).sum()),
        "missing_ce_rows": int((ce_kind == 0).sum()),
        "libraries": categories["ingest_lib"],
        "instruments": categories["instrument_type"],
        "seconds": round(time.monotonic() - start, 1),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.raw, args.cache, args.precise, args.output)


if __name__ == "__main__":
    main()
