"""Restore accurate peak m/z values aligned to the existing Saner cache.

The shared cache stores four peak features as float16, which loses too much
precision for formula assignment. This read-only pass replays its preprocessing
against official train.parquet and writes precise selected m/z as float32.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from model import spectrum_features, spectrum_metadata


def prepare(
    raw_path: Path, cache_path: Path, output: Path,
    *, resume_raw_index: int = 0, resume_cache_index: int = 0,
) -> None:
    spec = pq.read_table(
        cache_path / "spectra.parquet", columns=["length", "adduct_id", "mol_idx", "key"],
    )
    mol = pq.read_table(cache_path / "molecules.parquet", columns=["formula"])
    lengths = spec["length"].to_numpy()
    adducts = spec["adduct_id"].to_numpy()
    keys = spec["key"].to_pylist()
    formulas = mol["formula"].to_pylist()
    mol_idx = spec["mol_idx"].to_numpy()
    n = len(lengths)
    peaks = np.memmap(cache_path / "peaks.bin", mode="r", dtype="<f2", shape=(n, 96, 4))
    metadata = np.memmap(cache_path / "metadata.bin", mode="r", dtype="<f4", shape=(n, 4))
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    if resume_cache_index and temp.stat().st_size != resume_cache_index * 96 * 4:
        raise ValueError("Resume cache index does not match temporary file size")
    written = resume_cache_index
    raw_seen = 0
    start = time.monotonic()
    with temp.open("ab" if resume_cache_index else "wb") as handle:
        reader = pq.ParquetFile(raw_path)
        columns = ["ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
                   "ionization_mode", "collision_energy_ev", "molecular_formula", "inchikey14"]
        for batch in reader.iter_batches(batch_size=8192, columns=columns):
            if raw_seen + len(batch) <= resume_raw_index:
                raw_seen += len(batch)
                continue
            for row in batch.to_pylist():
                if raw_seen < resume_raw_index:
                    raw_seen += 1
                    continue
                raw_seen += 1
                if written == n:
                    break
                try:
                    x, length = spectrum_features(
                        row["ms2_mzs"], row["ms2_normalized_intensities"],
                        row["precursor_mz"],
                    )
                    meta, adduct = spectrum_metadata(row)
                except (ValueError, TypeError, OverflowError):
                    continue
                if (length != lengths[written] or adduct != adducts[written]
                        or (row["inchikey14"] != keys[written]
                            and row["molecular_formula"] != formulas[mol_idx[written]])
                        or not np.array_equal(x.astype("<f2"), peaks[written])
                        or not np.allclose(meta, metadata[written], rtol=0, atol=1e-6)):
                    continue
                (x[:, 0] * 1000).astype("<f4").tofile(handle)
                written += 1
            if written and written % 100000 < 8192:
                print(
                    "aligned", written, "/", n, "seconds",
                    round(time.monotonic() - start), flush=True,
                )
    if written != n:
        raise RuntimeError(f"Only aligned {written}/{n} spectra; {temp} retained for audit")
    if temp.stat().st_size != n * 96 * 4:
        raise RuntimeError("Precise m/z cache has incorrect byte count")
    os.replace(temp, output)
    print("saved", output, "spectra", n, "seconds", round(time.monotonic()-start), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume-raw-index", type=int, default=0)
    parser.add_argument("--resume-cache-index", type=int, default=0)
    args = parser.parse_args()
    prepare(
        args.raw, args.cache, args.output,
        resume_raw_index=args.resume_raw_index, resume_cache_index=args.resume_cache_index,
    )


if __name__ == "__main__":
    main()
