"""Align order-invariant, unit-resolved multi-CE signatures to frozen cache rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from model import spectrum_features, spectrum_metadata

NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
KNOWN_UNITS = {"ev", "nce", "v"}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_raw_numbers(value: str, count: int) -> tuple[str, ...] | None:
    try:
        numbers = [Decimal(token) for token in NUMBER.findall(value)]
    except InvalidOperation:
        return None
    if len(numbers) != count or any(not number.is_finite() for number in numbers):
        return None
    return tuple(format(number.normalize(), "f") for number in sorted(numbers))


def signature(row: dict) -> tuple[tuple, str | None]:
    values = row["collision_energy_ev"]
    if values is None or len(values) < 2:
        return (), "not_multi_numeric"
    try:
        energies = [float(value) for value in values]
    except (TypeError, ValueError):
        return (), "non_numeric_ev"
    if not all(np.isfinite(value) for value in energies):
        return (), "nonfinite_ev"
    unit = str(row["collision_energy_orig_units"] or "").strip().lower()
    if unit not in KNOWN_UNITS:
        return (), "unknown_unit"
    original = row["collision_energy_orig"]
    if not isinstance(original, str) or not original.strip():
        return (), "missing_original_string"
    raw = canonical_raw_numbers(original, len(energies))
    if raw is None:
        return (), "unparseable_original_string"
    # Both the reported original pattern and the converted eV values must match.
    # Exact float.hex avoids silently rounding distinct acquisition settings.
    key = (unit, raw, tuple(sorted(value.hex() for value in energies)))
    return key, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--precise-mzs", type=Path, required=True)
    parser.add_argument("--source-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spec = pq.read_table(
        args.cache / "spectra.parquet", columns=["key", "length", "adduct_id", "mol_idx"]
    )
    mol = pq.read_table(args.cache / "molecules.parquet", columns=["formula"])
    keys = spec["key"].to_pylist()
    lengths = spec["length"].to_numpy()
    adducts = spec["adduct_id"].to_numpy()
    mol_idx = spec["mol_idx"].to_numpy()
    formulas = mol["formula"].to_pylist()
    n = len(keys)
    peaks = np.memmap(args.cache / "peaks.bin", mode="r", dtype="<f2", shape=(n, 96, 4))
    metadata = np.memmap(args.cache / "metadata.bin", mode="r", dtype="<f4", shape=(n, 4))
    mzs = np.memmap(args.precise_mzs, mode="r", dtype="<f4", shape=(n, 96))
    ce_kind = np.load(args.source_metadata / "ce_kind.npy", mmap_mode="r")
    ingest_ids = np.load(args.source_metadata / "ingest_id.npy", mmap_mode="r")
    instrument_ids = np.load(args.source_metadata / "instrument_id.npy", mmap_mode="r")
    categories = json.loads((args.source_metadata / "categories.json").read_text())
    source_map = {value: index for index, value in enumerate(categories["ingest_lib"])}
    instrument_map = {value: index for index, value in enumerate(categories["instrument_type"])}
    if any(len(array) != n for array in (ce_kind, ingest_ids, instrument_ids)):
        raise ValueError("Source sidecars do not match cache rows")
    pattern_id = np.full(n, -1, dtype=np.int32)
    raw_row_index = np.full(n, -1, dtype=np.int32)
    patterns: dict[tuple, int] = {}
    pattern_examples: dict[int, dict] = {}
    exclusions = Counter()
    eligible_by_source = Counter()
    kind2_by_source = Counter()
    ce_column_hash = hashlib.sha256()
    matched = raw_seen = 0
    start = time.monotonic()
    reader = pq.ParquetFile(args.raw)
    columns = [
        "ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct",
        "ionization_mode", "collision_energy_ev", "collision_energy_orig",
        "collision_energy_orig_units", "molecular_formula", "inchikey14",
        "ingest_lib", "instrument_type",
    ]
    for batch in reader.iter_batches(batch_size=8192, columns=columns):
        for row in batch.to_pylist():
            raw_seen += 1
            ce_column_hash.update(json.dumps(
                [row["collision_energy_ev"], row["collision_energy_orig"],
                 row["collision_energy_orig_units"]],
                separators=(",", ":"), allow_nan=True,
            ).encode() + b"\n")
            if matched >= n:
                break
            if (
                row["inchikey14"] != keys[matched]
                and row["molecular_formula"] != formulas[mol_idx[matched]]
            ):
                continue
            try:
                feature, length = spectrum_features(
                    row["ms2_mzs"], row["ms2_normalized_intensities"],
                    row["precursor_mz"],
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
            if source_map.get(row["ingest_lib"], -1) != int(ingest_ids[matched]):
                raise AssertionError("Source sidecar mismatch during CE-pattern alignment")
            expected_instrument = instrument_map.get(row["instrument_type"], -1)
            if str(row["instrument_type"]).strip().lower() in {
                "nan", "n/a", "na", "none", "unknown"
            }:
                expected_instrument = -1
            if expected_instrument != int(instrument_ids[matched]):
                raise AssertionError("Instrument sidecar mismatch during CE-pattern alignment")
            values = row["collision_energy_ev"]
            observed_kind = (
                0 if values is None or len(values) == 0 else
                1 if len(values) == 1 and values[0] is not None
                and np.isfinite(values[0]) else 2
            )
            if observed_kind != int(ce_kind[matched]):
                raise AssertionError("CE-kind sidecar mismatch during CE-pattern alignment")
            raw_row_index[matched] = raw_seen - 1
            if observed_kind == 2:
                source = str(row["ingest_lib"])
                kind2_by_source[source] += 1
                key, reason = signature(row)
                if reason is not None:
                    exclusions[reason] += 1
                else:
                    identity = patterns.setdefault(key, len(patterns))
                    pattern_id[matched] = identity
                    eligible_by_source[source] += 1
                    if identity not in pattern_examples:
                        pattern_examples[identity] = {
                            "id": identity,
                            "unit": key[0],
                            "original_values_sorted": list(key[1]),
                            "ev_values_sorted_hex": list(key[2]),
                            "example_original_string": row["collision_energy_orig"],
                            "example_ev_array": values,
                            "example_source": source,
                        }
            matched += 1
            if matched % 200_000 == 0:
                print(json.dumps({
                    "matched": matched, "raw_seen": raw_seen,
                    "seconds": round(time.monotonic() - start),
                }), flush=True)
        if matched >= n:
            break
    if matched != n or raw_seen != reader.metadata.num_rows:
        raise RuntimeError(f"Aligned {matched}/{n}; scanned {raw_seen}/{reader.metadata.num_rows}")
    if np.any(raw_row_index < 0):
        raise AssertionError("A cache row lacks a raw-row index")
    if int((ce_kind == 2).sum()) != sum(kind2_by_source.values()):
        raise AssertionError("Kind-2 source counts do not cover sidecar")
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "pattern_id.npy", pattern_id)
    np.save(args.output / "raw_row_index.npy", raw_row_index)
    (args.output / "patterns.json").write_text(json.dumps(
        [pattern_examples[index] for index in range(len(patterns))], indent=2
    ) + "\n")
    report = {
        "raw_train_sha256": sha(args.raw),
        "raw_ce_columns_sha256": ce_column_hash.hexdigest(),
        "raw_ce_columns_hash_method": (
            "UTF-8 newline-delimited compact JSON of [collision_energy_ev, "
            "collision_energy_orig, collision_energy_orig_units] in original row order"
        ),
        "cache_spectra_sha256": sha(args.cache / "spectra.parquet"),
        "source_metadata_manifest_sha256": sha(args.source_metadata / "manifest.json"),
        "ce_kind_sidecar_sha256": sha(args.source_metadata / "ce_kind.npy"),
        "raw_rows_seen": raw_seen,
        "matched_cache_rows": matched,
        "cache_rows": n,
        "alignment_rate": matched / n,
        "kind2_cache_rows": int((ce_kind == 2).sum()),
        "kind2_eligible_rows": int((pattern_id >= 0).sum()),
        "kind2_exclusion_reasons": dict(sorted(exclusions.items())),
        "kind2_by_source": dict(sorted(kind2_by_source.items())),
        "eligible_by_source": dict(sorted(eligible_by_source.items())),
        "unique_patterns": len(patterns),
        "signature_definition": (
            "Only kind2 with >=2 finite eV values, known original units eV/NCE/V, "
            "and parseable original numeric tokens of equal length. Pattern equality "
            "requires equal unit, sorted original Decimal numeric tokens, and sorted "
            "exact IEEE-754 eV float.hex values; list order is ignored."
        ),
        "pattern_id_sha256": sha(args.output / "pattern_id.npy"),
        "raw_row_index_sha256": sha(args.output / "raw_row_index.npy"),
        "patterns_sha256": sha(args.output / "patterns.json"),
        "seconds": round(time.monotonic() - start, 1),
    }
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
