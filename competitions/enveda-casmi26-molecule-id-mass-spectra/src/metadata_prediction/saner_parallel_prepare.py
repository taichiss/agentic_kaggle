"""Parallelize upstream structure caching without changing serial row processing.

Run using the staged upstream prepare.py and features.py as working directory.
An output directory must be new. Original parsing, row order, deduplication,
binary packing and split assignment are performed by the upstream code itself.
"""

from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
from pathlib import Path

import pyarrow.parquet as pq

_STRUCTURE = None
_BITS = None


def _init_worker(bundle: str, bits: int) -> None:
    global _STRUCTURE, _BITS
    import sys

    sys.path.insert(0, bundle)
    _STRUCTURE = importlib.import_module("features").structure
    _BITS = bits


def _calculate(smiles: str):
    try:
        return smiles, _STRUCTURE(smiles, _BITS)
    except (ValueError, RuntimeError):
        return smiles, None


def run(bundle: Path, config_path: Path, work: Path, workers: int) -> None:
    import sys

    if work.exists():
        raise FileExistsError(work)
    config = json.loads(config_path.read_text())
    if workers < 1:
        raise ValueError("workers must be positive")
    sys.path.insert(0, str(bundle.resolve()))
    upstream = importlib.import_module("prepare")
    seen = set()
    for input_path in [config["train_path"], *config.get("external_paths", [])]:
        parquet = pq.ParquetFile(input_path)
        names = parquet.schema_arrow.names
        column = next((c for c in upstream.ALIASES["smiles"] if c in names), None)
        if column is None:
            raise ValueError(f"No structure column recognized in {input_path}")
        # Match upstream's str() handling of each structure value.
        for batch in parquet.iter_batches(batch_size=8192, columns=[column]):
            seen.update(str(x) for x in batch.to_pandas()[column].tolist())
    initargs = (str(bundle.resolve()), config["fp_bits"])
    with mp.Pool(workers, initializer=_init_worker, initargs=initargs) as pool:
        lookup = dict(pool.imap_unordered(_calculate, seen, chunksize=64))
    print(f"parallel structure cache: {len(lookup)} unique structures", flush=True)

    # The upstream keeps a second, lazily filled cache with its original first
    # occurrence rules. The lookup only changes how a cache miss is calculated.
    def cached_structure(smiles, bits):
        if bits != config["fp_bits"] or smiles not in lookup:
            raise KeyError("Unexpected structure cache lookup")
        value = lookup[smiles]
        if value is None:
            raise ValueError("Invalid structure")
        return value

    upstream.structure = cached_structure
    config["work"] = str(work.resolve())
    upstream.prepare(config)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.bundle, args.config, args.work, args.workers)


if __name__ == "__main__":
    main()
