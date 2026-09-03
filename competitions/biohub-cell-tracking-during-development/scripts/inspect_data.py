"""Inventory downloaded Biohub assets without loading microscopy arrays."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]


def inspect(root: Path) -> int:
    if not root.exists():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    suffixes = Counter(".zarr-member" if ".zarr" in path.parts else path.suffix for path in files)
    total_bytes = sum(path.stat().st_size for path in files)
    zarr_datasets = sorted(path.relative_to(root) for path in root.rglob("*.zarr") if path.is_dir())
    geff_datasets = sorted(path.relative_to(root) for path in root.rglob("*.geff") if path.is_dir())
    print(f"root: {root}")
    print(f"files: {len(files)}")
    print(f"bytes: {total_bytes}")
    print(f"zarr datasets: {len(zarr_datasets)}")
    for path in zarr_datasets:
        print(f"  zarr {path}")
    print(f"geff datasets: {len(geff_datasets)}")
    for path in geff_datasets:
        print(f"  geff {path}")
    print("suffix counts:")
    for suffix, count in sorted(suffixes.items()):
        print(f"  {suffix or '<none>'}: {count}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=WORKSPACE / "data" / "raw")
    args = parser.parse_args(argv)
    try:
        return inspect(args.root)
    except OSError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
