"""Write and validate Biohub mixed node/edge submission CSV files."""

from __future__ import annotations

import argparse
import csv
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

COLUMNS = (
    "id",
    "dataset",
    "row_type",
    "node_id",
    "t",
    "z",
    "y",
    "x",
    "source_id",
    "target_id",
)


@dataclass(frozen=True)
class Node:
    dataset: str
    node_id: int
    t: int
    z: int
    y: int
    x: int


@dataclass(frozen=True)
class Edge:
    dataset: str
    source_id: int
    target_id: int


def write_submission(path: Path, nodes: Iterable[Node], edges: Iterable[Edge]) -> None:
    """Write nodes followed by edges with the official sentinel values."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, int | str]] = []
    for node in nodes:
        values = asdict(node)
        rows.append(
            {
                "dataset": values["dataset"],
                "row_type": "node",
                "node_id": values["node_id"],
                "t": values["t"],
                "z": values["z"],
                "y": values["y"],
                "x": values["x"],
                "source_id": -1,
                "target_id": -1,
            }
        )
    for edge in edges:
        values = asdict(edge)
        rows.append(
            {
                "dataset": values["dataset"],
                "row_type": "edge",
                "node_id": -1,
                "t": -1,
                "z": -1,
                "y": -1,
                "x": -1,
                "source_id": values["source_id"],
                "target_id": values["target_id"],
            }
        )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for row_id, row in enumerate(rows):
            writer.writerow({"id": row_id, **row})


def _integer(row: dict[str, str], column: str, row_number: int) -> int:
    try:
        return int(row[column])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {column} must be an integer") from exc


def validate_submission(path: Path, expected_datasets: Sequence[str] = ()) -> dict[str, int]:
    """Validate structural failure modes before spending a submission."""
    nodes: dict[str, set[int]] = {}
    edges: list[tuple[int, str, int, int]] = []
    seen_edges: set[tuple[str, int, int]] = set()

    with path.open(encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"header must exactly match: {','.join(COLUMNS)}")
        for expected_id, row in enumerate(reader):
            row_number = expected_id + 2
            if _integer(row, "id", row_number) != expected_id:
                raise ValueError(f"row {row_number}: id must be consecutive from 0")
            dataset = row["dataset"].strip()
            if not dataset:
                raise ValueError(f"row {row_number}: dataset must not be empty")
            row_type = row["row_type"]
            if row_type == "node":
                node_id = _integer(row, "node_id", row_number)
                if node_id < 0:
                    raise ValueError(f"row {row_number}: node_id must be non-negative")
                for column in ("t", "z", "y", "x"):
                    if _integer(row, column, row_number) < 0:
                        raise ValueError(f"row {row_number}: {column} must be non-negative")
                if any(
                    _integer(row, column, row_number) != -1
                    for column in ("source_id", "target_id")
                ):
                    raise ValueError(f"row {row_number}: node source_id and target_id must be -1")
                dataset_nodes = nodes.setdefault(dataset, set())
                if node_id in dataset_nodes:
                    raise ValueError(f"row {row_number}: duplicate node_id {node_id} in {dataset}")
                dataset_nodes.add(node_id)
            elif row_type == "edge":
                for column in ("node_id", "t", "z", "y", "x"):
                    if _integer(row, column, row_number) != -1:
                        raise ValueError(f"row {row_number}: edge {column} must be -1")
                source_id = _integer(row, "source_id", row_number)
                target_id = _integer(row, "target_id", row_number)
                if min(source_id, target_id) < 0 or source_id == target_id:
                    raise ValueError(
                        f"row {row_number}: edge endpoints must be distinct non-negative IDs"
                    )
                edge = (dataset, source_id, target_id)
                if edge in seen_edges:
                    raise ValueError(
                        f"row {row_number}: duplicate edge {source_id}->{target_id} in {dataset}"
                    )
                seen_edges.add(edge)
                edges.append((row_number, dataset, source_id, target_id))
            else:
                raise ValueError(f"row {row_number}: row_type must be node or edge")

    if not nodes:
        raise ValueError("submission contains no node rows")
    for row_number, dataset, source_id, target_id in edges:
        dataset_nodes = nodes.get(dataset, set())
        missing = {source_id, target_id} - dataset_nodes
        if missing:
            raise ValueError(
                f"row {row_number}: edge references missing nodes {sorted(missing)} in {dataset}"
            )
    missing_datasets = set(expected_datasets) - set(nodes)
    if missing_datasets:
        raise ValueError(f"missing test datasets: {sorted(missing_datasets)}")
    return {
        "datasets": len(nodes),
        "nodes": sum(len(dataset_nodes) for dataset_nodes in nodes.values()),
        "edges": len(edges),
    }


def _test_datasets(test_dir: Path | None) -> list[str]:
    if test_dir is None:
        return []
    if not test_dir.is_dir():
        raise ValueError(f"test directory does not exist: {test_dir}")
    return sorted(path.name.removesuffix(".zarr") for path in test_dir.glob("*.zarr"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate a submission CSV")
    validate.add_argument("path", type=Path)
    validate.add_argument("--test-dir", type=Path)
    args = parser.parse_args(argv)
    try:
        summary = validate_submission(args.path, _test_datasets(args.test_dir))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(
        "submission valid: "
        f"{summary['datasets']} datasets, {summary['nodes']} nodes, {summary['edges']} edges"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
