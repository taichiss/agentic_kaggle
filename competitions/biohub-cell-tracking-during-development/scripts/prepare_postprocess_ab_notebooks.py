#!/usr/bin/env python
"""Build controlled Kaggle post-processing A/B notebooks from a TOML config."""

from __future__ import annotations

import argparse
import hashlib
import json
import tomllib
from pathlib import Path

COMPETITION_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = COMPETITION_ROOT / "configs/exp-0010-postprocess-ab.toml"
DEFAULT_OUTPUT = COMPETITION_ROOT / "data/kaggle-submission-EXP-0010-postprocess-ab"
COMPETITION_SLUG = "biohub-cell-tracking-during-development"


def _inference_patch_code(config: dict, variant_name: str, variant: dict) -> tuple[str, dict]:
    base = config["base"]
    patches = [dict(patch) for patch in variant["patches"]]
    trace = {
        "experiment_id": config["experiment_id"],
        "variant": variant_name,
        "base_dataset": base["kaggle_dataset"],
        "checkpoint_sha256": base["checkpoint_sha256"],
        "base_inference_script_sha256": base["inference_script_sha256"],
        "patches": patches,
    }
    code = f"""from pathlib import Path
import hashlib
import json
import subprocess
import sys

weights = list(Path('/kaggle/input').rglob('edge_predictor_best.pth'))
assert len(weights) == 1, f'expected one model checkpoint, found {{weights}}'
bundle = weights[0].parent
assert hashlib.sha256(weights[0].read_bytes()).hexdigest() == {base['checkpoint_sha256']!r}
source_script = bundle / 'run_kaggle_inference.py'
assert hashlib.sha256(source_script.read_bytes()).hexdigest() == {base['inference_script_sha256']!r}
script = source_script.read_text(encoding='utf-8')
patches = {patches!r}
for patch in patches:
    old, new = patch['old'], patch['new']
    assert script.count(old) == 1, (patch['parameter'], script.count(old))
    script = script.replace(old, new)
variant_script = Path('/kaggle/working/run_kaggle_inference_variant.py')
variant_script.write_text(script, encoding='utf-8')
trace = {trace!r}
trace['patched_inference_script_sha256'] = hashlib.sha256(
    variant_script.read_bytes()
).hexdigest()
Path('/kaggle/working/postprocess-profile.json').write_text(
    json.dumps(trace, indent=2) + '\\n', encoding='utf-8'
)
print(json.dumps(trace, indent=2), flush=True)
test_dir = Path('/kaggle/input/competitions/{COMPETITION_SLUG}/test')
output = Path('/kaggle/working/submission.csv')
command = [
    sys.executable,
    str(variant_script),
    '--bundle-dir',
    str(bundle),
    '--test-dir',
    str(test_dir),
    '--output',
    str(output),
    '--det-threshold',
    '0.99',
    '--edge-threshold',
    '0.5',
    '--postprocess-profile',
    'public-applicable-v1',
]
subprocess.run(command, check=True)
assert output.exists() and output.stat().st_size > 0
print(f'submission ready: {{output}} ({{output.stat().st_size:,}} bytes)')
"""
    return code, trace


def _component_prune_code(config: dict, variant_name: str, variant: dict) -> tuple[str, dict]:
    base = config["base"]
    patches = [dict(patch) for patch in variant["patches"]]
    minimum = int(patches[0]["new_value"])
    base_kernel = base["base_submission_kernel"]
    base_slug = base_kernel.split("/", 1)[-1]
    trace = {
        "experiment_id": config["experiment_id"],
        "variant": variant_name,
        "base_kernel": base_kernel,
        "minimum_component_nodes": minimum,
        "preserve_division_components": True,
        "patches": patches,
    }
    code = f"""from pathlib import Path
import csv
import json

columns = ('id', 'dataset', 'row_type', 'node_id', 't', 'z', 'y', 'x', 'source_id', 'target_id')
candidates = [
    path for path in Path('/kaggle/input').rglob('submission.csv')
    if {base_slug!r} in path.as_posix()
]
assert len(candidates) == 1, f'expected one base submission, found {{candidates}}'
with candidates[0].open(newline='', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    assert tuple(reader.fieldnames or ()) == columns
    rows = list(reader)

dataset_order = []
nodes = {{}}
edges = {{}}
for row in rows:
    dataset = row['dataset']
    if dataset not in nodes:
        dataset_order.append(dataset)
        nodes[dataset] = {{}}
        edges[dataset] = []
    if row['row_type'] == 'node':
        nodes[dataset][int(row['node_id'])] = tuple(
            int(row[key]) for key in ('t', 'z', 'y', 'x')
        )
    elif row['row_type'] == 'edge':
        edges[dataset].append((int(row['source_id']), int(row['target_id'])))
    else:
        raise ValueError(row['row_type'])

output = Path('/kaggle/working/submission.csv')
stats_path = Path('/kaggle/working/postprocess_stats.json')
row_id = total_nodes = total_edges = 0
dataset_stats = {{}}
with output.open('w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(columns)
    for dataset in dataset_order:
        parent = {{node_id: node_id for node_id in nodes[dataset]}}
        def find(node_id):
            while parent[node_id] != node_id:
                parent[node_id] = parent[parent[node_id]]
                node_id = parent[node_id]
            return node_id
        outdegree = {{}}
        for source, target in edges[dataset]:
            left, right = find(source), find(target)
            if left != right:
                parent[left] = right
            outdegree[source] = outdegree.get(source, 0) + 1
        components = {{}}
        for node_id in nodes[dataset]:
            components.setdefault(find(node_id), []).append(node_id)
        kept = set()
        for members in components.values():
            has_division = any(outdegree.get(node_id, 0) >= 2 for node_id in members)
            if len(members) >= {minimum} or has_division:
                kept.update(members)
        assert kept
        ordered = sorted(kept)
        remap = {{old: new for new, old in enumerate(ordered)}}
        kept_edges = [
            (remap[source], remap[target]) for source, target in edges[dataset]
            if source in kept and target in kept
        ]
        for old in ordered:
            frame, z, y, x = nodes[dataset][old]
            writer.writerow([row_id, dataset, 'node', remap[old], frame, z, y, x, -1, -1])
            row_id += 1
        for source, target in kept_edges:
            writer.writerow([row_id, dataset, 'edge', -1, -1, -1, -1, -1, source, target])
            row_id += 1
        dataset_stats[dataset] = {{
            'raw_nodes': len(nodes[dataset]),
            'raw_edges': len(edges[dataset]),
            'nodes': len(ordered),
            'edges': len(kept_edges),
        }}
        total_nodes += len(ordered)
        total_edges += len(kept_edges)
result = {{
    'min_component_nodes': {minimum},
    'keep_division_components': True,
    'datasets': dataset_stats,
    'rows': row_id,
    'nodes': total_nodes,
    'edges': total_edges,
}}
stats_path.write_text(json.dumps(result, indent=2, sort_keys=True) + '\\n', encoding='utf-8')
assert output.is_file() and output.stat().st_size > 0
print(json.dumps(result, indent=2, sort_keys=True))
print(f'submission ready: {{output}} ({{output.stat().st_size:,}} bytes)')
"""
    return code, trace


def _notebook(config: dict, variant_name: str, variant: dict) -> dict:
    mode = variant["execution_mode"]
    if mode == "inference_script_patch":
        code, trace = _inference_patch_code(config, variant_name, variant)
    elif mode == "submission_component_prune":
        code, trace = _component_prune_code(config, variant_name, variant)
    else:
        raise ValueError(f"unsupported execution_mode: {mode}")
    compile(code, f"{variant_name}.ipynb", "exec")
    cell_id = hashlib.sha256(variant_name.encode()).hexdigest()
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "id": cell_id[:12],
                "metadata": {},
                "source": [
                    f"# {variant['kernel_title']}\n",
                    "\n",
                    "Frozen epoch-30 inference with one post-processing parameter changed.\n",
                ],
            },
            {
                "cell_type": "code",
                "id": cell_id[12:24],
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3"},
            "agentic_kaggle": trace,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def prepare(config_path: Path, output_root: Path) -> dict:
    with config_path.open("rb") as file:
        config = tomllib.load(file)
    generated = {}
    for variant_name, variant in config["variants"].items():
        kernel_id = variant["kernel_id"]
        kernel_slug = kernel_id.split("/", 1)[-1]
        kernel_dir = output_root / variant_name / "kernel"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        notebook_name = f"{kernel_slug}.ipynb"
        notebook = _notebook(config, variant_name, variant)
        (kernel_dir / notebook_name).write_text(
            json.dumps(notebook, indent=2) + "\n", encoding="utf-8"
        )
        component_prune = variant["execution_mode"] == "submission_component_prune"
        metadata = {
            "id": kernel_id,
            "title": variant["kernel_title"],
            "code_file": notebook_name,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": "true",
            "enable_gpu": "false" if component_prune else "true",
            "enable_tpu": "false",
            "enable_internet": "false",
            "machine_shape": None if component_prune else "NvidiaTeslaT4",
            "dataset_sources": (
                [] if component_prune else [config["base"]["kaggle_dataset"]]
            ),
            "competition_sources": [COMPETITION_SLUG],
            "kernel_sources": (
                [config["base"]["base_submission_kernel"]]
                if component_prune
                else []
            ),
            "model_sources": [],
        }
        (kernel_dir / "kernel-metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        generated[variant_name] = {
            "kernel_id": kernel_id,
            "kernel_dir": str(kernel_dir),
            "submission_message": variant["submission_message"],
        }
    result = {
        "experiment_id": config["experiment_id"],
        "control_public_score": config["base"].get("control_public_score"),
        "generated": generated,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    prepare(args.config.resolve(), args.output_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
