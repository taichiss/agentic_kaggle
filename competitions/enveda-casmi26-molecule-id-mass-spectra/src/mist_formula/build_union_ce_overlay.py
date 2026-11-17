"""Append fixed 0.479 MIST A reranking to the exact G5 v2 CE-union parent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_dynamic_ce_overlay import (  # noqa: E402
    A_CHECKPOINT_SHA256,
    CONTROL_SHA256,
    INFERENCE_SHA256,
    audit_source,
    sha,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public_notebooks"))
from build_trained_model_overlay import build as build_overlay  # noqa: E402

G5_NOTEBOOK_SHA256 = "4a52c7780c24196feb7d3200da988bf245f8a93b10bef8a0cc5b0b0251be81c9"
G5_FROZEN_SHA256 = "e43233781e82141fc38add8d19cb279c0c31433dd6370a2f84357a2701eab9af"
G5_ADAPTER_SHA256 = "de98761e89a2b0d3b50bdfb40deda32743b0a3b3baa6e7ca152caade05997057"
G5_HELPER_SHA256 = "178adad68f7af279464c3786bd25d2b13777cd070e1c0ee426177e3b594cd13a"
G5_METADATA_SHA256 = "3b2d5270c50d1b7f1d9d4e65acf3adb81482c2f9b8c8d6ed02213fc0cbfea541"
A_DATASET = "suzukitaichi/casmi26-mist-formula-12k-20261107"
FUSION_WEIGHT = 0.25

G5_AUDIT = r"""
# Verify the parent G5 union generator log and bind it to the overlay audit.
_g5_path = _P('/kaggle/working/ce-union-generator-audit.json')
_g5 = _j.loads(_g5_path.read_text())
assert _g5['adapter_source_sha256'] == @@G5_ADAPTER_SHA@@
assert _g5['generator_calls'] == len(_g5['queries'])
assert _g5['baseline_generated_total'] == sum(
    query['baseline_generated'] for query in _g5['queries'])
assert _g5['extra_generated_total'] == sum(
    query['extra_generated'] for query in _g5['queries'])
assert _g5['generated_candidates_total'] == sum(
    query['generated_candidates'] for query in _g5['queries'])
assert _g5['generated_candidates_total'] == (
    _g5['baseline_generated_total'] + _g5['extra_generated_total'])
assert _g5['eligible_queries'] == sum(
    len(query['eligible_ce_views']) >= 2
    for query in _g5['queries'])
_audit_path = _P('/kaggle/working/' + _model_name + '-dynamic-audit.json')
_g5_final_audit = _j.loads(_audit_path.read_text())
_g5_final_audit['g5_generator_audit_sha256'] = _file_sha(_g5_path)
_g5_final_audit['g5_adapter_source_sha256'] = _g5['adapter_source_sha256']
_g5_final_audit['g5_generator_calls'] = _g5['generator_calls']
_g5_final_audit['g5_eligible_queries'] = _g5['eligible_queries']
_g5_final_audit['g5_baseline_generated_total'] = _g5['baseline_generated_total']
_g5_final_audit['g5_extra_generated_total'] = _g5['extra_generated_total']
_g5_final_audit['g5_generated_candidates_total'] = _g5['generated_candidates_total']
_audit_path.write_text(_j.dumps(_g5_final_audit, indent=2))
print('G5 + FIXED A AUDIT', _j.dumps(_g5_final_audit), flush=True)
"""


def build(args: argparse.Namespace) -> dict:
    if args.dataset != A_DATASET:
        raise ValueError("Fixed A dataset differs from the G1+A overlay")
    parent_sha = sha(args.parent_notebook)
    if parent_sha != G5_NOTEBOOK_SHA256:
        raise ValueError("G5 parent Notebook differs from the fixed source pin")
    parent_receipt = json.loads(args.parent_build_receipt.read_text())
    expected_parent_receipt = {
        "control_notebook_sha256": CONTROL_SHA256,
        "variant_frozen_sha256": G5_FROZEN_SHA256,
        "adapter_source_sha256": G5_ADAPTER_SHA256,
        "helper_source_sha256": G5_HELPER_SHA256,
        "changed_frozen_cells": [37, 39],
        "audit_file": "ce-union-generator-audit.json",
        "mode": "union",
    }
    for field, expected in expected_parent_receipt.items():
        if parent_receipt.get(field) != expected:
            raise ValueError(f"G5 parent receipt mismatch: {field}")
    parent_metadata = json.loads(args.parent_metadata.read_text())
    if (
        sha(args.parent_metadata) != G5_METADATA_SHA256
        or parent_metadata.get("code_file") != "ce-generator-union.ipynb"
        or not parent_metadata.get("is_private")
        or parent_metadata.get("id") != parent_receipt.get("kernel")
    ):
        raise ValueError("G5 parent metadata does not match the build receipt")
    if sha(args.inference_module) != INFERENCE_SHA256:
        raise ValueError("Fixed A inference module differs from the G1+A overlay")
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != A_CHECKPOINT_SHA256:
        raise ValueError("Fixed A checkpoint differs from the G1+A overlay")

    receipt = build_overlay(SimpleNamespace(
        control_notebook=args.parent_notebook,
        control_sha256=parent_sha,
        control_metadata=args.parent_metadata,
        module=args.inference_module,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        kernel=args.kernel,
        model_name=args.model_name,
        output=args.output,
        weight=FUSION_WEIGHT,
        protect_top_n=1,
        within_formula=True,
    ))
    notebook_path = args.output / "trained-model-overlay.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_notebook.read_text(encoding="utf-8"))
    if notebook["cells"][: len(parent["cells"])] != parent["cells"]:
        raise ValueError("Overlay builder changed G5 parent cells")
    notebook["cells"][len(parent["cells"])]["source"] = [
        "## Fixed MIST A on G5 CE-union parents\n",
        "The G5 parent creates candidates first; fixed A then reranks within "
        "molecular formula using RRF 0.25 while preserving the parent top one.\n",
    ]
    audit = audit_source(
        model_name=args.model_name,
        experiment_id=args.experiment_id,
        parent_sha=parent_sha,
        frozen_sha=G5_FROZEN_SHA256,
        checkpoint_sha=checkpoint_sha,
        weight=FUSION_WEIGHT,
    ) + G5_AUDIT.replace("@@G5_ADAPTER_SHA@@", repr(G5_ADAPTER_SHA256))
    compile(audit, "g5_fixed_a_audit", "exec")
    notebook["cells"].append({
        "cell_type": "code", "execution_count": None,
        "metadata": {}, "outputs": [], "source": audit.splitlines(keepends=True),
    })
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    receipt.update({
        "experiment_id": args.experiment_id,
        "parent_notebook_sha256": parent_sha,
        "parent_frozen_sha256": G5_FROZEN_SHA256,
        "parent_adapter_sha256": G5_ADAPTER_SHA256,
        "candidate_set_sha256": "computed from the parent output at runtime",
        "final_notebook_sha256": sha(notebook_path),
    })
    (args.output / "build-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-notebook", "parent-build-receipt", "parent-metadata",
        "inference-module", "checkpoint", "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("dataset", "kernel", "model-name"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--experiment-id")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
