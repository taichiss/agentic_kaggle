"""Append fixed 0.479 MIST A reranking to the exact G4 CE-diverse parent."""

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

G4_NOTEBOOK_SHA256 = "cc78ad16b5a126e422d2423c67cbbf39557b89a703434cc677b6e4c44f6f8302"
G4_FROZEN_SHA256 = "3b505080bcf92f1f520cbd45e75001af6bcbf1f0a84a10b3dce5a4f959773a39"
G4_ADAPTER_SHA256 = "178adad68f7af279464c3786bd25d2b13777cd070e1c0ee426177e3b594cd13a"
G4_METADATA_SHA256 = "2fe4ffced4742dbab1324aea9ace07d1a18c2a0c5971f82fdc3678339912e7b6"
A_DATASET = "suzukitaichi/casmi26-mist-formula-12k-20261107"
FUSION_WEIGHT = 0.25

G4_AUDIT = r"""
# Verify the parent G4 generator log and bind it to the overlay audit.
_g4_path = _P('/kaggle/working/ce-diverse-generator-audit.json')
_g4 = _j.loads(_g4_path.read_text())
assert _g4['adapter_source_sha256'] == @@G4_ADAPTER_SHA@@
assert _g4['generator_calls'] == len(_g4['queries'])
assert _g4['generated_candidates_total'] == sum(
    query['generated_candidates'] for query in _g4['queries'])
assert _g4['changed_parent_queries'] == sum(
    query['baseline_parent_sids'] != [p['sid'] for p in query['generator_parents']]
    for query in _g4['queries'])
_audit_path = _P('/kaggle/working/' + _model_name + '-dynamic-audit.json')
_g4_final_audit = _j.loads(_audit_path.read_text())
_g4_final_audit['g4_generator_audit_sha256'] = _file_sha(_g4_path)
_g4_final_audit['g4_adapter_source_sha256'] = _g4['adapter_source_sha256']
_g4_final_audit['g4_generator_calls'] = _g4['generator_calls']
_g4_final_audit['g4_changed_parent_queries'] = _g4['changed_parent_queries']
_g4_final_audit['g4_generated_candidates_total'] = _g4['generated_candidates_total']
_audit_path.write_text(_j.dumps(_g4_final_audit, indent=2))
print('G4 + FIXED A AUDIT', _j.dumps(_g4_final_audit), flush=True)
"""


def build(args: argparse.Namespace) -> dict:
    if args.dataset != A_DATASET:
        raise ValueError("Fixed A dataset differs from the G1+A overlay")
    parent_sha = sha(args.parent_notebook)
    if parent_sha != G4_NOTEBOOK_SHA256:
        raise ValueError("G4 parent Notebook differs from the fixed source pin")
    parent_receipt = json.loads(args.parent_build_receipt.read_text())
    expected_parent_receipt = {
        "control_notebook_sha256": CONTROL_SHA256,
        "variant_frozen_sha256": G4_FROZEN_SHA256,
        "adapter_source_sha256": G4_ADAPTER_SHA256,
        "changed_frozen_cells": [37, 39],
        "audit_file": "ce-diverse-generator-audit.json",
        "mode": "diverse",
    }
    for field, expected in expected_parent_receipt.items():
        if parent_receipt.get(field) != expected:
            raise ValueError(f"G4 parent receipt mismatch: {field}")
    parent_metadata = json.loads(args.parent_metadata.read_text())
    if (
        sha(args.parent_metadata) != G4_METADATA_SHA256
        or parent_metadata.get("code_file") != "ce-generator-diverse.ipynb"
        or not parent_metadata.get("is_private")
        or parent_metadata.get("id") != parent_receipt.get("kernel")
    ):
        raise ValueError("G4 parent metadata does not match the build receipt")
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
        raise ValueError("Overlay builder changed G4 parent cells")
    notebook["cells"][len(parent["cells"])]["source"] = [
        "## Fixed MIST A on G4 CE-diverse parents\n",
        "The G4 parent creates candidates first; fixed A then reranks within "
        "molecular formula using RRF 0.25 while preserving the parent top one.\n",
    ]
    audit = audit_source(
        model_name=args.model_name,
        experiment_id=args.experiment_id,
        parent_sha=parent_sha,
        frozen_sha=G4_FROZEN_SHA256,
        checkpoint_sha=checkpoint_sha,
        weight=FUSION_WEIGHT,
    ) + G4_AUDIT.replace("@@G4_ADAPTER_SHA@@", repr(G4_ADAPTER_SHA256))
    compile(audit, "g4_fixed_a_audit", "exec")
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
        "parent_frozen_sha256": G4_FROZEN_SHA256,
        "parent_adapter_sha256": G4_ADAPTER_SHA256,
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
