"""Append fixed MIST A to the exact G2 generator-only separate-CE parent."""

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

G2_NOTEBOOK_SHA256 = "f68984e184a6c1b2267473f96f658ce9a33f1b7b5f182b59c01155f97843f40b"
G2_FROZEN_SHA256 = "2a3a582ad8c90f609067b242f6c94976bc554ded9d2e3bf979841cfcef66f58f"
G2_ADAPTER_SHA256 = "5e220e374c312d8e1f94a3f9fe1ffca919836b3a65c5afe3043cf493011ca348"
G2_METADATA_SHA256 = "0c867eece5233820dc3f16fd07b65ecffa7936082f4b4db8c39e43e8e729b06c"
A_DATASET = "suzukitaichi/casmi26-mist-formula-12k-20261107"
FUSION_WEIGHT = 0.25

G2_AUDIT = r"""
# Bind the generator-only separate-CE parent audit to this overlay run.
_g2_path = _P('/kaggle/working/ce-generator-audit.json')
_g2 = _j.loads(_g2_path.read_text())
assert _g2['mode'] == 'separate'
assert _g2['adapter_source_sha256'] == @@G2_ADAPTER_SHA@@
assert _g2['mixed_views_total'] == 0 and _g2['queries_with_mixed_views'] == 0
assert 0 <= _g2['changed_generator_parent_queries'] <= _g2['generator_calls']
assert _g2['generated_candidates_total'] >= 0
_audit_path = _P('/kaggle/working/' + _model_name + '-dynamic-audit.json')
_g2_final_audit = _j.loads(_audit_path.read_text())
_g2_final_audit['g2_generator_audit_sha256'] = _file_sha(_g2_path)
_g2_final_audit['g2_adapter_source_sha256'] = _g2['adapter_source_sha256']
_g2_final_audit['g2_generator_calls'] = _g2['generator_calls']
_g2_final_audit['g2_changed_parent_queries'] = _g2['changed_generator_parent_queries']
_g2_final_audit['g2_generated_candidates_total'] = _g2['generated_candidates_total']
_audit_path.write_text(_j.dumps(_g2_final_audit, indent=2))
print('G2 + FIXED A AUDIT', _j.dumps(_g2_final_audit), flush=True)
"""


def build(args: argparse.Namespace) -> dict:
    if args.dataset != A_DATASET:
        raise ValueError("Fixed A dataset differs from the G1+A overlay")
    parent_sha = sha(args.parent_notebook)
    if parent_sha != G2_NOTEBOOK_SHA256:
        raise ValueError("G2 parent Notebook differs from the fixed source pin")
    parent_receipt = json.loads(args.parent_build_receipt.read_text())
    expected_parent_receipt = {
        "control_notebook_sha256": CONTROL_SHA256,
        "variant_frozen_sha256": G2_FROZEN_SHA256,
        "adapter_source_sha256": G2_ADAPTER_SHA256,
        "changed_frozen_cells": [37, 39],
        "audit_file": "ce-generator-audit.json",
        "mode": "separate",
    }
    for field, expected in expected_parent_receipt.items():
        if parent_receipt.get(field) != expected:
            raise ValueError(f"G2 parent receipt mismatch: {field}")
    parent_metadata = json.loads(args.parent_metadata.read_text())
    if (
        sha(args.parent_metadata) != G2_METADATA_SHA256
        or parent_metadata.get("code_file") != "ce-generator-separate.ipynb"
        or not parent_metadata.get("is_private")
        or parent_metadata.get("id") != parent_receipt.get("kernel")
    ):
        raise ValueError("G2 parent metadata does not match the build receipt")
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
        raise ValueError("Overlay builder changed G2 parent cells")
    notebook["cells"][len(parent["cells"])]["source"] = [
        "## Fixed MIST A on G2 generator-only separate CE views\n",
        "The G2 parent creates candidates first; fixed A then reranks within "
        "molecular formula using RRF 0.25 while preserving the parent top one.\n",
    ]
    audit = audit_source(
        model_name=args.model_name,
        experiment_id=args.experiment_id,
        parent_sha=parent_sha,
        frozen_sha=G2_FROZEN_SHA256,
        checkpoint_sha=checkpoint_sha,
        weight=FUSION_WEIGHT,
    ) + G2_AUDIT.replace("@@G2_ADAPTER_SHA@@", repr(G2_ADAPTER_SHA256))
    compile(audit, "g2_fixed_a_audit", "exec")
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
        "parent_frozen_sha256": G2_FROZEN_SHA256,
        "parent_adapter_sha256": G2_ADAPTER_SHA256,
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
