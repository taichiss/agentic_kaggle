"""Append the fixed A MIST reranker to a pinned CE-view parent Notebook.

The CE-view parent may produce a different candidate set from the 0.477
control. Its *source* is pinned here, while the actual parent output and
candidate set are hashed after the Kaggle Notebook runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public_notebooks"))
from build_trained_model_overlay import build as build_overlay  # noqa: E402

CONTROL_SHA256 = "ef88fe151bf99fac82c8e3ad4b401d20aa0b6626880a7c3a8ed53a9022e0f938"
INFERENCE_SHA256 = "b40d1ab02c8135831f9b17b002a228f133fe7164c8365113744ea5e126a13c50"
A_CHECKPOINT_SHA256 = "a63e61f2d1dc19df98f7acb5a65199e3c3377e56fcb4da1347553ff68db7f6bb"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


AUDIT = r"""# Audit the dynamic CE-view parent and the fixed A MIST overlay.
import hashlib as _h
import json as _j
from pathlib import Path as _P
import pandas as _pd

def _file_sha(path):
    digest = _h.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()

_model_name = @@MODEL_NAME@@
_parent = _P('/kaggle/working/' + _model_name + '-control.csv')
_final = _P('/kaggle/working/submission.csv')
_model_report_path = _P('/kaggle/working/' + _model_name + '-overlay-report.json')
_engine_receipt_path = _P('/kaggle/working/execution_receipt.json')
_base = _pd.read_csv(_parent, dtype=str, keep_default_na=False)
_out = _pd.read_csv(_final, dtype=str, keep_default_na=False)
_sample = _pd.read_csv(_P(COMP) / 'sample_submission.csv', dtype=str,
                       keep_default_na=False)
assert list(_base.columns) == list(_out.columns) == ['molecule_id', 'smiles']
assert _base.molecule_id.tolist() == _out.molecule_id.tolist() == _sample.molecule_id.tolist()
assert len(_base) > 0
for _old, _new in zip(_base.smiles, _out.smiles):
    _a, _b = _old.split(';'), _new.split(';')
    assert 1 <= len(_a) <= 25 and len(_a) == len(set(_a)) and all(_a)
    assert 1 <= len(_b) <= 25 and len(_b) == len(set(_b)) and all(_b)
    assert sorted(_a) == sorted(_b), 'overlay changed the dynamic parent candidate set'
    assert _a[0] == _b[0], 'overlay changed parent rank one'
_members = '\n'.join(str(mid) + '\t' + ';'.join(sorted(smiles.split(';')))
                     for mid, smiles in zip(_base.molecule_id, _base.smiles))
_candidate_sha = _h.sha256(_members.encode()).hexdigest()
_model_report = _j.loads(_model_report_path.read_text())
assert _model_report['checkpoint_sha256'] == @@CHECKPOINT_SHA@@
assert _model_report['inference_sha256'] == @@INFERENCE_SHA@@
assert _model_report['weight'] == @@WEIGHT@@
assert _model_report['protected_top_n'] == 1 and _model_report['within_formula']
assert not _model_report['errors'], _model_report['errors'][:3]
assert _model_report['top1_changed'] == 0
assert _model_report['submission_sha256'] == _file_sha(_final)
_engine_receipt = _j.loads(_engine_receipt_path.read_text())
assert _engine_receipt['source_sha256'] == @@FROZEN_SHA@@, 'CE-view parent source changed'
_audit = {
    'experiment_id': @@EXPERIMENT_ID@@,
    'parent_control_notebook_sha256': @@CONTROL_SHA@@,
    'ce_parent_notebook_sha256': @@CE_PARENT_SHA@@,
    'ce_parent_frozen_sha256': @@FROZEN_SHA@@,
    'ce_parent_output_sha256': _file_sha(_parent),
    'dynamic_candidate_set_sha256': _candidate_sha,
    'A_checkpoint_sha256': @@CHECKPOINT_SHA@@,
    'A_inference_sha256': @@INFERENCE_SHA@@,
    'rank_fusion_weight': @@WEIGHT@@,
    'final_submission_sha256': _file_sha(_final),
    'rows': len(_out),
    'parent_processed_molecules': _engine_receipt.get('processed_molecules'),
    'parent_is_full_run': _engine_receipt.get('is_full_run'),
    'overlay_attempted_rows': _model_report['attempted_rows'],
    'overlay_changed_rows': _model_report['changed_rows'],
    'top1_changed': _model_report['top1_changed'],
    'errors': len(_model_report['errors']),
}
_P('/kaggle/working/' + _model_name + '-dynamic-audit.json').write_text(
    _j.dumps(_audit, indent=2))
print('DYNAMIC CE OVERLAY AUDIT', _j.dumps(_audit), flush=True)
"""


def audit_source(
    *,
    model_name: str,
    experiment_id: str | None,
    parent_sha: str,
    frozen_sha: str,
    checkpoint_sha: str,
    weight: float,
) -> str:
    replacements = {
        "@@MODEL_NAME@@": repr(model_name),
        "@@EXPERIMENT_ID@@": repr(experiment_id),
        "@@CONTROL_SHA@@": repr(CONTROL_SHA256),
        "@@CE_PARENT_SHA@@": repr(parent_sha),
        "@@FROZEN_SHA@@": repr(frozen_sha),
        "@@CHECKPOINT_SHA@@": repr(checkpoint_sha),
        "@@INFERENCE_SHA@@": repr(INFERENCE_SHA256),
        "@@WEIGHT@@": repr(weight),
    }
    source = AUDIT
    for before, after in replacements.items():
        source = source.replace(before, after)
    compile(source, "dynamic_ce_overlay_audit", "exec")
    return source


def build(args: argparse.Namespace) -> dict:
    parent_sha = sha(args.parent_notebook)
    if parent_sha != args.parent_notebook_sha256:
        raise ValueError("CE-view parent Notebook differs from its explicit pin")
    parent_receipt = json.loads(args.parent_build_receipt.read_text())
    if (
        parent_receipt.get("control_notebook_sha256") != CONTROL_SHA256
        or parent_receipt.get("changed_frozen_cells") != [37]
        or parent_receipt.get("analog_query") != "each"
    ):
        raise ValueError("Expected the single-cell, each-query 0.477 parent")
    if sha(args.inference_module) != INFERENCE_SHA256:
        raise ValueError("The fixed A inference module differs")
    checkpoint_sha = sha(args.checkpoint)
    if checkpoint_sha != A_CHECKPOINT_SHA256:
        raise ValueError("The fixed A checkpoint differs")
    receipt = build_overlay(
        SimpleNamespace(
            control_notebook=args.parent_notebook,
            control_sha256=parent_sha,
            control_metadata=args.parent_metadata,
            module=args.inference_module,
            checkpoint=args.checkpoint,
            dataset=args.dataset,
            kernel=args.kernel,
            model_name=args.model_name,
            output=args.output,
            weight=args.weight,
            protect_top_n=1,
            within_formula=True,
        )
    )
    notebook_path = args.output / "trained-model-overlay.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    parent = json.loads(args.parent_notebook.read_text(encoding="utf-8"))
    if notebook["cells"][: len(parent["cells"])] != parent["cells"]:
        raise ValueError("Overlay builder changed the CE-view parent cells")
    notebook["cells"][len(parent["cells"])]["source"] = [
        "## Fixed A MIST reranker on the CE-view parent\n",
        "The parent creates its own candidates before same-formula reranking. "
        "The 0.479 historical A score does not describe this variant.\n",
    ]
    audit = audit_source(
        model_name=args.model_name,
        experiment_id=args.experiment_id,
        parent_sha=parent_sha,
        frozen_sha=parent_receipt["variant_frozen_sha256"],
        checkpoint_sha=checkpoint_sha,
        weight=args.weight,
    )
    notebook["cells"].append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": audit.splitlines(keepends=True),
        }
    )
    notebook_path.write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    receipt.update(
        experiment_id=args.experiment_id,
        parent_notebook_sha256=parent_sha,
        parent_frozen_sha256=parent_receipt["variant_frozen_sha256"],
        candidate_set_sha256="computed from the parent output at runtime",
        final_notebook_sha256=sha(notebook_path),
    )
    (args.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-notebook",
        "parent-build-receipt",
        "parent-metadata",
        "inference-module",
        "checkpoint",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("parent-notebook-sha256", "dataset", "kernel", "model-name"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--experiment-id")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
