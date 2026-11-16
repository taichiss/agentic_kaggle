"""Build a Mixup checkpoint submission from the exact 0.479 parent pipeline."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public_notebooks"))
from build_trained_model_overlay import build as build_overlay  # noqa: E402

PARENT_SHA = "ef88fe151bf99fac82c8e3ad4b401d20aa0b6626880a7c3a8ed53a9022e0f938"
PARENT_OUTPUT_SHA = "39309f6c46d25e9c934dad8871d211ad8afebda07f4441c596714cbe7b6fd9b2"
CANDIDATE_SET_SHA = "e8de1f1b07ab6d56685805d01878185241a9a06bc1603b9fef13e537fc408542"
INFERENCE_SHA = "b40d1ab02c8135831f9b17b002a228f133fe7164c8365113744ea5e126a13c50"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict:
    experiment_id = getattr(args, "experiment_id", None)
    experiment_line = (
        f"          'experiment_id': {experiment_id!r},\n" if experiment_id else ""
    )
    if not re.fullmatch(r"[a-z0-9-]+", args.model_name):
        raise ValueError("model-name must be a lowercase Kaggle-safe slug")
    if sha(args.inference_module) != INFERENCE_SHA:
        raise ValueError("Mixup submission must use frozen A 0.479 inference source")
    receipt = build_overlay(
        SimpleNamespace(
            control_notebook=args.parent,
            control_sha256=PARENT_SHA,
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
    audit = f"""# Verify exact A 0.479 control and unchanged candidate set before submitting.
import hashlib as _h, json as _j
from pathlib import Path as _P
import pandas as _pd

def _digest(path):
    d = _h.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            d.update(block)
    return d.hexdigest()

_control = _P('/kaggle/working/{args.model_name}-control.csv')
_submission = _P('/kaggle/working/submission.csv')
assert _digest(_control) == {PARENT_OUTPUT_SHA!r}, '0.477 parent output differs from A'
_base = _pd.read_csv(_control, dtype=str, keep_default_na=False)
_final = _pd.read_csv(_submission, dtype=str, keep_default_na=False)
assert list(_base.columns) == ['molecule_id', 'smiles']
assert list(_final.columns) == ['molecule_id', 'smiles']
assert len(_base) == len(_final) == 400
assert _base.molecule_id.tolist() == _final.molecule_id.tolist()
_members = '\\n'.join(str(m)+'\\t'+';'.join(sorted(s.split(';'))) for m,s in zip(_base.molecule_id,_base.smiles))
assert _h.sha256(_members.encode()).hexdigest() == {CANDIDATE_SET_SHA!r}, 'candidate set differs from A'
for old, new in zip(_base.smiles, _final.smiles):
    a, b = old.split(';'), new.split(';')
    assert 1 <= len(b) <= 25 and sorted(a) == sorted(b)
    assert a[0] == b[0]
_overlay = _j.loads(_P('/kaggle/working/{args.model_name}-overlay-report.json').read_text())
assert not _overlay['errors'], _overlay['errors'][:3]
_audit = {{'parent_notebook_sha256': {PARENT_SHA!r},
{experiment_line}          'parent_output_sha256': _digest(_control),
          'candidate_set_sha256': {CANDIDATE_SET_SHA!r},
          'checkpoint_sha256': {sha(args.checkpoint)!r},
          'inference_sha256': {INFERENCE_SHA!r},
          'rank_fusion_weight': {args.weight!r},
          'submission_sha256': _digest(_submission),
          'rows': len(_final), 'changed': _overlay['changed_rows'],
          'top1_changed': _overlay['top1_changed'], 'errors': len(_overlay['errors'])}}
_P('/kaggle/working/{args.model_name}-mixup-audit.json').write_text(_j.dumps(_audit, indent=2))
print('MIXUP AUDIT', _j.dumps(_audit), flush=True)
"""
    compile(audit, "mixup_audit", "exec")
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
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    receipt.update(
        parent_output_sha256=PARENT_OUTPUT_SHA,
        candidate_set_sha256=CANDIDATE_SET_SHA,
        final_notebook_sha256=sha(notebook_path),
    )
    if experiment_id:
        receipt["experiment_id"] = experiment_id
    (args.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "parent-metadata", "inference-module", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("dataset", "kernel", "model-name"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--experiment-id")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
