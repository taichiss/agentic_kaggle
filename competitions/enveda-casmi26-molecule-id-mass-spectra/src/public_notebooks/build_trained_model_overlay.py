"""Append a checked trained-model reranker to the pinned 0.477 notebook."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"[a-z0-9-]+", args.model_name):
        raise ValueError("model_name must be a simple lowercase slug")
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", args.kernel):
        raise ValueError("Invalid Kaggle kernel ref")
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", args.dataset):
        raise ValueError("Invalid Kaggle dataset ref")
    if args.weight < 0 or args.weight > 1:
        raise ValueError("weight must be between 0 and 1")
    if args.protect_top_n < 0 or args.protect_top_n > 25:
        raise ValueError("protect_top_n must be between 0 and 25")
    control_hash = digest(args.control_notebook)
    if control_hash != args.control_sha256:
        raise ValueError(f"Control notebook SHA-256 mismatch: {control_hash}")
    module_source = args.module.read_text(encoding="utf-8")
    compile(module_source, str(args.module), "exec")
    module_hash = digest(args.module)
    checkpoint_hash = digest(args.checkpoint)
    notebook = json.loads(args.control_notebook.read_text(encoding="utf-8"))
    metadata = json.loads(args.control_metadata.read_text(encoding="utf-8"))
    if not notebook.get("cells") or "dataset_sources" not in metadata:
        raise ValueError("Invalid parent notebook or Kaggle metadata")

    code = f'''# Trained-model overlay: {args.model_name}
import hashlib as _hashlib
import importlib.util as _importlib_util
import json as _json
import os as _os
import shutil as _shutil
import sys as _sys
import time as _time
from pathlib import Path as _Path
import numpy as _np
import pandas as _pd

_model_name = {args.model_name!r}
_model_source = {module_source!r}
_model_source_sha = {module_hash!r}
assert _hashlib.sha256(_model_source.encode('utf-8')).hexdigest() == _model_source_sha
_extension = _Path('/kaggle/working/trained_model_overlay')
_extension.mkdir(exist_ok=True)
_module_path = _extension / 'inference.py'
_module_path.write_text(_model_source, encoding='utf-8')
_spec = _importlib_util.spec_from_file_location('trained_model_inference', _module_path)
_model = _importlib_util.module_from_spec(_spec)
_sys.modules[_spec.name] = _model
_spec.loader.exec_module(_model)
_dataset_slug = {args.dataset.split('/')[-1]!r}
_matches = sorted(_Path('/kaggle/input').glob(
    '**/' + _dataset_slug + '/**/' + {args.checkpoint.name!r}))
assert len(_matches) == 1, ('model checkpoint missing or ambiguous', _matches)
_checkpoint = _matches[0]
def _sha256_file(path):
    d = _hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            d.update(chunk)
    return d.hexdigest()
assert _sha256_file(_checkpoint) == {checkpoint_hash!r}, 'model checkpoint hash mismatch'

_original = _Path('/kaggle/working/submission.csv')
_control = _Path('/kaggle/working/' + _model_name + '-control.csv')
_shutil.copyfile(_original, _control)
_pred = _pd.read_csv(_control, dtype=str, keep_default_na=False)
assert list(_pred.columns) == ['molecule_id', 'smiles']
_test = _pd.read_parquet(COMP / 'test.parquet')
_groups = {{k: v.to_dict('records') for k, v in _test.groupby('molecule_id', sort=False)}}
_device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
_start = _time.monotonic()
_errors = []
_attempted = 0
_changed = 0
_top1_changed = 0
_weight = {args.weight!r}
_protected = {args.protect_top_n!r}
_within_formula = {args.within_formula!r}
if _within_formula:
    from rdkit import Chem as _Chem
    from rdkit.Chem import rdMolDescriptors as _rdMolDescriptors
    def _formula(smiles):
        mol = _Chem.MolFromSmiles(smiles)
        return None if mol is None else _rdMolDescriptors.CalcMolFormula(mol)
_new_lists = []
for _row in _pred.itertuples(index=False):
    _candidates = _row.smiles.split(';')
    if not _candidates or any(not s for s in _candidates):
        raise ValueError('empty candidate in parent submission')
    if len(_candidates) == 1:
        _new_lists.append(_row.smiles)
        continue
    _attempted += 1
    _spectra = _groups.get(_row.molecule_id, [])
    if not _spectra:
        _new_lists.append(_row.smiles)
        _errors.append([_row.molecule_id, 'missing spectra'])
        continue
    try:
        _scores = _np.asarray(
            _model.score_candidates(_spectra, _candidates, _checkpoint, _device),
            dtype=float,
        )
        if (_scores.shape != (len(_candidates),) or
                _np.any(_np.isnan(_scores)) or _np.any(_np.isposinf(_scores)) or
                not _np.any(_np.isfinite(_scores))):
            raise ValueError('invalid scores shape or no finite candidate score')
        if _within_formula:
            _order = list(range(len(_candidates)))
            _formula_groups = {{}}
            for _i in range(_protected, len(_candidates)):
                _f = _formula(_candidates[_i])
                if _f is not None:
                    _formula_groups.setdefault(_f, []).append(_i)
            for _indices in _formula_groups.values():
                _model_order = sorted(_indices, key=lambda i: (-_scores[i], i))
                _model_rank = {{index: rank for rank, index in enumerate(_model_order)}}
                _local_score = {{
                    index: (1 - _weight) / (1 + base_rank) +
                    _weight / (1 + _model_rank[index])
                    for base_rank, index in enumerate(_indices)
                }}
                _sorted = sorted(_indices, key=lambda i: (-_local_score[i], i))
                for _slot, _candidate_index in zip(_indices, _sorted):
                    _order[_slot] = _candidate_index
        else:
            _model_order = _np.argsort(-_scores, kind='stable')
            _model_rank = _np.empty(len(_candidates), dtype=int)
            _model_rank[_model_order] = _np.arange(len(_candidates))
            _base_rank = _np.arange(len(_candidates))
            _fused = (1 - _weight) / (1 + _base_rank) + _weight / (1 + _model_rank)
            _order = list(range(_protected)) + sorted(
                range(_protected, len(_candidates)),
                key=lambda i: (-_fused[i], i),
            )
        _reranked = [_candidates[i] for i in _order]
    except Exception as exc:
        _errors.append([_row.molecule_id, repr(exc)[:300]])
        _reranked = _candidates
    _changed += _reranked != _candidates
    _top1_changed += _reranked[0] != _candidates[0]
    _new_lists.append(';'.join(_reranked))
if _attempted == 0 or len(_errors) > max(2, _attempted // 20):
    raise RuntimeError('too many model inference failures: ' + repr(_errors[:5]))
_pred['smiles'] = _new_lists
_temp = _Path('/kaggle/working/' + _model_name + '-submission.csv')
_pred.to_csv(_temp, index=False)
_check = _pd.read_csv(_temp, dtype=str, keep_default_na=False)
assert _check.molecule_id.tolist() == _pd.read_csv(_control, dtype=str).molecule_id.tolist()
for _old, _new in zip(_pd.read_csv(_control, dtype=str).smiles, _check.smiles):
    assert sorted(_old.split(';')) == sorted(_new.split(';'))
    assert _old.split(';')[:_protected] == _new.split(';')[:_protected]
    if _within_formula:
        assert [_formula(s) for s in _old.split(';')] == [
            _formula(s) for s in _new.split(';')]
_os.replace(_temp, _original)
_report = {{
    'model': _model_name,
    'checkpoint_sha256': {checkpoint_hash!r},
    'inference_sha256': _model_source_sha,
    'rows': len(_pred),
    'attempted_rows': _attempted,
    'changed_rows': int(_changed),
    'top1_changed': int(_top1_changed),
    'errors': _errors,
    'weight': _weight,
    'protected_top_n': _protected,
    'within_formula': _within_formula,
    'elapsed_seconds': round(_time.monotonic() - _start, 3),
    'submission_sha256': _sha256_file(_original),
}}
_Path('/kaggle/working/' + _model_name + '-overlay-report.json').write_text(
    _json.dumps(_report, indent=2), encoding='utf-8')
print('TRAINED MODEL OVERLAY:', _json.dumps({{k:v for k,v in _report.items() if k != 'errors'}}))
'''
    compile(code, "trained_model_overlay", "exec")
    notebook["cells"].extend(
        [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"## Trained {args.model_name} reranker\n",
                    "This variant appends a new trained model after the unchanged 0.477 "
                    "pipeline. The parent's score does not describe this variant.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            },
        ]
    )
    metadata["id"] = args.kernel
    metadata["title"] = args.kernel.split("/", 1)[1].replace("-", " ")
    metadata["code_file"] = "trained-model-overlay.ipynb"
    metadata["dataset_sources"] = list(
        dict.fromkeys([*metadata["dataset_sources"], args.dataset])
    )
    metadata["is_private"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / metadata["code_file"]).write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (args.output / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    receipt = {
        "model": args.model_name,
        "control_sha256": control_hash,
        "module_sha256": module_hash,
        "checkpoint_sha256": checkpoint_hash,
        "dataset": args.dataset,
        "kernel": args.kernel,
        "weight": args.weight,
        "protected_top_n": args.protect_top_n,
        "within_formula": args.within_formula,
    }
    (args.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-notebook", type=Path, required=True)
    parser.add_argument("--control-sha256", required=True)
    parser.add_argument("--control-metadata", type=Path, required=True)
    parser.add_argument("--module", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--protect-top-n", type=int, default=1)
    parser.add_argument("--within-formula", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
