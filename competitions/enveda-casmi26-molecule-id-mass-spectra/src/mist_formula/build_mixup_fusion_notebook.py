"""Build an A-parent notebook that fuses two Mixup candidate-score arrays once."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from build_mixup_notebook import (
    CANDIDATE_SET_SHA,
    INFERENCE_SHA,
    PARENT_OUTPUT_SHA,
    PARENT_SHA,
    sha,
)


def build(args: argparse.Namespace) -> dict:
    if not re.fullmatch(r"[a-z0-9-]+", args.model_name):
        raise ValueError("model-name must be a lowercase Kaggle-safe slug")
    for ref in (args.dataset_a, args.dataset_b, args.kernel):
        if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", ref):
            raise ValueError(f"Invalid Kaggle reference: {ref}")
    if not 0 <= args.weight <= 1:
        raise ValueError("rank-fusion weight must lie in [0,1]")
    if not 0 <= args.score_alpha_a <= 1:
        raise ValueError("score alpha for model A must lie in [0,1]")
    if sha(args.parent) != PARENT_SHA or sha(args.inference_module) != INFERENCE_SHA:
        raise ValueError("Fusion must use the exact A parent and inference source")
    inference_source = args.inference_module.read_text(encoding="utf-8")
    fusion_source = args.fusion_module.read_text(encoding="utf-8")
    compile(inference_source, "frozen_a_inference", "exec")
    compile(fusion_source, "candidate_score_fusion", "exec")
    parent = json.loads(args.parent.read_text(encoding="utf-8"))
    metadata = json.loads(args.parent_metadata.read_text(encoding="utf-8"))
    a_sha, b_sha = sha(args.checkpoint_a), sha(args.checkpoint_b)
    fusion_sha = hashlib.sha256(fusion_source.encode()).hexdigest()
    code = f"""# Two 2048-bit Mixup checkpoints; score-level fusion, then one A-parent RRF.
import hashlib as _h, importlib.util as _iu, json as _j, os as _o, shutil as _sh, sys as _sys, time as _t
from pathlib import Path as _P
import numpy as _np, pandas as _pd

_name = {args.model_name!r}
_sources = {{'mixup_inference': {inference_source!r}, 'mixup_score_fusion': {fusion_source!r}}}
_hashes = {{'mixup_inference': {INFERENCE_SHA!r}, 'mixup_score_fusion': {fusion_sha!r}}}
_work = _P('/kaggle/working/mixup-fusion-source'); _work.mkdir(exist_ok=True)
def _load(name):
    source = _sources[name]
    assert _h.sha256(source.encode()).hexdigest() == _hashes[name]
    path = _work / (name + '.py'); path.write_text(source, encoding='utf-8')
    spec = _iu.spec_from_file_location(name, path)
    module = _iu.module_from_spec(spec); _sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
_model = _load('mixup_inference')
_fusion = _load('mixup_score_fusion')
_alpha_a = {args.score_alpha_a!r}
def _weighted_scores(names, scores_a, scores_b):
    fused = _np.full(len(names), -_np.inf)
    groups = {{}}
    for index, smiles in enumerate(names):
        key = _fusion.formula(smiles)
        if key is not None:
            groups.setdefault(key, []).append(index)
    for indices in groups.values():
        if sum(index > 0 for index in indices) < 2:
            continue
        a, b = scores_a[indices], scores_b[indices]
        if not _np.isfinite(a).all() or not _np.isfinite(b).all():
            continue
        def _z(values):
            sd = values.std()
            return (values - values.mean()) / sd if sd > 1e-9 else _np.zeros_like(values)
        values = _alpha_a * _z(a) + (1 - _alpha_a) * _z(b)
        for index, value in zip(indices, values):
            if index > 0:
                fused[index] = value
    return fused
def _sha(path):
    digest = _h.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()
def _checkpoint(slug, expected):
    hits = sorted(_P('/kaggle/input').glob('**/' + slug + '/**/final.pt'))
    assert len(hits) == 1, ('checkpoint missing or ambiguous', slug, hits)
    assert _sha(hits[0]) == expected, ('checkpoint hash mismatch', slug)
    return hits[0]
_checkpoint_a = _checkpoint({args.dataset_a.split("/")[-1]!r}, {a_sha!r})
_checkpoint_b = _checkpoint({args.dataset_b.split("/")[-1]!r}, {b_sha!r})
_control = _P('/kaggle/working/' + _name + '-control.csv')
_sh.copyfile('/kaggle/working/submission.csv', _control)
assert _sha(_control) == {PARENT_OUTPUT_SHA!r}, '0.477 parent output differs from A'
_base = _pd.read_csv(_control, dtype=str, keep_default_na=False)
assert list(_base.columns) == ['molecule_id', 'smiles'] and len(_base) == 400
_test = _pd.read_parquet(COMP / 'test.parquet')
_groups = {{str(k): g.to_dict('records') for k, g in _test.groupby('molecule_id', sort=False)}}
_device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
_start = _t.monotonic()
_out = []
_report = {{'model': _name, 'rows': len(_base), 'attempted_rows': 0, 'changed_rows': 0,
           'top1_changed': 0, 'eligible_groups': 0, 'missing_groups': 0, 'errors': [],
           'score_fusion': 'within-formula z-score weighted candidate scores',
           'score_alpha_a': _alpha_a, 'score_alpha_b': 1-_alpha_a,
           'selection_note': {args.selection_note!r},
           'rank_fusion_weight': {args.weight!r}}}
for _row in _base.itertuples(index=False):
    names = _row.smiles.split(';')
    if len(names) == 1:
        _out.append(_row.smiles); continue
    _report['attempted_rows'] += 1
    spectra = _groups.get(str(_row.molecule_id), [])
    if not spectra:
        _report['errors'].append([str(_row.molecule_id), 'missing spectra'])
        _out.append(_row.smiles); continue
    try:
        scores_a = _np.asarray(_model.score_candidates(spectra, names, _checkpoint_a, _device), float)
        scores_b = _np.asarray(_model.score_candidates(spectra, names, _checkpoint_b, _device), float)
        if scores_a.shape != (len(names),) or scores_b.shape != (len(names),):
            raise ValueError('invalid score shape')
        fused_scores = _weighted_scores(names, scores_a, scores_b)
        order, diag = _fusion.rerank(names, fused_scores, weight={args.weight!r})
        _report['eligible_groups'] += diag['eligible_groups']
        _report['missing_groups'] += diag['missing_groups']
        new = [names[i] for i in order]
        assert new[0] == names[0] and sorted(new) == sorted(names)
        assert [_fusion.formula(s) for s in new] == [_fusion.formula(s) for s in names]
        _report['changed_rows'] += (new != names)
        _out.append(';'.join(new))
    except Exception as exc:
        _report['errors'].append([str(_row.molecule_id), repr(exc)[:300]])
        _out.append(_row.smiles)
if not _report['attempted_rows'] or len(_report['errors']) > max(2, _report['attempted_rows'] // 20):
    raise RuntimeError('too many inference failures: ' + repr(_report['errors'][:3]))
_final = _base.copy(); _final['smiles'] = _out
_tmp = _P('/kaggle/working/' + _name + '-submission.csv'); _final.to_csv(_tmp, index=False)
_check = _pd.read_csv(_tmp, dtype=str, keep_default_na=False)
assert _check.equals(_final)
_o.replace(_tmp, '/kaggle/working/submission.csv')
_report.update(parent_notebook_sha256={PARENT_SHA!r}, parent_output_sha256=_sha(_control),
               checkpoint_a_sha256={a_sha!r}, checkpoint_b_sha256={b_sha!r},
               inference_sha256={INFERENCE_SHA!r}, fusion_module_sha256={fusion_sha!r},
               submission_sha256=_sha(_P('/kaggle/working/submission.csv')),
               seconds=round(_t.monotonic() - _start, 2))
_P('/kaggle/working/' + _name + '-overlay-report.json').write_text(_j.dumps(_report, indent=2))
print('MIXUP FUSION REPORT', _j.dumps({{k:v for k,v in _report.items() if k != 'errors'}}), flush=True)
"""
    compile(code, "mixup_score_fusion_overlay", "exec")
    audit = f"""# Verify exact A parent, candidate set, protected rank 1 and both checkpoints.
import hashlib as _h, json as _j
from pathlib import Path as _P
import pandas as _pd
_base = _pd.read_csv(_P('/kaggle/working/{args.model_name}-control.csv'), dtype=str, keep_default_na=False)
_final = _pd.read_csv(_P('/kaggle/working/submission.csv'), dtype=str, keep_default_na=False)
assert len(_base) == len(_final) == 400
assert _base.molecule_id.tolist() == _final.molecule_id.tolist()
_members = '\\n'.join(str(m)+'\\t'+';'.join(sorted(s.split(';'))) for m,s in zip(_base.molecule_id,_base.smiles))
assert _h.sha256(_members.encode()).hexdigest() == {CANDIDATE_SET_SHA!r}
for old, new in zip(_base.smiles, _final.smiles):
    a, b = old.split(';'), new.split(';')
    assert len(a) == len(b) and sorted(a) == sorted(b) and a[0] == b[0]
_report = _j.loads(_P('/kaggle/working/{args.model_name}-overlay-report.json').read_text())
assert not _report['errors'], _report['errors'][:3]
assert _report['parent_output_sha256'] == {PARENT_OUTPUT_SHA!r}
assert _report['checkpoint_a_sha256'] == {a_sha!r}
assert _report['checkpoint_b_sha256'] == {b_sha!r}
_audit = {{'parent_notebook_sha256': {PARENT_SHA!r}, 'parent_output_sha256': {PARENT_OUTPUT_SHA!r},
          'candidate_set_sha256': {CANDIDATE_SET_SHA!r},
          'checkpoint_a_sha256': {a_sha!r}, 'checkpoint_b_sha256': {b_sha!r},
          'score_fusion': 'formula-local z weighted', 'score_alpha_a': {args.score_alpha_a!r},
          'score_alpha_b': {1 - args.score_alpha_a!r},
          'selection_note': {args.selection_note!r},
          'parent_rank_fusion_weight': {args.weight!r},
          'changed_rows': _report['changed_rows'], 'top1_changed': _report['top1_changed'],
          'submission_sha256': _report['submission_sha256']}}
_P('/kaggle/working/{args.model_name}-mixup-audit.json').write_text(_j.dumps(_audit, indent=2))
print('MIXUP FUSION AUDIT', _j.dumps(_audit), flush=True)
"""
    compile(audit, "mixup_score_fusion_audit", "exec")
    notebook = json.loads(json.dumps(parent))
    for title, source in (("Two-checkpoint score fusion", code), ("Submission audit", audit)):
        notebook["cells"].append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"## {title}\n"],
            }
        )
        notebook["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            }
        )
    metadata["id"] = args.kernel
    metadata["title"] = args.kernel.split("/", 1)[1].replace("-", " ")
    metadata["code_file"] = "mixup-fusion.ipynb"
    metadata["dataset_sources"] = list(
        dict.fromkeys(
            [
                *metadata["dataset_sources"],
                args.dataset_a,
                args.dataset_b,
            ]
        )
    )
    metadata["is_private"] = True
    args.output.mkdir(parents=True, exist_ok=True)
    notebook_path = args.output / metadata["code_file"]
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")))
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    receipt = {
        "parent_notebook_sha256": PARENT_SHA,
        "parent_output_sha256": PARENT_OUTPUT_SHA,
        "candidate_set_sha256": CANDIDATE_SET_SHA,
        "inference_sha256": INFERENCE_SHA,
        "fusion_module_sha256": fusion_sha,
        "checkpoint_a_sha256": a_sha,
        "checkpoint_b_sha256": b_sha,
        "dataset_a": args.dataset_a,
        "dataset_b": args.dataset_b,
        "kernel": args.kernel,
        "score_alpha_a": args.score_alpha_a,
        "score_alpha_b": 1 - args.score_alpha_a,
        "selection_note": args.selection_note,
        "rank_fusion_weight": args.weight,
        "notebook_sha256": sha(notebook_path),
    }
    (args.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent",
        "parent-metadata",
        "inference-module",
        "fusion-module",
        "checkpoint-a",
        "checkpoint-b",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("dataset-a", "dataset-b", "kernel", "model-name"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--score-alpha-a", type=float, default=0.5)
    parser.add_argument("--selection-note", default="Predeclared score-level fusion")
    print(json.dumps(build(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
