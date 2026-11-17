"""Build a local-only 0.477 variant with fixed generation and CE rank features.

Only frozen cells 37 (adapter) and 39 (label-free audit) change. The outer
Notebook retains its exact parent pipeline and input datasets. This builder
never pushes a Notebook or creates a Kaggle submission.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
from pathlib import Path

from build_ce_analog_notebook import CONTROL_FROZEN_SHA256, CONTROL_NOTEBOOK_SHA256


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(control_bytes: bytes, adapter_bytes: bytes) -> tuple[dict, dict]:
    if digest(control_bytes) != CONTROL_NOTEBOOK_SHA256:
        raise ValueError("0.477 control Notebook differs from pin")
    adapter = adapter_bytes.decode("utf-8")
    compile(adapter, "ce_analog_rank_only.py", "exec")
    adapter_sha = digest(adapter_bytes)
    notebook = json.loads(control_bytes)
    source = "".join(notebook["cells"][8]["source"])
    old_hash = re.search(r"^FROZEN_SHA256 = '([0-9a-f]{64})'$", source, re.M)
    old_b64 = re.search(r"^FROZEN_B64 = '([A-Za-z0-9+/=]+)'$", source, re.M)
    if old_hash is None or old_b64 is None:
        raise ValueError("Control's embedded frozen-source contract changed")
    frozen_bytes = base64.b64decode(old_b64.group(1), validate=True)
    if digest(frozen_bytes) != old_hash.group(1) or digest(frozen_bytes) != CONTROL_FROZEN_SHA256:
        raise ValueError("0.477 frozen Notebook differs from pin")
    frozen = json.loads(frozen_bytes)
    original_cells = copy.deepcopy(frozen["cells"])
    lines = "".join(frozen["cells"][37]["source"]).splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith("V = v1engine.V1FE(")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one V1FE construction in frozen cell 37")
    injection = [
        "# Local causal audit: each-CE rank features, original merged-CE generation.\n",
        "import hashlib as _ce_hashlib, types as _ce_types\n",
        f"_ce_adapter_source = {adapter!r}\n",
        "assert _ce_hashlib.sha256(_ce_adapter_source.encode('utf-8')).hexdigest() "
        f"== '{adapter_sha}'\n",
        "_ce_adapter_module = _ce_types.ModuleType('ce_analog_rank_only')\n",
        "exec(compile(_ce_adapter_source, 'ce_analog_rank_only.py', 'exec'), "
        "_ce_adapter_module.__dict__)\n",
        "_ce_adapter_module.install(E)\n",
        f"print('CE rank-only adapter SHA-256:', '{adapter_sha}', flush=True)\n",
    ]
    lines[matches[0] + 1:matches[0] + 1] = injection
    compile("".join(lines), "ce_analog_rank_only_cell_37", "exec")
    frozen["cells"][37]["source"] = lines
    audit = [
        "\n# Label-free CE analog rank-only diagnostics.\n",
        "import json as _ce_json\n",
        "with open('/kaggle/working/ce-rank-only-audit.json', 'w', encoding='utf-8') as _ce_file:\n",
        "    _ce_json.dump(E._ce_rank_only_history, _ce_file, sort_keys=True, indent=2)\n",
        "print('CE RANK-ONLY AUDIT', len(E._ce_rank_only_history), flush=True)\n",
    ]
    frozen["cells"][39]["source"] = original_cells[39]["source"] + audit
    if [i for i, (a, b) in enumerate(zip(original_cells, frozen["cells"], strict=True)) if a != b] != [37, 39]:
        raise ValueError("Unexpected frozen cell mutation")
    payload = json.dumps(frozen, ensure_ascii=False, separators=(",", ":")).encode()
    variant_sha = digest(payload)
    source = source.replace(old_hash.group(0), f"FROZEN_SHA256 = '{variant_sha}'")
    source = source.replace(old_b64.group(0), f"FROZEN_B64 = '{base64.b64encode(payload).decode()}'")
    notebook["cells"][8]["source"] = source.splitlines(keepends=True)
    notebook["cells"].insert(0, {
        "cell_type": "markdown", "metadata": {},
        "source": ["# Local CE analog rank-feature ablation\n", "\n",
                   "Frozen 0.477 parent; merged analogs generate candidates and "
                   "individual-CE analogs feed the unchanged ranker.\n"],
    })
    receipt = {
        "control_notebook_sha256": digest(control_bytes),
        "control_frozen_sha256": digest(frozen_bytes),
        "variant_frozen_sha256": variant_sha,
        "adapter_source_sha256": adapter_sha,
        "changed_frozen_cells": [37, 39],
        "local_only": True,
    }
    return notebook, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-notebook", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=Path(__file__).with_name("ce_analog_rank_only.py"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    notebook, receipt = build(args.control_notebook.read_bytes(), args.adapter.read_bytes())
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "ce-analog-rank-only-local.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    receipt["variant_notebook_sha256"] = digest(path.read_bytes())
    (args.output / "build-receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
