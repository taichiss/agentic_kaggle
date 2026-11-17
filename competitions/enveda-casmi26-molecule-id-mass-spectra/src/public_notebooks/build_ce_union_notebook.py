"""Build a pinned 0.477 private Notebook with CE-union generator parents."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import re
from pathlib import Path

CONTROL_NOTEBOOK_SHA256 = "ef88fe151bf99fac82c8e3ad4b401d20aa0b6626880a7c3a8ed53a9022e0f938"
CONTROL_FROZEN_SHA256 = "d1e757c2b71b46e6e8a5777f14302a4737d0fdbcd56d8365093c039b93c5fd88"
ADAPTER_PATH = Path(__file__).with_name("ce_union_generator.py")
HELPER_PATH = Path(__file__).with_name("ce_diverse_generator.py")
V1FE_PREFIX = "V = v1engine.V1FE("


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_variant(
    control_bytes: bytes, metadata: dict, adapter_bytes: bytes, *, kernel: str
) -> tuple[dict, dict, dict]:
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", kernel):
        raise ValueError(f"Invalid Kaggle kernel slug: {kernel}")
    control_sha = sha256(control_bytes)
    if control_sha != CONTROL_NOTEBOOK_SHA256:
        raise ValueError("0.477 control Notebook SHA-256 differs from its pin")
    helper_source = HELPER_PATH.read_text(encoding="utf-8")
    compile(helper_source, "ce_diverse_generator.py", "exec")
    helper_sha = sha256(HELPER_PATH.read_bytes())
    adapter_source = adapter_bytes.decode("utf-8")
    compile(adapter_source, "ce_union_generator.py", "exec")
    adapter_sha = sha256(adapter_bytes)

    notebook = json.loads(control_bytes)
    source = "".join(notebook["cells"][8]["source"])
    old_hash = re.search(r"^FROZEN_SHA256 = '([0-9a-f]{64})'$", source, re.M)
    old_b64 = re.search(r"^FROZEN_B64 = '([A-Za-z0-9+/=]+)'$", source, re.M)
    if old_hash is None or old_b64 is None:
        raise ValueError("The control Notebook's frozen-source contract changed")
    frozen_bytes = base64.b64decode(old_b64.group(1), validate=True)
    if sha256(frozen_bytes) != old_hash.group(1) or old_hash.group(1) != CONTROL_FROZEN_SHA256:
        raise ValueError("Embedded frozen Notebook differs from the 0.477 control pin")
    frozen = json.loads(frozen_bytes)
    original_cells = copy.deepcopy(frozen["cells"])

    cell37 = "".join(frozen["cells"][37]["source"])
    lines = cell37.splitlines(keepends=True)
    matches = [index for index, line in enumerate(lines) if line.startswith(V1FE_PREFIX)]
    if len(matches) != 1:
        raise ValueError("Expected exactly one V1FE construction in frozen cell 37")
    injection = [
        "# CE-union products; merged analogs still enter rank features.\n",
        "import hashlib as _ce_hashlib, types as _ce_types, sys as _ce_sys\n",
        f"_ce_helper_source = {helper_source!r}\n",
        "assert _ce_hashlib.sha256(_ce_helper_source.encode('utf-8')).hexdigest() "
        f"== {helper_sha!r}\n",
        "_ce_helper_module = _ce_types.ModuleType('ce_diverse_generator')\n",
        "exec(compile(_ce_helper_source, 'ce_diverse_generator.py', 'exec'),\n"
        "     _ce_helper_module.__dict__)\n",
        "_ce_sys.modules['ce_diverse_generator'] = _ce_helper_module\n",
        f"_ce_adapter_source = {adapter_source!r}\n",
        "assert _ce_hashlib.sha256(_ce_adapter_source.encode('utf-8')).hexdigest() "
        f"== {adapter_sha!r}\n",
        "_ce_adapter_module = _ce_types.ModuleType('ce_union_generator')\n",
        "exec(compile(_ce_adapter_source, 'ce_union_generator.py', 'exec'),\n"
        "     _ce_adapter_module.__dict__)\n",
        "_ce_adapter_module.install(E)\n",
        f"print('CE-union generator adapter SHA-256:', {adapter_sha!r}, flush=True)\n",
    ]
    lines[matches[0] + 1 : matches[0] + 1] = injection
    compile("".join(lines), "ce_union_frozen_cell_37", "exec")
    frozen["cells"][37]["source"] = lines

    cell39 = "".join(frozen["cells"][39]["source"])
    old_spectrum_end = "ce_n=int(len(ce_arr)) if len(ce_arr) else 1))"
    new_spectrum_end = (
        "ce_n=int(len(ce_arr)) if len(ce_arr) else 1, "
        "instrument=str(r.instrument_type)))"
    )
    if cell39.count(old_spectrum_end) != 1:
        raise ValueError("Expected one spectrum construction in frozen cell 39")
    cell39 = cell39.replace(old_spectrum_end, new_spectrum_end)
    audit = [
        "\n# Label-free CE-union generator diagnostics.\n",
        "import json as _ce_json\n",
        "_ce_history = E._ce_union_history\n",
        "_ce_audit = {\n",
        f"    'adapter_source_sha256': {adapter_sha!r},\n",
        "    'generator_calls': len(_ce_history),\n",
        "    'eligible_queries': sum(len(row['eligible_ce_views']) >= 2 for row in _ce_history),\n",
        "    'baseline_generated_total': sum(row['baseline_generated'] for row in _ce_history),\n",
        "    'extra_generated_total': sum(row['extra_generated'] for row in _ce_history),\n",
        "    'generated_candidates_total': sum(\n",
        "        row['generated_candidates'] for row in _ce_history\n",
        "    ),\n",
        "    'queries': _ce_history,\n",
        "}\n",
        "with open('ce-union-generator-audit.json', 'w', encoding='utf-8') as _ce_file:\n",
        "    _ce_json.dump(_ce_audit, _ce_file, sort_keys=True, indent=2)\n",
        "print('CE UNION GENERATOR AUDIT', _ce_json.dumps({\n",
        "    key: value for key, value in _ce_audit.items() if key != 'queries'\n",
        "}, sort_keys=True), flush=True)\n",
    ]
    compile(cell39 + "".join(audit), "ce_union_frozen_cell_39", "exec")
    frozen["cells"][39]["source"] = (cell39 + "".join(audit)).splitlines(keepends=True)
    changed = [
        index for index, (before, after) in enumerate(
            zip(original_cells, frozen["cells"], strict=True)
        ) if before != after
    ]
    if changed != [37, 39]:
        raise ValueError(f"Unexpected frozen cell changes: {changed}")
    payload = json.dumps(frozen, ensure_ascii=False, separators=(",", ":")).encode()
    frozen_sha = sha256(payload)
    source = source.replace(old_hash.group(0), f"FROZEN_SHA256 = '{frozen_sha}'")
    source = source.replace(
        old_b64.group(0), f"FROZEN_B64 = '{base64.b64encode(payload).decode()}'"
    )
    notebook["cells"][8]["source"] = source.splitlines(keepends=True)
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# CE-union generator parents\n",
            "\n",
            "Pinned 0.477 parent; original merged analog rank features and candidate "
            "pool stay intact. Generator parents are selected across compatible "
            "single-CE views. The original 0.477 score is historical.\n",
            f"Adapter source SHA-256: `{adapter_sha}`.\n",
            f"Frozen Notebook SHA-256: `{frozen_sha}`.\n",
        ],
    })
    output_metadata = copy.deepcopy(metadata)
    output_metadata["id"] = kernel
    output_metadata["title"] = kernel.split("/", 1)[1].replace("-", " ")
    output_metadata["code_file"] = "ce-generator-union.ipynb"
    output_metadata["is_private"] = True
    receipt = {
        "control_notebook_sha256": control_sha,
        "control_frozen_sha256": old_hash.group(1),
        "variant_frozen_sha256": frozen_sha,
        "adapter_source_sha256": adapter_sha,
        "helper_source_sha256": helper_sha,
        "changed_frozen_cells": changed,
        "audit_file": "ce-union-generator-audit.json",
        "mode": "union",
        "kernel": kernel,
    }
    return notebook, output_metadata, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-notebook", required=True, type=Path)
    parser.add_argument("--control-metadata", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    notebook, metadata, receipt = build_variant(
        args.control_notebook.read_bytes(),
        json.loads(args.control_metadata.read_text()),
        args.adapter.read_bytes(), kernel=args.kernel,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / metadata["code_file"]).write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    (args.output / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "build-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
