"""Build pinned 0.477 parent variants for generator-only CE views and mild Mixup."""

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
ADAPTER_PATH = Path(__file__).with_name("ce_mild_mixup.py")
V1FE_PREFIX = "V = v1engine.V1FE("


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_variant(
    control_bytes: bytes, metadata: dict, adapter_bytes: bytes, *, mode: str, kernel: str
) -> tuple[dict, dict, dict]:
    if mode not in {"separate", "mild"}:
        raise ValueError(f"Unsupported CE generator mode: {mode}")
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", kernel):
        raise ValueError(f"Invalid Kaggle kernel slug: {kernel}")
    control_sha = sha256(control_bytes)
    if control_sha != CONTROL_NOTEBOOK_SHA256:
        raise ValueError("0.477 control Notebook SHA-256 differs from its pin")
    adapter_text = adapter_bytes.decode("utf-8")
    compile(adapter_text, "ce_mild_mixup.py", "exec")
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
    target = "".join(frozen["cells"][37]["source"])
    lines = target.splitlines(keepends=True)
    matching = [index for index, line in enumerate(lines) if line.startswith(V1FE_PREFIX)]
    if len(matching) != 1:
        raise ValueError("Expected exactly one V1FE construction in frozen cell 37")
    include_mixup = mode == "mild"
    injection = [
        "# Generator-only CE-view experiment; install after V1FE recording hooks.\n",
        "import hashlib as _ce_hashlib, types as _ce_types\n",
        f"_ce_adapter_source = {adapter_text!r}\n",
        "assert _ce_hashlib.sha256(_ce_adapter_source.encode('utf-8')).hexdigest() "
        f"== '{adapter_sha}'\n",
        "_ce_adapter_module = _ce_types.ModuleType('ce_mild_mixup')\n",
        "exec(compile(_ce_adapter_source, 'ce_mild_mixup.py', 'exec'),\n"
        "     _ce_adapter_module.__dict__)\n",
        f"_ce_adapter_module.install(E, include_mixup={include_mixup})\n",
        f"print('generator CE view mode:', {mode!r},\n"
        f"      'module SHA-256:', {adapter_sha!r}, flush=True)\n",
    ]
    lines[matching[0] + 1 : matching[0] + 1] = injection
    patched = "".join(lines)
    compile(patched, "ce_generator_frozen_cell_37", "exec")
    frozen["cells"][37]["source"] = lines
    audit = [
        "\n# Aggregate label-free CE generator diagnostics for the Notebook output.\n",
        "import json as _ce_json\n",
        "_ce_history = E._ce_view_history\n",
        "_ce_audit = {\n",
        f"    'mode': {mode!r},\n",
        f"    'adapter_source_sha256': {adapter_sha!r},\n",
        "    'generator_calls': len(_ce_history),\n",
        "    'mixed_views_total': sum(row['mixed_views'] for row in _ce_history),\n",
        "    'queries_with_mixed_views': sum(row['mixed_views'] > 0 for row in _ce_history),\n",
        "    'changed_generator_parent_queries': sum(\n",
        "        row['baseline_analog_sids'] != row['generator_analog_sids']\n",
        "        for row in _ce_history\n",
        "    ),\n",
        "    'generated_candidates_total': sum(\n",
        "        row['generated_candidates'] for row in _ce_history\n",
        "    ),\n",
        "}\n",
        "with open('ce-generator-audit.json', 'w', encoding='utf-8') as _ce_file:\n",
        "    _ce_json.dump(_ce_audit, _ce_file, sort_keys=True, indent=2)\n",
        "print('CE GENERATOR AUDIT', _ce_json.dumps(_ce_audit, sort_keys=True), flush=True)\n",
    ]
    cell39 = "".join(frozen["cells"][39]["source"])
    old_spectrum_end = "ce_n=int(len(ce_arr)) if len(ce_arr) else 1))"
    new_spectrum_end = (
        "ce_n=int(len(ce_arr)) if len(ce_arr) else 1, "
        "instrument=str(r.instrument_type)))"
    )
    if cell39.count(old_spectrum_end) != 1:
        raise ValueError("Expected one spectrum construction in frozen cell 39")
    cell39 = cell39.replace(old_spectrum_end, new_spectrum_end)
    compile(cell39 + "".join(audit), "ce_generator_frozen_cell_39", "exec")
    frozen["cells"][39]["source"] = (cell39 + "".join(audit)).splitlines(keepends=True)
    changed = [
        i for i, (before, after) in enumerate(zip(original_cells, frozen["cells"], strict=True))
        if before != after
    ]
    if changed != [37, 39]:
        raise ValueError(f"Unexpected frozen cell changes: {changed}")
    payload = json.dumps(frozen, ensure_ascii=False, separators=(",", ":")).encode()
    new_sha = sha256(payload)
    source = source.replace(old_hash.group(0), f"FROZEN_SHA256 = '{new_sha}'")
    source = source.replace(
        old_b64.group(0), f"FROZEN_B64 = '{base64.b64encode(payload).decode()}'"
    )
    notebook["cells"][8]["source"] = source.splitlines(keepends=True)
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# Generator CE views: {mode}\n",
            "\n",
            "Frozen 0.477 parent with only the analog-derived candidate path changed. "
            "The original 0.477 score is historical and does not apply to this run.\n",
            f"Adapter source SHA-256: `{adapter_sha}`.\n",
            f"Frozen Notebook SHA-256: `{new_sha}`.\n",
        ],
    })
    output_metadata = copy.deepcopy(metadata)
    output_metadata["id"] = kernel
    output_metadata["title"] = kernel.split("/", 1)[1].replace("-", " ")
    output_metadata["code_file"] = f"ce-generator-{mode}.ipynb"
    output_metadata["is_private"] = True
    receipt = {
        "control_notebook_sha256": control_sha,
        "control_frozen_sha256": old_hash.group(1),
        "variant_frozen_sha256": new_sha,
        "adapter_source_sha256": adapter_sha,
        "changed_frozen_cells": changed,
        "audit_file": "ce-generator-audit.json",
        "mode": mode,
        "kernel": kernel,
    }
    return notebook, output_metadata, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-notebook", required=True, type=Path)
    parser.add_argument("--control-metadata", required=True, type=Path)
    parser.add_argument("--adapter", type=Path, default=ADAPTER_PATH)
    parser.add_argument("--mode", required=True, choices=("separate", "mild"))
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    notebook, metadata, receipt = build_variant(
        args.control_notebook.read_bytes(),
        json.loads(args.control_metadata.read_text()),
        args.adapter.read_bytes(), mode=args.mode, kernel=args.kernel
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
