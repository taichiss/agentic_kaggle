"""Build a pinned 0.477 parent variant using separate CE views for analog search.

The frozen parent engine merges all spectra of the same polarity before selecting
analog structures for its rule-based candidate generator. ``each`` changes only
that selection to score every measured spectrum separately and take the best
score per analog. The downstream candidate generation and ranking stay in the
original parent notebook.
"""

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
ENGINE_INIT = "L, E = _external_reference_engine(_EXTERNAL_REFERENCE_STATE, L, P, tfp, bank)\n"
ANALOG_QUERY_PATCH = (
    "assert E.cfg.analog_query == 'merged', 'Unexpected control analog-query mode'\n"
    "E.cfg.analog_query = 'each'\n"
    "print('engine analog query:', E.cfg.analog_query, flush=True)\n"
)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_variant(
    control_bytes: bytes,
    control_metadata: dict,
    *,
    analog_query: str,
    kernel: str,
    expected_control_sha256: str = CONTROL_NOTEBOOK_SHA256,
    expected_frozen_sha256: str = CONTROL_FROZEN_SHA256,
) -> tuple[dict, dict, dict]:
    """Return notebook, Kaggle metadata, and a source comparison receipt."""
    if analog_query not in {"merged", "each"}:
        raise ValueError(f"Unsupported analog query: {analog_query}")
    if not re.fullmatch(r"[a-z0-9-]+/[a-z0-9-]+", kernel):
        raise ValueError(f"Invalid Kaggle kernel slug: {kernel}")
    control_hash = sha256(control_bytes)
    if control_hash != expected_control_sha256:
        raise ValueError("0.477 control Notebook SHA-256 differs from its pin")

    notebook = json.loads(control_bytes)
    source = "".join(notebook["cells"][8]["source"])
    old_hash = re.search(r"^FROZEN_SHA256 = '([0-9a-f]{64})'$", source, re.M)
    old_b64 = re.search(r"^FROZEN_B64 = '([A-Za-z0-9+/=]+)'$", source, re.M)
    if old_hash is None or old_b64 is None:
        raise ValueError("The control Notebook's frozen-source contract changed")
    frozen_bytes = base64.b64decode(old_b64.group(1), validate=True)
    if sha256(frozen_bytes) != old_hash.group(1):
        raise ValueError("Embedded frozen Notebook failed its own SHA-256 pin")
    if old_hash.group(1) != expected_frozen_sha256:
        raise ValueError("Embedded frozen Notebook differs from the 0.477 control pin")

    changed_cells: list[int] = []
    new_frozen_bytes = frozen_bytes
    if analog_query == "each":
        frozen = json.loads(frozen_bytes)
        original_cells = copy.deepcopy(frozen["cells"])
        target = "".join(frozen["cells"][37]["source"])
        if target.count(ENGINE_INIT) != 1:
            raise ValueError("The main engine initialization differs from the control")
        patched = target.replace(ENGINE_INIT, ENGINE_INIT + ANALOG_QUERY_PATCH)
        compile(patched, "ce_analog_frozen_cell_37", "exec")
        frozen["cells"][37]["source"] = patched.splitlines(keepends=True)
        changed_cells = [i for i, (before, after) in enumerate(
            zip(original_cells, frozen["cells"], strict=True)
        ) if before != after]
        if changed_cells != [37]:
            raise ValueError(f"Unexpected frozen cell changes: {changed_cells}")
        new_frozen_bytes = json.dumps(frozen, ensure_ascii=False, separators=(",", ":")).encode()

    new_frozen_hash = sha256(new_frozen_bytes)
    source = source.replace(old_hash.group(0), f"FROZEN_SHA256 = '{new_frozen_hash}'")
    source = source.replace(
        old_b64.group(0),
        f"FROZEN_B64 = '{base64.b64encode(new_frozen_bytes).decode()}'",
    )
    notebook["cells"][8]["source"] = source.splitlines(keepends=True)
    notebook["cells"].insert(0, {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# CE analog-query experiment: {analog_query}\n",
            "\n",
            "This Notebook runs the pinned 0.477 parent pipeline. The original "
            "published 0.477 score is a historical control, not a score for "
            "this private run.\n",
            f"Frozen parent SHA-256: `{new_frozen_hash}`.\n",
            f"Analog-query mode: `{analog_query}`.\n",
        ],
    })

    metadata = copy.deepcopy(control_metadata)
    metadata["id"] = kernel
    metadata["title"] = kernel.split("/", 1)[1].replace("-", " ")
    metadata["code_file"] = f"ce-analog-{analog_query}.ipynb"
    metadata["is_private"] = True
    receipt = {
        "control_notebook_sha256": control_hash,
        "control_frozen_sha256": old_hash.group(1),
        "variant_frozen_sha256": new_frozen_hash,
        "changed_frozen_cells": changed_cells,
        "analog_query": analog_query,
        "kernel": kernel,
    }
    return notebook, metadata, receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-notebook", required=True, type=Path)
    parser.add_argument("--control-metadata", required=True, type=Path)
    parser.add_argument("--analog-query", required=True, choices=("merged", "each"))
    parser.add_argument("--kernel", required=True, help="Kaggle owner/kernel-slug")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    notebook, metadata, receipt = build_variant(
        args.control_notebook.read_bytes(),
        json.loads(args.control_metadata.read_text()),
        analog_query=args.analog_query,
        kernel=args.kernel,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / metadata["code_file"]).write_text(
        json.dumps(notebook, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
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
