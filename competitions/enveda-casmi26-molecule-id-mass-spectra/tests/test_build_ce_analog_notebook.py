"""The CE analog variant changes only the pinned parent generator's query view."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "src/public_notebooks/build_ce_analog_notebook.py"
)
spec = importlib.util.spec_from_file_location("build_ce_analog_notebook", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def fake_control() -> tuple[bytes, dict, str]:
    frozen = {"cells": [{"source": [f"# cell {i}\n"]} for i in range(38)]}
    frozen["cells"][37]["source"] = [
        "bank = object()\n",
        module.ENGINE_INIT,
        "V = v1engine.V1FE(E)\n",
    ]
    frozen_bytes = json.dumps(frozen).encode()
    frozen_hash = hashlib.sha256(frozen_bytes).hexdigest()
    source = (
        f"FROZEN_SHA256 = '{frozen_hash}'\n"
        f"FROZEN_B64 = '{base64.b64encode(frozen_bytes).decode()}'\n"
    )
    control = {"cells": [{"source": ["# unrelated\n"]} for _ in range(9)]}
    control["cells"][8]["source"] = [source]
    metadata = {
        "id": "author/control",
        "code_file": "control.ipynb",
        "dataset_sources": ["owner/reference"],
        "enable_gpu": True,
        "enable_internet": False,
    }
    return json.dumps(control).encode(), metadata, frozen_hash


def read_embedded(notebook: dict) -> dict:
    # The added title moves the pinned source cell from 8 to 9.
    source = "".join(notebook["cells"][9]["source"])
    encoded = source.split("FROZEN_B64 = '", 1)[1].split("'", 1)[0]
    return json.loads(base64.b64decode(encoded))


def test_each_changes_only_parent_generator_query_and_keeps_environment():
    control, metadata, frozen_hash = fake_control()
    notebook, built_metadata, receipt = module.build_variant(
        control,
        metadata,
        analog_query="each",
        kernel="owner/ce-analog-each",
        expected_control_sha256=module.sha256(control),
        expected_frozen_sha256=frozen_hash,
    )
    frozen = read_embedded(notebook)
    original = read_embedded({"cells": [None] + json.loads(control)["cells"]})
    assert receipt["changed_frozen_cells"] == [37]
    assert frozen["cells"][:37] == original["cells"][:37]
    assert "E.cfg.analog_query = 'each'" in "".join(frozen["cells"][37]["source"])
    assert built_metadata["dataset_sources"] == metadata["dataset_sources"]
    assert built_metadata["enable_gpu"] is True
    assert built_metadata["enable_internet"] is False
    assert built_metadata["is_private"] is True
    assert metadata["id"] == "author/control"


def test_merged_is_an_exact_frozen_source_control():
    control, metadata, frozen_hash = fake_control()
    notebook, _, receipt = module.build_variant(
        control,
        metadata,
        analog_query="merged",
        kernel="owner/ce-analog-merged",
        expected_control_sha256=module.sha256(control),
        expected_frozen_sha256=frozen_hash,
    )
    assert receipt["changed_frozen_cells"] == []
    assert receipt["variant_frozen_sha256"] == frozen_hash
    reference = {"cells": [None] + json.loads(control)["cells"]}
    assert read_embedded(notebook) == read_embedded(reference)


def test_wrong_parent_pin_or_changed_engine_fails_before_writing():
    control, metadata, frozen_hash = fake_control()
    with pytest.raises(ValueError, match="control Notebook SHA-256"):
        module.build_variant(
            control,
            metadata,
            analog_query="each",
            kernel="owner/ce-analog-each",
        )
    wrong = json.loads(control)
    encoded = "".join(wrong["cells"][8]["source"])
    original_bytes = base64.b64decode(
        encoded.split("FROZEN_B64 = '", 1)[1].split("'", 1)[0]
    )
    raw = json.loads(original_bytes)
    raw["cells"][37]["source"] = ["E = object()\n"]
    raw_bytes = json.dumps(raw).encode()
    new_hash = module.sha256(raw_bytes)
    encoded = encoded.replace(frozen_hash, new_hash)
    encoded = encoded.replace(
        base64.b64encode(original_bytes).decode(),
        base64.b64encode(raw_bytes).decode(),
    )
    wrong["cells"][8]["source"] = [encoded]
    wrong_bytes = json.dumps(wrong).encode()
    with pytest.raises(ValueError, match="main engine initialization"):
        module.build_variant(
            wrong_bytes,
            metadata,
            analog_query="each",
            kernel="owner/ce-analog-each",
            expected_control_sha256=module.sha256(wrong_bytes),
            expected_frozen_sha256=new_hash,
        )


def test_actual_0477_control_is_pinned_and_patchable():
    control = (
        Path(__file__).resolve().parents[1]
        / "data/derived/kaggle-public-0477-control"
        / "lb-0-477-two-0-472-fixes-merged-the-right-way.ipynb"
    )
    metadata = control.with_name("kernel-metadata.json")
    if not control.exists() or not metadata.exists():
        pytest.skip("Local 0.477 notebook asset is unavailable")
    notebook, built_metadata, receipt = module.build_variant(
        control.read_bytes(),
        json.loads(metadata.read_text()),
        analog_query="each",
        kernel="owner/ce-analog-each",
    )
    assert receipt["changed_frozen_cells"] == [37]
    assert built_metadata["code_file"] == "ce-analog-each.ipynb"
    assert "E.cfg.analog_query = 'each'" in "".join(
        read_embedded(notebook)["cells"][37]["source"]
    )
