"""The G4 Notebook preserves the pinned parent outside its two audited cells."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/public_notebooks/build_ce_diverse_notebook.py"
spec = importlib.util.spec_from_file_location("build_ce_diverse_notebook", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def fake_control() -> tuple[bytes, dict, bytes, str]:
    frozen = {"cells": [{"source": [f"# cell {i}\n"]} for i in range(40)]}
    frozen["cells"][37]["source"] = [
        "L, E = object(), object()\n", "V = v1engine.V1FE(E)\n", "print('ready')\n"
    ]
    frozen["cells"][39]["source"] = [
        "spectra.append(dict(ce_n=int(len(ce_arr)) if len(ce_arr) else 1))\n"
    ]
    frozen_bytes = json.dumps(frozen).encode()
    frozen_sha = hashlib.sha256(frozen_bytes).hexdigest()
    control = {"cells": [{"source": ["# unrelated\n"]} for _ in range(9)]}
    control["cells"][8]["source"] = [
        f"FROZEN_SHA256 = '{frozen_sha}'\n"
        f"FROZEN_B64 = '{base64.b64encode(frozen_bytes).decode()}'\n"
    ]
    metadata = {"id": "owner/control", "enable_gpu": True, "enable_internet": False}
    adapter = b"def install(engine):\n    pass\n"
    return json.dumps(control).encode(), metadata, adapter, frozen_sha


def embedded(notebook: dict) -> dict:
    source = "".join(notebook["cells"][9]["source"])
    b64 = source.split("FROZEN_B64 = '", 1)[1].split("'", 1)[0]
    return json.loads(base64.b64decode(b64))


def test_pinned_builder_changes_only_cells_37_and_39(monkeypatch):
    control, metadata, adapter, frozen_sha = fake_control()
    monkeypatch.setattr(module, "CONTROL_NOTEBOOK_SHA256", module.sha256(control))
    monkeypatch.setattr(module, "CONTROL_FROZEN_SHA256", frozen_sha)
    notebook, output_metadata, receipt = module.build_variant(
        control, metadata, adapter, kernel="owner/ce-diverse"
    )
    original_frozen = embedded({"cells": [None] + json.loads(control)["cells"]})
    patched = embedded(notebook)
    assert receipt["changed_frozen_cells"] == [37, 39]
    assert patched["cells"][:37] == original_frozen["cells"][:37]
    assert patched["cells"][38] == original_frozen["cells"][38]
    cell37 = "".join(patched["cells"][37]["source"])
    cell39 = "".join(patched["cells"][39]["source"])
    assert cell37.index("V = v1engine.V1FE") < cell37.index("_ce_adapter_module.install(E)")
    assert "instrument=str(r.instrument_type)" in cell39
    assert "'generator_parents'" in cell39
    assert "'generated_candidates'" in cell39
    assert receipt["adapter_source_sha256"] == module.sha256(adapter)
    assert output_metadata["enable_gpu"] and not output_metadata["enable_internet"]
    assert output_metadata["is_private"]
    assert metadata["id"] == "owner/control"


def test_wrong_parent_and_bad_slug_are_rejected(monkeypatch):
    control, metadata, adapter, frozen_sha = fake_control()
    with pytest.raises(ValueError, match="control Notebook SHA-256"):
        module.build_variant(control, metadata, adapter, kernel="owner/ce-diverse")
    monkeypatch.setattr(module, "CONTROL_NOTEBOOK_SHA256", module.sha256(control))
    monkeypatch.setattr(module, "CONTROL_FROZEN_SHA256", frozen_sha)
    with pytest.raises(ValueError, match="Invalid Kaggle kernel"):
        module.build_variant(control, metadata, adapter, kernel="invalid")


def test_actual_0477_control_and_adapter_are_patchable():
    control = (
        ROOT / "data/derived/kaggle-public-0477-control"
        / "lb-0-477-two-0-472-fixes-merged-the-right-way.ipynb"
    )
    metadata = control.with_name("kernel-metadata.json")
    adapter = ROOT / "src/public_notebooks/ce_diverse_generator.py"
    if not control.exists() or not metadata.exists():
        pytest.skip("Local 0.477 control Notebook is unavailable")
    notebook, _, receipt = module.build_variant(
        control.read_bytes(), json.loads(metadata.read_text()), adapter.read_bytes(),
        kernel="owner/ce-diverse",
    )
    assert receipt["changed_frozen_cells"] == [37, 39]
    assert receipt["adapter_source_sha256"] == module.sha256(adapter.read_bytes())
    assert "_ce_adapter_module.install(E)" in "".join(embedded(notebook)["cells"][37]["source"])
