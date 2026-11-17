"""Both CE generator arms preserve the frozen parent outside cell 37."""

from __future__ import annotations

import base64
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/public_notebooks/build_ce_mild_mixup_notebook.py"
spec = importlib.util.spec_from_file_location("build_ce_mild_mixup_notebook", MODULE_PATH)
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
    source = (
        f"FROZEN_SHA256 = '{frozen_sha}'\n"
        f"FROZEN_B64 = '{base64.b64encode(frozen_bytes).decode()}'\n"
    )
    control = {"cells": [{"source": ["# unrelated\n"]} for _ in range(9)]}
    control["cells"][8]["source"] = [source]
    metadata = {"id": "owner/control", "enable_gpu": True, "enable_internet": False}
    adapter = b"def install(engine, *, include_mixup):\n    pass\n"
    return json.dumps(control).encode(), metadata, adapter, frozen_sha


def embedded(notebook: dict) -> dict:
    source = "".join(notebook["cells"][9]["source"])
    b64 = source.split("FROZEN_B64 = '", 1)[1].split("'", 1)[0]
    return json.loads(base64.b64decode(b64))


@pytest.mark.parametrize("mode,expected", [("separate", "False"), ("mild", "True")])
def test_modes_patch_only_cell37_and_preserve_kaggle_settings(monkeypatch, mode, expected):
    control, metadata, adapter, frozen_sha = fake_control()
    monkeypatch.setattr(module, "CONTROL_NOTEBOOK_SHA256", module.sha256(control))
    monkeypatch.setattr(module, "CONTROL_FROZEN_SHA256", frozen_sha)
    notebook, output_metadata, receipt = module.build_variant(
        control, metadata, adapter, mode=mode, kernel=f"owner/ce-{mode}"
    )
    raw = embedded({"cells": [None] + json.loads(control)["cells"]})
    changed = embedded(notebook)
    assert receipt["changed_frozen_cells"] == [37, 39]
    assert changed["cells"][:37] == raw["cells"][:37]
    patched = "".join(changed["cells"][37]["source"])
    assert f"include_mixup={expected}" in patched
    assert patched.index("V = v1engine.V1FE") < patched.index("_ce_adapter_module.install")
    assert "changed_generator_parent_queries" in "".join(changed["cells"][39]["source"])
    assert "instrument=str(r.instrument_type)" in "".join(changed["cells"][39]["source"])
    assert receipt["audit_file"] == "ce-generator-audit.json"
    assert output_metadata["enable_gpu"] and not output_metadata["enable_internet"]
    assert output_metadata["is_private"]
    assert metadata["id"] == "owner/control"


def test_wrong_parent_or_missing_v1fe_is_rejected(monkeypatch):
    control, metadata, adapter, frozen_sha = fake_control()
    with pytest.raises(ValueError, match="control Notebook SHA-256"):
        module.build_variant(control, metadata, adapter, mode="mild", kernel="owner/ce-mild")
    monkeypatch.setattr(module, "CONTROL_NOTEBOOK_SHA256", module.sha256(control))
    monkeypatch.setattr(module, "CONTROL_FROZEN_SHA256", frozen_sha)
    with pytest.raises(ValueError, match="Invalid Kaggle kernel"):
        module.build_variant(control, metadata, adapter, mode="mild", kernel="bad")


def test_actual_0477_control_and_adapter_are_patchable():
    control = (
        ROOT / "data/derived/kaggle-public-0477-control"
        / "lb-0-477-two-0-472-fixes-merged-the-right-way.ipynb"
    )
    metadata = control.with_name("kernel-metadata.json")
    adapter = ROOT / "src/public_notebooks/ce_mild_mixup.py"
    if not control.exists() or not metadata.exists():
        pytest.skip("Local 0.477 control Notebook is unavailable")
    notebook, _, receipt = module.build_variant(
        control.read_bytes(), json.loads(metadata.read_text()), adapter.read_bytes(),
        mode="mild", kernel="owner/ce-generator-mild"
    )
    assert receipt["changed_frozen_cells"] == [37, 39]
    assert receipt["adapter_source_sha256"] == module.sha256(adapter.read_bytes())
    assert "include_mixup=True" in "".join(embedded(notebook)["cells"][37]["source"])
