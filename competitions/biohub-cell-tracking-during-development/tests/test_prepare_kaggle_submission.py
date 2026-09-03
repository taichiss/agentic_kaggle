"""Focused tests for the generic host Kaggle package."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts/prepare_kaggle_submission.py"
SPEC = importlib.util.spec_from_file_location("prepare_kaggle_submission_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
packaging = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = packaging
SPEC.loader.exec_module(packaging)


def test_zip_manifest_binds_expanded_members(tmp_path: Path) -> None:
    archive_path = tmp_path / "models.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("package/__init__.py", b"VALUE = 1\n")
        archive.writestr("package/model.py", b"class Model: pass\n")

    entry = packaging._manifest_entry(archive_path)

    assert entry["members"] == {
        "package/__init__.py": {
            "bytes": 10,
            "sha256": hashlib.sha256(b"VALUE = 1\n").hexdigest(),
        },
        "package/model.py": {
            "bytes": 18,
            "sha256": hashlib.sha256(b"class Model: pass\n").hexdigest(),
        },
    }


def test_postprocess_argument_belongs_to_inference_command() -> None:
    notebook = packaging._notebook(
        "owner/dataset", "title", postprocess_profile="public-applicable-v1"
    )
    source = "".join(notebook["cells"][1]["source"])
    before_command, command = source.split("command = [", 1)

    compile(source, "generated.ipynb", "exec")
    assert "--postprocess-profile" not in before_command
    assert "--postprocess-profile" in command
