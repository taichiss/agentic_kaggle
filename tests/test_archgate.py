from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.archgate import validate_repo


def write_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class ArchGateTest(unittest.TestCase):
    def test_missing_src_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            self.assertEqual(validate_repo(tmp_root), [])

    def test_allowed_dependency_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            write_file(tmp_root / "src" / "core" / "config.py", "VALUE = 1\n")
            write_file(tmp_root / "src" / "data" / "loader.py", "from core.config import VALUE\n")
            self.assertEqual(validate_repo(tmp_root), [])

    def test_forbidden_dependency_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            write_file(tmp_root / "src" / "training" / "pipeline.py", "def train() -> None:\n    pass\n")
            write_file(
                tmp_root / "src" / "core" / "logic.py",
                "from training.pipeline import train\n",
            )
            violations = validate_repo(tmp_root)
            self.assertEqual(len(violations), 1)
            self.assertIn("core layer cannot import training", violations[0].message)
