from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Find the repository root without depending on the current shell layout."""
    starts = [Path.cwd() if start is None else start, Path(__file__).resolve()]
    for candidate in starts:
        candidate = candidate.resolve()
        if candidate.is_file():
            candidate = candidate.parent
        for directory in (candidate, *candidate.parents):
            if (directory / "pyproject.toml").exists() and (
                directory / "templates" / "competition"
            ).is_dir():
                return directory
    raise RuntimeError("agentic_kaggle repository root was not found")
