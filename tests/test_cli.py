from __future__ import annotations

import shutil
import tomllib
from pathlib import Path

import pytest

from agentic_kaggle.cli import initialize_competition


def test_initialize_competition_expands_template(tmp_path: Path) -> None:
    source_template = Path(__file__).parents[1] / "templates" / "competition"
    shutil.copytree(source_template, tmp_path / "templates" / "competition")
    (tmp_path / "competitions").mkdir()

    destination = initialize_competition(
        tmp_path,
        "sample-competition",
        title="Sample Competition",
        metric="AUC",
        metric_direction="maximize",
        competition_url="https://www.kaggle.com/competitions/sample-competition",
    )

    with (destination / "competition.toml").open("rb") as file:
        manifest = tomllib.load(file)
    assert manifest["competition"]["slug"] == "sample-competition"
    assert manifest["competition"]["metric"] == "AUC"
    assert (destination / "strategy" / "current.md").is_file()
    assert (destination / "data" / "input").is_dir()
    assert "{{" not in (destination / "README.md").read_text(encoding="utf-8")


def test_initialize_competition_does_not_overwrite(tmp_path: Path) -> None:
    (tmp_path / "templates" / "competition").mkdir(parents=True)
    (tmp_path / "competitions" / "taken").mkdir(parents=True)

    with pytest.raises(FileExistsError):
        initialize_competition(
            tmp_path,
            "taken",
            title="Taken",
            metric="RMSE",
            metric_direction="minimize",
            competition_url="https://example.test/taken",
        )
