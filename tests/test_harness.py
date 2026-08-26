from pathlib import Path

from agentic_kaggle.harness.checks import run_checks


def test_repository_satisfies_harness_contract() -> None:
    root = Path(__file__).parents[1]
    assert run_checks(root) == []
