from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def code_targets(root: Path) -> list[str]:
    candidates = ["scripts", "tests", "src"]
    return [name for name in candidates if (root / name).exists()]


def run(command: list[str], cwd: Path) -> int:
    printable = " ".join(command)
    print(f"+ {printable}")
    completed = subprocess.run(command, cwd=cwd, check=False)
    return completed.returncode


def format_command(root: Path) -> list[str] | None:
    targets = code_targets(root)
    if not targets:
        return None
    return [sys.executable, "-m", "ruff", "format", *targets]


def lint_command(root: Path) -> list[str] | None:
    targets = code_targets(root)
    if not targets:
        return None
    return [sys.executable, "-m", "ruff", "check", *targets]


def typecheck_command(root: Path) -> list[str] | None:
    targets = code_targets(root)
    if not targets:
        return None
    return [sys.executable, "-m", "mypy", *targets]


def arch_command(root: Path) -> list[str]:
    return [sys.executable, "scripts/archgate.py"]


def test_command(root: Path) -> list[str] | None:
    tests_dir = root / "tests"
    if not tests_dir.exists():
        return None
    return [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]


def commands_for(action: str, root: Path) -> list[list[str] | None]:
    if action == "format":
        return [format_command(root)]
    if action == "lint":
        return [lint_command(root)]
    if action == "typecheck":
        return [typecheck_command(root)]
    if action == "arch":
        return [arch_command(root)]
    if action == "test":
        return [test_command(root)]
    if action == "all":
        return [
            lint_command(root),
            typecheck_command(root),
            arch_command(root),
            test_command(root),
        ]
    raise ValueError(f"unknown action: {action}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action",
        choices=["all", "arch", "format", "lint", "test", "typecheck"],
    )
    args = parser.parse_args()

    root = repo_root()
    for command in commands_for(args.action, root):
        if command is None:
            continue
        exit_code = run(command, cwd=root)
        if exit_code != 0:
            return exit_code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
