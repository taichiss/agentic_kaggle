from __future__ import annotations

import re
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

from agentic_kaggle.paths import find_repo_root

ADR_FILENAME = re.compile(r"^(?P<number>\d{4})-[a-z0-9]+(?:-[a-z0-9]+)*\.md$")
ADR_STATUSES = {"Proposed", "Accepted", "Rejected", "Deprecated", "Superseded"}
ADR_SECTIONS = ("## Context", "## Decision", "## Consequences", "## Validation")
SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

REQUIRED_ROOT_FILES = (
    "AGENTS.md",
    "README.md",
    "pyproject.toml",
    ".agents/development-workflow.md",
    ".agents/harness/README.md",
    ".agents/harness/commands.md",
    ".agents/state/session.example.json",
    "docs/adr/README.md",
    "templates/competition/competition.toml",
    ".github/workflows/ci.yml",
)
REQUIRED_COMPETITION_PATHS = (
    "README.md",
    "competition.toml",
    "strategy/current.md",
    "strategy/experiments.md",
    "strategy/todo.md",
    "docs/overview",
    "docs/discussion",
    "docs/kernel",
    "configs",
    "src",
    "tests",
    "evals",
    "notebooks",
)
REQUIRED_IGNORE_RULES = (
    "kaggle.json",
    ".env",
    "competitions/*/data/**",
    "competitions/*/artifacts/**",
    "competitions/*/submissions/**",
)


@dataclass(frozen=True)
class CheckFailure:
    path: str
    message: str

    def render(self) -> str:
        return f"{self.path}: {self.message}"


def check_required_files(root: Path) -> list[CheckFailure]:
    return [
        CheckFailure(relative_path, "required platform file is missing")
        for relative_path in REQUIRED_ROOT_FILES
        if not (root / relative_path).exists()
    ]


def check_adrs(root: Path) -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    adr_directory = root / "docs" / "adr"
    index_path = adr_directory / "README.md"
    index = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
    seen_numbers: set[str] = set()

    for path in sorted(adr_directory.glob("*.md")):
        if path.name == "README.md":
            continue
        relative = str(path.relative_to(root))
        match = ADR_FILENAME.fullmatch(path.name)
        if match is None:
            failures.append(CheckFailure(relative, "ADR filename must be NNNN-kebab-case-title.md"))
            continue
        number = match.group("number")
        if number in seen_numbers:
            failures.append(CheckFailure(relative, f"ADR number {number} is duplicated"))
        seen_numbers.add(number)

        text = path.read_text(encoding="utf-8")
        if not text.startswith(f"# ADR {number}:"):
            failures.append(CheckFailure(relative, f"title must start with '# ADR {number}:'"))
        metadata = _metadata(text)
        status = metadata.get("Status")
        if status not in ADR_STATUSES:
            failures.append(
                CheckFailure(relative, f"Status must be one of {sorted(ADR_STATUSES)}")
            )
        for field in ("Date", "Decision owners", "Supersedes", "Superseded by"):
            if not metadata.get(field):
                failures.append(CheckFailure(relative, f"metadata '{field}' is required"))
        for section in ADR_SECTIONS:
            if section not in text:
                failures.append(CheckFailure(relative, f"required section is missing: {section}"))
        if path.name not in index:
            failures.append(CheckFailure(relative, "ADR is not linked from docs/adr/README.md"))
        if status == "Accepted" and "{{" in text:
            failures.append(
                CheckFailure(relative, "Accepted ADR contains an unresolved placeholder")
            )

    if not seen_numbers:
        failures.append(CheckFailure("docs/adr", "at least one ADR is required"))
    return failures


def _metadata(text: str) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"- ([A-Za-z ]+):\s*(.+)", line)
        if match:
            metadata[match.group(1)] = match.group(2).strip()
    return metadata


def check_competitions(root: Path) -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    competitions = root / "competitions"
    if not competitions.is_dir():
        return [CheckFailure("competitions", "competition workspace directory is missing")]

    for workspace in sorted(path for path in competitions.iterdir() if path.is_dir()):
        relative_workspace = str(workspace.relative_to(root))
        slug = workspace.name
        if not SLUG_PATTERN.fullmatch(slug):
            failures.append(
                CheckFailure(relative_workspace, "directory name must be a Kaggle-style slug")
            )
        for required_path in REQUIRED_COMPETITION_PATHS:
            if not (workspace / required_path).exists():
                failures.append(
                    CheckFailure(
                        f"{relative_workspace}/{required_path}",
                        "required competition path is missing",
                    )
                )

        manifest_path = workspace / "competition.toml"
        if not manifest_path.is_file():
            continue
        try:
            with manifest_path.open("rb") as file:
                manifest = tomllib.load(file)
        except tomllib.TOMLDecodeError as exc:
            failures.append(
                CheckFailure(str(manifest_path.relative_to(root)), f"invalid TOML: {exc}")
            )
            continue
        failures.extend(_check_manifest(root, manifest_path, manifest, slug))
    return failures


def _check_manifest(
    root: Path, manifest_path: Path, manifest: dict[str, object], slug: str
) -> list[CheckFailure]:
    failures: list[CheckFailure] = []
    relative = str(manifest_path.relative_to(root))
    competition = manifest.get("competition")
    if not isinstance(competition, dict):
        return [CheckFailure(relative, "[competition] table is required")]
    required_values = ("slug", "title", "competition_url", "metric", "metric_direction")
    for key in required_values:
        if not isinstance(competition.get(key), str) or not competition[key].strip():
            failures.append(CheckFailure(relative, f"competition.{key} must be a non-empty string"))
    if competition.get("slug") != slug:
        failures.append(CheckFailure(relative, "competition.slug must match its directory name"))
    if competition.get("metric_direction") not in {"maximize", "minimize"}:
        failures.append(
            CheckFailure(relative, "competition.metric_direction must be maximize or minimize")
        )

    submission = manifest.get("submission")
    if not isinstance(submission, dict):
        failures.append(CheckFailure(relative, "[submission] table is required"))
    else:
        for key in ("id_columns", "prediction_columns"):
            value = submission.get(key)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                failures.append(CheckFailure(relative, f"submission.{key} must be a string array"))
    return failures


def check_gitignore(root: Path) -> list[CheckFailure]:
    path = root / ".gitignore"
    if not path.exists():
        return [CheckFailure(".gitignore", "file is missing")]
    rules = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return [
        CheckFailure(".gitignore", f"required ignore rule is missing: {rule}")
        for rule in REQUIRED_IGNORE_RULES
        if rule not in rules
    ]


def run_checks(root: Path) -> list[CheckFailure]:
    return [
        *check_required_files(root),
        *check_adrs(root),
        *check_competitions(root),
        *check_gitignore(root),
    ]


def main() -> int:
    root = find_repo_root()
    failures = run_checks(root)
    if failures:
        print("Harness checks failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure.render()}", file=sys.stderr)
        return 1
    print("Harness checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
