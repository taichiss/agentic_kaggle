from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from agentic_kaggle.paths import find_repo_root

SLUG_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TEXT_SUFFIXES = {".md", ".toml", ".json", ".py", ".txt", ".yaml", ".yml"}


def initialize_competition(
    root: Path,
    slug: str,
    *,
    title: str,
    metric: str,
    metric_direction: str,
    competition_url: str,
) -> Path:
    if not SLUG_PATTERN.fullmatch(slug):
        raise ValueError("slug must contain lowercase letters, numbers, and single hyphens")
    if metric_direction not in {"maximize", "minimize"}:
        raise ValueError("metric_direction must be maximize or minimize")

    template = root / "templates" / "competition"
    destination = root / "competitions" / slug
    if destination.exists():
        raise FileExistsError(f"competition workspace already exists: {destination}")
    if not template.is_dir():
        raise FileNotFoundError(f"competition template was not found: {template}")

    shutil.copytree(template, destination)
    replacements = {
        "{{SLUG}}": slug,
        "{{TITLE}}": title,
        "{{METRIC}}": metric,
        "{{METRIC_DIRECTION}}": metric_direction,
        "{{COMPETITION_URL}}": competition_url,
    }
    for path in destination.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        content = path.read_text(encoding="utf-8")
        for source, value in replacements.items():
            content = content.replace(source, value)
        path.write_text(content, encoding="utf-8")

    for runtime_directory in ("data/input", "artifacts", "submissions"):
        (destination / runtime_directory).mkdir(parents=True, exist_ok=True)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kaggle-init", description="Create an isolated Kaggle competition workspace."
    )
    parser.add_argument("slug", help="Kaggle competition URL slug")
    parser.add_argument("--title", required=True, help="competition display title")
    parser.add_argument("--metric", required=True, help="official leaderboard metric")
    parser.add_argument(
        "--metric-direction", required=True, choices=("maximize", "minimize")
    )
    parser.add_argument("--competition-url", required=True, help="official competition URL")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        destination = initialize_competition(
            find_repo_root(),
            args.slug,
            title=args.title,
            metric=args.metric,
            metric_direction=args.metric_direction,
            competition_url=args.competition_url,
        )
    except (ValueError, FileExistsError, FileNotFoundError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    print(f"Created competition workspace: {destination}")
    print(f"Next: verify {destination / 'competition.toml'} against official Kaggle pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
