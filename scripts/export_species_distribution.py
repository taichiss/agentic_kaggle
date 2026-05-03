from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BirdCLEF species distribution figure")
    parser.add_argument(
        "--competition-root",
        default="data/input/BirdCLEF+ 2026",
        help="directory that contains train.csv and audio folders",
    )
    parser.add_argument(
        "--output",
        default="doc/overview/2026/species_distribution.svg",
        help="path to the output SVG figure",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=18,
        help="number of species to show in each ranking panel",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    from data.audio_catalog import AudioCatalog
    from providers.audio_visualization import render_species_distribution_svg

    args = build_argument_parser().parse_args(argv)
    competition_root = Path(args.competition_root).resolve()
    output_path = Path(args.output).resolve()

    catalog = AudioCatalog.from_competition_root(competition_root)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M %Z").strip()
    summary = catalog.summary
    subtitle = (
        f"generated {generated_at} | train_audio classes={summary['train_audio_classes']} | "
        f"soundscape active labels={summary['soundscape_active_labels']} | "
        f"labeled files={summary['labeled_soundscape_files']}"
    )
    svg_payload = render_species_distribution_svg(
        list(catalog.species_stats.values()),
        summary=summary,
        top_n=max(args.top_n, 1),
        title="BirdCLEF+ 2026 species distribution",
        subtitle=subtitle,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(svg_payload)

    print(f"Saved species distribution figure: {output_path}")
    print(
        "Summary: "
        f"train_audio_classes={summary['train_audio_classes']}, "
        f"soundscape_active_labels={summary['soundscape_active_labels']}, "
        f"soundscape_windows={summary['soundscape_windows']}, "
        f"labeled_soundscape_files={summary['labeled_soundscape_files']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
