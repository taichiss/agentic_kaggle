from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NOTEBOOK = REPO_ROOT / "notebook.ipynb"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output" / "kaggle_kernel" / "full_fit_probe_submit"
DEFAULT_SLUG = "birdclef-2026-full-fit-probe-submit"
DEFAULT_TITLE = "BirdCLEF 2026 Full-Fit Probe Submit"
DEFAULT_MODEL_SOURCE = "google/bird-vocalization-classifier/tensorflow2/perch_v2_cpu/1"
DEFAULT_DATASET_SOURCES = (
    "suzukitaichi/birdclef-2026-perch-probe-cv-models",
    "rishikeshjani/perch-onnx-for-birdclef-2026",
)
DEFAULT_KERNEL_SOURCES = ("ashok205/tf-wheels",)
DEFAULT_COMPETITION_SOURCES = ("birdclef-2026",)


def load_kaggle_username(config_path: Path) -> str:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    username = payload.get("username")
    if not isinstance(username, str) or not username:
        raise ValueError(f"username missing in {config_path}")
    return username


def build_metadata(
    owner: str,
    slug: str,
    title: str,
    notebook_name: str,
) -> dict[str, object]:
    return {
        "id": f"{owner}/{slug}",
        "title": title,
        "code_file": notebook_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "false",
        "enable_tpu": "false",
        "enable_internet": "false",
        "competition_sources": list(DEFAULT_COMPETITION_SOURCES),
        "dataset_sources": list(DEFAULT_DATASET_SOURCES),
        "kernel_sources": list(DEFAULT_KERNEL_SOURCES),
        "model_sources": [DEFAULT_MODEL_SOURCE],
    }


def prepare_bundle(
    notebook_path: Path,
    output_dir: Path,
    owner: str,
    slug: str,
    title: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    target_notebook = output_dir / notebook_path.name
    shutil.copy2(notebook_path, target_notebook)

    metadata = build_metadata(
        owner=owner,
        slug=slug,
        title=title,
        notebook_name=target_notebook.name,
    )
    metadata_path = output_dir / "kernel-metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kaggle-config", type=Path, default=REPO_ROOT / "kaggle.json")
    parser.add_argument("--owner", type=str, default=None)
    parser.add_argument("--slug", type=str, default=DEFAULT_SLUG)
    parser.add_argument("--title", type=str, default=DEFAULT_TITLE)
    args = parser.parse_args()

    owner = args.owner or load_kaggle_username(args.kaggle_config)
    metadata_path = prepare_bundle(
        notebook_path=args.notebook,
        output_dir=args.output_dir,
        owner=owner,
        slug=args.slug,
        title=args.title,
    )
    print(metadata_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
