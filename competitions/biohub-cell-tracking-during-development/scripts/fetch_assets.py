"""Fetch Biohub competition assets into Git-ignored runtime directories."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tomllib
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
SOURCES = WORKSPACE / "asset-sources.toml"


def _run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def _load_sources() -> dict:
    with SOURCES.open("rb") as file:
        return tomllib.load(file)


def _require(command: str) -> None:
    if shutil.which(command) is None:
        raise RuntimeError(f"required command was not found: {command}")


def fetch_baseline(config: dict, *, force: bool) -> None:
    _require("git")
    source = config["organizer_baseline"]
    destination = WORKSPACE / source["destination"]
    if destination.exists():
        if force:
            _run(
                [
                    "git",
                    "-C",
                    str(destination),
                    "fetch",
                    "--depth",
                    "1",
                    "origin",
                    source["revision"],
                ]
            )
            _run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])
        else:
            print(f"skip existing baseline: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir()
    _run(["git", "-C", str(destination), "init"])
    _run(["git", "-C", str(destination), "remote", "add", "origin", source["repository"]])
    _run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", source["revision"]])
    _run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"])


def fetch_data(config: dict, *, force: bool) -> None:
    _require("kaggle")
    slug = config["competition"]["slug"]
    downloads = WORKSPACE / "data" / "downloads"
    raw = WORKSPACE / "data" / "raw"
    downloads.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    command = ["kaggle", "competitions", "download", slug, "-p", str(downloads)]
    if force:
        command.append("--force")
    _run(command)
    archives = sorted(downloads.glob("*.zip"))
    if not archives:
        print(f"no ZIP archive found in {downloads}; inspect the Kaggle CLI output")
        return
    for archive in archives:
        print(f"extract: {archive} -> {raw}")
        with zipfile.ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                destination = (raw / member.filename).resolve()
                if raw.resolve() not in destination.parents and destination != raw.resolve():
                    raise RuntimeError(f"unsafe archive path: {member.filename}")
                if destination.exists() and not force:
                    continue
                zip_file.extract(member, raw)


def _fetch_public_notebook(source: dict, destination: Path) -> None:
    request = urllib.request.Request(
        source["api_url"], headers={"User-Agent": "agentic-kaggle-asset-fetcher/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    metadata = payload.get("metadata", {})
    blob = payload.get("blob", {})
    notebook_source = blob.get("sourceNullable") or blob.get("source")
    if metadata.get("ref") != source["ref"]:
        raise RuntimeError(f"public API returned unexpected notebook ref: {metadata.get('ref')}")
    expected_version = source.get("expected_version")
    actual_version = metadata.get("currentVersionNumber")
    if expected_version is not None and actual_version != expected_version:
        raise RuntimeError(
            f"notebook {source['ref']} version changed: "
            f"expected {expected_version}, got {actual_version}"
        )
    if not isinstance(notebook_source, str) or not notebook_source.strip():
        raise RuntimeError(f"public API returned no source for {source['ref']}")
    try:
        json.loads(notebook_source)
    except json.JSONDecodeError as exc:
        message = f"public API returned invalid notebook JSON for {source['ref']}"
        raise RuntimeError(message) from exc
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "notebook.ipynb").write_text(notebook_source, encoding="utf-8")
    (destination / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"saved public notebook version {actual_version}: {destination}")


def fetch_notebooks(config: dict, *, force: bool) -> None:
    for source in config.get("public_notebooks", []):
        destination = WORKSPACE / source["destination"]
        if destination.exists() and any(destination.iterdir()) and not force:
            print(f"skip existing notebook: {destination}")
            continue
        if source.get("api_url"):
            _fetch_public_notebook(source, destination)
        else:
            _require("kaggle")
            destination.mkdir(parents=True, exist_ok=True)
            _run(
                ["kaggle", "kernels", "pull", source["ref"], "-p", str(destination), "-m"]
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("asset", choices=("baseline", "data", "notebooks", "all"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="update an existing baseline or overwrite existing downloads/extracted files",
    )
    args = parser.parse_args(argv)
    config = _load_sources()
    try:
        if args.asset in {"baseline", "all"}:
            fetch_baseline(config, force=args.force)
        if args.asset in {"data", "all"}:
            fetch_data(config, force=args.force)
        if args.asset in {"notebooks", "all"}:
            fetch_notebooks(config, force=args.force)
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        urllib.error.URLError,
        zipfile.BadZipFile,
    ) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
