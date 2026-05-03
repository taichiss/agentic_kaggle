from __future__ import annotations

import ast
import hashlib
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from core.audio_eda_types import RecordingEntry, SoundscapeWindow, TaxonomyEntry

SITE_PATTERN = re.compile(r"_(S\d{2})_")


@dataclass
class _SpeciesAccumulator:
    primary_label: str
    common_name: str
    scientific_name: str
    class_name: str
    train_audio_recordings: int = 0
    soundscape_window_count: int = 0
    ratings: list[float] = field(default_factory=list)
    soundscape_files: set[str] = field(default_factory=set)


class AudioCatalog:
    def __init__(
        self,
        entries: list[RecordingEntry],
        taxonomy: dict[str, TaxonomyEntry],
        competition_root: Path,
    ) -> None:
        self.entries = sorted(entries, key=_sort_key)
        self.taxonomy = taxonomy
        self.competition_root = competition_root
        self.entry_by_id = {entry.recording_id: entry for entry in self.entries}
        self.species_stats = self._build_species_stats()
        self.summary = self._build_summary()

    @classmethod
    def from_competition_root(cls, competition_root: Path) -> AudioCatalog:
        taxonomy = _load_taxonomy(competition_root / "taxonomy.csv")
        entries = [
            *_load_train_audio_entries(competition_root, taxonomy),
            *_load_soundscape_entries(competition_root, taxonomy, dataset="train_soundscapes"),
            *_load_soundscape_entries(competition_root, taxonomy, dataset="test_soundscapes"),
        ]
        return cls(entries=entries, taxonomy=taxonomy, competition_root=competition_root)

    def search(
        self,
        query: str = "",
        dataset: str = "all",
        limit: int = 50,
        offset: int = 0,
    ) -> dict[str, object]:
        query_lower = query.strip().lower()
        matched: list[RecordingEntry] = []
        for entry in self.entries:
            if dataset != "all" and entry.dataset != dataset:
                continue
            if query_lower and query_lower not in entry.search_text:
                continue
            matched.append(entry)

        total = len(matched)
        page = matched[offset : offset + limit]
        return {
            "items": [self.serialize_summary(entry) for entry in page],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def get_entry(self, recording_id: str) -> RecordingEntry:
        try:
            return self.entry_by_id[recording_id]
        except KeyError as exc:
            raise KeyError(f"unknown recording id: {recording_id}") from exc

    def resolve_path(self, recording_id: str) -> Path:
        entry = self.get_entry(recording_id)
        return self.competition_root / entry.relative_path

    def serialize_summary(self, entry: RecordingEntry) -> dict[str, object]:
        label_count = len({label for window in entry.soundscape_windows for label in window.labels})
        return {
            "recording_id": entry.recording_id,
            "dataset": entry.dataset,
            "filename": entry.filename,
            "title": entry.title,
            "primary_label": entry.primary_label,
            "common_name": entry.common_name,
            "scientific_name": entry.scientific_name,
            "class_name": entry.class_name,
            "collection": entry.collection,
            "author": entry.author,
            "rating": entry.rating,
            "site": entry.site,
            "window_count": len(entry.soundscape_windows),
            "label_count": label_count,
        }

    def serialize_detail(self, entry: RecordingEntry) -> dict[str, object]:
        default_duration = 15.0
        default_start = 0.0
        if entry.soundscape_windows:
            first_window = entry.soundscape_windows[0]
            default_start = first_window.start_sec
            default_duration = max(first_window.end_sec - first_window.start_sec, 1.0)

        return {
            **self.serialize_summary(entry),
            "relative_path": entry.relative_path,
            "license_name": entry.license_name,
            "latitude": entry.latitude,
            "longitude": entry.longitude,
            "secondary_labels": list(entry.secondary_labels),
            "tags": list(entry.tags),
            "default_start_sec": default_start,
            "default_duration_sec": default_duration,
            "soundscape_windows": [
                self._serialize_window(window) for window in entry.soundscape_windows
            ],
        }

    def serialize_species_batch(self, labels: list[str]) -> list[dict[str, object]]:
        unique_labels = list(dict.fromkeys(label for label in labels if label))
        return [self.species_stats[label] for label in unique_labels if label in self.species_stats]

    def _serialize_window(self, window: SoundscapeWindow) -> dict[str, object]:
        labels = list(window.labels)
        return {
            **asdict(window),
            "labels": labels,
            "label_display": [self.label_display_name(label) for label in labels],
        }

    def label_display_name(self, primary_label: str) -> str:
        taxonomy_entry = self.taxonomy.get(primary_label)
        if taxonomy_entry is None:
            return primary_label
        return f"{primary_label} · {taxonomy_entry.common_name}"

    def _build_summary(self) -> dict[str, object]:
        dataset_counts = Counter(entry.dataset for entry in self.entries)
        soundscape_windows = sum(len(entry.soundscape_windows) for entry in self.entries)
        soundscape_labels = {
            label
            for entry in self.entries
            for window in entry.soundscape_windows
            for label in window.labels
        }
        labeled_soundscape_files = sum(
            1
            for entry in self.entries
            if entry.dataset == "train_soundscapes" and entry.soundscape_windows
        )
        return {
            "datasets": dict(dataset_counts),
            "recordings": len(self.entries),
            "train_audio_classes": len(
                {entry.primary_label for entry in self.entries if entry.dataset == "train_audio"}
            ),
            "soundscape_windows": soundscape_windows,
            "soundscape_active_labels": len(soundscape_labels),
            "labeled_soundscape_files": labeled_soundscape_files,
        }

    def _build_species_stats(self) -> dict[str, dict[str, object]]:
        species: dict[str, _SpeciesAccumulator] = {}
        for label, taxonomy_entry in self.taxonomy.items():
            species[label] = _SpeciesAccumulator(
                primary_label=label,
                common_name=taxonomy_entry.common_name,
                scientific_name=taxonomy_entry.scientific_name,
                class_name=taxonomy_entry.class_name,
            )

        for entry in self.entries:
            if entry.dataset == "train_audio" and entry.primary_label is not None:
                stat = species.setdefault(
                    entry.primary_label,
                    _empty_species_stat(
                        entry.primary_label,
                        self.label_display_name(entry.primary_label),
                    ),
                )
                stat.train_audio_recordings += 1
                if entry.rating is not None:
                    stat.ratings.append(entry.rating)
                continue

            if entry.dataset != "train_soundscapes":
                continue
            seen_in_file: set[str] = set()
            for window in entry.soundscape_windows:
                for label in window.labels:
                    stat = species.setdefault(
                        label,
                        _empty_species_stat(label, self.label_display_name(label)),
                    )
                    stat.soundscape_window_count += 1
                    seen_in_file.add(label)
            for label in seen_in_file:
                species[label].soundscape_files.add(entry.filename)

        result: dict[str, dict[str, object]] = {}
        for label, stat in species.items():
            result[label] = {
                "primary_label": label,
                "common_name": stat.common_name,
                "scientific_name": stat.scientific_name,
                "class_name": stat.class_name,
                "train_audio_recordings": stat.train_audio_recordings,
                "soundscape_window_count": stat.soundscape_window_count,
                "soundscape_file_count": len(stat.soundscape_files),
                "rating_summary": _rating_summary(stat.ratings),
                "rating_histogram": _rating_histogram(stat.ratings),
            }
        return result


def _load_taxonomy(path: Path) -> dict[str, TaxonomyEntry]:
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    return {
        row["primary_label"]: TaxonomyEntry(
            primary_label=row["primary_label"],
            common_name=row["common_name"],
            scientific_name=row["scientific_name"],
            class_name=row["class_name"],
        )
        for row in table.to_dict(orient="records")
    }


def _load_train_audio_entries(
    competition_root: Path,
    taxonomy: dict[str, TaxonomyEntry],
) -> list[RecordingEntry]:
    table = pd.read_csv(competition_root / "train.csv", dtype=str, keep_default_na=False)
    entries: list[RecordingEntry] = []
    for row in table.to_dict(orient="records"):
        relative_path = f"train_audio/{row['filename']}"
        path = competition_root / relative_path
        if not path.exists():
            continue
        primary_label = row["primary_label"]
        taxonomy_entry = taxonomy.get(primary_label)
        secondary_labels = tuple(_parse_list_text(row["secondary_labels"]))
        tags = tuple(_parse_list_text(row["type"]))
        common_name = _coalesce(
            row["common_name"], taxonomy_entry.common_name if taxonomy_entry else None
        )
        scientific_name = _coalesce(
            row["scientific_name"],
            taxonomy_entry.scientific_name if taxonomy_entry else None,
        )
        class_name = _coalesce(
            row["class_name"], taxonomy_entry.class_name if taxonomy_entry else None
        )
        title = f"{common_name or primary_label} | {Path(row['filename']).name}"
        search_text = _search_blob(
            "train_audio",
            relative_path,
            row["filename"],
            primary_label,
            common_name,
            scientific_name,
            class_name,
            row["author"],
            row["collection"],
            *secondary_labels,
            *tags,
        )
        entries.append(
            RecordingEntry(
                recording_id=_recording_id("train_audio", row["filename"]),
                dataset="train_audio",
                relative_path=relative_path,
                filename=Path(row["filename"]).name,
                title=title,
                primary_label=primary_label,
                common_name=common_name,
                scientific_name=scientific_name,
                class_name=class_name,
                collection=_empty_to_none(row["collection"]),
                author=_empty_to_none(row["author"]),
                license_name=_empty_to_none(row["license"]),
                rating=_parse_optional_float(row["rating"]),
                latitude=_parse_optional_float(row["latitude"]),
                longitude=_parse_optional_float(row["longitude"]),
                secondary_labels=secondary_labels,
                tags=tags,
                site=None,
                soundscape_windows=(),
                search_text=search_text,
            )
        )
    return entries


def _load_soundscape_entries(
    competition_root: Path,
    taxonomy: dict[str, TaxonomyEntry],
    dataset: str,
) -> list[RecordingEntry]:
    directory = competition_root / dataset
    label_windows: dict[str, tuple[SoundscapeWindow, ...]] = {}
    if dataset == "train_soundscapes":
        labels = pd.read_csv(
            competition_root / "train_soundscapes_labels.csv",
            dtype=str,
            keep_default_na=False,
        ).drop_duplicates()
        grouped = labels.groupby("filename", sort=False)
        label_windows = {
            filename: tuple(
                sorted(
                    (
                        SoundscapeWindow(
                            start_sec=_clock_to_seconds(row["start"]),
                            end_sec=_clock_to_seconds(row["end"]),
                            labels=tuple(filter(None, row["primary_label"].split(";"))),
                        )
                        for row in group.to_dict(orient="records")
                    ),
                    key=lambda window: (window.start_sec, window.end_sec, window.labels),
                )
            )
            for filename, group in grouped
        }

    entries: list[RecordingEntry] = []
    for path in sorted(directory.glob("*.ogg")):
        relative_path = f"{dataset}/{path.name}"
        windows = label_windows.get(path.name, ())
        label_set = sorted({label for window in windows for label in window.labels})
        site = _extract_site(path.name)
        title = f"{dataset} | {site or 'site-unknown'} | {path.name}"
        search_text = _search_blob(
            dataset,
            relative_path,
            path.name,
            site,
            *(taxonomy[label].common_name for label in label_set if label in taxonomy),
            *label_set,
        )
        entries.append(
            RecordingEntry(
                recording_id=_recording_id(dataset, path.name),
                dataset=dataset,
                relative_path=relative_path,
                filename=path.name,
                title=title,
                primary_label=None,
                common_name=None,
                scientific_name=None,
                class_name=None,
                collection=None,
                author=None,
                license_name=None,
                rating=None,
                latitude=None,
                longitude=None,
                secondary_labels=tuple(label_set),
                tags=(),
                site=site,
                soundscape_windows=windows,
                search_text=search_text,
            )
        )
    return entries


def _sort_key(entry: RecordingEntry) -> tuple[str, str, str]:
    return (entry.dataset, entry.common_name or entry.title, entry.filename)


def _recording_id(dataset: str, relative_name: str) -> str:
    key = f"{dataset}::{relative_name}".encode()
    return hashlib.sha1(key).hexdigest()


def _search_blob(*values: str | None) -> str:
    return " ".join(value.lower() for value in values if value)


def _parse_list_text(value: str) -> list[str]:
    if value == "":
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return [value]
    if not isinstance(parsed, list):
        return [str(parsed)]
    return [str(item) for item in parsed]


def _parse_optional_float(value: str) -> float | None:
    if value == "":
        return None
    return float(value)


def _empty_to_none(value: str) -> str | None:
    return value or None


def _coalesce(first: str | None, second: str | None) -> str | None:
    if first:
        return first
    return second


def _extract_site(filename: str) -> str | None:
    match = SITE_PATTERN.search(filename)
    if match is None:
        return None
    return match.group(1)


def _clock_to_seconds(clock_text: str) -> float:
    hours_text, minutes_text, seconds_text = clock_text.split(":")
    return int(hours_text) * 3600 + int(minutes_text) * 60 + int(seconds_text)


def _rating_summary(ratings: list[float]) -> dict[str, float | int | None]:
    if not ratings:
        return {"count": 0, "mean": None, "median": None, "min": None, "max": None}
    ordered = sorted(ratings)
    count = len(ordered)
    midpoint = count // 2
    if count % 2 == 1:
        median = ordered[midpoint]
    else:
        median = (ordered[midpoint - 1] + ordered[midpoint]) / 2.0
    return {
        "count": count,
        "mean": sum(ordered) / count,
        "median": median,
        "min": ordered[0],
        "max": ordered[-1],
    }


def _rating_histogram(ratings: list[float]) -> list[dict[str, object]]:
    ranges: list[tuple[str, float, float]] = [
        ("0-1", 0.0, 1.0),
        ("1-2", 1.0, 2.0),
        ("2-3", 2.0, 3.0),
        ("3-4", 3.0, 4.0),
        ("4-5", 4.0, 5.0),
        ("5.0", 5.0, 5.1),
    ]
    counts = {label: 0 for label, _, _ in ranges}
    for rating in ratings:
        for label, lo, hi in ranges:
            if lo <= rating < hi:
                counts[label] += 1
                break
    return [{"label": label, "count": counts[label]} for label, _, _ in ranges]


def _empty_species_stat(label: str, display_name: str) -> _SpeciesAccumulator:
    common_name = display_name.split("·", 1)[-1].strip() if "·" in display_name else display_name
    return _SpeciesAccumulator(
        primary_label=label,
        common_name=common_name,
        scientific_name="",
        class_name="",
    )
