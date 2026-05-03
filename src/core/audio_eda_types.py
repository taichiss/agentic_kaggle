from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaxonomyEntry:
    primary_label: str
    common_name: str
    scientific_name: str
    class_name: str


@dataclass(frozen=True)
class SoundscapeWindow:
    start_sec: float
    end_sec: float
    labels: tuple[str, ...]


@dataclass(frozen=True)
class RecordingEntry:
    recording_id: str
    dataset: str
    relative_path: str
    filename: str
    title: str
    primary_label: str | None
    common_name: str | None
    scientific_name: str | None
    class_name: str | None
    collection: str | None
    author: str | None
    license_name: str | None
    rating: float | None
    latitude: float | None
    longitude: float | None
    secondary_labels: tuple[str, ...]
    tags: tuple[str, ...]
    site: str | None
    soundscape_windows: tuple[SoundscapeWindow, ...]
    search_text: str


@dataclass(frozen=True)
class AudioInfo:
    sample_rate: int
    channels: int
    frames: int
    duration_sec: float


@dataclass(frozen=True)
class AnnotationEntry:
    annotation_id: str
    created_at: str
    recording_id: str
    dataset: str
    filename: str
    relative_path: str
    start_sec: float
    duration_sec: float
    category: str
    labels: tuple[str, ...]
    note: str
