from __future__ import annotations

import json
import threading
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from core.audio_eda_types import AnnotationEntry


class AnnotationStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()

    def list_annotations(self, recording_id: str | None = None) -> list[dict[str, object]]:
        records = self._read_entries()
        if recording_id is not None:
            records = [entry for entry in records if entry.recording_id == recording_id]
        return [self._serialize(entry) for entry in sorted(records, key=_sort_key, reverse=True)]

    def add_annotation(
        self,
        *,
        recording_id: str,
        dataset: str,
        filename: str,
        relative_path: str,
        start_sec: float,
        duration_sec: float,
        category: str,
        labels: list[str],
        note: str,
    ) -> dict[str, object]:
        cleaned_note = note.strip()
        if cleaned_note == "":
            raise ValueError("annotation note is required")
        entry = AnnotationEntry(
            annotation_id=uuid.uuid4().hex,
            created_at=datetime.now(tz=UTC).isoformat(timespec="seconds"),
            recording_id=recording_id,
            dataset=dataset,
            filename=filename,
            relative_path=relative_path,
            start_sec=max(float(start_sec), 0.0),
            duration_sec=max(float(duration_sec), 0.0),
            category=category.strip() or "other",
            labels=tuple(label for label in labels if label),
            note=cleaned_note,
        )
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(asdict(entry), ensure_ascii=False))
                handle.write("\n")
        return self._serialize(entry)

    def _read_entries(self) -> list[AnnotationEntry]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        entries: list[AnnotationEntry] = []
        for line in lines:
            if line.strip() == "":
                continue
            payload = json.loads(line)
            entries.append(
                AnnotationEntry(
                    annotation_id=str(payload["annotation_id"]),
                    created_at=str(payload["created_at"]),
                    recording_id=str(payload["recording_id"]),
                    dataset=str(payload["dataset"]),
                    filename=str(payload["filename"]),
                    relative_path=str(payload["relative_path"]),
                    start_sec=float(payload["start_sec"]),
                    duration_sec=float(payload["duration_sec"]),
                    category=str(payload["category"]),
                    labels=tuple(str(label) for label in payload.get("labels", [])),
                    note=str(payload["note"]),
                )
            )
        return entries

    def _serialize(self, entry: AnnotationEntry) -> dict[str, object]:
        return {
            **asdict(entry),
            "labels": list(entry.labels),
        }


def _sort_key(entry: AnnotationEntry) -> tuple[str, str]:
    return (entry.created_at, entry.annotation_id)
