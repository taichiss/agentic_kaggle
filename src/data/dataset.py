"""Dataset helpers for 20-second SED clips with 5-second Perch teacher targets."""

from __future__ import annotations

import ast
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from numpy.typing import NDArray

from core.config import AudioConfig, DistillConfig

REQUIRED_TEACHER_CACHE_KEYS = frozenset(
    {
        "class_labels",
        "chunk_end_sec",
        "chunk_index",
        "chunk_start_sec",
        "filenames",
        "logits",
        "n_chunks_per_file",
        "primary_labels_per_chunk",
        "secondary_labels_per_chunk",
    }
)


@dataclass(frozen=True)
class PerchTeacherChunk:
    filename: str
    chunk_index: int
    start_sec: float
    end_sec: float
    total_chunks: int
    logits: NDArray[np.float32]


@dataclass(frozen=True)
class PerchTeacherCache:
    class_labels: tuple[str, ...]
    chunks_by_filename: dict[str, tuple[PerchTeacherChunk, ...]]


@dataclass(frozen=True)
class TrainAudioClipRecord:
    audio_path: Path
    filename: str
    primary_label: str
    secondary_labels: tuple[str, ...]
    clip_index: int
    clip_start_sec: float
    clip_duration_sec: float
    valid_duration_sec: float
    label_target: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]
    teacher_mask: NDArray[np.float32]


@dataclass(frozen=True)
class TrainAudioPerchSample:
    filename: str
    primary_label: str
    secondary_labels: tuple[str, ...]
    clip_index: int
    clip_start_sec: float
    clip_duration_sec: float
    valid_duration_sec: float
    waveform: NDArray[np.float32]
    label_target: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]
    teacher_mask: NDArray[np.float32]


@dataclass(frozen=True)
class SoundscapeTeacherWindow:
    row_id: str
    filename: str
    end_sec: int
    logits: NDArray[np.float32]


@dataclass(frozen=True)
class SoundscapeTeacherCache:
    class_labels: tuple[str, ...]
    windows_by_row_id: dict[str, SoundscapeTeacherWindow]


@dataclass(frozen=True)
class SoundscapeClipRecord:
    audio_path: Path
    filename: str
    clip_index: int
    clip_start_sec: float
    clip_duration_sec: float
    window_end_secs: tuple[int, ...]
    window_targets: NDArray[np.float32]
    window_mask: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]
    teacher_mask: NDArray[np.float32]


@dataclass(frozen=True)
class SoundscapeSedSample:
    filename: str
    clip_index: int
    clip_start_sec: float
    clip_duration_sec: float
    window_end_secs: tuple[int, ...]
    waveform: NDArray[np.float32]
    window_targets: NDArray[np.float32]
    window_mask: NDArray[np.float32]
    teacher_logits: NDArray[np.float32]
    teacher_mask: NDArray[np.float32]


def load_perch_teacher_cache(cache_path: Path) -> PerchTeacherCache:
    """Load the full 5-second Perch cache used to supervise 20-second clips."""

    with np.load(cache_path, allow_pickle=False) as cache:
        missing = REQUIRED_TEACHER_CACHE_KEYS.difference(cache.files)
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(
                "Perch teacher cache is missing required keys: "
                f"{missing_display}. Generate the full 5-second chunk cache first."
            )

        class_labels = tuple(str(label) for label in cache["class_labels"])
        n_rows = len(cache["filenames"])
        grouped_chunks: dict[str, list[PerchTeacherChunk]] = defaultdict(list)
        for row_index in range(n_rows):
            filename = str(cache["filenames"][row_index])
            grouped_chunks[filename].append(
                PerchTeacherChunk(
                    filename=filename,
                    chunk_index=int(cache["chunk_index"][row_index]),
                    start_sec=float(cache["chunk_start_sec"][row_index]),
                    end_sec=float(cache["chunk_end_sec"][row_index]),
                    total_chunks=int(cache["n_chunks_per_file"][row_index]),
                    logits=cache["logits"][row_index].astype(np.float32, copy=True),
                )
            )

    return PerchTeacherCache(
        class_labels=class_labels,
        chunks_by_filename={
            filename: tuple(sorted(chunks, key=lambda chunk: chunk.chunk_index))
            for filename, chunks in grouped_chunks.items()
        },
    )


def build_label_target(
    primary_label: str,
    secondary_labels: tuple[str, ...],
    label_to_idx: dict[str, int],
    primary_weight: float = 1.0,
    secondary_weight: float = 0.7,
) -> NDArray[np.float32]:
    """Build the weak 234-class target shared by every 20-second clip in a file."""

    target = np.zeros(len(label_to_idx), dtype=np.float32)
    if primary_label in label_to_idx:
        target[label_to_idx[primary_label]] = primary_weight

    for label in secondary_labels:
        class_index = label_to_idx.get(label)
        if class_index is None:
            continue
        target[class_index] = max(target[class_index], secondary_weight)
    return target


def build_train_audio_perch_clip_index(
    data_dir: Path,
    teacher_cache_path: Path,
    *,
    audio_cfg: AudioConfig | None = None,
    distill_cfg: DistillConfig | None = None,
    primary_weight: float = 1.0,
    secondary_weight: float = 0.7,
) -> tuple[list[TrainAudioClipRecord], tuple[str, ...]]:
    """Create a 20-second clip index backed by 5-second Perch teacher windows."""

    audio_cfg = audio_cfg or AudioConfig()
    distill_cfg = distill_cfg or DistillConfig()
    if not math.isclose(audio_cfg.chunk_duration_sec, distill_cfg.clip_duration_sec):
        raise ValueError(
            "AudioConfig.chunk_duration_sec must match "
            "DistillConfig.teacher_window_sec * teacher_windows_per_clip"
        )

    teacher_cache = load_perch_teacher_cache(teacher_cache_path)
    label_to_idx = {label: idx for idx, label in enumerate(teacher_cache.class_labels)}
    train_df = pd.read_csv(data_dir / "train.csv", dtype=str, keep_default_na=False)

    clip_records: list[TrainAudioClipRecord] = []
    teacher_windows_per_clip = distill_cfg.teacher_windows_per_clip
    teacher_window_sec = distill_cfg.teacher_window_sec
    clip_duration_sec = audio_cfg.chunk_duration_sec

    for row in train_df.to_dict(orient="records"):
        filename = row["filename"]
        audio_path = data_dir / "train_audio" / filename
        if not audio_path.exists():
            continue

        teacher_chunks = teacher_cache.chunks_by_filename.get(filename)
        if not teacher_chunks:
            continue

        secondary_labels = _parse_list_text(row["secondary_labels"])
        label_target = build_label_target(
            primary_label=row["primary_label"],
            secondary_labels=secondary_labels,
            label_to_idx=label_to_idx,
            primary_weight=primary_weight,
            secondary_weight=secondary_weight,
        )

        chunk_lookup = {chunk.chunk_index: chunk for chunk in teacher_chunks}
        expected_chunks = max(chunk.total_chunks for chunk in teacher_chunks)
        n_clips = max(1, math.ceil(expected_chunks / teacher_windows_per_clip))

        for clip_index in range(n_clips):
            start_chunk_index = clip_index * teacher_windows_per_clip
            teacher_logits = np.zeros(
                (teacher_windows_per_clip, len(teacher_cache.class_labels)),
                dtype=np.float32,
            )
            teacher_mask = np.zeros(teacher_windows_per_clip, dtype=np.float32)
            clip_start_sec = start_chunk_index * teacher_window_sec
            valid_end_sec = clip_start_sec

            for offset in range(teacher_windows_per_clip):
                chunk = chunk_lookup.get(start_chunk_index + offset)
                if chunk is None:
                    continue
                teacher_logits[offset] = chunk.logits
                teacher_mask[offset] = 1.0
                valid_end_sec = max(valid_end_sec, chunk.end_sec)

            if teacher_mask.sum() == 0.0:
                continue

            clip_records.append(
                TrainAudioClipRecord(
                    audio_path=audio_path,
                    filename=filename,
                    primary_label=row["primary_label"],
                    secondary_labels=secondary_labels,
                    clip_index=clip_index,
                    clip_start_sec=clip_start_sec,
                    clip_duration_sec=clip_duration_sec,
                    valid_duration_sec=max(0.0, valid_end_sec - clip_start_sec),
                    label_target=label_target.copy(),
                    teacher_logits=teacher_logits,
                    teacher_mask=teacher_mask,
                )
            )

    if not clip_records:
        raise RuntimeError("No train_audio clips could be indexed from the Perch teacher cache.")
    return clip_records, teacher_cache.class_labels


class TrainAudioPerchTeacherDataset:
    """Indexable waveform dataset for weak labels plus 4x5-second teacher supervision."""

    def __init__(
        self,
        clip_records: list[TrainAudioClipRecord],
        class_labels: tuple[str, ...],
        *,
        audio_cfg: AudioConfig | None = None,
    ) -> None:
        self.clip_records = clip_records
        self.class_labels = class_labels
        self.audio_cfg = audio_cfg or AudioConfig()

    @classmethod
    def from_paths(
        cls,
        data_dir: Path,
        teacher_cache_path: Path,
        *,
        audio_cfg: AudioConfig | None = None,
        distill_cfg: DistillConfig | None = None,
        primary_weight: float = 1.0,
        secondary_weight: float = 0.7,
    ) -> TrainAudioPerchTeacherDataset:
        clip_records, class_labels = build_train_audio_perch_clip_index(
            data_dir=data_dir,
            teacher_cache_path=teacher_cache_path,
            audio_cfg=audio_cfg,
            distill_cfg=distill_cfg,
            primary_weight=primary_weight,
            secondary_weight=secondary_weight,
        )
        return cls(clip_records=clip_records, class_labels=class_labels, audio_cfg=audio_cfg)

    def __len__(self) -> int:
        return len(self.clip_records)

    def __getitem__(self, index: int) -> TrainAudioPerchSample:
        record = self.clip_records[index]
        waveform = load_audio_clip(
            record.audio_path,
            sample_rate=self.audio_cfg.sample_rate,
            start_sec=record.clip_start_sec,
            duration_sec=record.clip_duration_sec,
        )
        return TrainAudioPerchSample(
            filename=record.filename,
            primary_label=record.primary_label,
            secondary_labels=record.secondary_labels,
            clip_index=record.clip_index,
            clip_start_sec=record.clip_start_sec,
            clip_duration_sec=record.clip_duration_sec,
            valid_duration_sec=record.valid_duration_sec,
            waveform=waveform,
            label_target=record.label_target.copy(),
            teacher_logits=record.teacher_logits.copy(),
            teacher_mask=record.teacher_mask.copy(),
        )


def load_soundscape_teacher_cache(
    cache_path: Path,
    class_labels: tuple[str, ...],
) -> SoundscapeTeacherCache:
    """Load labeled soundscape 5-second teacher logits."""

    with np.load(cache_path, allow_pickle=False) as cache:
        required = frozenset({"row_ids", "filenames", "logits"})
        missing = required.difference(cache.files)
        if missing:
            missing_display = ", ".join(sorted(missing))
            raise ValueError(
                f"Soundscape teacher cache is missing required keys: {missing_display}."
            )

        logits = cache["logits"]
        if logits.shape[1] != len(class_labels):
            raise ValueError(
                "Soundscape teacher cache class dimension does not match class_labels."
            )

        windows_by_row_id: dict[str, SoundscapeTeacherWindow] = {}
        for row_id, filename, row_logits in zip(
            cache["row_ids"].tolist(),
            cache["filenames"].tolist(),
            logits,
            strict=True,
        ):
            row_id_text = str(row_id)
            windows_by_row_id[row_id_text] = SoundscapeTeacherWindow(
                row_id=row_id_text,
                filename=str(filename),
                end_sec=int(row_id_text.rsplit("_", 1)[1]),
                logits=row_logits.astype(np.float32, copy=True),
            )

    return SoundscapeTeacherCache(
        class_labels=class_labels,
        windows_by_row_id=windows_by_row_id,
    )


def build_soundscape_clip_index(
    data_dir: Path,
    *,
    class_labels: tuple[str, ...],
    teacher_cache_path: Path | None = None,
    dataset_name: str = "all66",
    audio_cfg: AudioConfig | None = None,
    distill_cfg: DistillConfig | None = None,
) -> list[SoundscapeClipRecord]:
    """Create 20-second soundscape clips with 4x5-second strong labels."""

    audio_cfg = audio_cfg or AudioConfig()
    distill_cfg = distill_cfg or DistillConfig()
    if not math.isclose(audio_cfg.chunk_duration_sec, distill_cfg.clip_duration_sec):
        raise ValueError(
            "AudioConfig.chunk_duration_sec must match "
            "DistillConfig.teacher_window_sec * teacher_windows_per_clip"
        )

    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    teacher_cache = (
        load_soundscape_teacher_cache(teacher_cache_path, class_labels)
        if teacher_cache_path is not None
        else None
    )

    labeled_windows = _load_soundscape_label_rows(data_dir, label_to_idx)
    clip_records: list[SoundscapeClipRecord] = []
    teacher_windows_per_clip = distill_cfg.teacher_windows_per_clip
    clip_duration_sec = audio_cfg.chunk_duration_sec

    for filename, windows in sorted(labeled_windows.items()):
        labeled_count = sum(1 for window in windows if window is not None)
        if dataset_name == "full59" and labeled_count != 12:
            continue
        if dataset_name not in {"full59", "all66"}:
            raise ValueError(f"unsupported soundscape dataset: {dataset_name}")

        audio_path = data_dir / "train_soundscapes" / filename
        if not audio_path.exists():
            continue

        for clip_index in range(12 - teacher_windows_per_clip + 1):
            window_targets = np.zeros(
                (teacher_windows_per_clip, len(class_labels)),
                dtype=np.float32,
            )
            window_mask = np.zeros(teacher_windows_per_clip, dtype=np.float32)
            teacher_logits = np.zeros_like(window_targets)
            teacher_mask = np.zeros(teacher_windows_per_clip, dtype=np.float32)
            window_end_secs: list[int] = []

            for offset in range(teacher_windows_per_clip):
                window_index = clip_index + offset
                end_sec = (window_index + 1) * 5
                row_id = f"{Path(filename).stem}_{end_sec}"
                window_end_secs.append(end_sec)

                label_row = windows[window_index]
                if label_row is not None:
                    window_targets[offset] = label_row
                    window_mask[offset] = 1.0

                if teacher_cache is None:
                    continue
                teacher_window = teacher_cache.windows_by_row_id.get(row_id)
                if teacher_window is None:
                    continue
                teacher_logits[offset] = teacher_window.logits
                teacher_mask[offset] = 1.0

            if window_mask.sum() == 0.0:
                continue

            clip_records.append(
                SoundscapeClipRecord(
                    audio_path=audio_path,
                    filename=filename,
                    clip_index=clip_index,
                    clip_start_sec=float(clip_index * 5),
                    clip_duration_sec=clip_duration_sec,
                    window_end_secs=tuple(window_end_secs),
                    window_targets=window_targets,
                    window_mask=window_mask,
                    teacher_logits=teacher_logits,
                    teacher_mask=teacher_mask,
                )
            )

    if not clip_records:
        raise RuntimeError("No soundscape 20-second clips could be indexed.")
    return clip_records


class SoundscapeSedDataset:
    """Indexable 20-second soundscape clips with per-window labels."""

    def __init__(
        self,
        clip_records: list[SoundscapeClipRecord],
        class_labels: tuple[str, ...],
        *,
        audio_cfg: AudioConfig | None = None,
    ) -> None:
        self.clip_records = clip_records
        self.class_labels = class_labels
        self.audio_cfg = audio_cfg or AudioConfig()

    @classmethod
    def from_paths(
        cls,
        data_dir: Path,
        *,
        class_labels: tuple[str, ...],
        teacher_cache_path: Path | None = None,
        dataset_name: str = "all66",
        audio_cfg: AudioConfig | None = None,
        distill_cfg: DistillConfig | None = None,
    ) -> SoundscapeSedDataset:
        clip_records = build_soundscape_clip_index(
            data_dir=data_dir,
            class_labels=class_labels,
            teacher_cache_path=teacher_cache_path,
            dataset_name=dataset_name,
            audio_cfg=audio_cfg,
            distill_cfg=distill_cfg,
        )
        return cls(clip_records=clip_records, class_labels=class_labels, audio_cfg=audio_cfg)

    def __len__(self) -> int:
        return len(self.clip_records)

    def __getitem__(self, index: int) -> SoundscapeSedSample:
        record = self.clip_records[index]
        waveform = load_audio_clip(
            record.audio_path,
            sample_rate=self.audio_cfg.sample_rate,
            start_sec=record.clip_start_sec,
            duration_sec=record.clip_duration_sec,
        )
        return SoundscapeSedSample(
            filename=record.filename,
            clip_index=record.clip_index,
            clip_start_sec=record.clip_start_sec,
            clip_duration_sec=record.clip_duration_sec,
            window_end_secs=record.window_end_secs,
            waveform=waveform,
            window_targets=record.window_targets.copy(),
            window_mask=record.window_mask.copy(),
            teacher_logits=record.teacher_logits.copy(),
            teacher_mask=record.teacher_mask.copy(),
        )


def load_audio_clip(
    path: Path,
    *,
    sample_rate: int,
    start_sec: float,
    duration_sec: float,
) -> NDArray[np.float32]:
    """Load, resample, slice, and zero-pad a waveform clip."""

    audio, original_sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = _resample_audio(
        audio=audio, source_sample_rate=original_sample_rate, target_sample_rate=sample_rate
    )

    start_index = max(0, int(round(start_sec * sample_rate)))
    clip_length = int(round(duration_sec * sample_rate))
    stop_index = start_index + clip_length

    if start_index >= len(audio):
        return np.zeros(clip_length, dtype=np.float32)

    clip = audio[start_index:stop_index].astype(np.float32, copy=False)
    if len(clip) < clip_length:
        clip = np.pad(clip, (0, clip_length - len(clip)))
    return clip.astype(np.float32, copy=False)


def _parse_list_text(value: str) -> tuple[str, ...]:
    if not value:
        return ()
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return (value,)
    if not isinstance(parsed, list):
        return (str(parsed),)
    return tuple(str(item) for item in parsed if str(item))


def _resample_audio(
    audio: NDArray[np.float32],
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> NDArray[np.float32]:
    if source_sample_rate == target_sample_rate:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    ratio = target_sample_rate / source_sample_rate
    target_length = max(1, int(round(len(audio) * ratio)))
    source_positions = np.arange(len(audio), dtype=np.float32)
    target_positions = np.linspace(0, len(audio) - 1, target_length, dtype=np.float32)
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def _load_soundscape_label_rows(
    data_dir: Path,
    label_to_idx: dict[str, int],
) -> dict[str, list[NDArray[np.float32] | None]]:
    table = pd.read_csv(
        data_dir / "train_soundscapes_labels.csv",
        dtype=str,
        keep_default_na=False,
    ).drop_duplicates()

    grouped: dict[str, list[NDArray[np.float32] | None]] = defaultdict(lambda: [None] * 12)
    for row in table.to_dict(orient="records"):
        end_sec = _clock_to_seconds(row["end"])
        window_index = max(0, (end_sec // 5) - 1)
        target = grouped[row["filename"]][window_index]
        if target is None:
            target = np.zeros(len(label_to_idx), dtype=np.float32)
            grouped[row["filename"]][window_index] = target

        for label in row["primary_label"].split(";"):
            cleaned = label.strip()
            if not cleaned:
                continue
            class_index = label_to_idx.get(cleaned)
            if class_index is not None:
                target[class_index] = 1.0

    return grouped


def _clock_to_seconds(clock_text: str) -> int:
    hours_text, minutes_text, seconds_text = clock_text.split(":")
    return int(hours_text) * 3600 + int(minutes_text) * 60 + int(seconds_text)
