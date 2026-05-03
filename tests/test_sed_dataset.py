from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_test_targets() -> tuple[Any, ...]:
    from core.config import AudioConfig, DistillConfig
    from data.dataset import (
        SoundscapeSedDataset,
        TrainAudioPerchTeacherDataset,
        build_soundscape_clip_index,
        build_train_audio_perch_clip_index,
        load_perch_teacher_cache,
    )

    return (
        AudioConfig,
        DistillConfig,
        SoundscapeSedDataset,
        TrainAudioPerchTeacherDataset,
        build_train_audio_perch_clip_index,
        build_soundscape_clip_index,
        load_perch_teacher_cache,
    )


(
    AudioConfig,
    DistillConfig,
    SoundscapeSedDataset,
    TrainAudioPerchTeacherDataset,
    build_train_audio_perch_clip_index,
    build_soundscape_clip_index,
    load_perch_teacher_cache,
) = load_test_targets()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_wav(path: Path, sample_rate: int, duration_sec: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = np.sin(2 * np.pi * 440.0 * samples).astype(np.float32)
    sf.write(path, waveform, sample_rate, format="WAV")


def write_teacher_cache(
    path: Path, filename: str, class_labels: list[str], logits: np.ndarray
) -> None:
    n_chunks = logits.shape[0]
    np.savez(
        path,
        class_labels=np.asarray(class_labels, dtype=str),
        filenames=np.asarray([filename] * n_chunks, dtype=str),
        logits=logits.astype(np.float32),
        primary_labels_per_chunk=np.asarray(["sp1"] * n_chunks, dtype=str),
        secondary_labels_per_chunk=np.asarray(['["sp2"]'] * n_chunks, dtype=str),
        chunk_index=np.arange(n_chunks, dtype=np.int16),
        chunk_start_sec=np.arange(n_chunks, dtype=np.float32) * 5.0,
        chunk_end_sec=(np.arange(n_chunks, dtype=np.float32) + 1.0) * 5.0,
        n_chunks_per_file=np.asarray([n_chunks] * n_chunks, dtype=np.int16),
    )


def write_soundscape_teacher_cache(
    path: Path,
    row_ids: list[str],
    filenames: list[str],
    logits: np.ndarray,
) -> None:
    np.savez(
        path,
        row_ids=np.asarray(row_ids, dtype=str),
        filenames=np.asarray(filenames, dtype=str),
        logits=logits.astype(np.float32),
    )


class SedDatasetTest(unittest.TestCase):
    def test_clip_index_groups_teacher_chunks_into_20_second_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            competition_root = root / "competition"
            filename = "sp1/example.wav"
            write_text(
                competition_root / "train.csv",
                "\n".join(
                    [
                        "primary_label,secondary_labels,filename",
                        'sp1,"[""sp2""]",sp1/example.wav',
                    ]
                ),
            )
            write_wav(
                competition_root / "train_audio" / filename, sample_rate=8_000, duration_sec=25.0
            )

            teacher_logits = np.zeros((5, 3), dtype=np.float32)
            for chunk_index in range(5):
                teacher_logits[chunk_index, chunk_index % 3] = chunk_index + 1.0
            cache_path = root / "teacher_cache.npz"
            write_teacher_cache(
                cache_path,
                filename=filename,
                class_labels=["sp1", "sp2", "sp3"],
                logits=teacher_logits,
            )

            audio_cfg = AudioConfig(sample_rate=8_000, chunk_duration_sec=20.0)
            distill_cfg = DistillConfig(teacher_window_sec=5.0, teacher_windows_per_clip=4)
            clip_records, class_labels = build_train_audio_perch_clip_index(
                data_dir=competition_root,
                teacher_cache_path=cache_path,
                audio_cfg=audio_cfg,
                distill_cfg=distill_cfg,
            )

            self.assertEqual(class_labels, ("sp1", "sp2", "sp3"))
            self.assertEqual(len(clip_records), 2)

            first_clip, second_clip = clip_records
            self.assertEqual(first_clip.clip_start_sec, 0.0)
            self.assertEqual(first_clip.valid_duration_sec, 20.0)
            np.testing.assert_array_equal(
                first_clip.teacher_mask,
                np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            )
            np.testing.assert_array_equal(first_clip.teacher_logits[0], teacher_logits[0])
            np.testing.assert_array_equal(first_clip.teacher_logits[3], teacher_logits[3])
            self.assertEqual(float(first_clip.label_target[0]), 1.0)
            self.assertAlmostEqual(float(first_clip.label_target[1]), 0.7, places=6)

            self.assertEqual(second_clip.clip_start_sec, 20.0)
            self.assertEqual(second_clip.valid_duration_sec, 5.0)
            np.testing.assert_array_equal(
                second_clip.teacher_mask,
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )
            np.testing.assert_array_equal(second_clip.teacher_logits[0], teacher_logits[4])
            self.assertTrue(np.all(second_clip.teacher_logits[1:] == 0.0))

    def test_dataset_loads_waveform_and_zero_pads_tail_clip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            competition_root = root / "competition"
            filename = "sp1/example.wav"
            write_text(
                competition_root / "train.csv",
                "\n".join(
                    [
                        "primary_label,secondary_labels,filename",
                        'sp1,"[]",sp1/example.wav',
                    ]
                ),
            )
            write_wav(
                competition_root / "train_audio" / filename, sample_rate=8_000, duration_sec=25.0
            )

            teacher_logits = np.ones((5, 3), dtype=np.float32)
            cache_path = root / "teacher_cache.npz"
            write_teacher_cache(
                cache_path,
                filename=filename,
                class_labels=["sp1", "sp2", "sp3"],
                logits=teacher_logits,
            )

            audio_cfg = AudioConfig(sample_rate=8_000, chunk_duration_sec=20.0)
            dataset = TrainAudioPerchTeacherDataset.from_paths(
                data_dir=competition_root,
                teacher_cache_path=cache_path,
                audio_cfg=audio_cfg,
                distill_cfg=DistillConfig(teacher_window_sec=5.0, teacher_windows_per_clip=4),
            )

            sample = dataset[1]
            self.assertEqual(sample.waveform.shape, (160_000,))
            self.assertGreater(float(np.abs(sample.waveform[:40_000]).sum()), 0.0)
            self.assertEqual(float(np.abs(sample.waveform[40_000:]).sum()), 0.0)
            np.testing.assert_array_equal(
                sample.teacher_mask,
                np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    def test_old_single_chunk_cache_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "old_cache.npz"
            np.savez(
                cache_path,
                class_labels=np.asarray(["sp1", "sp2"], dtype=str),
                filenames=np.asarray(["sp1/example.wav"], dtype=str),
                logits=np.zeros((1, 2), dtype=np.float32),
                primary_labels_per_chunk=np.asarray(["sp1"], dtype=str),
                secondary_labels_per_chunk=np.asarray(["[]"], dtype=str),
            )

            with self.assertRaisesRegex(ValueError, "chunk_index"):
                load_perch_teacher_cache(cache_path)

    def test_soundscape_clip_index_supports_partial_and_full_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            competition_root = root / "competition"
            write_text(
                competition_root / "train_soundscapes_labels.csv",
                "\n".join(
                    [
                        "filename,start,end,primary_label",
                        "file_full.ogg,00:00:00,00:00:05,sp1",
                        "file_full.ogg,00:00:05,00:00:10,sp2",
                        "file_full.ogg,00:00:10,00:00:15,sp3",
                        "file_full.ogg,00:00:15,00:00:20,sp1;sp2",
                        "file_full.ogg,00:00:20,00:00:25,sp1",
                        "file_full.ogg,00:00:25,00:00:30,sp2",
                        "file_full.ogg,00:00:30,00:00:35,sp3",
                        "file_full.ogg,00:00:35,00:00:40,sp1",
                        "file_full.ogg,00:00:40,00:00:45,sp2",
                        "file_full.ogg,00:00:45,00:00:50,sp3",
                        "file_full.ogg,00:00:50,00:00:55,sp1",
                        "file_full.ogg,00:00:55,00:01:00,sp2",
                        "file_partial.ogg,00:00:00,00:00:05,sp1",
                        "file_partial.ogg,00:00:10,00:00:15,sp3",
                    ]
                ),
            )
            write_wav(
                competition_root / "train_soundscapes" / "file_full.ogg",
                sample_rate=8_000,
                duration_sec=60.0,
            )
            write_wav(
                competition_root / "train_soundscapes" / "file_partial.ogg",
                sample_rate=8_000,
                duration_sec=60.0,
            )

            teacher_logits = np.arange(14 * 3, dtype=np.float32).reshape(14, 3)
            row_ids = [
                "file_full_5",
                "file_full_10",
                "file_full_15",
                "file_full_20",
                "file_full_25",
                "file_full_30",
                "file_full_35",
                "file_full_40",
                "file_full_45",
                "file_full_50",
                "file_full_55",
                "file_full_60",
                "file_partial_5",
                "file_partial_15",
            ]
            filenames = ["file_full.ogg"] * 12 + ["file_partial.ogg"] * 2
            teacher_cache_path = root / "soundscape_teacher.npz"
            write_soundscape_teacher_cache(
                teacher_cache_path,
                row_ids=row_ids,
                filenames=filenames,
                logits=teacher_logits,
            )

            audio_cfg = AudioConfig(sample_rate=8_000, chunk_duration_sec=20.0)
            full59_records = build_soundscape_clip_index(
                competition_root,
                class_labels=("sp1", "sp2", "sp3"),
                teacher_cache_path=teacher_cache_path,
                dataset_name="full59",
                audio_cfg=audio_cfg,
                distill_cfg=DistillConfig(teacher_window_sec=5.0, teacher_windows_per_clip=4),
            )
            all66_records = build_soundscape_clip_index(
                competition_root,
                class_labels=("sp1", "sp2", "sp3"),
                teacher_cache_path=teacher_cache_path,
                dataset_name="all66",
                audio_cfg=audio_cfg,
                distill_cfg=DistillConfig(teacher_window_sec=5.0, teacher_windows_per_clip=4),
            )

            self.assertEqual(len(full59_records), 9)
            self.assertEqual(len(all66_records), 12)

            partial_clip = next(
                record for record in all66_records if record.filename == "file_partial.ogg"
            )
            np.testing.assert_array_equal(
                partial_clip.window_mask,
                np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            )
            np.testing.assert_array_equal(
                partial_clip.teacher_mask,
                np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
            )
            self.assertEqual(partial_clip.window_end_secs, (5, 10, 15, 20))

            dataset = SoundscapeSedDataset(
                clip_records=all66_records,
                class_labels=("sp1", "sp2", "sp3"),
                audio_cfg=audio_cfg,
            )
            sample = dataset[0]
            self.assertEqual(sample.waveform.shape, (160_000,))
            self.assertEqual(sample.window_targets.shape, (4, 3))
            self.assertEqual(sample.teacher_logits.shape, (4, 3))


if __name__ == "__main__":
    unittest.main()
