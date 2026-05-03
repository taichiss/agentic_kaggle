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
    from data.audio_catalog import AudioCatalog
    from providers.annotation_store import AnnotationStore
    from providers.audio_visualization import (
        compute_spectrum_analysis,
        inspect_audio,
        read_audio_segment,
        render_frequency_profile_svg,
        render_species_distribution_svg,
        render_spectrogram_png,
        render_waveform_svg,
    )

    return (
        AnnotationStore,
        AudioCatalog,
        compute_spectrum_analysis,
        inspect_audio,
        read_audio_segment,
        render_frequency_profile_svg,
        render_species_distribution_svg,
        render_spectrogram_png,
        render_waveform_svg,
    )


(
    AnnotationStore,
    AudioCatalog,
    compute_spectrum_analysis,
    inspect_audio,
    read_audio_segment,
    render_frequency_profile_svg,
    render_species_distribution_svg,
    render_spectrogram_png,
    render_waveform_svg,
) = load_test_targets()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_wav(path: Path, sample_rate: int = 8_000, duration_sec: float = 1.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = np.sin(2 * np.pi * 440 * samples).astype(np.float32)
    sf.write(path, waveform, sample_rate, format="WAV")


class AudioCatalogTest(unittest.TestCase):
    def test_catalog_deduplicates_soundscape_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            competition_root = root / "competition"
            write_text(
                competition_root / "taxonomy.csv",
                "\n".join(
                    [
                        "primary_label,inat_taxon_id,scientific_name,common_name,class_name",
                        "ashgre1,1,Aratinga weddellii,Ash-breasted Parakeet,Aves",
                        "22961,2,Leptodactylus podicipinus,Pointedbelly Frog,Amphibia",
                    ]
                ),
            )
            write_text(
                competition_root / "train.csv",
                "\n".join(
                    [
                        "primary_label,secondary_labels,type,latitude,longitude,scientific_name,common_name,class_name,inat_taxon_id,author,license,rating,url,filename,collection",
                        'ashgre1,"[]","[]",-1.1,-2.2,Aratinga weddellii,Ash-breasted Parakeet,Aves,1,Alice,cc-by,4.5,https://example.com,ashgre1/sample.wav,XC',
                        'ashgre1,"[]","[]",-1.0,-2.0,Aratinga weddellii,Ash-breasted Parakeet,Aves,1,Bob,cc-by,3.5,https://example.com,ashgre1/sample2.wav,XC',
                    ]
                ),
            )
            write_text(
                competition_root / "train_soundscapes_labels.csv",
                "\n".join(
                    [
                        "filename,start,end,primary_label",
                        "soundscape.ogg,00:00:00,00:00:05,22961",
                        "soundscape.ogg,00:00:00,00:00:05,22961",
                        "soundscape.ogg,00:00:05,00:00:10,22961;ashgre1",
                    ]
                ),
            )
            write_wav(competition_root / "train_audio" / "ashgre1" / "sample.wav")
            write_wav(competition_root / "train_audio" / "ashgre1" / "sample2.wav")
            write_wav(competition_root / "train_soundscapes" / "soundscape.ogg")
            write_wav(competition_root / "test_soundscapes" / "testscape.ogg")

            catalog = AudioCatalog.from_competition_root(competition_root)
            self.assertEqual(catalog.summary["recordings"], 4)
            self.assertEqual(catalog.summary["soundscape_windows"], 2)

            payload = catalog.search(query="Pointedbelly")
            self.assertEqual(payload["total"], 1)
            detail = catalog.serialize_detail(
                catalog.get_entry(payload["items"][0]["recording_id"])
            )
            self.assertEqual(len(detail["soundscape_windows"]), 2)
            self.assertEqual(detail["default_duration_sec"], 5.0)
            species = catalog.serialize_species_batch(["ashgre1", "22961"])
            self.assertEqual(species[0]["train_audio_recordings"], 2)
            self.assertEqual(species[0]["soundscape_window_count"], 1)
            self.assertEqual(species[0]["rating_summary"]["count"], 2)
            self.assertEqual(species[1]["soundscape_file_count"], 1)

    def test_audio_render_helpers_emit_nonempty_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "tone.wav"
            write_wav(audio_path, sample_rate=16_000, duration_sec=1.5)

            info = inspect_audio(audio_path)
            self.assertAlmostEqual(info.duration_sec, 1.5, places=2)
            audio, sample_rate = read_audio_segment(audio_path, start_sec=0.25, duration_sec=0.5)
            self.assertEqual(sample_rate, 16_000)
            self.assertGreater(audio.shape[0], 0)

            analysis = compute_spectrum_analysis(audio, sample_rate)
            waveform = render_waveform_svg(audio)
            frequency_profile = render_frequency_profile_svg(audio, sample_rate)
            spectrogram = render_spectrogram_png(audio, sample_rate)
            self.assertGreater(float(analysis["spectral_centroid_hz"]), 0.0)
            self.assertEqual(len(analysis["band_energy_share"]), 3)
            self.assertTrue(waveform.startswith(b"<svg"))
            self.assertTrue(frequency_profile.startswith(b"<svg"))
            self.assertTrue(spectrogram.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_species_distribution_svg_contains_rank_panels(self) -> None:
        svg_payload = render_species_distribution_svg(
            [
                {
                    "primary_label": "ashgre1",
                    "common_name": "Ash-breasted Parakeet",
                    "train_audio_recordings": 12,
                    "soundscape_window_count": 3,
                    "soundscape_file_count": 2,
                },
                {
                    "primary_label": "bucmot4",
                    "common_name": "Buckley's Forest-Falcon",
                    "train_audio_recordings": 4,
                    "soundscape_window_count": 7,
                    "soundscape_file_count": 5,
                },
            ],
            summary={"labeled_soundscape_files": 6},
            title="BirdCLEF+ 2026 species distribution",
        )
        self.assertTrue(svg_payload.startswith(b"<svg"))
        self.assertIn(b"train_audio recordings", svg_payload)
        self.assertIn(b"soundscape windows", svg_payload)
        self.assertIn(b"ashgre1 / Ash-breasted Parakeet", svg_payload)

    def test_annotation_store_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = AnnotationStore(Path(tmp_dir) / "annotations.jsonl")
            store.add_annotation(
                recording_id="abc",
                dataset="train_soundscapes",
                filename="soundscape.ogg",
                relative_path="train_soundscapes/soundscape.ogg",
                start_sec=5.0,
                duration_sec=10.0,
                category="false_positive",
                labels=["ashgre1", "22961"],
                note="帯域は鳥より虫に近い",
            )
            items = store.list_annotations(recording_id="abc")
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["category"], "false_positive")
            self.assertEqual(items[0]["labels"], ["ashgre1", "22961"])


if __name__ == "__main__":
    unittest.main()
