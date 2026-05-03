# ruff: noqa: E402
"""Extract Perch embeddings and mapped logits from 5-second train_audio chunks."""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import soundfile as sf

warnings.filterwarnings("ignore")

DEFAULT_DATA_DIR = REPO_ROOT / "data" / "input" / "BirdCLEF+ 2026"
DEFAULT_MODEL_DIR = REPO_ROOT / "data" / "models"
DEFAULT_CACHE_PATH = REPO_ROOT / "data" / "models" / "train_audio_perch_chunks_v1.npz"
TARGET_SAMPLE_RATE = 32_000
WINDOW_SECONDS = 5
WINDOW_SAMPLES = TARGET_SAMPLE_RATE * WINDOW_SECONDS


@dataclass(frozen=True)
class PendingChunk:
    audio: np.ndarray
    filename: str
    primary_label: str
    secondary_labels: str
    chunk_index: int
    chunk_start_sec: float
    chunk_end_sec: float
    total_chunks: int


def resample_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SAMPLE_RATE:
        return audio.astype(np.float32, copy=False)
    if len(audio) == 0:
        return np.zeros(0, dtype=np.float32)

    ratio = TARGET_SAMPLE_RATE / sample_rate
    target_length = max(1, int(round(len(audio) * ratio)))
    target_positions = np.linspace(0, len(audio) - 1, target_length)
    source_positions = np.arange(len(audio))
    return np.interp(target_positions, source_positions, audio).astype(np.float32)


def load_audio(path: Path) -> np.ndarray | None:
    try:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    except Exception as exc:
        print(f"Skipping unreadable file {path}: {exc}")
        return None

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return resample_audio(audio, sample_rate)


def chunk_audio(
    audio: np.ndarray,
    max_chunks_per_file: int | None,
) -> list[tuple[np.ndarray, int, float, float]]:
    if len(audio) == 0:
        audio = np.zeros(WINDOW_SAMPLES, dtype=np.float32)

    total_chunks = max(1, math.ceil(len(audio) / WINDOW_SAMPLES))
    if max_chunks_per_file is not None:
        total_chunks = min(total_chunks, max_chunks_per_file)

    chunks: list[tuple[np.ndarray, int, float, float]] = []
    audio_duration_sec = len(audio) / TARGET_SAMPLE_RATE
    for chunk_index in range(total_chunks):
        start = chunk_index * WINDOW_SAMPLES
        stop = min(start + WINDOW_SAMPLES, len(audio))
        chunk = audio[start:stop]
        if len(chunk) < WINDOW_SAMPLES:
            chunk = np.pad(chunk, (0, WINDOW_SAMPLES - len(chunk)))
        chunks.append(
            (
                chunk.astype(np.float32, copy=False),
                chunk_index,
                start / TARGET_SAMPLE_RATE,
                min((chunk_index + 1) * WINDOW_SECONDS, audio_duration_sec),
            )
        )
    return chunks


def flush_pending_chunks(
    infer_fn: Any,
    pending_chunks: list[PendingChunk],
    mapped_positions: np.ndarray,
    mapped_bc_indices: np.ndarray,
    n_classes: int,
    embeddings_batches: list[np.ndarray],
    logits_batches: list[np.ndarray],
    metadata: dict[str, list[Any]],
) -> None:
    if not pending_chunks:
        return

    import tensorflow as tf

    audio_batch = np.stack([chunk.audio for chunk in pending_chunks]).astype(np.float32)
    outputs = infer_fn(inputs=tf.convert_to_tensor(audio_batch))
    raw_logits = outputs["label"].numpy().astype(np.float32)
    embeddings = outputs["embedding"].numpy().astype(np.float32)

    mapped_logits = np.zeros((len(pending_chunks), n_classes), dtype=np.float32)
    mapped_logits[:, mapped_positions] = raw_logits[:, mapped_bc_indices]

    embeddings_batches.append(embeddings)
    logits_batches.append(mapped_logits)
    metadata["filenames"].extend(chunk.filename for chunk in pending_chunks)
    metadata["primary_labels_per_chunk"].extend(chunk.primary_label for chunk in pending_chunks)
    metadata["secondary_labels_per_chunk"].extend(
        chunk.secondary_labels for chunk in pending_chunks
    )
    metadata["chunk_index"].extend(chunk.chunk_index for chunk in pending_chunks)
    metadata["chunk_start_sec"].extend(chunk.chunk_start_sec for chunk in pending_chunks)
    metadata["chunk_end_sec"].extend(chunk.chunk_end_sec for chunk in pending_chunks)
    metadata["n_chunks_per_file"].extend(chunk.total_chunks for chunk in pending_chunks)

    pending_chunks.clear()


def extract_train_audio_embeddings(
    data_dir: Path,
    model_dir: Path,
    cache_path: Path,
    force_rebuild: bool = False,
    batch_size: int = 128,
    max_files: int | None = None,
    max_chunks_per_file: int | None = None,
) -> dict[str, np.ndarray]:
    if cache_path.exists() and not force_rebuild:
        print(f"Loading existing cache: {cache_path}")
        return dict(np.load(cache_path, allow_pickle=False))

    import os

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    from scripts.experiment.perch_probe_cv import build_perch_mapping, load_primary_labels

    train_df = pd.read_csv(data_dir / "train.csv")
    if max_files is not None:
        train_df = train_df.head(max_files).copy()

    model = tf.saved_model.load(str(model_dir))
    infer_fn = model.signatures["serving_default"]

    primary_labels = load_primary_labels(data_dir)
    bc_indices, mapped_mask = build_perch_mapping(data_dir, model_dir)
    mapped_positions = np.where(mapped_mask)[0]
    mapped_bc_indices = bc_indices[mapped_mask]
    n_classes = len(primary_labels)

    pending_chunks: list[PendingChunk] = []
    embeddings_batches: list[np.ndarray] = []
    logits_batches: list[np.ndarray] = []
    metadata: dict[str, list[Any]] = {
        "filenames": [],
        "primary_labels_per_chunk": [],
        "secondary_labels_per_chunk": [],
        "chunk_index": [],
        "chunk_start_sec": [],
        "chunk_end_sec": [],
        "n_chunks_per_file": [],
    }

    total_files = 0
    skipped_files = 0
    total_chunks = 0
    train_audio_dir = data_dir / "train_audio"

    for row in train_df.itertuples(index=False):
        path = train_audio_dir / row.filename
        if not path.exists():
            skipped_files += 1
            continue

        audio = load_audio(path)
        if audio is None:
            skipped_files += 1
            continue

        chunked_audio = chunk_audio(audio, max_chunks_per_file=max_chunks_per_file)
        total_files += 1
        total_chunks += len(chunked_audio)

        for chunk_audio_array, chunk_index, start_sec, end_sec in chunked_audio:
            pending_chunks.append(
                PendingChunk(
                    audio=chunk_audio_array,
                    filename=str(row.filename),
                    primary_label=str(row.primary_label),
                    secondary_labels=str(row.secondary_labels),
                    chunk_index=chunk_index,
                    chunk_start_sec=float(start_sec),
                    chunk_end_sec=float(end_sec),
                    total_chunks=len(chunked_audio),
                )
            )

        if len(pending_chunks) >= batch_size:
            flush_pending_chunks(
                infer_fn=infer_fn,
                pending_chunks=pending_chunks,
                mapped_positions=mapped_positions,
                mapped_bc_indices=mapped_bc_indices,
                n_classes=n_classes,
                embeddings_batches=embeddings_batches,
                logits_batches=logits_batches,
                metadata=metadata,
            )

        if total_files % 250 == 0:
            print(
                f"Processed {total_files} files / {total_chunks} chunks (skipped {skipped_files})"
            )

    flush_pending_chunks(
        infer_fn=infer_fn,
        pending_chunks=pending_chunks,
        mapped_positions=mapped_positions,
        mapped_bc_indices=mapped_bc_indices,
        n_classes=n_classes,
        embeddings_batches=embeddings_batches,
        logits_batches=logits_batches,
        metadata=metadata,
    )

    if not embeddings_batches or not logits_batches:
        raise RuntimeError("No train_audio chunks were extracted.")

    result = {
        "embeddings": np.concatenate(embeddings_batches, axis=0).astype(np.float32),
        "logits": np.concatenate(logits_batches, axis=0).astype(np.float32),
        "filenames": np.asarray(metadata["filenames"], dtype=str),
        "primary_labels_per_chunk": np.asarray(metadata["primary_labels_per_chunk"], dtype=str),
        "secondary_labels_per_chunk": np.asarray(metadata["secondary_labels_per_chunk"], dtype=str),
        "chunk_index": np.asarray(metadata["chunk_index"], dtype=np.int16),
        "chunk_start_sec": np.asarray(metadata["chunk_start_sec"], dtype=np.float32),
        "chunk_end_sec": np.asarray(metadata["chunk_end_sec"], dtype=np.float32),
        "n_chunks_per_file": np.asarray(metadata["n_chunks_per_file"], dtype=np.int16),
        "class_labels": np.asarray(primary_labels, dtype=str),
    }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **cast(dict[str, Any], result))

    print(f"Saved cache to {cache_path}")
    print(
        f"  files={total_files} chunks={len(result['filenames'])} "
        f"avg_chunks_per_file={len(result['filenames']) / max(total_files, 1):.2f} "
        f"skipped={skipped_files}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Perch features from train_audio")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-chunks-per-file", type=int, default=None)
    args = parser.parse_args()

    result = extract_train_audio_embeddings(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        cache_path=args.cache_path,
        force_rebuild=args.force_rebuild,
        batch_size=args.batch_size,
        max_files=args.max_files,
        max_chunks_per_file=args.max_chunks_per_file,
    )
    print(f"embeddings shape: {result['embeddings'].shape}")
    print(f"logits shape:     {result['logits'].shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
