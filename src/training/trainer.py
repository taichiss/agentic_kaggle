"""Training helpers for TensorFlow SED experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import tensorflow as tf

from core.config import AudioConfig, DistillConfig, TrainConfig
from data.dataset import (
    SoundscapeClipRecord,
    TrainAudioClipRecord,
    load_audio_clip,
)
from training.losses import clip_bce_with_logits, masked_bce_with_logits, masked_sigmoid_mse


class HasClipRecord(Protocol):
    audio_path: Path
    clip_start_sec: float
    clip_duration_sec: float


@dataclass(frozen=True)
class StageResult:
    name: str
    epochs: int
    train_loss: float
    val_loss: float | None


class ClipSequence(tf.keras.utils.Sequence):
    """Keras Sequence that loads 20-second clips lazily from disk."""

    def __init__(
        self,
        records: Sequence[HasClipRecord],
        *,
        audio_cfg: AudioConfig,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        self.records = list(records)
        self.audio_cfg = audio_cfg
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.records), dtype=np.int32)
        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        batch_indices = self.indices[index * self.batch_size : (index + 1) * self.batch_size]
        waveforms: list[np.ndarray] = []
        for record_index in batch_indices.tolist():
            record = self.records[record_index]
            waveforms.append(
                load_audio_clip(
                    record.audio_path,
                    sample_rate=self.audio_cfg.sample_rate,
                    start_sec=record.clip_start_sec,
                    duration_sec=record.clip_duration_sec,
                )
            )
        payload = self._build_payload(batch_indices.tolist())
        payload["waveform"] = np.stack(waveforms).astype(np.float32)
        return payload

    def on_epoch_end(self) -> None:
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _build_payload(self, batch_indices: list[int]) -> dict[str, np.ndarray]:
        raise NotImplementedError


def _apply_gradients(
    optimizer: tf.keras.optimizers.Optimizer,
    gradients: list[tf.Tensor | None],
    variables: list[tf.Variable],
) -> None:
    pairs = [
        (gradient, variable)
        for gradient, variable in zip(gradients, variables, strict=True)
        if gradient is not None
    ]
    if pairs:
        optimizer.apply_gradients(pairs)


class TrainAudioSequence(ClipSequence):
    def __init__(
        self,
        records: list[TrainAudioClipRecord],
        *,
        audio_cfg: AudioConfig,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        super().__init__(
            cast(Sequence[HasClipRecord], records),
            audio_cfg=audio_cfg,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.train_audio_records = records

    def _build_payload(self, batch_indices: list[int]) -> dict[str, np.ndarray]:
        label_targets = np.stack(
            [self.train_audio_records[i].label_target for i in batch_indices]
        ).astype(np.float32)
        teacher_logits = np.stack(
            [self.train_audio_records[i].teacher_logits for i in batch_indices]
        ).astype(np.float32)
        teacher_mask = np.stack(
            [self.train_audio_records[i].teacher_mask for i in batch_indices]
        ).astype(np.float32)
        return {
            "clip_targets": label_targets,
            "teacher_logits": teacher_logits,
            "teacher_mask": teacher_mask,
        }


class SoundscapeSequence(ClipSequence):
    def __init__(
        self,
        records: list[SoundscapeClipRecord],
        *,
        audio_cfg: AudioConfig,
        batch_size: int,
        shuffle: bool,
    ) -> None:
        super().__init__(
            cast(Sequence[HasClipRecord], records),
            audio_cfg=audio_cfg,
            batch_size=batch_size,
            shuffle=shuffle,
        )
        self.soundscape_records = records

    def _build_payload(self, batch_indices: list[int]) -> dict[str, np.ndarray]:
        window_targets = np.stack(
            [self.soundscape_records[i].window_targets for i in batch_indices]
        ).astype(np.float32)
        window_mask = np.stack(
            [self.soundscape_records[i].window_mask for i in batch_indices]
        ).astype(np.float32)
        teacher_logits = np.stack(
            [self.soundscape_records[i].teacher_logits for i in batch_indices]
        ).astype(np.float32)
        teacher_mask = np.stack(
            [self.soundscape_records[i].teacher_mask for i in batch_indices]
        ).astype(np.float32)
        clip_targets = window_targets.max(axis=1)
        return {
            "clip_targets": clip_targets,
            "window_targets": window_targets,
            "window_mask": window_mask,
            "teacher_logits": teacher_logits,
            "teacher_mask": teacher_mask,
        }


def run_train_audio_stage(
    model: tf.keras.Model,
    train_seq: TrainAudioSequence,
    *,
    train_cfg: TrainConfig,
    distill_cfg: DistillConfig,
    mapped_class_mask: np.ndarray | None = None,
) -> StageResult:
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg.lr)
    train_loss_metric = tf.keras.metrics.Mean()

    class_mask_tensor = (
        tf.convert_to_tensor(mapped_class_mask.astype(np.float32))
        if mapped_class_mask is not None and distill_cfg.mapped_classes_only
        else None
    )

    for _ in range(train_cfg.epochs):
        train_loss_metric.reset_state()
        for batch_index in range(len(train_seq)):
            batch = train_seq[batch_index]
            with tf.GradientTape() as tape:
                outputs = model(batch["waveform"], training=True)
                label_loss = clip_bce_with_logits(outputs["clip_logits"], batch["clip_targets"])
                distill_loss = masked_sigmoid_mse(
                    outputs["window_logits"],
                    batch["teacher_logits"],
                    batch["teacher_mask"],
                    class_mask=class_mask_tensor,
                )
                loss = ((1.0 - distill_cfg.teacher_loss_weight) * label_loss) + (
                    distill_cfg.teacher_loss_weight * distill_loss
                )
            gradients = tape.gradient(loss, model.trainable_variables)
            _apply_gradients(optimizer, gradients, model.trainable_variables)
            train_loss_metric.update_state(loss)

    return StageResult(
        name="train_audio_teacher",
        epochs=train_cfg.epochs,
        train_loss=float(train_loss_metric.result().numpy()),
        val_loss=None,
    )


def run_soundscape_stage(
    model: tf.keras.Model,
    train_seq: SoundscapeSequence,
    val_seq: SoundscapeSequence | None,
    *,
    train_cfg: TrainConfig,
    distill_cfg: DistillConfig,
    mapped_class_mask: np.ndarray | None = None,
    include_teacher_distill: bool = True,
) -> StageResult:
    optimizer = tf.keras.optimizers.Adam(learning_rate=train_cfg.lr)
    train_loss_metric = tf.keras.metrics.Mean()
    val_loss_metric = tf.keras.metrics.Mean()

    class_mask_tensor = (
        tf.convert_to_tensor(mapped_class_mask.astype(np.float32))
        if mapped_class_mask is not None and distill_cfg.mapped_classes_only
        else None
    )

    for _ in range(train_cfg.epochs):
        train_loss_metric.reset_state()
        for batch_index in range(len(train_seq)):
            batch = train_seq[batch_index]
            with tf.GradientTape() as tape:
                outputs = model(batch["waveform"], training=True)
                window_loss = masked_bce_with_logits(
                    outputs["window_logits"],
                    batch["window_targets"],
                    batch["window_mask"],
                )
                clip_loss = clip_bce_with_logits(outputs["clip_logits"], batch["clip_targets"])
                loss = 0.5 * window_loss + 0.5 * clip_loss
                if include_teacher_distill:
                    distill_loss = masked_sigmoid_mse(
                        outputs["window_logits"],
                        batch["teacher_logits"],
                        batch["teacher_mask"],
                        class_mask=class_mask_tensor,
                    )
                    loss = ((1.0 - distill_cfg.teacher_loss_weight) * loss) + (
                        distill_cfg.teacher_loss_weight * distill_loss
                    )
            gradients = tape.gradient(loss, model.trainable_variables)
            _apply_gradients(optimizer, gradients, model.trainable_variables)
            train_loss_metric.update_state(loss)

        if val_seq is None:
            continue

        val_loss_metric.reset_state()
        for batch_index in range(len(val_seq)):
            batch = val_seq[batch_index]
            outputs = model(batch["waveform"], training=False)
            window_loss = masked_bce_with_logits(
                outputs["window_logits"],
                batch["window_targets"],
                batch["window_mask"],
            )
            clip_loss = clip_bce_with_logits(outputs["clip_logits"], batch["clip_targets"])
            val_loss_metric.update_state(0.5 * window_loss + 0.5 * clip_loss)

    return StageResult(
        name="soundscape_finetune",
        epochs=train_cfg.epochs,
        train_loss=float(train_loss_metric.result().numpy()),
        val_loss=float(val_loss_metric.result().numpy()) if val_seq is not None else None,
    )


def predict_soundscape_windows(
    model: tf.keras.Model,
    records: list[SoundscapeClipRecord],
    *,
    audio_cfg: AudioConfig,
    batch_size: int,
) -> dict[tuple[str, int], np.ndarray]:
    sequence = SoundscapeSequence(
        records,
        audio_cfg=audio_cfg,
        batch_size=batch_size,
        shuffle=False,
    )
    aggregates: dict[tuple[str, int], list[np.ndarray]] = {}

    for batch_index in range(len(sequence)):
        batch = sequence[batch_index]
        outputs = model(batch["waveform"], training=False)
        window_probs = tf.math.sigmoid(outputs["window_logits"]).numpy()
        start = batch_index * batch_size
        stop = min(len(records), start + batch_size)
        batch_records = records[start:stop]

        for sample_index, record in enumerate(batch_records):
            for window_offset, end_sec in enumerate(record.window_end_secs):
                key = (record.filename, end_sec)
                aggregates.setdefault(key, []).append(window_probs[sample_index, window_offset])

    return {
        key: np.mean(np.stack(values).astype(np.float32), axis=0)
        for key, values in aggregates.items()
    }


def clone_weights(model: tf.keras.Model) -> list[np.ndarray]:
    return [weight.copy() for weight in model.get_weights()]


def restore_weights(model: tf.keras.Model, weights: list[np.ndarray]) -> None:
    model.set_weights(weights)
