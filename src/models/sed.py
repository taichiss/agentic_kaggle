"""TensorFlow/Keras SED model with 20-second waveform input and 4-window outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tensorflow as tf

from core.config import AudioConfig, ModelConfig

DEFAULT_EFFICIENTNET_B0_WEIGHTS = Path("/root/.keras/models/efficientnetb0_notop.h5")


@dataclass(frozen=True)
class SedOutput:
    clip_logits: tf.Tensor
    window_logits: tf.Tensor


class LogMelSpectrogram(tf.keras.layers.Layer):
    """Waveform-to-logmel layer for 20-second clips."""

    def __init__(self, audio_cfg: AudioConfig, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.audio_cfg = audio_cfg
        self._mel_matrix: tf.Tensor | None = None

    def build(self, input_shape: tf.TensorShape) -> None:
        del input_shape
        self._mel_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.audio_cfg.n_mels,
            num_spectrogram_bins=(self.audio_cfg.n_fft // 2) + 1,
            sample_rate=self.audio_cfg.sample_rate,
            lower_edge_hertz=float(self.audio_cfg.fmin),
            upper_edge_hertz=float(self.audio_cfg.fmax),
            dtype=tf.float32,
        )
        super().build(())

    def call(self, waveform: tf.Tensor) -> tf.Tensor:
        if waveform.shape.rank == 1:
            waveform = waveform[tf.newaxis, :]

        stft = tf.signal.stft(
            waveform,
            frame_length=self.audio_cfg.n_fft,
            frame_step=self.audio_cfg.hop_length,
            fft_length=self.audio_cfg.n_fft,
            pad_end=False,
        )
        power = tf.math.square(tf.abs(stft))
        mel = tf.tensordot(power, self._mel_matrix, axes=1)
        mel = tf.transpose(mel, perm=[0, 2, 1])
        mel = tf.math.log(mel + 1e-6)

        mel_min = tf.reduce_min(mel, axis=[1, 2], keepdims=True)
        mel_max = tf.reduce_max(mel, axis=[1, 2], keepdims=True)
        mel = (mel - mel_min) / (mel_max - mel_min + 1e-6)
        mel = mel[..., tf.newaxis]
        return tf.repeat(mel, repeats=3, axis=-1)


class ResizeWindowLogits(tf.keras.layers.Layer):
    """Resize variable backbone time resolution to 4 teacher windows."""

    def __init__(self, num_windows: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.num_windows = num_windows

    def call(self, time_logits: tf.Tensor) -> tf.Tensor:
        logits = tf.transpose(time_logits, perm=[0, 2, 1])
        logits = logits[..., tf.newaxis]
        resized = tf.image.resize(
            logits,
            size=(tf.shape(logits)[1], self.num_windows),
            method="bilinear",
        )
        resized = tf.squeeze(resized, axis=-1)
        return tf.transpose(resized, perm=[0, 2, 1])


class TemporalMeanPool(tf.keras.layers.Layer):
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(inputs, axis=1)


class AttentionClipPool(tf.keras.layers.Layer):
    def call(self, inputs: tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        attention_logits, frame_logits = inputs
        attention = tf.nn.softmax(attention_logits, axis=1)
        return tf.reduce_sum(attention * frame_logits, axis=1)


def build_sed_model(
    audio_cfg: AudioConfig,
    model_cfg: ModelConfig,
    *,
    num_windows: int,
    hidden_dim: int = 256,
    dropout_rate: float = 0.2,
    backbone_trainable_layers: int = 20,
) -> tf.keras.Model:
    """Build a 20-second SED model that predicts clip and 5-second window logits."""

    waveform = tf.keras.Input(
        shape=(audio_cfg.chunk_samples,),
        dtype=tf.float32,
        name="waveform",
    )
    mel = LogMelSpectrogram(audio_cfg, name="logmel")(waveform)

    weights_source: str | None = None
    if model_cfg.pretrained:
        weights_source = (
            str(DEFAULT_EFFICIENTNET_B0_WEIGHTS)
            if DEFAULT_EFFICIENTNET_B0_WEIGHTS.exists()
            else "imagenet"
        )

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=weights_source,
        input_tensor=mel,
    )
    for layer in backbone.layers[:-backbone_trainable_layers]:
        layer.trainable = False

    features = backbone.output
    time_features = TemporalMeanPool(name="freq_pool")(features)
    time_features = tf.keras.layers.Conv1D(
        hidden_dim,
        kernel_size=1,
        activation="swish",
        name="time_dense",
    )(time_features)
    time_features = tf.keras.layers.Dropout(dropout_rate, name="time_dropout")(time_features)

    attention_logits = tf.keras.layers.Conv1D(
        model_cfg.num_classes,
        kernel_size=1,
        activation="tanh",
        name="attention_logits",
    )(time_features)
    frame_logits = tf.keras.layers.Conv1D(
        model_cfg.num_classes,
        kernel_size=1,
        activation=None,
        name="frame_logits",
    )(time_features)

    clip_logits = AttentionClipPool(name="clip_logits")((attention_logits, frame_logits))
    window_logits = ResizeWindowLogits(num_windows=num_windows, name="window_logits")(frame_logits)

    return tf.keras.Model(
        inputs=waveform,
        outputs={
            "clip_logits": clip_logits,
            "window_logits": window_logits,
        },
        name="birdclef_sed_b0_teacher",
    )
