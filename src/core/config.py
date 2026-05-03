"""Experiment configuration dataclasses.

Centralises all hyper-parameters so that every script shares the same
schema and experiments are trivially reproducible from a YAML / dict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


# ---------------------------------------------------------------------------
# Audio / Mel config
# ---------------------------------------------------------------------------
@dataclass
class AudioConfig:
    """Raw audio and mel-spectrogram parameters."""

    sample_rate: int = 32_000
    chunk_duration_sec: float = 20.0  # 2025-1st: 20 s is optimal
    n_mels: int = 224
    fmin: int = 0
    fmax: int = 16_000
    n_fft: int = 4096
    hop_length: int = 1252  # yields ~512 frames for 20 s @ 32 kHz
    top_db: float = 80.0

    # Derived
    @property
    def chunk_samples(self) -> int:
        return int(self.sample_rate * self.chunk_duration_sec)


# ---------------------------------------------------------------------------
# Augmentation config
# ---------------------------------------------------------------------------
@dataclass
class AugConfig:
    """Data augmentation hyper-parameters."""

    mixup_prob: float = 0.5
    mixup_alpha: float = 1.0  # Beta(alpha, alpha); inf → weight=0.5
    sumix_prob: float = 0.0
    spec_aug_freq_mask: int = 20
    spec_aug_time_mask: int = 40
    filter_augment_prob: float = 0.3
    drop_path_rate: float = 0.0  # Stochastic Depth (>0 for self-training)


# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """SED model hyper-parameters."""

    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"
    pretrained: bool = True
    num_classes: int = 234
    in_channels: int = 3  # repeated mel channels
    sed_pool: Literal["gem", "avg", "max"] = "gem"


# ---------------------------------------------------------------------------
# Distillation config
# ---------------------------------------------------------------------------
@dataclass
class DistillConfig:
    """Perch teacher supervision for 20-second weakly-labeled clips."""

    enabled: bool = False
    teacher_window_sec: float = 5.0
    teacher_windows_per_clip: int = 4
    teacher_loss_weight: float = 0.3
    teacher_cache_path: str = "data/models/train_audio_perch_chunks_v1.npz"
    mapped_classes_only: bool = True

    @property
    def clip_duration_sec(self) -> float:
        return self.teacher_window_sec * self.teacher_windows_per_clip


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """Training loop hyper-parameters."""

    epochs: int = 15
    batch_size: int = 64
    lr: float = 5e-4
    lr_min: float = 1e-6
    weight_decay: float = 1e-4
    optimizer: Literal["adamw", "radam"] = "adamw"
    scheduler: Literal["cosine", "cosine_warm_restarts"] = "cosine_warm_restarts"
    cosine_restart_period: int = 5
    loss: Literal["ce", "focal", "bce"] = "ce"
    focal_gamma: float = 2.0
    label_smoothing: float = 0.0
    mixed_precision: bool = True
    gradient_clip_norm: float = 0.0  # 0 = disabled
    num_workers: int = 4
    seed: int = 42


# ---------------------------------------------------------------------------
# Self-training config
# ---------------------------------------------------------------------------
@dataclass
class SelfTrainConfig:
    """Iterative Noisy Student pseudo-labeling."""

    enabled: bool = False
    pseudo_ratio: float = 1.0  # fraction of batch mixed with pseudo
    power_transform: float = 1.0  # >1 to suppress noise
    confidence_threshold: float = 0.5  # min max-prob per soundscape
    low_prob_cutoff: float = 0.1  # zero-out below this
    iterations: int = 1


# ---------------------------------------------------------------------------
# CV config
# ---------------------------------------------------------------------------
@dataclass
class CVConfig:
    """Cross-validation split configuration."""

    strategy: Literal["file_gkf", "site_balanced", "site_holdout"] = "site_balanced"
    n_folds: int = 3


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    """Full experiment specification."""

    name: str = "sed_baseline"
    audio: AudioConfig = field(default_factory=AudioConfig)
    aug: AugConfig = field(default_factory=AugConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    distill: DistillConfig = field(default_factory=DistillConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    self_train: SelfTrainConfig = field(default_factory=SelfTrainConfig)
    cv: CVConfig = field(default_factory=CVConfig)

    # Paths (relative to project root)
    data_dir: str = "data/input/BirdCLEF+ 2026"
    output_dir: str = "output"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir) / self.name
