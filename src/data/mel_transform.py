"""Mel-spectrogram feature extraction.

Implements the mel parameters from the 2025-1st solution:
  - n_mels=224, n_fft=4096, hop_length=1252 for 20-second chunks
  - 0-1 normalisation, repeated to 3 channels
"""

from __future__ import annotations

import torch
import torchaudio
from torch import Tensor, nn

from core.config import AudioConfig


class MelSpecTransform(nn.Module):
    """Compute log-mel spectrogram and repeat to 3 channels.

    Output shape: (batch, 3, n_mels, time_frames)
    """

    def __init__(self, cfg: AudioConfig | None = None) -> None:
        super().__init__()
        if cfg is None:
            cfg = AudioConfig()
        self.cfg = cfg

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.fmin,
            f_max=cfg.fmax,
            power=2.0,
        )
        self.top_db = torchaudio.transforms.AmplitudeToDB(
            stype="power",
            top_db=cfg.top_db,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Convert waveform to 3-channel log-mel spectrogram.

        Parameters
        ----------
        waveform : (batch, samples) or (samples,)

        Returns
        -------
        mel : (batch, 3, n_mels, time_frames)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        mel = self.mel(waveform)  # (B, n_mels, T)
        mel = self.top_db(mel)  # dB scale

        # 0-1 normalisation per sample
        b = mel.shape[0]
        mel_flat = mel.view(b, -1)
        mn = mel_flat.min(dim=1, keepdim=True).values.unsqueeze(-1)
        mx = mel_flat.max(dim=1, keepdim=True).values.unsqueeze(-1)
        mel = (mel - mn) / (mx - mn + 1e-8)

        # Repeat to 3 channels (RGB-like for pretrained CNN)
        mel = mel.unsqueeze(1).repeat(1, 3, 1, 1)  # (B, 3, n_mels, T)
        return mel


def compute_mel(
    waveform: Tensor,
    cfg: AudioConfig | None = None,
) -> Tensor:
    """Functional wrapper around MelSpecTransform."""
    transform = MelSpecTransform(cfg)
    transform.eval()
    with torch.no_grad():
        return transform(waveform)
