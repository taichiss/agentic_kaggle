from __future__ import annotations

import html
import io
import struct
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram

from core.audio_eda_types import AudioInfo


def inspect_audio(path: Path) -> AudioInfo:
    info = sf.info(str(path))
    duration_sec = 0.0 if info.samplerate <= 0 else info.frames / info.samplerate
    return AudioInfo(
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        duration_sec=duration_sec,
    )


def read_audio_segment(
    path: Path,
    start_sec: float = 0.0,
    duration_sec: float | None = None,
) -> tuple[np.ndarray, int]:
    info = inspect_audio(path)
    safe_start = max(start_sec, 0.0)
    start_frame = min(int(safe_start * info.sample_rate), info.frames)
    remaining_frames = max(info.frames - start_frame, 0)
    if duration_sec is None:
        frame_count = remaining_frames
    else:
        frame_count = min(int(max(duration_sec, 0.0) * info.sample_rate), remaining_frames)
    if frame_count <= 0:
        raise ValueError("selected audio segment is empty")

    audio, sample_rate = sf.read(
        str(path),
        start=start_frame,
        frames=frame_count,
        dtype="float32",
        always_2d=True,
    )
    return audio, sample_rate


def render_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def render_waveform_svg(audio: np.ndarray, width: int = 960, height: int = 220) -> bytes:
    mono = _to_mono(audio)
    bucket_count = max(32, min(width, mono.shape[0]))
    buckets = np.array_split(mono, bucket_count)
    mins = np.array([bucket.min() if bucket.size else 0.0 for bucket in buckets], dtype=np.float32)
    maxs = np.array([bucket.max() if bucket.size else 0.0 for bucket in buckets], dtype=np.float32)

    mid_y = height / 2
    scale_y = max(height * 0.45, 1.0)
    lines: list[str] = []
    for idx, (min_value, max_value) in enumerate(zip(mins, maxs, strict=True)):
        x = idx * width / max(bucket_count - 1, 1)
        y1 = mid_y - max_value * scale_y
        y2 = mid_y - min_value * scale_y
        lines.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="{y1:.2f}" y2="{y2:.2f}" '
            'stroke="#0f172a" stroke-width="1.2" stroke-linecap="round" />'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="waveform">'
        '<rect width="100%" height="100%" fill="#f8fafc" rx="16" ry="16" />'
        f'<line x1="0" x2="{width}" y1="{mid_y:.2f}" y2="{mid_y:.2f}" stroke="#cbd5e1" />'
        f"{''.join(lines)}"
        "</svg>"
    )
    return svg.encode("utf-8")


def render_spectrogram_png(audio: np.ndarray, sample_rate: int) -> bytes:
    mono = _to_mono(audio)
    _, spec = _spectrogram_matrix(mono, sample_rate)
    if spec.size == 0:
        spec = np.zeros((1, 1), dtype=np.float32)

    spec_db = 20.0 * np.log10(spec + 1e-6)
    lo = float(np.percentile(spec_db, 5))
    hi = float(np.percentile(spec_db, 99))
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((spec_db - lo) / (hi - lo), 0.0, 1.0)
    image = np.flipud(_apply_colormap(scaled))
    image = _downsample_image(image, max_width=960, max_height=320)
    return _encode_png(image)


def render_frequency_profile_svg(
    audio: np.ndarray,
    sample_rate: int,
    width: int = 960,
    height: int = 220,
) -> bytes:
    analysis = compute_spectrum_analysis(audio, sample_rate)
    frequencies = np.asarray(analysis["frequencies_hz"], dtype=np.float32)
    power_db = np.asarray(analysis["mean_power_db"], dtype=np.float32)
    if frequencies.size == 0 or power_db.size == 0:
        frequencies = np.array([0.0, 1.0], dtype=np.float32)
        power_db = np.array([0.0, 0.0], dtype=np.float32)

    lo = float(power_db.min())
    hi = float(power_db.max())
    if hi <= lo:
        hi = lo + 1.0
    x_scale = max(float(frequencies[-1]), 1.0)
    points = []
    for frequency, value in zip(frequencies, power_db, strict=True):
        x = (frequency / x_scale) * width
        y = height - (((value - lo) / (hi - lo)) * (height - 30) + 15)
        points.append(f"{x:.2f},{y:.2f}")

    band_markers = []
    for boundary in (2_000.0, 6_000.0):
        if boundary >= x_scale:
            continue
        x = (boundary / x_scale) * width
        band_markers.append(
            f'<line x1="{x:.2f}" x2="{x:.2f}" y1="0" y2="{height}" '
            'stroke="#cbd5e1" stroke-dasharray="4 5" />'
        )

    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="frequency profile">'
        '<rect width="100%" height="100%" fill="#fffdf9" rx="16" ry="16" />'
        f"{''.join(band_markers)}"
        f'<polyline fill="none" stroke="#b45309" stroke-width="3" points="{" ".join(points)}" />'
        '<text x="16" y="24" fill="#6b7280" font-size="12">0 Hz</text>'
        f'<text x="{width - 84}" y="24" fill="#6b7280" font-size="12">{int(x_scale)} Hz</text>'
        "</svg>"
    )
    return svg.encode("utf-8")


def render_species_distribution_svg(
    species_stats: list[dict[str, object]],
    summary: dict[str, object] | None = None,
    *,
    width: int = 1_560,
    top_n: int = 18,
    title: str = "BirdCLEF species distribution",
    subtitle: str | None = None,
) -> bytes:
    items = list(species_stats)
    outer_margin = 36
    panel_gap = 24
    panel_width = int((width - outer_margin * 2 - panel_gap * 2) / 3)
    row_height = 24
    panel_height = 120 + row_height * max(top_n, 1)
    header_height = 188
    height = header_height + panel_height + 48

    train_audio_species = sum(1 for item in items if _stat_int(item, "train_audio_recordings") > 0)
    train_audio_recordings = sum(_stat_int(item, "train_audio_recordings") for item in items)
    soundscape_species = sum(1 for item in items if _stat_int(item, "soundscape_window_count") > 0)
    soundscape_windows = sum(_stat_int(item, "soundscape_window_count") for item in items)
    labeled_soundscape_files = 0
    if summary is not None:
        soundscape_windows = _stat_int(summary, "soundscape_windows")
        labeled_soundscape_files = _stat_int(summary, "labeled_soundscape_files")

    effective_subtitle = subtitle or "train_audio と labeled soundscape の class 偏りを比較"
    card_y = 92
    card_height = 72
    card_width = int((width - outer_margin * 2 - panel_gap * 2) / 3)
    card_specs = [
        (
            outer_margin,
            "#0f766e",
            "train_audio coverage",
            f"{train_audio_species} species",
            f"{train_audio_recordings:,} recordings",
        ),
        (
            outer_margin + card_width + panel_gap,
            "#b45309",
            "soundscape coverage",
            f"{soundscape_species} species",
            f"{soundscape_windows:,} labeled windows",
        ),
        (
            outer_margin + (card_width + panel_gap) * 2,
            "#475569",
            "labeled files",
            f"{labeled_soundscape_files} files",
            "deduplicated train_soundscapes subset",
        ),
    ]

    panel_specs = [
        (
            "train_audio recordings",
            "train_audio_recordings",
            "#0f766e",
            "train.csv focal recordings",
        ),
        (
            "soundscape windows",
            "soundscape_window_count",
            "#b45309",
            "deduplicated labeled windows",
        ),
        (
            "soundscape files",
            "soundscape_file_count",
            "#475569",
            "files that contain the label",
        ),
    ]

    svg_parts = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" role="img" '
            'aria-label="species distribution">'
        ),
        '<rect width="100%" height="100%" fill="#f8fafc" rx="28" ry="28" />',
        '<rect x="0" y="0" width="100%" height="164" fill="#eff6ff" rx="28" ry="28" />',
        (
            '<text x="36" y="46" fill="#0f172a" font-size="28" '
            'font-family="Helvetica, Arial, sans-serif" font-weight="700">'
            f"{html.escape(title)}"
            "</text>"
        ),
        (
            '<text x="36" y="72" fill="#475569" font-size="14" '
            'font-family="Helvetica, Arial, sans-serif">'
            f"{html.escape(effective_subtitle)}"
            "</text>"
        ),
    ]

    for card_x, accent, label, value, detail in card_specs:
        svg_parts.append(
            _render_summary_card(
                x=card_x,
                y=card_y,
                width=card_width,
                height=card_height,
                accent=accent,
                label=label,
                value=value,
                detail=detail,
            )
        )

    panel_y = header_height
    for index, (panel_title, metric, accent, note) in enumerate(panel_specs):
        panel_x = outer_margin + index * (panel_width + panel_gap)
        svg_parts.extend(
            _render_species_distribution_panel(
                items=items,
                x=panel_x,
                y=panel_y,
                width=panel_width,
                height=panel_height,
                title=panel_title,
                metric=metric,
                accent=accent,
                note=note,
                top_n=top_n,
                row_height=row_height,
            )
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts).encode("utf-8")


def compute_spectrum_analysis(audio: np.ndarray, sample_rate: int) -> dict[str, object]:
    mono = _to_mono(audio)
    frequencies, spec = _spectrogram_matrix(mono, sample_rate)
    if spec.size == 0:
        frequencies = np.array([0.0], dtype=np.float32)
        spec = np.zeros((1, 1), dtype=np.float32)

    mean_power = spec.mean(axis=1)
    power_sum = float(mean_power.sum()) + 1e-12
    centroid = float(np.sum(frequencies * mean_power) / power_sum)
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * mean_power) / power_sum))
    cumulative = np.cumsum(mean_power)
    rolloff_idx = int(np.searchsorted(cumulative, power_sum * 0.85))
    rolloff_idx = min(rolloff_idx, len(frequencies) - 1)
    dominant_idx = int(np.argmax(mean_power))
    dominant_frequency = float(frequencies[dominant_idx])

    rms = float(np.sqrt(np.mean(mono**2))) if mono.size else 0.0
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    zero_crossings = 0.0
    if mono.size >= 2:
        zero_crossings = float(np.mean(np.abs(np.diff(np.signbit(mono)).astype(np.float32))))
    flatness = float(
        np.exp(np.mean(np.log(mean_power + 1e-12))) / (float(np.mean(mean_power)) + 1e-12)
    )
    mean_power_db = 10.0 * np.log10(mean_power + 1e-12)

    band_specs = (
        ("0-2 kHz", 0.0, 2_000.0),
        ("2-6 kHz", 2_000.0, 6_000.0),
        ("6-16 kHz", 6_000.0, 16_000.0),
    )
    band_energy_share_raw: list[tuple[str, float]] = []
    for label, lo, hi in band_specs:
        mask = (frequencies >= lo) & (frequencies < hi)
        share = 0.0 if not np.any(mask) else float(mean_power[mask].sum() / power_sum)
        band_energy_share_raw.append((label, share))

    dominant_band_label, _ = max(band_energy_share_raw, key=lambda item: item[1])
    tendency_tags = []
    if dominant_band_label == "0-2 kHz":
        tendency_tags.append("低域優勢")
    elif dominant_band_label == "6-16 kHz":
        tendency_tags.append("高域優勢")
    else:
        tendency_tags.append("中域優勢")
    if flatness < 0.3:
        tendency_tags.append("トーナル")
    elif flatness > 0.6:
        tendency_tags.append("ブロードバンド")
    if bandwidth > 3_000.0:
        tendency_tags.append("広帯域")

    return {
        "rms_db": _safe_db(rms),
        "peak_db": _safe_db(peak),
        "spectral_centroid_hz": centroid,
        "rolloff_85_hz": float(frequencies[rolloff_idx]),
        "bandwidth_hz": bandwidth,
        "dominant_frequency_hz": dominant_frequency,
        "spectral_flatness": flatness,
        "zero_crossing_rate": zero_crossings,
        "band_energy_share": [
            {"label": label, "share": share} for label, share in band_energy_share_raw
        ],
        "tendency_tags": tendency_tags,
        "frequencies_hz": frequencies.round(2).tolist(),
        "mean_power_db": mean_power_db.round(3).tolist(),
    }


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return np.asarray(audio, dtype=np.float32)
    return np.asarray(audio.mean(axis=1), dtype=np.float32)


def _spectrogram_matrix(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    mono = np.asarray(audio, dtype=np.float32)
    if mono.shape[0] < 128:
        mono = np.pad(mono, (0, 128 - mono.shape[0]))
    nperseg = min(1024, max(128, mono.shape[0] // 8))
    noverlap = min(nperseg - 1, int(nperseg * 0.75))
    freqs, _, spec = spectrogram(
        mono,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    max_frequency = min(float(sample_rate) / 2.0, 16_000.0)
    mask = freqs <= max_frequency
    return freqs[mask].astype(np.float32), spec[mask].astype(np.float32)


def _apply_colormap(values: np.ndarray) -> np.ndarray:
    red = np.clip(255.0 * values**0.55, 0.0, 255.0)
    green = np.clip(255.0 * np.sqrt(values), 0.0, 255.0)
    blue = np.clip(255.0 * (1.0 - values) ** 1.4, 0.0, 255.0)
    return np.stack([red, green, blue], axis=2).astype(np.uint8)


def _downsample_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    height, width, _ = image.shape
    row_index = np.linspace(0, height - 1, num=min(height, max_height), dtype=np.int32)
    col_index = np.linspace(0, width - 1, num=min(width, max_width), dtype=np.int32)
    return image[row_index][:, col_index]


def _encode_png(image: np.ndarray) -> bytes:
    height, width, _ = image.shape
    raw_rows = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    payload = zlib.compress(raw_rows, level=9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", payload)
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(tag)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc & 0xFFFFFFFF)


def _safe_db(value: float) -> float:
    return float(20.0 * np.log10(max(value, 1e-12)))


def _render_summary_card(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    accent: str,
    label: str,
    value: str,
    detail: str,
) -> str:
    return (
        f'<g transform="translate({x},{y})">'
        f'<rect width="{width}" height="{height}" fill="#ffffff" stroke="#dbeafe" '
        'stroke-width="1.2" rx="18" ry="18" />'
        f'<rect x="18" y="18" width="10" height="{height - 36}" fill="{accent}" rx="5" ry="5" />'
        '<text x="44" y="28" fill="#64748b" font-size="12" '
        'font-family="Helvetica, Arial, sans-serif">'
        f"{html.escape(label)}"
        "</text>"
        '<text x="44" y="52" fill="#0f172a" font-size="24" '
        'font-family="Helvetica, Arial, sans-serif" font-weight="700">'
        f"{html.escape(value)}"
        "</text>"
        '<text x="44" y="66" fill="#475569" font-size="12" '
        'font-family="Helvetica, Arial, sans-serif">'
        f"{html.escape(detail)}"
        "</text>"
        "</g>"
    )


def _render_species_distribution_panel(
    *,
    items: list[dict[str, object]],
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    metric: str,
    accent: str,
    note: str,
    top_n: int,
    row_height: int,
) -> list[str]:
    active_count = sum(1 for item in items if _stat_int(item, metric) > 0)
    ranked_items = _top_species_by_metric(items, metric=metric, top_n=top_n)
    max_value = max((_stat_int(item, metric) for item in ranked_items), default=1)
    label_width = 200
    value_width = 56
    bar_x = x + label_width
    bar_width = max(width - label_width - value_width - 36, 40)
    rows_y = y + 76

    parts = [
        f'<g transform="translate({x},{y})">',
        f'<rect width="{width}" height="{height}" fill="#ffffff" stroke="#e2e8f0" '
        'stroke-width="1.2" rx="22" ry="22" />',
        (
            f'<rect x="22" y="22" width="{width - 44}" height="6" fill="{accent}" '
            'rx="3" ry="3" opacity="0.9" />'
        ),
        '<text x="22" y="50" fill="#0f172a" font-size="20" '
        'font-family="Helvetica, Arial, sans-serif" font-weight="700">'
        f"{html.escape(title)}"
        "</text>",
        '<text x="22" y="68" fill="#475569" font-size="12" '
        'font-family="Helvetica, Arial, sans-serif">'
        f"{html.escape(note)}"
        "</text>",
        '<text x="22" y="92" fill="#64748b" font-size="12" '
        'font-family="Helvetica, Arial, sans-serif">'
        f"{html.escape(f'top {len(ranked_items)} / {active_count} active species')}"
        "</text>",
    ]
    if not ranked_items:
        parts.extend(
            [
                '<text x="22" y="124" fill="#94a3b8" font-size="13" '
                'font-family="Helvetica, Arial, sans-serif">'
                "no observations"
                "</text>",
                "</g>",
            ]
        )
        return parts

    for index, item in enumerate(ranked_items):
        value = _stat_int(item, metric)
        fill_width = max(2.0, bar_width * (value / max_value))
        row_top = rows_y + index * row_height
        label = _truncate_text(_species_display_name(item), limit=34)
        parts.extend(
            [
                (
                    '<text x="22" '
                    f'y="{row_top + 13}" fill="#334155" font-size="12" '
                    'font-family="Helvetica, Arial, sans-serif">'
                    f"{html.escape(label)}"
                    "</text>"
                ),
                (
                    f'<rect x="{bar_x}" y="{row_top}" width="{bar_width}" height="14" '
                    'fill="#e2e8f0" rx="7" ry="7" />'
                ),
                (
                    f'<rect x="{bar_x}" y="{row_top}" width="{fill_width:.2f}" height="14" '
                    f'fill="{accent}" rx="7" ry="7" opacity="0.88" />'
                ),
                (
                    f'<text x="{bar_x + bar_width + 10}" y="{row_top + 12}" '
                    'fill="#0f172a" font-size="12" '
                    'font-family="Helvetica, Arial, sans-serif" text-anchor="start">'
                    f"{value:,}"
                    "</text>"
                ),
            ]
        )
    parts.append("</g>")
    return parts


def _top_species_by_metric(
    items: list[dict[str, object]],
    *,
    metric: str,
    top_n: int,
) -> list[dict[str, object]]:
    positive_items = [item for item in items if _stat_int(item, metric) > 0]
    return sorted(
        positive_items,
        key=lambda item: (_stat_int(item, metric), _species_display_name(item)),
        reverse=True,
    )[:top_n]


def _species_display_name(item: dict[str, object]) -> str:
    label = str(item.get("primary_label", "unknown"))
    common_name = str(item.get("common_name", "")).strip()
    if not common_name or common_name == label:
        return label
    return f"{label} / {common_name}"


def _stat_int(item: dict[str, object], key: str) -> int:
    raw_value = item.get(key, 0)
    if isinstance(raw_value, bool):
        return int(raw_value)
    if isinstance(raw_value, (int, float)):
        return int(raw_value)
    return int(str(raw_value))


def _truncate_text(value: str, *, limit: int) -> str:
    if len(value) <= limit:
        return value
    return f"{value[: max(limit - 1, 1)]}…"
