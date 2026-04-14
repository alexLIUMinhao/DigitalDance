from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import numpy as np
    import soundfile as sf
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("music-motion-lab audio analysis requires numpy and soundfile.") from exc


DEFAULT_FRAME_SIZE = 2048
DEFAULT_HOP_SIZE = 512

DEFAULT_BAND_RANGES_HZ: dict[str, tuple[float, float]] = {
    "low": (40.0, 180.0),
    "low_mid": (180.0, 500.0),
    "high_attack": (2000.0, 6000.0),
}


def load_audio(input_path: Path) -> tuple["np.ndarray", int]:
    audio, sample_rate = sf.read(str(input_path), always_2d=False)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"Audio file is empty: {input_path}")
    return audio, int(sample_rate)


def frame_audio(audio: "np.ndarray", frame_size: int, hop_size: int) -> "np.ndarray":
    if audio.size < frame_size:
        padding = np.zeros(frame_size - audio.size, dtype=np.float32)
        audio = np.concatenate([audio, padding], axis=0)
    frame_count = 1 + max(0, (audio.size - frame_size) // hop_size)
    frames = np.zeros((frame_count, frame_size), dtype=np.float32)
    for index in range(frame_count):
        start = index * hop_size
        frames[index, :] = audio[start:start + frame_size]
    return frames


def normalize(values: "np.ndarray") -> "np.ndarray":
    if values.size == 0:
        return values.astype(np.float32)
    peak = float(np.max(values))
    if peak <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return (values / peak).astype(np.float32)


def local_peak_indices(values: "np.ndarray", floor: float) -> list[int]:
    if values.size < 3:
        return []
    peak_indices: list[int] = []
    for index in range(1, len(values) - 1):
        if values[index] >= floor and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            peak_indices.append(index)
    return peak_indices


def estimate_bpm(onset_envelope: "np.ndarray", sample_rate: int, hop_size: int) -> tuple[float, int, float]:
    min_bpm = 60.0
    max_bpm = 180.0
    min_lag = max(1, int(round((60.0 / max_bpm) * sample_rate / hop_size)))
    max_lag = max(min_lag + 1, int(round((60.0 / min_bpm) * sample_rate / hop_size)))

    centered = onset_envelope.astype(np.float32) - float(onset_envelope.mean())
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    window = autocorr[min_lag:max_lag + 1]
    if window.size == 0 or float(window.max()) <= 1e-8:
        fallback_bpm = 100.0
        fallback_lag = max(1, int(round((60.0 / fallback_bpm) * sample_rate / hop_size)))
        return fallback_bpm, fallback_lag, 0.15

    best_offset = int(window.argmax())
    best_lag = min_lag + best_offset
    bpm = 60.0 * sample_rate / max(best_lag * hop_size, 1)
    confidence = float(window[best_offset] / max(float(autocorr[0]), 1e-8))
    return float(bpm), int(best_lag), max(0.05, min(1.0, confidence))


def build_beat_frames(
    peak_indices: list[int],
    spacing_frames: int,
    max_frame: int,
    start_frame: int = 0,
) -> list[int]:
    if spacing_frames <= 0:
        spacing_frames = 1
    if not peak_indices:
        return list(range(max(0, int(start_frame)), max_frame, spacing_frames))

    peak_set = np.asarray(peak_indices, dtype=np.int32)
    beat_frames: list[int] = []
    current = max(0, int(start_frame))
    tolerance = max(1, spacing_frames // 3)
    while current < max_frame:
        nearest_index = int(np.argmin(np.abs(peak_set - current)))
        candidate = int(peak_set[nearest_index])
        beat_frames.append(candidate if abs(candidate - current) <= tolerance else current)
        current += spacing_frames

    deduped: list[int] = []
    for frame in beat_frames:
        if not deduped or frame > deduped[-1]:
            deduped.append(frame)
    return deduped


def best_beat_offset(onset_envelope: "np.ndarray", spacing_frames: int) -> int:
    if spacing_frames <= 1 or onset_envelope.size == 0:
        return 0
    best_offset = 0
    best_score = -1.0
    max_offset = min(spacing_frames, onset_envelope.size)
    for offset in range(max_offset):
        grid_values = onset_envelope[offset::spacing_frames]
        score = float(grid_values.sum()) if grid_values.size else 0.0
        if score > best_score:
            best_score = score
            best_offset = offset
    return int(best_offset)


def build_frame_analysis(
    audio: "np.ndarray",
    sample_rate: int,
    frame_size: int = DEFAULT_FRAME_SIZE,
    hop_size: int = DEFAULT_HOP_SIZE,
    low_band_range_hz: tuple[float, float] = (40.0, 180.0),
) -> dict[str, Any]:
    frames = frame_audio(audio, frame_size=frame_size, hop_size=hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window

    rms = np.sqrt(np.mean(np.square(windowed), axis=1)).astype(np.float32)
    rms_norm = normalize(rms)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1)).astype(np.float32)
    spectral_diff = np.diff(spectrum, axis=0)
    spectral_flux = np.maximum(spectral_diff, 0.0).sum(axis=1)
    onset_envelope = np.concatenate([np.zeros(1, dtype=np.float32), spectral_flux.astype(np.float32)], axis=0)
    onset_envelope = normalize(onset_envelope + rms_norm)

    frequencies = np.fft.rfftfreq(frame_size, d=1.0 / max(sample_rate, 1))
    low_band_mask = (frequencies >= low_band_range_hz[0]) & (frequencies <= low_band_range_hz[1])
    if bool(low_band_mask.any()):
        low_band_envelope = normalize(spectrum[:, low_band_mask].mean(axis=1).astype(np.float32))
    else:
        low_band_envelope = np.zeros((spectrum.shape[0],), dtype=np.float32)

    band_envelopes: dict[str, np.ndarray] = {}
    for band_name, band_range in DEFAULT_BAND_RANGES_HZ.items():
        band_mask = (frequencies >= band_range[0]) & (frequencies <= band_range[1])
        if bool(band_mask.any()):
            band_envelopes[band_name] = normalize(spectrum[:, band_mask].mean(axis=1).astype(np.float32))
        else:
            band_envelopes[band_name] = np.zeros((spectrum.shape[0],), dtype=np.float32)

    frame_times_sec = (np.arange(len(rms_norm), dtype=np.float32) * float(hop_size)) / max(float(sample_rate), 1.0)
    return {
        "frame_size": int(frame_size),
        "hop_size": int(hop_size),
        "rms_norm": rms_norm,
        "spectrum": spectrum,
        "onset_envelope": onset_envelope,
        "low_band_envelope": low_band_envelope,
        "low_mid_envelope": band_envelopes["low_mid"].astype(np.float32),
        "high_attack_envelope": band_envelopes["high_attack"].astype(np.float32),
        "band_envelopes": {name: values.astype(np.float32) for name, values in band_envelopes.items()},
        "frame_times_sec": frame_times_sec.astype(np.float32),
        "duration_sec": float(audio.size / max(sample_rate, 1)),
        "sample_rate": int(sample_rate),
    }


def hz_to_mel(values_hz: "np.ndarray") -> "np.ndarray":
    return 2595.0 * np.log10(1.0 + (values_hz / 700.0))


def mel_to_hz(values_mel: "np.ndarray") -> "np.ndarray":
    return 700.0 * (np.power(10.0, values_mel / 2595.0) - 1.0)


def build_mel_filter_bank(
    sample_rate: int,
    frame_size: int,
    mel_bins: int,
    min_freq_hz: float = 30.0,
    max_freq_hz: float | None = None,
) -> "np.ndarray":
    nyquist = sample_rate / 2.0
    max_freq_hz = min(float(max_freq_hz or nyquist), nyquist)
    min_freq_hz = max(0.0, min(min_freq_hz, max_freq_hz))

    mel_points = np.linspace(
        float(hz_to_mel(np.asarray([min_freq_hz], dtype=np.float32))[0]),
        float(hz_to_mel(np.asarray([max_freq_hz], dtype=np.float32))[0]),
        mel_bins + 2,
        dtype=np.float32,
    )
    hz_points = mel_to_hz(mel_points)
    fft_bins = np.floor((frame_size + 1) * hz_points / max(sample_rate, 1)).astype(int)
    freq_bin_count = frame_size // 2 + 1
    filter_bank = np.zeros((mel_bins, freq_bin_count), dtype=np.float32)
    for band_index in range(mel_bins):
        left = max(0, min(freq_bin_count - 1, int(fft_bins[band_index])))
        center = max(left + 1, min(freq_bin_count - 1, int(fft_bins[band_index + 1])))
        right = max(center + 1, min(freq_bin_count, int(fft_bins[band_index + 2])))
        for freq_index in range(left, center):
            filter_bank[band_index, freq_index] = (freq_index - left) / max(center - left, 1)
        for freq_index in range(center, right):
            filter_bank[band_index, freq_index] = (right - freq_index) / max(right - center, 1)
    return filter_bank


def compute_mel_spectrogram(
    spectrum: "np.ndarray",
    sample_rate: int,
    frame_size: int,
    mel_bins: int = 48,
    min_freq_hz: float = 30.0,
    max_freq_hz: float | None = None,
) -> "np.ndarray":
    if spectrum.size == 0:
        return np.zeros((mel_bins, 0), dtype=np.float32)
    power_spectrum = np.square(np.asarray(spectrum, dtype=np.float32))
    filter_bank = build_mel_filter_bank(
        sample_rate=sample_rate,
        frame_size=frame_size,
        mel_bins=mel_bins,
        min_freq_hz=min_freq_hz,
        max_freq_hz=max_freq_hz,
    )
    mel = power_spectrum @ filter_bank.T
    mel = np.log1p(np.maximum(mel, 0.0))
    return mel.T.astype(np.float32)
