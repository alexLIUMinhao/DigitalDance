from __future__ import annotations

from pathlib import Path
from typing import Any

from ..audio_analysis import (
    build_beat_frames as _build_beat_frames,
    estimate_bpm as _estimate_bpm,
    frame_audio as _frame_audio,
    load_audio as _load_audio,
    local_peak_indices as _local_peak_indices,
    normalize as _normalize,
)
from ..contracts import SongEventMap
from ..utils import percentile, utc_now_iso

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("music-motion-lab music analysis requires numpy.") from exc


def _coerce_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return [round(float(value), 5) for value in values]
    return [round(float(values), 5)]


def _ensure_strictly_increasing(name: str, values: list[float]) -> None:
    for index in range(len(values) - 1):
        if values[index + 1] <= values[index]:
            raise ValueError(f"{name} must be strictly increasing at index {index}: {values[index]} -> {values[index + 1]}")


def _tempo_hypotheses(primary_bpm: float, beat_stability: float) -> list[dict[str, float | str]]:
    safe_bpm = max(primary_bpm, 1.0)
    return [
        {"label": "primary", "bpm": round(safe_bpm, 4), "confidence": round(max(0.05, beat_stability), 4)},
        {"label": "half_time", "bpm": round(max(1.0, safe_bpm / 2.0), 4), "confidence": round(max(0.05, beat_stability * 0.55), 4)},
        {"label": "double_time", "bpm": round(safe_bpm * 2.0, 4), "confidence": round(max(0.05, beat_stability * 0.4), 4)},
    ]


def _section_label(section_index: int, total_sections: int) -> str:
    if total_sections <= 1:
        return "full_song"
    if section_index == 0:
        return "intro"
    if section_index == total_sections - 1:
        return "outro"
    if section_index == total_sections // 2:
        return "chorus"
    return "verse"


def _derive_bpm_from_beats(beat_times: list[float], fallback_bpm: float) -> float:
    if len(beat_times) < 2:
        return fallback_bpm
    intervals = [max(1e-3, beat_times[index + 1] - beat_times[index]) for index in range(len(beat_times) - 1)]
    return 60.0 / max(1e-3, float(np.median(np.asarray(intervals, dtype=np.float32))))


def _build_beat_entries(
    beat_times: list[float],
    onset_envelope: "np.ndarray",
    sample_rate: int,
    hop_size: int,
    downbeats_override: list[float] | None,
    beats_per_bar: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    downbeat_lookup = {round(value, 5) for value in (downbeats_override or [])}
    beat_entries: list[dict[str, Any]] = []
    downbeats: list[dict[str, Any]] = []
    for index, beat_time in enumerate(beat_times):
        envelope_index = min(len(onset_envelope) - 1, int(round((beat_time * sample_rate) / hop_size)))
        strength = float(onset_envelope[max(0, envelope_index)])
        is_downbeat = round(float(beat_time), 5) in downbeat_lookup if downbeat_lookup else bool(index % max(1, beats_per_bar) == 0)
        entry = {
            "index": index,
            "time_sec": round(float(beat_time), 5),
            "strength": round(strength, 5),
            "is_downbeat": is_downbeat,
        }
        beat_entries.append(entry)
        if is_downbeat:
            downbeats.append({"index": len(downbeats), "time_sec": entry["time_sec"], "source_beat_index": index})
    return beat_entries, downbeats


def _build_auto_phrases(
    beat_entries: list[dict[str, Any]],
    beats_per_bar: int,
    phrase_bars: int,
) -> list[dict[str, Any]]:
    beats_per_phrase = max(1, beats_per_bar * phrase_bars)
    phrase_entries: list[dict[str, Any]] = []
    for phrase_start in range(0, len(beat_entries), beats_per_phrase):
        phrase_end = min(len(beat_entries), phrase_start + beats_per_phrase)
        if phrase_end <= phrase_start:
            continue
        phrase_entries.append(
            {
                "index": len(phrase_entries),
                "label": "phrase",
                "start_beat": phrase_start,
                "end_beat_exclusive": phrase_end,
                "start_time_sec": beat_entries[phrase_start]["time_sec"],
                "end_time_sec": beat_entries[phrase_end - 1]["time_sec"],
                "source": "automatic",
            }
        )
    return phrase_entries


def _build_auto_sections(
    phrase_entries: list[dict[str, Any]],
    section_phrases: int,
    beat_stability: float,
) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    phrases_per_section = max(1, section_phrases)
    total_sections = max(1, (len(phrase_entries) + phrases_per_section - 1) // phrases_per_section)
    for start_index in range(0, len(phrase_entries), phrases_per_section):
        end_index = min(len(phrase_entries), start_index + phrases_per_section)
        if end_index <= start_index:
            continue
        section_index = len(sections)
        first_phrase = phrase_entries[start_index]
        last_phrase = phrase_entries[end_index - 1]
        sections.append(
            {
                "index": section_index,
                "label": _section_label(section_index, total_sections),
                "start_beat": first_phrase["start_beat"],
                "end_beat_exclusive": last_phrase["end_beat_exclusive"],
                "start_time_sec": first_phrase["start_time_sec"],
                "end_time_sec": last_phrase["end_time_sec"],
                "confidence": round(max(0.35, beat_stability * 0.7), 5),
                "source": "automatic",
            }
        )
    return sections


def _coerce_range_entries(
    entries: Any,
    beat_entries: list[dict[str, Any]],
    default_label: str,
    include_confidence: bool = False,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    result: list[dict[str, Any]] = []
    beat_count = len(beat_entries)
    for index, raw_entry in enumerate(entries):
        if not isinstance(raw_entry, dict):
            raise ValueError(f"{default_label} override entries must be objects")
        start_beat = int(raw_entry.get("start_beat", raw_entry.get("startBeat", 0)))
        end_beat_exclusive = int(raw_entry.get("end_beat_exclusive", raw_entry.get("endBeatExclusive", 0)))
        if start_beat < 0 or end_beat_exclusive <= start_beat or end_beat_exclusive > beat_count:
            raise ValueError(f"Invalid {default_label} override range at index {index}: {start_beat} -> {end_beat_exclusive}")
        entry = {
            "index": index,
            "label": str(raw_entry.get("label", default_label)),
            "start_beat": start_beat,
            "end_beat_exclusive": end_beat_exclusive,
            "start_time_sec": beat_entries[start_beat]["time_sec"],
            "end_time_sec": beat_entries[end_beat_exclusive - 1]["time_sec"],
            "source": "override",
        }
        if include_confidence:
            entry["confidence"] = round(float(raw_entry.get("confidence", 1.0)), 5)
        result.append(entry)
    return result


def _coerce_accent_entries(
    entries: Any,
    onset_envelope: "np.ndarray",
    sample_rate: int,
    hop_size: int,
) -> list[dict[str, Any]]:
    if not entries:
        return []
    if isinstance(entries, list) and entries and isinstance(entries[0], dict):
        result: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(entries):
            time_sec = round(float(raw_entry.get("time_sec", raw_entry.get("timeSec"))), 5)
            result.append(
                {
                    "index": index,
                    "time_sec": time_sec,
                    "strength": round(float(raw_entry.get("strength", 1.0)), 5),
                    "kind": str(raw_entry.get("kind", "override")),
                }
            )
        _ensure_strictly_increasing("override accents", [entry["time_sec"] for entry in result])
        return result

    accent_times = _coerce_float_list(entries)
    _ensure_strictly_increasing("override accents", accent_times)
    result: list[dict[str, Any]] = []
    for index, time_sec in enumerate(accent_times):
        envelope_index = min(len(onset_envelope) - 1, int(round((time_sec * sample_rate) / hop_size)))
        strength = float(onset_envelope[max(0, envelope_index)])
        result.append(
            {
                "index": index,
                "time_sec": time_sec,
                "strength": round(strength, 5),
                "kind": "override",
            }
        )
    return result


def _meaningful_override_keys(overrides: dict[str, Any]) -> list[str]:
    ignored_keys = {"description"}
    keys: list[str] = []
    for key, value in overrides.items():
        if key in ignored_keys:
            continue
        if value is None:
            continue
        if isinstance(value, (list, dict)) and not value:
            continue
        keys.append(key)
    return sorted(keys)


def analyze_song(
    input_path: Path,
    song_id: str,
    beats_per_bar: int = 4,
    phrase_bars: int = 2,
    section_phrases: int = 2,
    overrides: dict[str, Any] | None = None,
    override_source: str | None = None,
) -> SongEventMap:
    overrides = dict(overrides or {})
    audio, sample_rate = _load_audio(input_path)
    duration_sec = float(audio.size / max(sample_rate, 1))

    frame_size = 2048
    hop_size = 512
    frames = _frame_audio(audio, frame_size=frame_size, hop_size=hop_size)
    window = np.hanning(frame_size).astype(np.float32)
    windowed = frames * window

    rms = np.sqrt(np.mean(np.square(windowed), axis=1))
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    spectral_diff = np.diff(spectrum, axis=0)
    spectral_flux = np.maximum(spectral_diff, 0.0).sum(axis=1)
    onset_envelope = np.concatenate([np.zeros(1, dtype=np.float32), spectral_flux.astype(np.float32)], axis=0)
    onset_envelope = _normalize(onset_envelope + _normalize(rms))

    primary_bpm, spacing_frames, tempo_confidence = _estimate_bpm(onset_envelope, sample_rate=sample_rate, hop_size=hop_size)
    peak_floor = float(np.percentile(onset_envelope, 80)) if onset_envelope.size else 0.0
    peak_indices = _local_peak_indices(onset_envelope, floor=peak_floor)
    beat_frames = _build_beat_frames(peak_indices=peak_indices, spacing_frames=spacing_frames, max_frame=len(onset_envelope))
    auto_beat_times = [(frame * hop_size) / max(sample_rate, 1) for frame in beat_frames]
    if len(auto_beat_times) < 2:
        fallback_spacing = 60.0 / max(primary_bpm, 1.0)
        auto_beat_times = list(np.arange(0.0, max(duration_sec, fallback_spacing), fallback_spacing, dtype=float))

    override_beat_times = _coerce_float_list(overrides.get("beats"))
    if override_beat_times:
        _ensure_strictly_increasing("override beats", override_beat_times)
        beat_times = override_beat_times
    else:
        beat_times = [round(float(value), 5) for value in auto_beat_times]

    beats_per_bar = int(overrides.get("beats_per_bar", beats_per_bar))
    if beats_per_bar <= 0:
        beats_per_bar = 4

    intervals = np.diff(np.asarray(beat_times, dtype=np.float32)) if len(beat_times) >= 2 else np.asarray([60.0 / max(primary_bpm, 1.0)], dtype=np.float32)
    interval_mean = float(intervals.mean()) if intervals.size else max(60.0 / max(primary_bpm, 1.0), 0.5)
    interval_std = float(intervals.std()) if intervals.size else 0.0
    beat_stability = max(0.0, min(1.0, 1.0 - min(1.0, interval_std / max(interval_mean, 1e-6))))
    beat_stability = max(beat_stability, tempo_confidence * 0.7)

    explicit_downbeats = _coerce_float_list(overrides.get("downbeats"))
    if explicit_downbeats:
        _ensure_strictly_increasing("override downbeats", explicit_downbeats)

    if overrides.get("bpm") is not None:
        primary_bpm = float(overrides["bpm"])
    elif override_beat_times:
        primary_bpm = _derive_bpm_from_beats(beat_times, primary_bpm)

    beat_entries, downbeats = _build_beat_entries(
        beat_times=beat_times,
        onset_envelope=onset_envelope,
        sample_rate=sample_rate,
        hop_size=hop_size,
        downbeats_override=explicit_downbeats,
        beats_per_bar=beats_per_bar,
    )

    accent_floor = percentile([float(onset_envelope[index]) for index in peak_indices], 0.7) if peak_indices else 0.0
    auto_accent_entries: list[dict[str, Any]] = []
    for frame_index in peak_indices:
        strength = float(onset_envelope[frame_index])
        if strength < accent_floor:
            continue
        auto_accent_entries.append(
            {
                "index": len(auto_accent_entries),
                "time_sec": round(float((frame_index * hop_size) / max(sample_rate, 1)), 5),
                "strength": round(strength, 5),
                "kind": "accent_peak",
            }
        )
    accent_entries = _coerce_accent_entries(overrides.get("accents"), onset_envelope, sample_rate, hop_size) or auto_accent_entries

    auto_phrase_entries = _build_auto_phrases(beat_entries=beat_entries, beats_per_bar=beats_per_bar, phrase_bars=phrase_bars)
    phrase_entries = _coerce_range_entries(overrides.get("phrases"), beat_entries, default_label="phrase") or auto_phrase_entries

    auto_sections = _build_auto_sections(phrase_entries=phrase_entries, section_phrases=section_phrases, beat_stability=beat_stability)
    sections = _coerce_range_entries(overrides.get("sections"), beat_entries, default_label="verse", include_confidence=True) or auto_sections

    accent_confidence = 0.0
    if accent_entries:
        strengths = [entry["strength"] for entry in accent_entries]
        accent_confidence = max(0.0, min(1.0, percentile(strengths, 0.85) / max(max(strengths), 1e-6)))
    if override_beat_times:
        beat_stability = 1.0
    if overrides.get("accents"):
        accent_confidence = 1.0
    phrase_confidence = 1.0 if overrides.get("phrases") else (0.45 if phrase_entries else 0.0)
    section_confidence = 1.0 if overrides.get("sections") else (0.4 if sections else 0.0)
    overall_confidence = round((beat_stability + accent_confidence + phrase_confidence + section_confidence) / 4.0, 5)

    automatic_manual_review = overall_confidence < 0.68 or primary_bpm < 95.0
    if overrides.get("manual_review_required") is not None:
        manual_review_required = bool(overrides.get("manual_review_required"))
    elif override_beat_times or explicit_downbeats:
        manual_review_required = False
    else:
        manual_review_required = automatic_manual_review

    applied_override_keys = _meaningful_override_keys(overrides)
    analysis_mode = "manual_override" if applied_override_keys else "automatic"

    notes = [
        "Layered analysis is heuristic-first in v1 and uses numpy-based onset, tempo, and phrase estimates.",
    ]
    if applied_override_keys:
        notes.append(f"Applied manual overrides: {', '.join(applied_override_keys)}.")
    if manual_review_required:
        notes.append("Manual review is recommended before using this song for automatic choreography.")

    return SongEventMap(
        schema_version=1,
        song_id=song_id,
        source_audio_path=str(input_path),
        duration_sec=round(duration_sec, 5),
        beats_per_bar=beats_per_bar,
        tempo_hypotheses=_tempo_hypotheses(primary_bpm, beat_stability),
        beats=beat_entries,
        downbeats=downbeats,
        accents=accent_entries,
        phrases=phrase_entries,
        sections=sections,
        confidence={
            "beat_stability": round(beat_stability, 5),
            "accent_clarity": round(accent_confidence, 5),
            "phrase_confidence": round(phrase_confidence, 5),
            "section_confidence": round(section_confidence, 5),
            "overall": overall_confidence,
        },
        manual_review_required=manual_review_required,
        analysis_mode=analysis_mode,
        override_source=override_source,
        applied_override_keys=applied_override_keys,
        notes=notes,
        generated_at_utc=utc_now_iso(),
    )
