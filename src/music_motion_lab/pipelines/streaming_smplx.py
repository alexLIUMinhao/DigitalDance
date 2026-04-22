from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..utils import slugify, stable_digest, utc_now_iso, write_json
from .smplx_mesh_stitch_renderer import build_smplx_mesh_stitch_visual_preview


DEFAULT_INITIAL_BUFFER_SEC = 2.0
DEFAULT_LOOKAHEAD_SEC = 2.0
DEFAULT_LOOKFRONT_SEC = 1.0
DEFAULT_CHUNK_MS = 46.44
DEFAULT_ROLLING_WINDOW_SEC = 8.0
DEFAULT_TEMPO_WINDOW_SEC = 16.0
DEFAULT_STREAM_FPS = 30
DEFAULT_BLEND_FRAMES = 10
DEFAULT_SAFE_RETIME_MIN = 0.85
DEFAULT_SAFE_RETIME_MAX = 1.15
DEFAULT_M15_COHORT_SIZE = 10
DEFAULT_INITIAL_POSE_MODE = "neutral_idle"
SYNTHETIC_NEUTRAL_IDLE_SOURCE = "smplx_neutral_idle"
SYNTHETIC_FREEZE_FIRST_SOURCE = "smplx_freeze_first"
SYNTHETIC_NEUTRAL_REST_SOURCE = "smplx_neutral_rest"
SYNTHETIC_SOURCE_SEQUENCE = "__synthetic__"
M15_LOW_CONFIDENCE_THRESHOLD = 0.38
M15_MIN_SEQUENCE_DWELL_STEPS = 2
M15_SEQUENCE_SWITCH_MARGIN = 0.055
M15_CROSS_SEQUENCE_TRANSITION_FLOOR = 0.62
M15_NON_TAIL_SPEED_MIN = 0.90
M15_NON_TAIL_SPEED_MAX = 1.10
M15_RHYTHM_HARD_MIN = 0.48
M15_TRANSITION_HARD_MIN = 0.50
M15_LOW_CONFIDENCE_FALLBACK_BEATS = 4
M17_LOW_CONFIDENCE_THRESHOLD = 0.55
M17_NON_TAIL_SPEED_MIN = 0.92
M17_NON_TAIL_SPEED_MAX = 1.08
M17_RHYTHM_HARD_MIN = 0.55
M17_TRANSITION_HARD_MIN = 0.55
M17_LOW_CONFIDENCE_FALLBACK_BEATS = 4
M17_TEMPO_HYSTERESIS_MARGIN = 0.12
M17_TEMPO_STABLE_SWITCH_TICKS = 3
M17_EVENT_CONFIDENCE_FLOOR = 0.58
M17_INITIAL_BOUNDARY_CONFIDENCE_FLOOR = 0.48
M17_MAX_INITIAL_IDLE_EXTENSION_SEC = 1.0
M17_IDEAL_MAX_CONSECUTIVE_SAME_UNIT = 2
M17_MAX_CONSECUTIVE_SAME_UNIT = 3


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round_time(value: float) -> float:
    return round(float(value), 5)


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _nearest_frame_index(frame_times: np.ndarray, time_sec: float) -> int:
    if frame_times.size == 0:
        return 0
    return int(np.clip(np.searchsorted(frame_times, float(time_sec)), 0, len(frame_times) - 1))


def _frame_value(values: np.ndarray, frame_times: np.ndarray, time_sec: float) -> float:
    if values.size == 0:
        return 0.0
    return float(values[_nearest_frame_index(frame_times, time_sec)])


def _indices_in_window(frame_times: np.ndarray, indices: Iterable[int], start_sec: float, end_sec: float) -> list[int]:
    result: list[int] = []
    for index in indices:
        if index < 0 or index >= frame_times.size:
            continue
        time_sec = float(frame_times[index])
        if start_sec <= time_sec <= end_sec:
            result.append(int(index))
    return result


def _dynamic_floor(values: np.ndarray, frame_times: np.ndarray, start_sec: float, end_sec: float, ratio: float) -> float:
    if values.size == 0:
        return 1.0
    start_index = _nearest_frame_index(frame_times, start_sec)
    end_index = _nearest_frame_index(frame_times, end_sec)
    if end_index <= start_index:
        return float(values[start_index])
    window = values[start_index : end_index + 1]
    return float(np.percentile(window, ratio)) if window.size else 1.0


def _tempo_state_for_window(
    onset: np.ndarray,
    frame_times: np.ndarray,
    sample_rate: int,
    hop_size: int,
    window_start_sec: float,
    available_until_sec: float,
) -> dict[str, Any]:
    from ..audio_analysis import best_beat_offset, estimate_bpm

    start_index = _nearest_frame_index(frame_times, window_start_sec)
    end_index = _nearest_frame_index(frame_times, available_until_sec)
    visible = onset[start_index : end_index + 1]
    if visible.size < 8:
        bpm = 120.0
        spacing_frames = max(1, int(round((60.0 / bpm) * sample_rate / hop_size)))
        confidence = 0.15
        offset_frames = 0
    else:
        bpm, spacing_frames, confidence = estimate_bpm(visible, sample_rate=sample_rate, hop_size=hop_size)
        offset_frames = best_beat_offset(visible, spacing_frames)
    offset_global_frame = start_index + offset_frames
    offset_sec = float(frame_times[min(max(0, offset_global_frame), len(frame_times) - 1)]) if frame_times.size else 0.0
    return {
        "bpm": round(float(bpm), 5),
        "spacing_frames": int(max(1, spacing_frames)),
        "spacing_sec": round(float(60.0 / max(float(bpm), 1e-6)), 5),
        "offset_sec": round(float(offset_sec), 5),
        "confidence": round(float(_clamp(confidence, 0.0, 1.0)), 5),
    }


def _normalize_stream_tempo_state(
    tempo_state: dict[str, Any],
    sample_rate: int,
    hop_size: int,
    previous_bpm: float | None,
    tracker_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    primary_bpm = max(1.0, _safe_float(tempo_state.get("bpm"), 120.0))
    primary_confidence = _safe_float(tempo_state.get("confidence"), 0.0)
    candidates: list[dict[str, Any]] = []
    for label, factor, confidence_scale in (
        ("stream_half_time", 0.5, 0.92 if primary_bpm > 150.0 else 0.55),
        ("stream_primary", 1.0, 1.0),
        ("stream_double_time", 2.0, 0.88 if primary_bpm < 75.0 else 0.4),
    ):
        bpm = primary_bpm * factor
        if bpm < 45.0 or bpm > 220.0:
            continue
        range_penalty = 0.0
        if bpm < 78.0:
            range_penalty = (78.0 - bpm) / 78.0
        elif bpm > 155.0:
            range_penalty = (bpm - 155.0) / 155.0
        continuity_penalty = 0.0
        if previous_bpm is not None and previous_bpm > 1.0:
            continuity_penalty = min(0.45, abs(bpm - previous_bpm) / 120.0)
        score = _clamp(primary_confidence * confidence_scale - range_penalty * 0.45 - continuity_penalty, 0.0, 1.0)
        candidates.append(
            {
                "label": label,
                "bpm": round(float(bpm), 5),
                "spacing_sec": round(float(60.0 / max(bpm, 1e-6)), 5),
                "spacing_frames": int(max(1, round((60.0 / max(bpm, 1e-6)) * sample_rate / max(1, hop_size)))),
                "offset_sec": tempo_state.get("offset_sec", 0.0),
                "confidence": round(float(score), 5),
            }
        )
    candidates.sort(key=lambda item: (_safe_float(item.get("confidence")), item.get("label") == "stream_primary"), reverse=True)
    raw_selected = dict(candidates[0] if candidates else tempo_state)
    tracker = dict(tracker_state or {})
    previous_label = str(tracker.get("selected_tempo_label") or raw_selected.get("label") or "stream_primary")
    previous_candidate = next((dict(item) for item in candidates if str(item.get("label")) == previous_label), None)
    stability_hold = False
    switch_pending_ticks = _safe_int(tracker.get("switch_pending_ticks"))
    stable_ticks = _safe_int(tracker.get("stable_ticks"), 1)
    if previous_candidate is not None and str(raw_selected.get("label")) != previous_label:
        previous_confidence = _safe_float(previous_candidate.get("confidence"))
        raw_confidence = _safe_float(raw_selected.get("confidence"))
        if previous_confidence >= max(0.30, raw_confidence - M17_TEMPO_HYSTERESIS_MARGIN):
            raw_selected = previous_candidate
            stability_hold = True
            switch_pending_ticks = 0
        else:
            switch_pending_ticks += 1
            if switch_pending_ticks < M17_TEMPO_STABLE_SWITCH_TICKS:
                raw_selected = previous_candidate
                stability_hold = True
            else:
                switch_pending_ticks = 0
    else:
        switch_pending_ticks = 0
    if str(raw_selected.get("label")) == previous_label:
        stable_ticks = stable_ticks + 1 if tracker else max(stable_ticks, 1)
    else:
        stable_ticks = 1
    selected = dict(raw_selected)
    selected["selected_tempo_label"] = selected.get("label", "stream_primary")
    selected["stability_hold"] = bool(stability_hold)
    selected["stable_ticks"] = int(stable_ticks)
    updated_tracker = {
        "selected_tempo_label": selected.get("selected_tempo_label", "stream_primary"),
        "selected_bpm": _safe_float(selected.get("bpm"), primary_bpm),
        "stable_ticks": int(stable_ticks),
        "switch_pending_ticks": int(switch_pending_ticks),
        "stability_hold": bool(stability_hold),
    }
    return selected, candidates, updated_tracker


def _planner_score_weights(planner_version: str) -> dict[str, float]:
    version = str(planner_version or "m9").lower()
    if version == "m17":
        return {
            "rhythm_lock": 0.60,
            "transition_smoothness": 0.25,
            "style_energy_bpm": 0.10,
            "source_quality_weight": 0.03,
            "diversity": 0.02,
        }
    if version == "m15":
        return {
            "rhythm_lock": 0.50,
            "transition_smoothness": 0.30,
            "style_energy_bpm": 0.10,
            "source_quality_weight": 0.05,
            "diversity": 0.05,
        }
    if version == "m12":
        return {
            "rhythm_lock": 0.55,
            "transition_smoothness": 0.30,
            "style_energy_bpm": 0.10,
            "source_quality_weight": 0.0,
            "diversity": 0.05,
        }
    return {
        "rhythm_lock": 0.45,
        "transition_smoothness": 0.25,
        "style_energy_bpm": 0.20,
        "source_quality_weight": 0.0,
        "diversity": 0.10,
    }


def _planner_name(planner_version: str) -> str:
    version = str(planner_version or "m9").lower()
    if version == "m17":
        return "streaming_retrieval_v4_any_song_rhythm_stable"
    if version == "m15":
        return "streaming_retrieval_v3_style_cohort"
    if version == "m12":
        return "streaming_retrieval_v2_phrase_aware"
    return "streaming_retrieval_v1"


def _beat_times_for_window(
    tempo_state: dict[str, Any],
    start_sec: float,
    end_sec: float,
    frame_times: np.ndarray,
    onset: np.ndarray,
    beats_per_bar: int,
) -> list[dict[str, Any]]:
    spacing = max(1e-3, _safe_float(tempo_state.get("spacing_sec"), 0.5))
    offset = _safe_float(tempo_state.get("offset_sec"))
    first_index = int(math.floor((start_sec - offset) / spacing)) - 1
    last_index = int(math.ceil((end_sec - offset) / spacing)) + 1
    beats: list[dict[str, Any]] = []
    for beat_index in range(first_index, last_index + 1):
        time_sec = offset + beat_index * spacing
        if time_sec < start_sec - 1e-8 or time_sec > end_sec + 1e-8:
            continue
        strength = _frame_value(onset, frame_times, time_sec)
        confidence = _clamp(0.25 + 0.55 * _safe_float(tempo_state.get("confidence")) + 0.2 * strength, 0.0, 1.0)
        local_index = len(beats)
        absolute_beat_index = max(0, beat_index)
        beats.append(
            {
                "index": absolute_beat_index,
                "time_sec": _round_time(time_sec),
                "strength": round(float(strength), 5),
                "confidence": round(float(confidence), 5),
                "is_downbeat": bool(absolute_beat_index % max(1, beats_per_bar) == 0),
                "count": int(absolute_beat_index % max(1, beats_per_bar)) + 1,
                "window_index": local_index,
            }
        )
    return beats


def _peak_events(
    frame_times: np.ndarray,
    values: np.ndarray,
    peak_indices: list[int],
    start_sec: float,
    end_sec: float,
    floor: float,
    kind: str,
    max_events: int,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for frame_index in _indices_in_window(frame_times, peak_indices, start_sec, end_sec):
        strength = float(values[frame_index]) if frame_index < values.size else 0.0
        if strength < floor:
            continue
        events.append(
            {
                "time_sec": _round_time(float(frame_times[frame_index])),
                "strength": round(strength, 5),
                "kind": kind,
                "analysis_frame": int(frame_index),
            }
        )
    events.sort(key=lambda item: (float(item["time_sec"]), -float(item["strength"])))
    if len(events) > max_events:
        strongest = sorted(events, key=lambda item: float(item["strength"]), reverse=True)[:max_events]
        return sorted(strongest, key=lambda item: float(item["time_sec"]))
    return events


def build_streaming_song_event_records(
    audio_path: Path,
    song_id: str,
    initial_buffer_sec: float = DEFAULT_INITIAL_BUFFER_SEC,
    lookahead_sec: float = DEFAULT_LOOKAHEAD_SEC,
    lookfront_sec: float = DEFAULT_LOOKFRONT_SEC,
    chunk_ms: float = DEFAULT_CHUNK_MS,
    beats_per_bar: int = 4,
    rolling_window_sec: float = DEFAULT_ROLLING_WINDOW_SEC,
    tempo_window_sec: float = DEFAULT_TEMPO_WINDOW_SEC,
) -> list[dict[str, Any]]:
    from ..audio_analysis import build_frame_analysis, load_audio, local_peak_indices

    audio, sample_rate = load_audio(audio_path)
    analysis = build_frame_analysis(audio, sample_rate)
    frame_times = np.asarray(analysis["frame_times_sec"], dtype=np.float32)
    onset = np.asarray(analysis["onset_envelope"], dtype=np.float32)
    low = np.asarray(analysis["low_band_envelope"], dtype=np.float32)
    high = np.asarray(analysis["high_attack_envelope"], dtype=np.float32)
    duration_sec = float(analysis["duration_sec"])
    hop_size = int(analysis["hop_size"])

    onset_peaks = local_peak_indices(onset, floor=float(np.percentile(onset, 45.0)) if onset.size else 1.0)
    low_peaks = local_peak_indices(low, floor=float(np.percentile(low, 55.0)) if low.size else 1.0)
    high_peaks = local_peak_indices(high, floor=float(np.percentile(high, 60.0)) if high.size else 1.0)

    chunk_sec = max(0.001, float(chunk_ms) / 1000.0)
    tick_count = int(math.ceil(duration_sec / chunk_sec)) + 1
    planning_future_sec = max(0.0, float(lookahead_sec))
    transport_future_sec = max(0.0, float(lookfront_sec))
    total_future_sec = planning_future_sec + transport_future_sec
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "kind": "stream_header",
            "song_id": song_id,
            "source_audio_path": str(audio_path),
            "duration_sec": round(duration_sec, 5),
            "sample_rate": int(sample_rate),
            "frame_size": int(analysis["frame_size"]),
            "hop_size": hop_size,
            "chunk_ms": round(float(chunk_ms), 5),
            "initial_buffer_sec": round(float(initial_buffer_sec), 5),
            "lookahead_sec": round(float(planning_future_sec), 5),
            "lookfront_sec": round(float(transport_future_sec), 5),
            "total_future_sec": round(float(total_future_sec), 5),
            "rolling_window_sec": round(float(rolling_window_sec), 5),
            "beats_per_bar": int(beats_per_bar),
            "analysis_mode": "streaming_simulation",
            "tempo_tracker": "m17_online_hypotheses_with_hysteresis",
            "generated_at_utc": utc_now_iso(),
        }
    ]

    tempo_cache: dict[str, Any] | None = None
    previous_selected_bpm: float | None = None
    tempo_tracker_state: dict[str, Any] | None = None
    tempo_update_ticks = max(1, int(round(0.25 / chunk_sec)))
    for tick_index in range(tick_count):
        playhead_sec = min(duration_sec, tick_index * chunk_sec)
        planning_until = min(duration_sec, playhead_sec + planning_future_sec)
        available_until = min(duration_sec, playhead_sec + total_future_sec)
        window_start = max(0.0, playhead_sec - max(0.0, float(rolling_window_sec)))
        tempo_start = max(0.0, available_until - max(1.0, float(tempo_window_sec)))
        if tempo_cache is None or tick_index % tempo_update_ticks == 0:
            tempo_cache = _tempo_state_for_window(
                onset=onset,
                frame_times=frame_times,
                sample_rate=sample_rate,
                hop_size=hop_size,
                window_start_sec=tempo_start,
                available_until_sec=available_until,
            )
        tempo_state, tempo_hypotheses, tempo_tracker_state = _normalize_stream_tempo_state(
            dict(tempo_cache),
            sample_rate=sample_rate,
            hop_size=hop_size,
            previous_bpm=previous_selected_bpm,
            tracker_state=tempo_tracker_state,
        )
        previous_selected_bpm = _safe_float(tempo_state.get("bpm"), previous_selected_bpm or 120.0)
        beat_window_start = max(0.0, window_start)
        beats = _beat_times_for_window(
            tempo_state=tempo_state,
            start_sec=beat_window_start,
            end_sec=available_until,
            frame_times=frame_times,
            onset=onset,
            beats_per_bar=beats_per_bar,
        )
        downbeats = [dict(beat) for beat in beats if beat.get("is_downbeat")]
        low_floor = _dynamic_floor(low, frame_times, window_start, available_until, 76.0)
        high_floor = _dynamic_floor(high, frame_times, window_start, available_until, 82.0)
        onset_floor = _dynamic_floor(onset, frame_times, window_start, available_until, 80.0)
        kick_events = _peak_events(frame_times, low, low_peaks, window_start, available_until, low_floor, "kick", 24)
        high_events = _peak_events(frame_times, high, high_peaks, window_start, available_until, high_floor, "high_attack", 24)
        accent_events = _peak_events(frame_times, onset, onset_peaks, window_start, available_until, onset_floor, "accent_peak", 32)
        drum_hits = sorted([*kick_events, *high_events], key=lambda item: (float(item["time_sec"]), str(item["kind"])))
        for index, item in enumerate(drum_hits):
            item["index"] = index
        for index, item in enumerate(accent_events):
            item["index"] = index
        downbeat_confidence = max([_safe_float(item.get("confidence")) for item in downbeats] or [0.0])
        current_onset = _frame_value(onset, frame_times, playhead_sec)
        current_low = _frame_value(low, frame_times, playhead_sec)
        current_high = _frame_value(high, frame_times, playhead_sec)
        records.append(
            {
                "schema_version": 1,
                "kind": "tick",
                "tick_index": tick_index,
                "playhead_sec": _round_time(playhead_sec),
                "available_audio_until_sec": _round_time(available_until),
                "planning_audio_until_sec": _round_time(planning_until),
                "lookahead_sec": round(float(planning_future_sec), 5),
                "lookfront_sec": round(float(transport_future_sec), 5),
                "total_future_sec": round(float(max(0.0, available_until - playhead_sec)), 5),
                "visible_window_sec": {"start": _round_time(window_start), "end": _round_time(available_until)},
                "tempo_hypotheses": [
                    {
                        "label": item["label"],
                        "bpm": item["bpm"],
                        "confidence": item["confidence"],
                    }
                    for item in tempo_hypotheses
                ],
                "beat_phase": {
                    "bpm": tempo_state["bpm"],
                    "spacing_sec": tempo_state["spacing_sec"],
                    "offset_sec": tempo_state["offset_sec"],
                    "confidence": tempo_state["confidence"],
                    "selected_tempo_label": tempo_state.get("selected_tempo_label", "stream_primary"),
                    "stable_ticks": _safe_int(tempo_state.get("stable_ticks"), 1),
                    "stability_hold": bool(tempo_state.get("stability_hold")),
                },
                "downbeat_confidence": round(float(downbeat_confidence), 5),
                "analysis_frame": {
                    "onset": round(float(current_onset), 5),
                    "low_band": round(float(current_low), 5),
                    "high_attack": round(float(current_high), 5),
                },
                "beats": beats,
                "downbeats": downbeats,
                "drum_hits": drum_hits,
                "accents": accent_events,
                "segment_hypotheses": _segment_hypotheses(beats, available_until),
            }
        )
    return records


def _segment_hypotheses(beats: list[dict[str, Any]], available_until: float) -> list[dict[str, Any]]:
    if not beats:
        return []
    hypotheses: list[dict[str, Any]] = []
    downbeat_candidates = [beat for beat in beats if bool(beat.get("is_downbeat")) and _safe_float(beat.get("confidence")) >= 0.5]
    if not downbeat_candidates:
        downbeat_candidates = [beat for beat in beats if _safe_float(beat.get("confidence")) >= 0.58] or beats[:1]
    for beat in downbeat_candidates[-3:]:
        start = _safe_float(beat.get("time_sec"))
        spacing = 0.5
        index = beats.index(beat)
        if index + 1 < len(beats):
            spacing = max(0.001, _safe_float(beats[index + 1].get("time_sec")) - start)
        for duration_beats in (2, 4, 8):
            end = start + duration_beats * spacing
            if start <= available_until:
                hypotheses.append(
                    {
                        "start_time_sec": _round_time(start),
                        "end_time_sec": _round_time(end),
                        "duration_beats": duration_beats,
                        "boundary_source": "downbeat" if beat.get("is_downbeat") else "beat",
                        "confidence": beat.get("confidence", 0.0),
                    }
                )
    return hypotheses[-12:]


def _resolve_reference_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _count_grid_for_unit(unit: dict[str, Any]) -> list[dict[str, Any]]:
    frame_range = dict(unit.get("frame_range", {}) or {})
    start_frame = _safe_int(frame_range.get("start"))
    end_frame = max(start_frame + 1, _safe_int(frame_range.get("end_exclusive"), start_frame + 1))
    duration_beats = max(1, _safe_int(unit.get("duration_beats"), _safe_int(unit.get("duration_beats_estimate"), 4)))
    frame_span = max(1, end_frame - start_frame - 1)
    keyframes_by_offset = {
        _safe_int(item.get("beat_offset"), -1): dict(item)
        for item in list(unit.get("keyframes", []) or [])
        if _safe_int(item.get("beat_offset"), -1) >= 0
    }
    grid: list[dict[str, Any]] = []
    for beat_offset in range(duration_beats + 1):
        keyframe = keyframes_by_offset.get(beat_offset)
        if keyframe is not None:
            source_frame = _safe_int(keyframe.get("frame_index"), start_frame)
            strength = _safe_float(keyframe.get("strength"))
        else:
            source_frame = start_frame + int(round(frame_span * beat_offset / max(1, duration_beats)))
            strength = 0.0
        count = int(beat_offset % 4) + 1
        grid.append(
            {
                "beat_offset": int(beat_offset),
                "count": count,
                "phrase_count": int(beat_offset + 1),
                "source_frame": int(min(end_frame - 1, max(start_frame, source_frame))),
                "is_downbeat": bool(count == 1),
                "role": "count_1_downbeat" if count == 1 else ("count_3_backbeat" if count == 3 else "weak_count"),
                "strength": round(float(strength), 5),
            }
        )
    return grid


def _motion_segment(project_root: Path, unit: dict[str, Any], motion_cache: dict[str, np.ndarray]) -> np.ndarray | None:
    source_motion_path = str(dict(unit.get("reference_artifacts", {}) or {}).get("source_motion_path", "") or "")
    if not source_motion_path:
        return None
    path = _resolve_reference_path(project_root, source_motion_path)
    if not path.exists():
        return None
    cache_key = str(path)
    if cache_key not in motion_cache:
        motion_cache[cache_key] = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
    motion = motion_cache[cache_key]
    frame_range = dict(unit.get("frame_range", {}) or {})
    start = max(0, _safe_int(frame_range.get("start")))
    end = min(int(motion.shape[0]), max(start + 1, _safe_int(frame_range.get("end_exclusive"), start + 1)))
    return np.asarray(motion[start:end, :], dtype=np.float32)


def _contact_windows_from_activity(activity: np.ndarray, source: str) -> list[dict[str, int | str]]:
    if activity.size == 0:
        return []
    contact_floor = float(np.percentile(activity, 30.0))
    windows: list[dict[str, int | str]] = []
    start: int | None = None
    for index, value in enumerate(activity.tolist()):
        if value <= contact_floor:
            if start is None:
                start = index
        elif start is not None:
            if index - start >= 2:
                windows.append({"start_local_frame": start, "end_local_frame_exclusive": index + 1, "source": source})
            start = None
    if start is not None and len(activity) - start >= 2:
        windows.append({"start_local_frame": start, "end_local_frame_exclusive": len(activity) + 1, "source": source})
    return windows


def _root_speed_profile(segment: np.ndarray | None, fps: float = 30.0, contact_mode: str = "root") -> dict[str, Any]:
    if segment is None or segment.shape[0] < 2 or segment.shape[1] < 3:
        return {
            "mean": 0.0,
            "p90": 0.0,
            "entry": [0.0, 0.0, 0.0],
            "exit": [0.0, 0.0, 0.0],
            "displacement": 0.0,
            "curve": [0.0] * 8,
            "motion_accent_local_frames": [],
            "foot_contact_windows": [],
            "contact_mode": contact_mode,
        }
    root = np.asarray(segment[:, :3], dtype=np.float32)
    velocity = np.diff(root, axis=0) * float(fps)
    speed = np.linalg.norm(velocity, axis=1)
    displacement = float(np.linalg.norm(root[-1] - root[0]))
    sample_indices = np.linspace(0, len(speed) - 1, num=8).astype(int) if speed.size else np.zeros((8,), dtype=int)
    max_speed = float(speed.max()) if speed.size else 1.0
    curve = [round(float(speed[index] / max(max_speed, 1e-6)), 5) for index in sample_indices]
    accent_frames: list[int] = []
    if speed.size >= 3:
        floor = float(np.percentile(speed, 72.0))
        for index in range(1, len(speed) - 1):
            if speed[index] >= floor and speed[index] >= speed[index - 1] and speed[index] >= speed[index + 1]:
                accent_frames.append(index + 1)
        accent_frames = sorted(accent_frames, key=lambda idx: float(speed[max(0, idx - 1)]), reverse=True)[:8]
        accent_frames.sort()
    contact_mode = str(contact_mode or "root").lower()
    contact_windows: list[dict[str, int | str]] = []
    if contact_mode == "joints" and segment.shape[1] > 45:
        pose_slice = np.asarray(segment[:, 9 : min(segment.shape[1], 159)], dtype=np.float32)
        pose_activity = np.linalg.norm(np.diff(pose_slice, axis=0), axis=1) if pose_slice.shape[0] > 1 else np.asarray([], dtype=np.float32)
        contact_windows = _contact_windows_from_activity(pose_activity, source="joint_pose_velocity_proxy")
    if not contact_windows and speed.size:
        contact_windows = _contact_windows_from_activity(speed, source="root_speed_proxy")
    return {
        "mean": round(float(speed.mean()) if speed.size else 0.0, 5),
        "p90": round(float(np.percentile(speed, 90.0)) if speed.size else 0.0, 5),
        "entry": [round(float(value), 5) for value in velocity[0].tolist()] if velocity.size else [0.0, 0.0, 0.0],
        "exit": [round(float(value), 5) for value in velocity[-1].tolist()] if velocity.size else [0.0, 0.0, 0.0],
        "displacement": round(displacement, 5),
        "curve": curve,
        "motion_accent_local_frames": accent_frames,
        "foot_contact_windows": contact_windows[:8],
        "contact_mode": contact_mode,
    }


def _movement_quality(unit: dict[str, Any], root_profile: dict[str, Any]) -> dict[str, Any]:
    displacement = _safe_float(root_profile.get("displacement"))
    mean_speed = _safe_float(root_profile.get("mean"))
    p90_speed = _safe_float(root_profile.get("p90"))
    yaw_delta = abs(_safe_float(dict(unit.get("transition_profile", {}) or {}).get("yaw_delta_deg")))
    return {
        "size": "large_motion" if displacement >= 1.0 or p90_speed >= 1.6 else ("small_motion" if displacement <= 0.25 else "medium_motion"),
        "attack": "sharp" if p90_speed >= max(0.9, mean_speed * 1.8) else "smooth",
        "travel": "traveling" if displacement >= 0.55 else "in_place",
        "turning": "turning" if yaw_delta >= 35.0 else "forward",
        "energy": str(unit.get("energy", "mid_energy") or "mid_energy"),
    }


def annotate_finedance_motion_units(motion_library: dict[str, Any], project_root: Path, contact_mode: str = "root") -> dict[str, Any]:
    motion_cache: dict[str, np.ndarray] = {}
    annotated_units: list[dict[str, Any]] = []
    for unit in list(motion_library.get("units", []) or []):
        annotated = dict(unit)
        segment = _motion_segment(project_root, annotated, motion_cache)
        root_profile = _root_speed_profile(segment, contact_mode=contact_mode)
        count_grid = _count_grid_for_unit(annotated)
        accent_locks = [
            {
                "role": item["role"],
                "beat_offset": item["beat_offset"],
                "count": item["count"],
                "source_frame": item["source_frame"],
                "strength": item["strength"],
            }
            for item in count_grid
            if item["role"] in {"count_1_downbeat", "count_3_backbeat"} or _safe_float(item.get("strength")) >= 0.55
        ]
        if not accent_locks and count_grid:
            accent_locks = [dict(count_grid[0])]
        frame_start = _safe_int(dict(annotated.get("frame_range", {}) or {}).get("start"))
        motion_accent_frames = [
            {
                "source_frame": int(frame_start + local_frame),
                "local_frame": int(local_frame),
                "kind": "root_speed_peak",
            }
            for local_frame in list(root_profile.get("motion_accent_local_frames", []) or [])
        ]
        annotated.update(
            {
                "annotation_schema_version": 1,
                "annotation_profile": {
                    "milestone": "M10" if str(contact_mode).lower() == "joints" else "M9",
                    "contact_mode": root_profile.get("contact_mode", contact_mode),
                },
                "count_grid": count_grid,
                "accent_lock_frames": accent_locks,
                "motion_accent_frames": motion_accent_frames,
                "foot_contact_windows": list(root_profile.get("foot_contact_windows", []) or []),
                "entry_pose_anchor": {
                    **dict(annotated.get("entry_anchor", {}) or {}),
                    "pose_fingerprint": annotated.get("entry_pose_fingerprint")
                    or stable_digest({"unit_id": annotated.get("unit_id"), "entry": annotated.get("entry_anchor")}),
                },
                "exit_pose_anchor": {
                    **dict(annotated.get("exit_anchor", {}) or {}),
                    "pose_fingerprint": annotated.get("exit_pose_fingerprint")
                    or stable_digest({"unit_id": annotated.get("unit_id"), "exit": annotated.get("exit_anchor")}),
                },
                "root_velocity": {
                    "mean_speed": root_profile["mean"],
                    "p90_speed": root_profile["p90"],
                    "entry_velocity": root_profile["entry"],
                    "exit_velocity": root_profile["exit"],
                    "displacement": root_profile["displacement"],
                },
                "yaw_delta": round(_safe_float(dict(annotated.get("transition_profile", {}) or {}).get("yaw_delta_deg")), 5),
                "energy_curve": list(root_profile.get("curve", []) or []),
                "movement_quality": _movement_quality(annotated, root_profile),
                "safe_retime_range": {"min": DEFAULT_SAFE_RETIME_MIN, "max": DEFAULT_SAFE_RETIME_MAX},
            }
        )
        annotated_units.append(annotated)
    result = dict(motion_library)
    base_id = str(result.get("library_id", "finedance_rhythmic_smplx_library_v1"))
    suffix = "_m10_annotated" if str(contact_mode).lower() == "joints" else "_m9_annotated"
    result["library_id"] = base_id if base_id.endswith(suffix) else f"{base_id}{suffix}"
    result["units"] = annotated_units
    result["notes"] = [
        *list(result.get("notes", []) or []),
        f"{'M10' if str(contact_mode).lower() == 'joints' else 'M9'} annotations add count-grid locks, movement quality, transition anchors, contact windows, and safe retime ranges for streaming retrieval.",
    ]
    result["generated_at_utc"] = utc_now_iso()
    return result


def _records_by_kind(records: list[dict[str, Any]], kind: str) -> list[dict[str, Any]]:
    return [record for record in records if record.get("kind") == kind]


def _header(records: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    for record in records:
        if record.get("kind") == kind:
            return dict(record)
    return {}


def _find_tick_for_decision(ticks: list[dict[str, Any]], decision_time_sec: float) -> dict[str, Any]:
    if not ticks:
        raise ValueError("stream event records contain no ticks")
    for tick in ticks:
        if _safe_float(tick.get("playhead_sec")) + 1e-8 >= decision_time_sec:
            return tick
    return ticks[-1]


def _events_in_time_window(tick: dict[str, Any], key: str, start_sec: float, end_sec: float) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in list(tick.get(key, []) or []):
        time_sec = _safe_float(event.get("time_sec"), -1.0)
        if start_sec - 1e-8 <= time_sec <= end_sec + 1e-8:
            result.append(dict(event))
    return result


def _target_energy_for_tick(tick: dict[str, Any], start_sec: float, visible_end_sec: float) -> str:
    window_end = min(visible_end_sec, start_sec + 2.0)
    accents = _events_in_time_window(tick, "accents", start_sec, window_end)
    drums = _events_in_time_window(tick, "drum_hits", start_sec, window_end)
    strengths = [_safe_float(item.get("strength")) for item in [*accents, *drums]]
    if len(strengths) >= 5 or max(strengths or [0.0]) >= 0.82:
        return "high_energy"
    if len(strengths) <= 1 and max(strengths or [0.0]) <= 0.38:
        return "low_energy"
    return "mid_energy"


def _target_style_profile_for_tick(
    tick: dict[str, Any],
    start_sec: float,
    visible_end_sec: float,
    previous_unit: dict[str, Any] | None,
) -> dict[str, Any]:
    window_end = min(visible_end_sec, start_sec + 2.0)
    accents = _events_in_time_window(tick, "accents", start_sec, window_end)
    drums = _events_in_time_window(tick, "drum_hits", start_sec, window_end)
    target_energy = _target_energy_for_tick(tick, start_sec=start_sec, visible_end_sec=visible_end_sec)
    sharp_activity = len(accents) + len(drums)
    target_attack = "sharp" if sharp_activity >= 4 or max([_safe_float(item.get("strength")) for item in [*accents, *drums]] or [0.0]) >= 0.82 else "smooth"
    target_size = "large_motion" if target_energy == "high_energy" else ("small_motion" if target_energy == "low_energy" else "medium_motion")
    preferred_tags = list(dict(previous_unit or {}).get("style_tags", []) or [])
    if not preferred_tags:
        preferred_tags = ["street"] if _safe_float(dict(tick.get("beat_phase", {}) or {}).get("bpm"), 120.0) >= 110.0 else ["jazz"]
    return {
        "energy": target_energy,
        "attack": target_attack,
        "size": target_size,
        "preferred_tags": preferred_tags[:4],
    }


def _sequence_profiles(units: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[str(unit.get("source_sequence", "unknown"))].append(unit)
    for sequence_id, sequence_units in grouped.items():
        bpm_values = [_safe_float(unit.get("source_song_bpm"), _safe_float(dict(unit.get("rhythm_profile", {}) or {}).get("source_bpm"), 120.0)) for unit in sequence_units]
        quality_weights = [_safe_float(unit.get("source_song_quality_weight"), 0.65) for unit in sequence_units]
        energy_counts = Counter(str(unit.get("source_song_energy", unit.get("energy", "unknown")) or "unknown") for unit in sequence_units)
        style_counts = Counter(str(tag) for unit in sequence_units for tag in list(unit.get("source_song_style_tags", unit.get("style_tags", [])) or []))
        attack_counts = Counter(str(dict(unit.get("movement_quality", {}) or {}).get("attack", "smooth")) for unit in sequence_units)
        size_counts = Counter(str(dict(unit.get("movement_quality", {}) or {}).get("size", "medium_motion")) for unit in sequence_units)
        profiles[sequence_id] = {
            "sequence_id": sequence_id,
            "unit_count": len(sequence_units),
            "bpm": round(float(np.median(np.asarray(bpm_values or [120.0], dtype=np.float32))), 5),
            "dominant_energy": energy_counts.most_common(1)[0][0] if energy_counts else "unknown",
            "priority_tier": str(sequence_units[0].get("priority_tier", "unknown") or "unknown"),
            "quality_weight": round(float(np.mean(np.asarray(quality_weights or [0.65], dtype=np.float32))), 5),
            "style_tags": [key for key, _ in style_counts.most_common(4)],
            "dominant_attack": attack_counts.most_common(1)[0][0] if attack_counts else "smooth",
            "dominant_size": size_counts.most_common(1)[0][0] if size_counts else "medium_motion",
        }
    return profiles


def _sequence_cohort_score(
    profile: dict[str, Any],
    target_bpm: float,
    style_profile: dict[str, Any],
    previous_unit: dict[str, Any] | None,
) -> float:
    bpm_score = 1.0 - min(1.0, abs(_safe_float(profile.get("bpm"), target_bpm) - target_bpm) / 40.0)
    target_energy = str(style_profile.get("energy", "mid_energy"))
    dominant_energy = str(profile.get("dominant_energy", "unknown"))
    energy_score = 1.0 if dominant_energy == target_energy else (0.75 if "mid" in {dominant_energy, target_energy} else 0.35)
    attack_score = 1.0 if str(profile.get("dominant_attack")) == str(style_profile.get("attack")) else 0.72
    size_score = 1.0 if str(profile.get("dominant_size")) == str(style_profile.get("size")) else 0.76
    preferred_tags = {str(tag) for tag in list(style_profile.get("preferred_tags", []) or [])}
    style_tags = {str(tag) for tag in list(profile.get("style_tags", []) or [])}
    tag_score = 0.82 if not preferred_tags else (1.0 if preferred_tags & style_tags else 0.58)
    quality_weight = _safe_float(profile.get("quality_weight"), 0.65)
    continuity = 1.0 if previous_unit and str(previous_unit.get("source_sequence")) == str(profile.get("sequence_id")) else 0.78
    return round(float(bpm_score * 0.28 + energy_score * 0.26 + attack_score * 0.14 + size_score * 0.10 + tag_score * 0.12 + quality_weight * 0.10) * continuity, 5)


def _transition_score(previous_unit: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if previous_unit is None:
        return 1.0
    if candidate.get("unit_id") in previous_unit.get("compatible_next_units", []):
        return 1.0
    previous_exit = dict(previous_unit.get("exit_pose_anchor") or previous_unit.get("exit_anchor") or {})
    entry = dict(candidate.get("entry_pose_anchor") or candidate.get("entry_anchor") or {})
    speed_delta = abs(_safe_float(previous_exit.get("planar_speed")) - _safe_float(entry.get("planar_speed")))
    yaw_delta = abs((_safe_float(previous_exit.get("root_yaw_deg")) - _safe_float(entry.get("root_yaw_deg")) + 180.0) % 360.0 - 180.0)
    previous_energy = str(previous_unit.get("energy", "mid_energy") or "mid_energy")
    candidate_energy = str(candidate.get("energy", "mid_energy") or "mid_energy")
    energy_score = 1.0 if previous_energy == candidate_energy else (0.74 if "mid" in {previous_energy, candidate_energy} else 0.42)
    previous_styles = set(str(tag) for tag in list(previous_unit.get("style_tags", []) or []))
    candidate_styles = set(str(tag) for tag in list(candidate.get("style_tags", []) or []))
    style_score = 1.0 if previous_styles & candidate_styles else 0.68
    previous_contacts = list(previous_unit.get("foot_contact_windows", []) or [])
    candidate_contacts = list(candidate.get("foot_contact_windows", []) or [])
    contact_score = 1.0 if previous_contacts and candidate_contacts else (0.84 if not previous_contacts and not candidate_contacts else 0.62)
    same_sequence = 0.12 if previous_unit.get("source_sequence") == candidate.get("source_sequence") else -0.06
    score = 0.34 + max(0.0, 0.26 - speed_delta * 0.22) + max(0.0, 0.18 - yaw_delta / 180.0) + energy_score * 0.16 + style_score * 0.12 + contact_score * 0.10 + same_sequence
    return round(float(_clamp(score, 0.0, 1.0)), 5)


def _music_style_score(candidate: dict[str, Any], target_style_profile: dict[str, Any], target_bpm: float) -> float:
    target_energy = str(target_style_profile.get("energy", "mid_energy") or "mid_energy")
    energy = str(candidate.get("energy", "mid_energy") or "mid_energy")
    energy_score = 1.0 if energy == target_energy else (0.72 if "mid" in {energy, target_energy} else 0.32)
    source_bpm = _safe_float(candidate.get("source_song_bpm"), _safe_float(dict(candidate.get("rhythm_profile", {}) or {}).get("source_bpm"), target_bpm))
    bpm_score = 1.0 - min(1.0, abs(source_bpm - target_bpm) / 45.0)
    quality = dict(candidate.get("movement_quality", {}) or {})
    attack_score = 1.0 if quality.get("attack") == target_style_profile.get("attack") else 0.7
    size_score = 1.0 if quality.get("size") == target_style_profile.get("size") else 0.74
    preferred_tags = set(str(tag) for tag in list(target_style_profile.get("preferred_tags", []) or []))
    candidate_tags = set(str(tag) for tag in list(candidate.get("style_tags", []) or []))
    tag_score = 0.85 if not preferred_tags else (1.0 if preferred_tags & candidate_tags else 0.6)
    quality_weight = _safe_float(candidate.get("source_song_quality_weight"), 0.65)
    return round(float((energy_score * 0.32) + (bpm_score * 0.28) + (attack_score * 0.14) + (size_score * 0.11) + (tag_score * 0.10) + (quality_weight * 0.05)), 5)


def _rhythm_score(
    candidate: dict[str, Any],
    tick: dict[str, Any],
    target_start_sec: float,
    beat_spacing_sec: float,
    visible_until_sec: float,
    target_beats: int,
    planner_version: str = "m9",
) -> tuple[float, list[dict[str, Any]], list[float]]:
    locks = list(candidate.get("accent_lock_frames", []) or [])
    if not locks:
        locks = [{"beat_offset": 0, "role": "count_1_downbeat", "count": 1}]
    locks = [
        lock
        for lock in locks
        if _safe_float(lock.get("beat_offset")) <= float(target_beats) + 1e-8
    ] or [{"beat_offset": 0, "role": "count_1_downbeat", "count": 1}]
    version = str(planner_version or "m9").lower()
    is_m17 = version == "m17"
    beats = list(tick.get("beats", []) or [])
    downbeats = list(tick.get("downbeats", []) or [])
    drums = list(tick.get("drum_hits", []) or [])
    accents = list(tick.get("accents", []) or [])
    lock_reports: list[dict[str, Any]] = []
    expected_hits: list[float] = []
    scores: list[float] = []
    beat_confidence = _safe_float(dict(tick.get("beat_phase", {}) or {}).get("confidence"), 0.0)
    for lock in locks[:6]:
        beat_offset = _safe_float(lock.get("beat_offset"))
        lock_time = target_start_sec + beat_offset * beat_spacing_sec
        expected_hits.append(_round_time(lock_time))
        role = str(lock.get("role", "accent"))
        is_predicted = lock_time > visible_until_sec + 1e-8
        if role == "count_1_downbeat":
            prioritized_events = [
                *((event, 1.0) for event in downbeats),
                *((event, 0.95) for event in drums),
                *((event, 0.82) for event in beats),
                *((event, 0.68) for event in accents),
            ]
        elif role == "count_3_backbeat":
            prioritized_events = [
                *((event, 0.98) for event in drums),
                *((event, 0.88) for event in beats),
                *((event, 0.72) for event in accents),
                *((event, 0.65) for event in downbeats),
            ]
        else:
            prioritized_events = [
                *((event, 0.92) for event in drums),
                *((event, 0.84) for event in accents),
                *((event, 0.80) for event in beats),
                *((event, 0.70) for event in downbeats),
            ]
        best_visible_score = 0.0
        nearest_delta = beat_spacing_sec
        best_event_kind = "predicted"
        visible_tolerance = max(beat_spacing_sec * (0.28 if is_m17 else 0.35), 1e-3)
        for event, priority_weight in prioritized_events:
            event_time = _safe_float(event.get("time_sec"), 99999.0)
            if event_time > visible_until_sec + 1e-8:
                continue
            delta = abs(event_time - lock_time)
            confidence = _safe_float(event.get("confidence"), _safe_float(event.get("strength"), 0.0))
            event_score = _clamp(1.0 - delta / visible_tolerance, 0.0, 1.0)
            event_score *= priority_weight
            event_score *= 0.72 + 0.28 * _clamp(confidence, 0.0, 1.0)
            if event_score > best_visible_score + 1e-8:
                best_visible_score = event_score
                nearest_delta = delta
                best_event_kind = str(event.get("kind", "beat"))
        if role == "count_1_downbeat":
            downbeat_bonus = 0.18 if int(round(beat_offset)) % 4 == 0 else 0.0
        elif role == "count_3_backbeat":
            downbeat_bonus = 0.08
        else:
            downbeat_bonus = 0.0
        if is_predicted:
            predicted_base = 0.45 + downbeat_bonus
            if is_m17 and beat_confidence < M17_LOW_CONFIDENCE_THRESHOLD:
                predicted_base = 0.20 + downbeat_bonus * 0.25
            score = _clamp(predicted_base, 0.0, 1.0)
            score *= 0.75 + 0.25 * _clamp(_safe_float(lock.get("strength"), 0.0), 0.0, 1.0)
        else:
            score = _clamp(best_visible_score + downbeat_bonus, 0.0, 1.0)
            score *= 0.72 + 0.28 * _clamp(_safe_float(lock.get("strength"), 0.0), 0.0, 1.0)
        scores.append(score)
        lock_reports.append(
            {
                "kind": "stream_rhythm_lock",
                "role": role,
                "time_sec": _round_time(lock_time),
                "beat_offset": round(float(beat_offset), 5),
                "visible_at_decision": not is_predicted,
                "nearest_visible_event_delta_sec": round(float(nearest_delta), 5),
                "nearest_visible_event_kind": best_event_kind,
                "score": round(float(score), 5),
            }
        )
    return round(float(sum(scores) / max(1, len(scores))), 5), lock_reports, sorted(set(expected_hits))


def _candidate_source_duration(candidate: dict[str, Any]) -> float:
    if candidate.get("duration_sec") is not None:
        return max(1e-3, _safe_float(candidate.get("duration_sec"), 1.0))
    frame_range = dict(candidate.get("frame_range", {}) or {})
    frame_count = max(1, _safe_int(frame_range.get("end_exclusive"), 1) - _safe_int(frame_range.get("start")))
    return frame_count / 30.0


def _choose_target_beats(units: list[dict[str, Any]], spacing_sec: float, preferred_beats: int) -> int:
    options: list[int] = []
    for value in (preferred_beats, 2, 4, 8):
        if value > 0 and value not in options:
            options.append(value)
    best_option = options[0]
    best_penalty = float("inf")
    for target_beats in options:
        target_duration = max(1e-3, float(target_beats) * spacing_sec)
        exact = [unit for unit in units if _safe_int(unit.get("duration_beats"), 0) == target_beats]
        pool = exact or units
        penalties = []
        for unit in pool:
            speed_scale = _candidate_source_duration(unit) / target_duration
            safe_range = dict(unit.get("safe_retime_range", {}) or {})
            low = _safe_float(safe_range.get("min"), 0.85)
            high = _safe_float(safe_range.get("max"), 1.15)
            penalties.append(abs(speed_scale - _clamp(speed_scale, low, high)))
        option_penalty = min(penalties or [999.0])
        if option_penalty < best_penalty - 1e-8:
            best_penalty = option_penalty
            best_option = target_beats
    return int(best_option)


def _m15_recovery_rank(
    score: float,
    candidate: dict[str, Any],
    breakdown: dict[str, Any],
    previous_unit: dict[str, Any] | None,
) -> tuple[float, float, float, float]:
    rhythm = _safe_float(breakdown.get("rhythm_lock"), 0.0)
    transition = _safe_float(breakdown.get("transition_smoothness"), 0.0)
    speed = _safe_float(breakdown.get("speed_scale"), 1.0)
    speed_closeness = max(0.0, 1.0 - abs(speed - 1.0))
    same_sequence = 1.0 if previous_unit and str(previous_unit.get("source_sequence")) == str(candidate.get("source_sequence")) else 0.0
    return (rhythm, transition, speed_closeness, same_sequence + score * 0.001)


def _m17_recovery_rank(
    score: float,
    candidate: dict[str, Any],
    breakdown: dict[str, Any],
    previous_unit: dict[str, Any] | None,
) -> tuple[float, float, float, float, float]:
    rhythm = _safe_float(breakdown.get("rhythm_lock"), 0.0)
    transition = _safe_float(breakdown.get("transition_smoothness"), 0.0)
    speed = _safe_float(breakdown.get("speed_scale"), 1.0)
    speed_closeness = max(0.0, 1.0 - abs(speed - 1.0))
    same_unit = 1.0 if previous_unit and str(previous_unit.get("unit_id")) == str(candidate.get("unit_id")) else 0.0
    same_sequence = 1.0 if previous_unit and str(previous_unit.get("source_sequence")) == str(candidate.get("source_sequence")) else 0.0
    return (same_unit, same_sequence, rhythm, transition, speed_closeness + score * 0.001)


def _max_consecutive_motion_unit_run(decisions: list[dict[str, Any]]) -> int:
    max_run = 0
    current_id: str | None = None
    current_run = 0
    for decision in decisions:
        if str(decision.get("pose_source") or "finedance_motion_unit") != "finedance_motion_unit":
            current_id = None
            current_run = 0
            continue
        unit_id = str(decision.get("selected_unit_id") or "")
        if unit_id and unit_id == current_id:
            current_run += 1
        else:
            current_id = unit_id
            current_run = 1 if unit_id else 0
        max_run = max(max_run, current_run)
    return max_run


def _initial_pose_source(initial_pose_mode: str) -> str:
    mode = str(initial_pose_mode or DEFAULT_INITIAL_POSE_MODE).lower()
    if mode == "freeze_first":
        return SYNTHETIC_FREEZE_FIRST_SOURCE
    if mode == "neutral_rest":
        return SYNTHETIC_NEUTRAL_REST_SOURCE
    return SYNTHETIC_NEUTRAL_IDLE_SOURCE


def _planner_boundary_events(
    tick: dict[str, Any],
    start_sec: float,
    available_until_sec: float,
    confidence_floor: float,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for kind, priority in (("downbeats", 0), ("drum_hits", 1), ("beats", 2), ("accents", 3)):
        for event in list(tick.get(kind, []) or []):
            event_time = _safe_float(event.get("time_sec"), -1.0)
            if event_time + 1e-8 < start_sec or event_time > available_until_sec + 1e-8:
                continue
            confidence = _safe_float(event.get("confidence"), _safe_float(event.get("strength"), 0.0))
            if confidence < confidence_floor:
                continue
            candidates.append(
                {
                    "time_sec": _round_time(event_time),
                    "kind": "downbeat" if kind == "downbeats" else ("beat" if kind == "beats" else kind[:-1]),
                    "priority": priority,
                    "confidence": round(float(confidence), 5),
                }
            )
    candidates.sort(key=lambda item: (float(item["time_sec"]), int(item["priority"]), -float(item["confidence"])))
    return candidates


def _synthetic_idle_decision(
    *,
    index: int,
    planner_name: str,
    planner_version: str,
    initial_pose_mode: str,
    start_sec: float,
    end_sec: float,
    playhead_sec: float,
    available_audio_until_sec: float,
    planning_lookahead_sec: float,
    lookfront_sec: float,
    total_future_sec: float,
    decision_tick_index: int,
    decision_time_sec: float,
    target_bpm: float,
    target_energy: str,
    switch_mode: str,
    extra_switch_reason: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pose_source = _initial_pose_source(initial_pose_mode)
    return {
        "schema_version": 1,
        "kind": "decision",
        "index": int(index),
        "decision_time_sec": _round_time(decision_time_sec),
        "decision_tick_index": int(decision_tick_index),
        "playhead_sec": _round_time(playhead_sec),
        "available_audio_until_sec": _round_time(available_audio_until_sec),
        "future_visibility_guard": {
            "lookahead_sec": round(float(planning_lookahead_sec), 5),
            "lookfront_sec": round(float(lookfront_sec), 5),
            "total_future_sec": round(float(total_future_sec), 5),
            "used_audio_until_sec": _round_time(available_audio_until_sec),
            "passed": True,
        },
        "target_time_sec": {"start": _round_time(start_sec), "end": _round_time(end_sec)},
        "target_beats": 0,
        "target_bpm": round(float(target_bpm), 5),
        "target_energy": target_energy,
        "selected_unit_id": f"{pose_source}_{slugify(f'{start_sec:.2f}_{end_sec:.2f}')}",
        "source_sequence": SYNTHETIC_SOURCE_SEQUENCE,
        "selected_from_sequence": SYNTHETIC_SOURCE_SEQUENCE,
        "selected_from_tier": "synthetic_pose",
        "cohort_source_sequences": [],
        "cohort_rankings": [],
        "source_frame_range": {"start": 0, "end_exclusive": 1},
        "source_beat_range": {"start": 0, "end_exclusive": 0},
        "source_motion_path": None,
        "speed_scale": 1.0,
        "score": 1.0,
        "pose_source": pose_source,
        "initial_pose_mode": str(initial_pose_mode or DEFAULT_INITIAL_POSE_MODE),
        "score_breakdown": {
            "rhythm_lock": 1.0,
            "transition_smoothness": 1.0,
            "style_energy_bpm": 1.0,
            "source_quality_weight": 1.0,
            "diversity": 1.0,
            "weighted_total": 1.0,
            "speed_scale": 1.0,
        },
        "switch_reason": {
            "planner": planner_name,
            "planner_version": planner_version,
            "mode": switch_mode,
            "same_sequence_as_previous": False,
            **dict(extra_switch_reason or {}),
        },
        "expected_accent_hits": [],
        "rhythm_locks": [],
        "reference_artifacts": {},
        "rejected_top_candidates": [],
    }


def _score_candidate(
    candidate: dict[str, Any],
    previous_unit: dict[str, Any] | None,
    recent_units: Counter[str],
    tick: dict[str, Any],
    target_start_sec: float,
    target_end_sec: float,
    target_beats: int,
    target_style_profile: dict[str, Any],
    planner_version: str = "m9",
) -> tuple[float, dict[str, Any], list[dict[str, Any]], list[float]]:
    version = str(planner_version or "m9").lower()
    beat_phase = dict(tick.get("beat_phase", {}) or {})
    spacing = max(1e-3, _safe_float(beat_phase.get("spacing_sec"), (target_end_sec - target_start_sec) / max(1, target_beats)))
    visible_until = _safe_float(tick.get("available_audio_until_sec"), target_start_sec)
    rhythm_score, lock_reports, expected_hits = _rhythm_score(
        candidate=candidate,
        tick=tick,
        target_start_sec=target_start_sec,
        beat_spacing_sec=spacing,
        visible_until_sec=visible_until,
        target_beats=target_beats,
        planner_version=version,
    )
    transition = _transition_score(previous_unit, candidate)
    style = _music_style_score(candidate, target_style_profile=target_style_profile, target_bpm=_safe_float(beat_phase.get("bpm"), 120.0))
    repeat_penalty = 0.5 if version in {"m12", "m15"} else (0.65 if version == "m17" else 0.35)
    diversity = max(0.0, 1.0 - recent_units[str(candidate.get("unit_id"))] * repeat_penalty)
    is_m12 = version == "m12"
    is_m15 = version == "m15"
    is_m17 = version == "m17"
    weights = _planner_score_weights(version)
    total = (
        rhythm_score * weights["rhythm_lock"]
        + transition * weights["transition_smoothness"]
        + style * weights["style_energy_bpm"]
        + _safe_float(candidate.get("source_song_quality_weight"), 0.65) * weights["source_quality_weight"]
        + diversity * weights["diversity"]
    )
    source_duration = _candidate_source_duration(candidate)
    target_duration = max(1e-3, target_end_sec - target_start_sec)
    speed_scale = source_duration / target_duration
    safe_range = dict(candidate.get("safe_retime_range", {}) or {})
    min_retime = _safe_float(safe_range.get("min"), 0.85)
    max_retime = _safe_float(safe_range.get("max"), 1.15)
    if speed_scale < min_retime or speed_scale > max_retime:
        speed_penalty_gain = 2.8 if is_m17 else (2.4 if is_m12 else 1.8)
        total -= min(1.0, abs(speed_scale - _clamp(speed_scale, min_retime, max_retime)) * speed_penalty_gain)
    if abs(_safe_float(candidate.get("duration_beats"), target_beats) - target_beats) > 0.1:
        total -= 0.12
    if is_m17:
        if rhythm_score < 0.50:
            total -= 0.32
        if transition < 0.55:
            total -= 0.22
    elif is_m12 or is_m15:
        if rhythm_score < 0.35:
            total -= 0.25
        if transition < 0.45:
            total -= 0.18
    breakdown = {
        "rhythm_lock": round(float(rhythm_score), 5),
        "transition_smoothness": round(float(transition), 5),
        "style_energy_bpm": round(float(style), 5),
        "diversity": round(float(diversity), 5),
        "weighted_total": round(float(total), 5),
        "speed_scale": round(float(speed_scale), 5),
        "source_quality_weight": round(float(_safe_float(candidate.get("source_song_quality_weight"), 0.65)), 5),
    }
    return round(float(total), 5), breakdown, lock_reports, expected_hits


def simulate_streaming_smplx_plan_records(
    stream_event_records: list[dict[str, Any]],
    annotated_library: dict[str, Any],
    stream_events_path: str | None = None,
    max_steps: int = 0,
    planner_version: str = "m9",
    tail_policy: str = "none",
    source_sequence_allowlist: list[str] | tuple[str, ...] | None = None,
    initial_hold_sec: float = 0.0,
    initial_pose_mode: str = DEFAULT_INITIAL_POSE_MODE,
    cohort_size: int = DEFAULT_M15_COHORT_SIZE,
) -> list[dict[str, Any]]:
    header = _header(stream_event_records, "stream_header")
    ticks = _records_by_kind(stream_event_records, "tick")
    if not ticks:
        raise ValueError("stream_event_records contain no tick records")
    units = [dict(unit) for unit in list(annotated_library.get("units", []) or [])]
    allowed_sequences = {str(value) for value in list(source_sequence_allowlist or []) if str(value)}
    if allowed_sequences:
        units = [unit for unit in units if str(unit.get("source_sequence")) in allowed_sequences]
    if not units:
        raise ValueError("annotated motion library contains no units")

    sequence_profiles = _sequence_profiles(units)
    song_id = str(header.get("song_id") or "stream_song")
    duration_sec = _safe_float(header.get("duration_sec"), _safe_float(ticks[-1].get("available_audio_until_sec")))
    initial_buffer_sec = _safe_float(header.get("initial_buffer_sec"), DEFAULT_INITIAL_BUFFER_SEC)
    planning_lookahead_sec = _safe_float(header.get("lookahead_sec"), initial_buffer_sec)
    lookfront_sec = _safe_float(header.get("lookfront_sec"), 0.0)
    total_future_sec = _safe_float(header.get("total_future_sec"), planning_lookahead_sec + lookfront_sec)
    if total_future_sec <= 0.0:
        total_future_sec = initial_buffer_sec
    counts = sorted({_safe_int(unit.get("duration_beats"), 0) for unit in units if _safe_int(unit.get("duration_beats"), 0) > 0})
    preferred_beats = 4 if 4 in counts else (2 if 2 in counts else (8 if 8 in counts else counts[0]))
    planner_version = str(planner_version or "m9").lower()
    is_m12 = planner_version == "m12"
    is_m15 = planner_version == "m15"
    is_m17 = planner_version == "m17"
    tail_policy = str(tail_policy or "none").lower()
    initial_pose_mode = str(initial_pose_mode or DEFAULT_INITIAL_POSE_MODE).lower()
    cohort_size = max(1, int(cohort_size or DEFAULT_M15_COHORT_SIZE))
    initial_hold_sec = _clamp(_safe_float(initial_hold_sec), 0.0, max(0.0, duration_sec - 0.35))
    planner_name = _planner_name(planner_version)
    score_weights = _planner_score_weights(planner_version)

    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "kind": "stream_plan_header",
            "song_id": song_id,
            "library_id": annotated_library.get("library_id"),
            "source_stream_events_path": stream_events_path,
            "duration_sec": round(float(duration_sec), 5),
            "initial_buffer_sec": round(float(initial_buffer_sec), 5),
            "lookahead_sec": round(float(planning_lookahead_sec), 5),
            "lookfront_sec": round(float(lookfront_sec), 5),
            "total_future_sec": round(float(total_future_sec), 5),
            "planner": planner_name,
            "planner_version": planner_version,
            "tail_policy": tail_policy,
            "initial_hold_sec": round(float(initial_hold_sec), 5),
            "initial_pose_mode": initial_pose_mode,
            "cohort_size": int(cohort_size),
            "source_sequence_allowlist": sorted(allowed_sequences),
            "score_weights": score_weights,
            "generated_at_utc": utc_now_iso(),
        }
    ]

    next_start_sec = initial_hold_sec
    previous_unit: dict[str, Any] | None = None
    current_sequence_run = 0
    current_unit_run = 0
    recent_units: Counter[str] = Counter()
    step_index = 0

    while next_start_sec < duration_sec - 0.20:
        if max_steps > 0 and step_index >= max_steps:
            break

        decision_time = max(0.0, next_start_sec - total_future_sec)
        tick = _find_tick_for_decision(ticks, decision_time)
        beat_phase = dict(tick.get("beat_phase", {}) or {})
        tempo_confidence = _safe_float(beat_phase.get("confidence"), 0.0)
        if tempo_confidence < 0.28:
            effective_bpm = 120.0
            spacing = 0.5
            scoring_tick = dict(tick)
            scoring_tick["beat_phase"] = {
                **beat_phase,
                "bpm": effective_bpm,
                "spacing_sec": spacing,
                "confidence": tempo_confidence,
            }
        else:
            effective_bpm = _safe_float(beat_phase.get("bpm"), 120.0)
            spacing = max(0.18, _safe_float(beat_phase.get("spacing_sec"), 0.5))
            scoring_tick = tick
        visible_until_sec = _safe_float(tick.get("available_audio_until_sec"), next_start_sec)
        playhead = _safe_float(tick.get("playhead_sec"), 0.0)

        if is_m17 and previous_unit is None and next_start_sec >= initial_hold_sec - 1e-8:
            visible_boundaries = _planner_boundary_events(
                scoring_tick,
                start_sec=next_start_sec,
                available_until_sec=visible_until_sec,
                confidence_floor=M17_INITIAL_BOUNDARY_CONFIDENCE_FLOOR,
            )
            can_extend_initial_idle = next_start_sec < initial_hold_sec + M17_MAX_INITIAL_IDLE_EXTENSION_SEC - 1e-8
            if visible_boundaries:
                next_start_sec = max(next_start_sec, _safe_float(visible_boundaries[0].get("time_sec"), next_start_sec))
                decision_time = max(0.0, next_start_sec - total_future_sec)
                tick = _find_tick_for_decision(ticks, decision_time)
                beat_phase = dict(tick.get("beat_phase", {}) or {})
                tempo_confidence = _safe_float(beat_phase.get("confidence"), tempo_confidence)
                effective_bpm = _safe_float(beat_phase.get("bpm"), effective_bpm)
                spacing = max(0.18, _safe_float(beat_phase.get("spacing_sec"), spacing))
                if tempo_confidence >= 0.28:
                    scoring_tick = tick
                visible_until_sec = _safe_float(tick.get("available_audio_until_sec"), next_start_sec)
                playhead = _safe_float(tick.get("playhead_sec"), playhead)
            elif can_extend_initial_idle and (tempo_confidence < M17_LOW_CONFIDENCE_THRESHOLD or next_start_sec >= visible_until_sec - 1e-8):
                idle_end_sec = min(
                    duration_sec,
                    initial_hold_sec + M17_MAX_INITIAL_IDLE_EXTENSION_SEC,
                    max(next_start_sec + max(0.5, spacing), visible_until_sec),
                )
                if idle_end_sec <= next_start_sec + 1e-3:
                    idle_end_sec = min(duration_sec, next_start_sec + max(0.5, spacing))
                records.append(
                    _synthetic_idle_decision(
                        index=step_index,
                        planner_name=planner_name,
                        planner_version=planner_version,
                        initial_pose_mode=initial_pose_mode,
                        start_sec=next_start_sec,
                        end_sec=idle_end_sec,
                        playhead_sec=playhead,
                        available_audio_until_sec=visible_until_sec,
                        planning_lookahead_sec=planning_lookahead_sec,
                        lookfront_sec=lookfront_sec,
                        total_future_sec=total_future_sec,
                        decision_tick_index=_safe_int(tick.get("tick_index")),
                        decision_time_sec=decision_time,
                        target_bpm=effective_bpm,
                        target_energy="neutral_idle",
                        switch_mode="low_confidence_upright_idle",
                        extra_switch_reason={
                            "low_confidence_continuation": True,
                            "tempo_confidence": round(float(tempo_confidence), 5),
                        },
                    )
                )
                next_start_sec = idle_end_sec
                step_index += 1
                continue

        target_beats = (
            _choose_target_beats(units, spacing_sec=spacing, preferred_beats=preferred_beats)
            if tempo_confidence >= 0.28
            else min(preferred_beats, 4)
        )
        if is_m12 or is_m15 or is_m17:
            hypotheses = list(scoring_tick.get("segment_hypotheses", []) or [])
            viable = [
                item
                for item in hypotheses
                if _safe_float(item.get("start_time_sec"), -999.0) <= next_start_sec + spacing * 0.55
                and _safe_int(item.get("duration_beats"), 0) in counts
            ]
            viable.sort(key=lambda item: (_safe_float(item.get("confidence")), _safe_int(item.get("duration_beats")) in {4, 8}), reverse=True)
            if viable:
                target_beats = _safe_int(viable[0].get("duration_beats"), target_beats)
            if is_m15 and tempo_confidence < M15_LOW_CONFIDENCE_THRESHOLD:
                steady_options = [count for count in counts if count >= M15_LOW_CONFIDENCE_FALLBACK_BEATS]
                target_beats = min(steady_options, key=lambda count: abs(count - M15_LOW_CONFIDENCE_FALLBACK_BEATS)) if steady_options else max(counts)
            if is_m17 and tempo_confidence < M17_LOW_CONFIDENCE_THRESHOLD:
                steady_options = [count for count in counts if count >= M17_LOW_CONFIDENCE_FALLBACK_BEATS]
                target_beats = min(steady_options, key=lambda count: abs(count - M17_LOW_CONFIDENCE_FALLBACK_BEATS)) if steady_options else max(counts)
            if tail_policy == "recover":
                remaining_total = duration_sec - next_start_sec
                if remaining_total < target_beats * spacing * 0.75:
                    tail_options = [count for count in counts if count <= target_beats] or counts
                    target_beats = min(tail_options, key=lambda count: abs(count * spacing - remaining_total))

        target_end_sec = min(duration_sec, next_start_sec + target_beats * spacing)
        tail_extended = False
        if (is_m12 or is_m15 or is_m17) and tail_policy == "recover" and target_end_sec < duration_sec:
            remaining_after = duration_sec - target_end_sec
            if remaining_after < max(0.75, spacing * 1.25):
                target_end_sec = duration_sec
                tail_extended = True
        if target_end_sec - next_start_sec < 0.35:
            break

        target_style_profile = _target_style_profile_for_tick(
            scoring_tick,
            start_sec=next_start_sec,
            visible_end_sec=visible_until_sec,
            previous_unit=previous_unit,
        )
        target_energy = str(target_style_profile.get("energy", "mid_energy"))

        cohort_source_sequences: list[str] = []
        cohort_rankings: list[dict[str, Any]] = []
        if is_m15 or is_m17:
            target_bpm = _safe_float(dict(scoring_tick.get("beat_phase", {}) or {}).get("bpm"), 120.0)
            cohort_rankings = sorted(
                [
                    {
                        "sequence_id": sequence_id,
                        "score": _sequence_cohort_score(
                            profile,
                            target_bpm=target_bpm,
                            style_profile=target_style_profile,
                            previous_unit=previous_unit,
                        ),
                        "priority_tier": profile.get("priority_tier"),
                        "quality_weight": profile.get("quality_weight"),
                    }
                    for sequence_id, profile in sequence_profiles.items()
                ],
                key=lambda item: item["score"],
                reverse=True,
            )
            cohort_source_sequences = [str(item.get("sequence_id")) for item in cohort_rankings[:cohort_size]]

        candidates = [
            unit
            for unit in units
            if (not cohort_source_sequences or str(unit.get("source_sequence")) in cohort_source_sequences)
            and _safe_int(unit.get("duration_beats"), target_beats) == target_beats
        ] or units

        scored: list[tuple[float, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[float]]] = []
        for candidate in candidates:
            score, breakdown, lock_reports, expected_hits = _score_candidate(
                    candidate=candidate,
                    previous_unit=previous_unit,
                    recent_units=recent_units,
                    tick=scoring_tick,
                    target_start_sec=next_start_sec,
                    target_end_sec=target_end_sec,
                    target_beats=target_beats,
                    target_style_profile=target_style_profile,
                    planner_version=planner_version,
                )
            scored.append((score, candidate, breakdown, lock_reports, expected_hits))
        scored.sort(key=lambda item: item[0], reverse=True)

        hard_rejects: list[dict[str, Any]] = []
        valid_scored: list[tuple[float, dict[str, Any], dict[str, Any], list[dict[str, Any]], list[float]]] = []
        for score, candidate, breakdown, lock_reports, expected_hits in scored:
            reject_reasons: list[str] = []
            speed_scale = _safe_float(breakdown.get("speed_scale"), 1.0)
            if is_m15 and _safe_float(breakdown.get("rhythm_lock")) < M15_RHYTHM_HARD_MIN:
                reject_reasons.append("rhythm_lock_below_0_48")
            if is_m15 and _safe_float(breakdown.get("transition_smoothness")) < M15_TRANSITION_HARD_MIN:
                reject_reasons.append("transition_smoothness_below_0_50")
            if is_m17 and _safe_float(breakdown.get("rhythm_lock")) < M17_RHYTHM_HARD_MIN:
                reject_reasons.append("rhythm_lock_below_0_55")
            if is_m17 and _safe_float(breakdown.get("transition_smoothness")) < M17_TRANSITION_HARD_MIN:
                reject_reasons.append("transition_smoothness_below_0_55")
            if is_m15 and not tail_extended and (speed_scale < M15_NON_TAIL_SPEED_MIN or speed_scale > M15_NON_TAIL_SPEED_MAX):
                reject_reasons.append("non_tail_speed_outside_0_90_1_10")
            if is_m17 and not tail_extended and (speed_scale < M17_NON_TAIL_SPEED_MIN or speed_scale > M17_NON_TAIL_SPEED_MAX):
                reject_reasons.append("non_tail_speed_outside_0_92_1_08")
            if (
                is_m17
                and previous_unit is not None
                and str(previous_unit.get("unit_id")) == str(candidate.get("unit_id"))
            ):
                if current_unit_run >= M17_MAX_CONSECUTIVE_SAME_UNIT:
                    reject_reasons.append("same_unit_run_limit_3")
                elif current_unit_run >= M17_IDEAL_MAX_CONSECUTIVE_SAME_UNIT:
                    reject_reasons.append("same_unit_preferred_limit_2")
            if (
                is_m15
                and previous_unit is not None
                and str(previous_unit.get("source_sequence")) != str(candidate.get("source_sequence"))
                and _safe_float(breakdown.get("transition_smoothness")) < M15_CROSS_SEQUENCE_TRANSITION_FLOOR
            ):
                reject_reasons.append("cross_sequence_transition_below_0_62")
            if reject_reasons:
                hard_rejects.append(
                    {
                        "unit_id": candidate.get("unit_id"),
                        "source_sequence": candidate.get("source_sequence"),
                        "score": score,
                        "score_breakdown": breakdown,
                        "reasons": reject_reasons,
                    }
                )
                continue
            valid_scored.append((score, candidate, breakdown, lock_reports, expected_hits))

        effective_scored = valid_scored or scored
        if is_m17 and not valid_scored and scored:
            effective_scored = sorted(
                scored,
                key=lambda item: _m17_recovery_rank(
                    score=item[0],
                    candidate=item[1],
                    breakdown=item[2],
                    previous_unit=previous_unit,
                ),
                reverse=True,
            )
        elif is_m15 and not valid_scored and scored:
            effective_scored = sorted(
                scored,
                key=lambda item: _m15_recovery_rank(
                    score=item[0],
                    candidate=item[1],
                    breakdown=item[2],
                    previous_unit=previous_unit,
                ),
                reverse=True,
            )

        if previous_unit is not None and (is_m15 or is_m17):
            previous_unit_id = str(previous_unit.get("unit_id"))
            previous_sequence = str(previous_unit.get("source_sequence"))
            repeat_prefer_change_active = bool(is_m17 and current_unit_run >= M17_IDEAL_MAX_CONSECUTIVE_SAME_UNIT)
            repeat_limit_active = bool(is_m17 and current_unit_run >= M17_MAX_CONSECUTIVE_SAME_UNIT)
            same_unit_recovery = [item for item in effective_scored if str(item[1].get("unit_id")) == previous_unit_id]
            same_sequence_recovery = [item for item in effective_scored if str(item[1].get("source_sequence")) == previous_sequence]
            if is_m17 and same_unit_recovery and not repeat_prefer_change_active:
                effective_scored = same_unit_recovery + [item for item in effective_scored if item not in same_unit_recovery]
            if same_sequence_recovery:
                effective_scored = same_sequence_recovery + [item for item in effective_scored if item not in same_sequence_recovery]
            if repeat_prefer_change_active or repeat_limit_active:
                non_same_unit = [item for item in effective_scored if str(item[1].get("unit_id")) != previous_unit_id]
                if non_same_unit:
                    effective_scored = non_same_unit + [item for item in effective_scored if item not in non_same_unit]

        if not effective_scored:
            break

        score, selected, breakdown, lock_reports, expected_hits = effective_scored[0]
        if previous_unit is not None and (is_m15 or is_m17):
            previous_sequence = str(previous_unit.get("source_sequence"))
            selected_sequence = str(selected.get("source_sequence"))
            same_sequence_candidate = next((item for item in effective_scored if str(item[1].get("source_sequence")) == previous_sequence), None)
            if selected_sequence != previous_sequence and same_sequence_candidate is not None:
                same_score, same_selected, same_breakdown, same_lock_reports, same_expected_hits = same_sequence_candidate
                score_margin = score - same_score
                switch_margin = 0.075 if is_m17 else M15_SEQUENCE_SWITCH_MARGIN
                should_stick = current_sequence_run < M15_MIN_SEQUENCE_DWELL_STEPS or score_margin < switch_margin
                if should_stick:
                    hard_rejects.append(
                        {
                            "unit_id": selected.get("unit_id"),
                            "source_sequence": selected.get("source_sequence"),
                            "score": score,
                            "score_breakdown": breakdown,
                            "reasons": ["sequence_switch_margin_not_met"],
                        }
                    )
                    score, selected, breakdown, lock_reports, expected_hits = (
                        same_score,
                        same_selected,
                        same_breakdown,
                        same_lock_reports,
                        same_expected_hits,
                    )

        source_artifacts = dict(selected.get("reference_artifacts", {}) or {})
        available_until = _safe_float(tick.get("available_audio_until_sec"), 0.0)
        playhead = _safe_float(tick.get("playhead_sec"), 0.0)
        guard_passed = available_until <= playhead + total_future_sec + 1e-5 and next_start_sec <= available_until + 1e-5
        previous_unit_id = str(previous_unit.get("unit_id")) if previous_unit is not None else None
        selected_unit_id = str(selected.get("unit_id"))
        selected_unit_run = current_unit_run + 1 if previous_unit_id == selected_unit_id else 1
        low_confidence_continuation = bool(
            is_m17
            and tempo_confidence < M17_LOW_CONFIDENCE_THRESHOLD
            and previous_unit is not None
            and str(previous_unit.get("source_sequence")) == str(selected.get("source_sequence"))
        )
        decision = {
            "schema_version": 1,
            "kind": "decision",
            "index": step_index,
            "decision_time_sec": _round_time(decision_time),
            "decision_tick_index": _safe_int(tick.get("tick_index")),
            "playhead_sec": _round_time(playhead),
            "available_audio_until_sec": _round_time(available_until),
            "future_visibility_guard": {
                "lookahead_sec": round(float(planning_lookahead_sec), 5),
                "lookfront_sec": round(float(lookfront_sec), 5),
                "total_future_sec": round(float(total_future_sec), 5),
                "used_audio_until_sec": _round_time(available_until),
                "passed": bool(guard_passed),
            },
            "target_time_sec": {"start": _round_time(next_start_sec), "end": _round_time(target_end_sec)},
            "target_beats": int(target_beats),
            "target_bpm": round(float(effective_bpm), 5),
            "target_energy": target_energy,
            "selected_unit_id": selected.get("unit_id"),
            "source_sequence": selected.get("source_sequence"),
            "selected_from_sequence": selected.get("source_sequence"),
            "selected_from_tier": str(selected.get("priority_tier", "unknown") or "unknown"),
            "pose_source": "finedance_motion_unit",
            "initial_pose_mode": initial_pose_mode,
            "cohort_source_sequences": cohort_source_sequences,
            "cohort_rankings": cohort_rankings[: min(len(cohort_rankings), max(4, cohort_size))],
            "source_frame_range": dict(selected.get("frame_range", {}) or {}),
            "source_beat_range": dict(selected.get("beat_range", {}) or {}),
            "source_motion_path": source_artifacts.get("source_motion_path"),
            "speed_scale": breakdown["speed_scale"],
            "score": score,
            "score_breakdown": breakdown,
            "switch_reason": {
                "planner": planner_name,
                "planner_version": planner_version,
                "mode": "retrieval",
                "fallback": bool((tempo_confidence < (M17_LOW_CONFIDENCE_THRESHOLD if is_m17 else 0.28)) or not valid_scored),
                "target_energy": target_energy,
                "candidate_energy": selected.get("energy"),
                "compatible_from_previous": bool(previous_unit and selected.get("unit_id") in previous_unit.get("compatible_next_units", [])),
                "tail_policy": tail_policy,
                "tail_extended_to_song_end": bool(tail_extended),
                "same_sequence_as_previous": bool(previous_unit and previous_unit.get("source_sequence") == selected.get("source_sequence")),
                "same_unit_as_previous": bool(previous_unit_id == selected_unit_id),
                "same_unit_run": int(selected_unit_run),
                "same_unit_ideal_run_limit": int(M17_IDEAL_MAX_CONSECUTIVE_SAME_UNIT) if is_m17 else None,
                "same_unit_hard_run_limit": int(M17_MAX_CONSECUTIVE_SAME_UNIT) if is_m17 else None,
                "low_confidence_continuation": low_confidence_continuation,
                "tempo_confidence": round(float(tempo_confidence), 5),
            },
            "expected_accent_hits": expected_hits,
            "rhythm_locks": lock_reports,
            "reference_artifacts": source_artifacts,
            "rejected_top_candidates": hard_rejects[:3],
        }
        records.append(decision)
        selected_sequence = str(selected.get("source_sequence"))
        previous_sequence = str(previous_unit.get("source_sequence")) if previous_unit is not None else None
        current_sequence_run = current_sequence_run + 1 if previous_sequence == selected_sequence else 1
        current_unit_run = selected_unit_run
        previous_unit = selected
        recent_units.update([str(selected.get("unit_id"))])
        next_start_sec = target_end_sec
        step_index += 1

    if initial_hold_sec > 0.0:
        decisions = [record for record in records if record.get("kind") == "decision"]
        insert_at = next((index for index, record in enumerate(records) if record.get("kind") == "decision"), len(records))
        first_decision_start_sec = (
            _safe_float(dict(decisions[0].get("target_time_sec", {}) or {}).get("start"))
            if decisions
            else initial_hold_sec
        )
        initial_hold_end_sec = max(initial_hold_sec, first_decision_start_sec)
        if initial_pose_mode == "freeze_first" and decisions:
            first_decision = decisions[0]
            first_range = dict(first_decision.get("source_frame_range", {}) or {})
            hold_start_frame = _safe_int(first_range.get("start"))
            first_guard = dict(first_decision.get("future_visibility_guard", {}) or {})
            records.insert(
                insert_at,
                {
                    "schema_version": 1,
                    "kind": "decision",
                    "index": 0,
                    "decision_time_sec": 0.0,
                    "decision_tick_index": 0,
                    "playhead_sec": 0.0,
                    "available_audio_until_sec": round(float(total_future_sec), 5),
                    "future_visibility_guard": {
                        "lookahead_sec": round(float(planning_lookahead_sec), 5),
                        "lookfront_sec": round(float(lookfront_sec), 5),
                        "total_future_sec": round(float(total_future_sec), 5),
                        "used_audio_until_sec": first_guard.get("used_audio_until_sec", round(float(total_future_sec), 5)),
                        "passed": True,
                    },
                    "target_time_sec": {"start": 0.0, "end": _round_time(initial_hold_end_sec)},
                    "target_beats": 0,
                    "target_bpm": first_decision.get("target_bpm"),
                    "target_energy": "hold",
                    "selected_unit_id": f"{first_decision.get('selected_unit_id')}_initial_hold",
                    "source_sequence": first_decision.get("source_sequence"),
                    "selected_from_sequence": first_decision.get("selected_from_sequence"),
                    "selected_from_tier": first_decision.get("selected_from_tier"),
                    "pose_source": SYNTHETIC_FREEZE_FIRST_SOURCE,
                    "initial_pose_mode": initial_pose_mode,
                    "source_frame_range": {"start": hold_start_frame, "end_exclusive": hold_start_frame + 1},
                    "source_beat_range": dict(first_decision.get("source_beat_range", {}) or {}),
                    "source_motion_path": first_decision.get("source_motion_path"),
                    "speed_scale": 0.0,
                    "score": 1.0,
                    "score_breakdown": {
                        "rhythm_lock": 1.0,
                        "transition_smoothness": 1.0,
                        "style_energy_bpm": 1.0,
                        "source_quality_weight": 1.0,
                        "diversity": 1.0,
                        "weighted_total": 1.0,
                        "speed_scale": 0.0,
                    },
                    "switch_reason": {
                        "planner": planner_name,
                        "planner_version": planner_version,
                        "mode": "initial_hold",
                        "hold_until_sec": _round_time(initial_hold_end_sec),
                    },
                    "expected_accent_hits": [],
                    "rhythm_locks": [],
                    "reference_artifacts": dict(first_decision.get("reference_artifacts", {}) or {}),
                    "rejected_top_candidates": [],
                },
            )
        else:
            records.insert(
                insert_at,
                _synthetic_idle_decision(
                    index=0,
                    planner_name=planner_name,
                    planner_version=planner_version,
                    initial_pose_mode=initial_pose_mode,
                    start_sec=0.0,
                    end_sec=initial_hold_end_sec,
                    playhead_sec=0.0,
                    available_audio_until_sec=min(duration_sec, total_future_sec),
                    planning_lookahead_sec=planning_lookahead_sec,
                    lookfront_sec=lookfront_sec,
                    total_future_sec=total_future_sec,
                    decision_tick_index=0,
                    decision_time_sec=0.0,
                    target_bpm=120.0,
                    target_energy="neutral_idle",
                    switch_mode="initial_upright_hold",
                    extra_switch_reason={"hold_until_sec": _round_time(initial_hold_end_sec)},
                ),
            )
        for new_index, decision in enumerate(record for record in records if record.get("kind") == "decision"):
            decision["index"] = new_index

    decisions = [record for record in records if record.get("kind") == "decision"]
    records.append(
        {
            "schema_version": 1,
            "kind": "stream_plan_summary",
            "song_id": song_id,
            "step_count": len(decisions),
            "planned_until_sec": _round_time(next_start_sec),
            "duration_sec": round(float(duration_sec), 5),
            "future_visibility_violations": sum(
                1
                for item in decisions
                if not bool(dict(item.get("future_visibility_guard", {}) or {}).get("passed"))
            ),
            "generated_at_utc": utc_now_iso(),
        }
    )
    return records


def _dedupe_timed_events(records: list[dict[str, Any]], key: str, confidence_floor: float = 0.0) -> list[dict[str, Any]]:
    by_time: dict[float, dict[str, Any]] = {}
    previous_available_until = -1e-4
    for tick in sorted(_records_by_kind(records, "tick"), key=lambda item: _safe_float(item.get("available_audio_until_sec"), 0.0)):
        available_until = _safe_float(tick.get("available_audio_until_sec"), 0.0)
        if available_until + 1e-8 < previous_available_until:
            previous_available_until = available_until
        for event in list(tick.get(key, []) or []):
            time_sec = _round_time(_safe_float(event.get("time_sec")))
            if time_sec > available_until + 1e-8:
                continue
            if time_sec <= previous_available_until + 1e-8:
                continue
            if _safe_float(event.get("confidence"), _safe_float(event.get("strength"))) < confidence_floor:
                continue
            slot = round(time_sec, 3)
            current = by_time.get(slot)
            if current is None or _safe_float(event.get("strength"), _safe_float(event.get("confidence"))) > _safe_float(current.get("strength"), _safe_float(current.get("confidence"))):
                by_time[slot] = dict(event, time_sec=time_sec)
        previous_available_until = max(previous_available_until, available_until)
    events = sorted(by_time.values(), key=lambda item: _safe_float(item.get("time_sec")))
    for index, event in enumerate(events):
        event["index"] = index
    return events


def stream_events_to_song_event_map(stream_records: list[dict[str, Any]]) -> dict[str, Any]:
    header = _header(stream_records, "stream_header")
    beats = _dedupe_timed_events(stream_records, "beats", confidence_floor=0.18)
    beats_per_bar = max(1, _safe_int(header.get("beats_per_bar"), 4))
    for index, beat in enumerate(beats):
        beat["index"] = index
        beat["count"] = int(index % beats_per_bar) + 1
        beat["is_downbeat"] = bool(index % beats_per_bar == 0)
    downbeats = [
        {
            "index": len([item for item in beats[:idx] if item.get("is_downbeat")]),
            "time_sec": beat["time_sec"],
            "source_beat_index": idx,
            "confidence": beat.get("confidence", 0.0),
        }
        for idx, beat in enumerate(beats)
        if bool(beat.get("is_downbeat"))
    ]
    accents = _dedupe_timed_events(stream_records, "accents", confidence_floor=0.0)
    drum_hits = _dedupe_timed_events(stream_records, "drum_hits", confidence_floor=0.0)
    duration_sec = _safe_float(header.get("duration_sec"), max([_safe_float(item.get("time_sec")) for item in beats] or [0.0]))
    return {
        "schema_version": 1,
        "song_id": header.get("song_id", "stream_song"),
        "source_audio_path": header.get("source_audio_path", ""),
        "duration_sec": round(float(duration_sec), 5),
        "beats_per_bar": beats_per_bar,
        "tempo_hypotheses": [
            {
                "label": "stream_primary",
                "bpm": round(60.0 / max(1e-3, float(np.median(np.diff([_safe_float(item.get("time_sec")) for item in beats])))) if len(beats) > 1 else 120.0, 5),
                "confidence": round(float(np.mean([_safe_float(item.get("confidence")) for item in beats])) if beats else 0.0, 5),
            }
        ],
        "beats": beats,
        "downbeats": downbeats,
        "accents": accents,
        "drum_hits": drum_hits,
        "phrases": [],
        "sections": [],
        "confidence": {
            "beat_stability": round(float(np.mean([_safe_float(item.get("confidence")) for item in beats])) if beats else 0.0, 5),
            "streaming": True,
        },
        "manual_review_required": True,
        "analysis_mode": "streaming_simulation",
        "override_source": None,
        "applied_override_keys": [],
        "notes": ["Aggregated from M9 streaming tick records; event times never exceed each tick's available_audio_until_sec."],
        "generated_at_utc": utc_now_iso(),
    }


def _cache_request(plan_id: str, sequence_id: str, source_motion_path: str, frame_ranges: list[dict[str, int]]) -> dict[str, Any]:
    if frame_ranges:
        start_frame = min(item["start"] for item in frame_ranges)
        end_frame = max(item["end_exclusive"] for item in frame_ranges)
    else:
        start_frame = 0
        end_frame = 0
    cache_name = f"{slugify(sequence_id)}_{start_frame:05d}_{max(start_frame, end_frame - 1):05d}.npz"
    return {
        "sequence_id": sequence_id,
        "source_motion_path": source_motion_path,
        "frame_ranges": frame_ranges,
        "cache_path": f"outputs/smplx_mesh_previews/cache/{slugify(plan_id)}/{cache_name}",
        "cache_format": "npz: vertices float32 [frames, verts, 3], joints float32 [frames, joints, 3], faces int32 [faces, 3]",
    }


def stream_plan_to_stitch_manifest(
    stream_plan_records: list[dict[str, Any]],
    fps: int = DEFAULT_STREAM_FPS,
    blend_frames: int = DEFAULT_BLEND_FRAMES,
) -> dict[str, Any]:
    header = _header(stream_plan_records, "stream_plan_header")
    decisions = _records_by_kind(stream_plan_records, "decision")
    if not decisions:
        raise ValueError("stream plan contains no decision records")
    plan_digest = stable_digest(
        {
            "song_id": header.get("song_id"),
            "planner_version": header.get("planner_version"),
            "cohort_size": header.get("cohort_size"),
            "initial_hold_sec": header.get("initial_hold_sec"),
            "decision_signature": [
                {
                    "unit_id": item.get("selected_unit_id"),
                    "pose_source": item.get("pose_source"),
                    "source_sequence": item.get("source_sequence"),
                    "source_frame_range": item.get("source_frame_range"),
                    "target_time_sec": item.get("target_time_sec"),
                }
                for item in decisions
            ],
        }
    )[:8]
    plan_id = f"{header.get('song_id', 'stream_song')}_{plan_digest}_streaming_smplx_plan"
    ranges_by_sequence: dict[str, list[dict[str, int]]] = defaultdict(list)
    source_path_by_sequence: dict[str, str] = {}
    steps: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []
    previous_step: dict[str, Any] | None = None
    for decision in decisions:
        index = _safe_int(decision.get("index"), len(steps))
        target = dict(decision.get("target_time_sec", {}) or {})
        start_time = _safe_float(target.get("start"))
        end_time = max(start_time + 1e-3, _safe_float(target.get("end"), start_time + 1.0))
        scene_start = int(round(start_time * fps)) + 1 if previous_step is None else _safe_int(previous_step.get("scene_frame_end")) + 1
        scene_end = max(scene_start, int(round(end_time * fps)))
        source_range = dict(decision.get("source_frame_range", {}) or {})
        source_start = _safe_int(source_range.get("start"))
        source_end = max(source_start + 1, _safe_int(source_range.get("end_exclusive"), source_start + 1))
        source_sequence = str(decision.get("source_sequence", "unknown") or "unknown")
        pose_source = str(decision.get("pose_source") or "finedance_motion_unit")
        is_synthetic_pose = pose_source in {
            SYNTHETIC_NEUTRAL_IDLE_SOURCE,
            SYNTHETIC_NEUTRAL_REST_SOURCE,
        }
        source_motion_path = str(decision.get("source_motion_path") or dict(decision.get("reference_artifacts", {}) or {}).get("source_motion_path") or "")
        if not source_motion_path and not is_synthetic_pose:
            raise FileNotFoundError(f"stream decision {index} has no source_motion_path")
        if source_motion_path:
            ranges_by_sequence[source_sequence].append({"start": source_start, "end_exclusive": source_end})
            source_path_by_sequence[source_sequence] = source_motion_path
        rhythm_locks = []
        for lock_index, lock in enumerate(list(decision.get("rhythm_locks", []) or [])):
            lock_time = _safe_float(lock.get("time_sec"))
            if lock_time < start_time - 1e-8 or lock_time > end_time + (0.5 / max(1, fps)):
                continue
            rhythm_locks.append(
                {
                    "index": lock_index,
                    "kind": str(lock.get("kind", "stream_rhythm_lock")),
                    "role": lock.get("role"),
                    "time_sec": _round_time(lock_time),
                    "scene_frame": int(round(lock_time * fps)) + 1,
                    "visible_at_decision": bool(lock.get("visible_at_decision")),
                    "score": _safe_float(lock.get("score")),
                }
            )
        step = {
            "index": index,
            "unit_id": decision.get("selected_unit_id"),
            "source_sequence": source_sequence,
            "source_motion_path": source_motion_path or None,
            "pose_source": pose_source,
            "initial_pose_mode": decision.get("initial_pose_mode"),
            "source_frame_start": source_start,
            "source_frame_end_exclusive": source_end,
            "source_beat_range": dict(decision.get("source_beat_range", {}) or {}),
            "scene_frame_start": scene_start,
            "scene_frame_end": scene_end,
            "start_time_sec": _round_time(start_time),
            "end_time_sec": _round_time(end_time),
            "speed_scale": _safe_float(decision.get("speed_scale"), 1.0),
            "blend_in_frames": min(blend_frames if previous_step is not None else 0, max(0, (scene_end - scene_start) // 3)),
            "blend_out_frames": min(blend_frames if index < len(decisions) - 1 else 0, max(0, (scene_end - scene_start) // 3)),
            "rhythm_locks": rhythm_locks,
            "root_alignment": {
                "mode": "streaming_pelvis_xyz_velocity_continuity",
                "preserve_beat_locked_frames": True,
            },
            "transition_score": _safe_float(dict(decision.get("score_breakdown", {}) or {}).get("transition_smoothness")),
            "section_label": "streaming",
            "streaming_decision": {
                "decision_time_sec": decision.get("decision_time_sec"),
                "playhead_sec": decision.get("playhead_sec"),
                "available_audio_until_sec": decision.get("available_audio_until_sec"),
                "score": decision.get("score"),
                "score_breakdown": dict(decision.get("score_breakdown", {}) or {}),
                "target_energy": decision.get("target_energy"),
                "future_visibility_guard": dict(decision.get("future_visibility_guard", {}) or {}),
            },
        }
        if previous_step is not None:
            transitions.append(
                {
                    "index": len(transitions),
                    "outgoing_step": previous_step["index"],
                    "incoming_step": index,
                    "boundary_frame": step["scene_frame_start"],
                    "blend_frames": min(blend_frames, step["blend_in_frames"], previous_step["blend_out_frames"]),
                    "incoming_transition_score": step["transition_score"],
                    "mode": "streaming_root_continuity_plus_visual_smoothing",
                }
            )
        steps.append(step)
        previous_step = step
    cache_requests = [
        _cache_request(plan_id=plan_id, sequence_id=sequence_id, source_motion_path=source_path_by_sequence[sequence_id], frame_ranges=frame_ranges)
        for sequence_id, frame_ranges in sorted(ranges_by_sequence.items())
    ]
    return {
        "schema_version": 1,
        "manifest_id": f"{plan_id}_smplx_stitch_preview",
        "plan_id": plan_id,
        "song_id": header.get("song_id"),
        "library_id": header.get("library_id"),
        "generated_at_utc": utc_now_iso(),
        "fps": int(fps),
        "mesh_backend": "smplx_neutral",
        "blend_policy": {
            "default_blend_frames": int(blend_frames),
            "rotation": "streaming_joint_retime",
            "root_translation": "continuous_xyz_offset",
            "beat_locks": "preserve_target_scene_frames",
        },
        "scene": {"frame_start": 1, "frame_end": max(step["scene_frame_end"] for step in steps)},
        "steps": steps,
        "transitions": transitions,
        "cache_requests": cache_requests,
        "streaming_review": {
            "header": header,
            "decisions": [
                {
                    "index": item.get("index"),
                    "decision_time_sec": item.get("decision_time_sec"),
                    "playhead_sec": item.get("playhead_sec"),
                    "available_audio_until_sec": item.get("available_audio_until_sec"),
                    "target_time_sec": item.get("target_time_sec"),
                    "target_beats": item.get("target_beats"),
                    "selected_unit_id": item.get("selected_unit_id"),
                    "pose_source": item.get("pose_source"),
                    "initial_pose_mode": item.get("initial_pose_mode"),
                    "source_sequence": item.get("source_sequence"),
                    "selected_from_sequence": item.get("selected_from_sequence"),
                    "selected_from_tier": item.get("selected_from_tier"),
                    "cohort_source_sequences": item.get("cohort_source_sequences"),
                    "cohort_rankings": item.get("cohort_rankings"),
                    "source_frame_range": item.get("source_frame_range"),
                    "target_bpm": item.get("target_bpm"),
                    "target_energy": item.get("target_energy"),
                    "speed_scale": item.get("speed_scale"),
                    "score": item.get("score"),
                    "score_breakdown": item.get("score_breakdown"),
                    "switch_reason": item.get("switch_reason"),
                    "future_visibility_guard": item.get("future_visibility_guard"),
                    "rejected_top_candidates": item.get("rejected_top_candidates"),
                }
                for item in decisions
            ],
        },
        "notes": [
            "M9 streaming SMPL-X stitch manifest generated from decision JSONL.",
            "Streaming decisions are made with a two-second visible-audio guard; target durations may extend beyond the visible window via beat-phase prediction.",
        ],
    }


def render_streaming_smplx_mesh_review(
    stream_plan_records: list[dict[str, Any]],
    stream_event_records: list[dict[str, Any]] | None,
    project_root: Path,
    motion_base_assets_root: Path,
    audio_path: Path,
    output_video: Path,
    output_strip: Path,
    output_report: Path,
    output_html: Path,
    manifest_output: Path | None = None,
    fps: int = DEFAULT_STREAM_FPS,
    blend_frames: int = DEFAULT_BLEND_FRAMES,
    force_cache: bool = False,
    batch_size: int = 128,
    face_stride: int = 30,
    render_frame_stride: int = 2,
    max_render_frames: int = 0,
    transition_smooth_frames: int = 12,
    transition_smooth_passes: int = 2,
) -> dict[str, Any]:
    manifest = stream_plan_to_stitch_manifest(stream_plan_records, fps=fps, blend_frames=blend_frames)
    if manifest_output is not None:
        write_json(manifest_output, manifest)
    song_event_map = stream_events_to_song_event_map(stream_event_records or []) if stream_event_records else None
    report = build_smplx_mesh_stitch_visual_preview(
        manifest=manifest,
        project_root=project_root,
        motion_base_assets_root=motion_base_assets_root,
        output_video=output_video,
        output_strip=output_strip,
        output_report=output_report,
        output_html=output_html,
        song_event_map=song_event_map,
        audio_path=audio_path,
        force_cache=force_cache,
        batch_size=batch_size,
        face_stride=face_stride,
        render_frame_stride=render_frame_stride,
        max_render_frames=max_render_frames,
        transition_smooth_frames=transition_smooth_frames,
        transition_smooth_passes=transition_smooth_passes,
    )
    report["streaming"] = manifest.get("streaming_review", {})
    decisions = list(report["streaming"].get("decisions", []) or [])
    report["metrics"]["future_visibility_violations"] = sum(
        1
        for item in decisions
        if not bool(dict(item.get("future_visibility_guard", {}) or {}).get("passed"))
    )
    speed_values = [_safe_float(item.get("speed_scale"), 1.0) for item in decisions]
    rejected_candidates = [candidate for item in decisions for candidate in list(item.get("rejected_top_candidates", []) or [])]
    report["metrics"]["streaming_decision_count"] = len(decisions)
    report["metrics"]["speed_scale_outside_default_range_count"] = sum(1 for value in speed_values if value < 0.85 or value > 1.15)
    report["metrics"]["max_streaming_speed_scale"] = round(max(speed_values or [1.0]), 6)
    report["metrics"]["min_streaming_speed_scale"] = round(min(speed_values or [1.0]), 6)
    report["metrics"]["cross_sequence_transition_count"] = sum(
        1 for previous, current in zip(decisions, decisions[1:]) if str(previous.get("source_sequence")) != str(current.get("source_sequence"))
    )
    report["metrics"]["rhythm_hard_reject_count"] = sum(
        1
        for item in rejected_candidates
        if "rhythm_lock_below_0_40" in list(item.get("reasons", []) or [])
        or "rhythm_lock_below_0_48" in list(item.get("reasons", []) or [])
        or "rhythm_lock_below_0_55" in list(item.get("reasons", []) or [])
    )
    report["metrics"]["non_tail_speed_hard_reject_count"] = sum(
        1
        for item in rejected_candidates
        if "non_tail_speed_outside_0_90_1_10" in list(item.get("reasons", []) or [])
        or "non_tail_speed_outside_0_92_1_08" in list(item.get("reasons", []) or [])
    )
    report["metrics"]["transition_hard_reject_count"] = sum(
        1
        for item in rejected_candidates
        if "transition_smoothness_below_0_45" in list(item.get("reasons", []) or [])
        or "transition_smoothness_below_0_50" in list(item.get("reasons", []) or [])
        or "transition_smoothness_below_0_55" in list(item.get("reasons", []) or [])
        or "cross_sequence_transition_below_0_62" in list(item.get("reasons", []) or [])
    )
    report["metrics"]["repeat_unit_hard_reject_count"] = sum(
        1 for item in rejected_candidates if "same_unit_run_limit_3" in list(item.get("reasons", []) or [])
    )
    report["metrics"]["repeat_unit_preferred_reject_count"] = sum(
        1 for item in rejected_candidates if "same_unit_preferred_limit_2" in list(item.get("reasons", []) or [])
    )
    report["metrics"]["max_consecutive_motion_unit_run"] = _max_consecutive_motion_unit_run(decisions)
    report["metrics"]["low_confidence_continuation_count"] = sum(
        1 for item in decisions if bool(dict(item.get("switch_reason", {}) or {}).get("low_confidence_continuation"))
    )
    report["metrics"]["initial_upright_hold_sec"] = round(
        sum(
            _safe_float(dict(item.get("target_time_sec", {}) or {}).get("end"))
            - _safe_float(dict(item.get("target_time_sec", {}) or {}).get("start"))
            for item in decisions
            if str(dict(item.get("switch_reason", {}) or {}).get("mode")) == "initial_upright_hold"
        ),
        5,
    )
    dance_starts = [
        _safe_float(dict(item.get("target_time_sec", {}) or {}).get("start"))
        for item in decisions
        if str(item.get("pose_source") or "finedance_motion_unit") == "finedance_motion_unit"
    ]
    report["metrics"]["first_dance_start_sec"] = round(float(min(dance_starts) if dance_starts else 0.0), 5)
    report["metrics"]["cohort_song_count"] = len(
        {str(sequence_id) for decision in decisions for sequence_id in list(decision.get("cohort_source_sequences", []) or [])}
    )
    report["metrics"]["max_streaming_lookahead_sec"] = round(
        max(
            [
                _safe_float(item.get("available_audio_until_sec")) - _safe_float(item.get("playhead_sec"))
                for item in decisions
            ]
            or [0.0]
        ),
        6,
    )
    write_json(output_report, report)
    return report


def calibrate_song_event_rail(
    stream_event_records: list[dict[str, Any]],
    manual_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    event_map = stream_events_to_song_event_map(stream_event_records)
    overrides = dict(manual_overrides or {})
    applied: list[str] = []
    for key in ("beats", "downbeats", "accents", "drum_hits", "phrases", "sections"):
        if key in overrides:
            values = [dict(item) for item in list(overrides.get(key) or [])]
            values.sort(key=lambda item: _safe_float(item.get("time_sec"), _safe_float(item.get("start_time_sec"))))
            for index, item in enumerate(values):
                item.setdefault("index", index)
            event_map[key] = values
            applied.append(key)
    if "tempo_hypotheses" in overrides:
        event_map["tempo_hypotheses"] = list(overrides.get("tempo_hypotheses") or [])
        applied.append("tempo_hypotheses")
    event_map["manual_review_required"] = not bool(applied)
    event_map["override_source"] = overrides.get("override_source")
    event_map["applied_override_keys"] = applied
    event_map["calibration_schema_version"] = 1
    event_map["notes"] = [
        *list(event_map.get("notes", []) or []),
        "M11 calibration map; manual override keys replace the corresponding streaming aggregate rails.",
    ]
    event_map["generated_at_utc"] = utc_now_iso()
    return event_map


def _event_times(payload: dict[str, Any], key: str) -> list[float]:
    return sorted(_safe_float(item.get("time_sec")) for item in list(payload.get(key, []) or []))


def _match_time_series(detected: list[float], reference: list[float], tolerance_sec: float) -> dict[str, Any]:
    used: set[int] = set()
    deltas: list[float] = []
    for detected_time in detected:
        best_index = -1
        best_delta = float("inf")
        for index, reference_time in enumerate(reference):
            if index in used:
                continue
            delta = abs(detected_time - reference_time)
            if delta < best_delta:
                best_delta = delta
                best_index = index
        if best_index >= 0 and best_delta <= tolerance_sec:
            used.add(best_index)
            deltas.append(best_delta)
    return {
        "detected_count": len(detected),
        "reference_count": len(reference),
        "matched_count": len(deltas),
        "precision": round(float(len(deltas) / max(1, len(detected))), 5),
        "recall": round(float(len(deltas) / max(1, len(reference))), 5),
        "mae_sec": round(float(np.mean(deltas)) if deltas else 0.0, 6),
        "max_error_sec": round(float(max(deltas)) if deltas else 0.0, 6),
    }


def evaluate_streaming_event_rail(
    stream_event_records: list[dict[str, Any]],
    reference_event_map: dict[str, Any] | None = None,
    tolerance_sec: float = 2.0 / DEFAULT_STREAM_FPS,
) -> dict[str, Any]:
    aggregate = stream_events_to_song_event_map(stream_event_records)
    reference = dict(reference_event_map or aggregate)
    ticks = _records_by_kind(stream_event_records, "tick")
    future_event_violations = 0
    for tick in ticks:
        available_until = _safe_float(tick.get("available_audio_until_sec"), 0.0)
        for key in ("beats", "downbeats", "drum_hits", "accents"):
            for event in list(tick.get(key, []) or []):
                if _safe_float(event.get("time_sec")) > available_until + 1e-8:
                    future_event_violations += 1
    beat_eval = _match_time_series(_event_times(aggregate, "beats"), _event_times(reference, "beats"), tolerance_sec)
    downbeat_eval = _match_time_series(_event_times(aggregate, "downbeats"), _event_times(reference, "downbeats"), tolerance_sec)
    drum_eval = _match_time_series(_event_times(aggregate, "drum_hits"), _event_times(reference, "drum_hits"), tolerance_sec)
    reference_beat_count = max(1, len(_event_times(reference, "beats")))
    overcount_ratio = len(_event_times(aggregate, "beats")) / reference_beat_count
    return {
        "schema_version": 1,
        "report_id": f"{aggregate.get('song_id', 'stream_song')}_streaming_event_rail_eval",
        "song_id": aggregate.get("song_id"),
        "generated_at_utc": utc_now_iso(),
        "tolerance_sec": round(float(tolerance_sec), 6),
        "metrics": {
            "beat_timing_mae_sec": beat_eval["mae_sec"],
            "beat_precision": beat_eval["precision"],
            "beat_recall": beat_eval["recall"],
            "beat_overcount_ratio": round(float(overcount_ratio), 5),
            "downbeat_hit_rate": downbeat_eval["recall"],
            "drum_hit_precision": drum_eval["precision"],
            "future_event_violations": future_event_violations,
            "high_confidence_lock_error_frames": round(float(beat_eval["mae_sec"] * DEFAULT_STREAM_FPS), 5),
        },
        "beats": beat_eval,
        "downbeats": downbeat_eval,
        "drum_hits": drum_eval,
        "acceptance": {
            "future_safe": future_event_violations == 0,
            "high_confidence_lock_error_within_2_frames": beat_eval["mae_sec"] * DEFAULT_STREAM_FPS <= 2.0,
            "no_obvious_double_time_overcount": overcount_ratio <= 1.35,
        },
        "notes": [
            "M11 streaming event-rail evaluation compares aggregate streaming events against a calibrated reference map when provided.",
            "When no reference is provided, the aggregate stream rail is used as a self-consistency baseline.",
        ],
    }


def evaluate_streaming_planner_records(stream_plan_records: list[dict[str, Any]]) -> dict[str, Any]:
    header = _header(stream_plan_records, "stream_plan_header")
    decisions = _records_by_kind(stream_plan_records, "decision")
    speeds = [_safe_float(item.get("speed_scale"), 1.0) for item in decisions]
    gaps: list[dict[str, Any]] = []
    for previous, current in zip(decisions, decisions[1:]):
        previous_end = _safe_float(dict(previous.get("target_time_sec", {}) or {}).get("end"))
        current_start = _safe_float(dict(current.get("target_time_sec", {}) or {}).get("start"))
        if abs(current_start - previous_end) > 1e-4:
            gaps.append({"previous_index": previous.get("index"), "current_index": current.get("index"), "gap_sec": round(current_start - previous_end, 6)})
    non_tail_speeds = speeds[:-1] if len(speeds) > 1 else speeds
    outside = [value for value in speeds if value < DEFAULT_SAFE_RETIME_MIN or value > DEFAULT_SAFE_RETIME_MAX]
    non_tail_outside = [value for value in non_tail_speeds if value < DEFAULT_SAFE_RETIME_MIN or value > DEFAULT_SAFE_RETIME_MAX]
    rejected_candidates = [item for decision in decisions for item in list(decision.get("rejected_top_candidates", []) or [])]
    rhythm_reject_count = sum(
        1
        for item in rejected_candidates
        if "rhythm_lock_below_0_40" in list(item.get("reasons", []) or [])
        or "rhythm_lock_below_0_48" in list(item.get("reasons", []) or [])
        or "rhythm_lock_below_0_55" in list(item.get("reasons", []) or [])
    )
    non_tail_speed_reject_count = sum(
        1
        for item in rejected_candidates
        if "non_tail_speed_outside_0_90_1_10" in list(item.get("reasons", []) or [])
        or "non_tail_speed_outside_0_92_1_08" in list(item.get("reasons", []) or [])
    )
    transition_reject_count = sum(
        1
        for item in rejected_candidates
        if "transition_smoothness_below_0_45" in list(item.get("reasons", []) or [])
        or "transition_smoothness_below_0_50" in list(item.get("reasons", []) or [])
        or "transition_smoothness_below_0_55" in list(item.get("reasons", []) or [])
    )
    transition_reject_count += sum(1 for item in rejected_candidates if "cross_sequence_transition_below_0_62" in list(item.get("reasons", []) or []))
    repeat_reject_count = sum(1 for item in rejected_candidates if "same_unit_run_limit_3" in list(item.get("reasons", []) or []))
    repeat_preferred_reject_count = sum(1 for item in rejected_candidates if "same_unit_preferred_limit_2" in list(item.get("reasons", []) or []))
    cross_sequence_transition_count = sum(
        1
        for previous, current in zip(decisions, decisions[1:])
        if str(previous.get("source_sequence")) != str(current.get("source_sequence"))
    )
    initial_upright_hold_sec = sum(
        _safe_float(dict(item.get("target_time_sec", {}) or {}).get("end"))
        - _safe_float(dict(item.get("target_time_sec", {}) or {}).get("start"))
        for item in decisions
        if str(dict(item.get("switch_reason", {}) or {}).get("mode")) == "initial_upright_hold"
    )
    first_dance_start_candidates = [
        _safe_float(dict(item.get("target_time_sec", {}) or {}).get("start"))
        for item in decisions
        if str(item.get("pose_source") or "finedance_motion_unit") == "finedance_motion_unit"
    ]
    return {
        "schema_version": 1,
        "report_id": f"{header.get('song_id', 'stream_song')}_streaming_planner_eval",
        "song_id": header.get("song_id"),
        "planner": header.get("planner"),
        "planner_version": header.get("planner_version"),
        "tail_policy": header.get("tail_policy"),
        "generated_at_utc": utc_now_iso(),
        "metrics": {
            "decision_count": len(decisions),
            "speed_scale_outside_default_range_count": len(outside),
            "speed_scale_outside_default_range_ratio": round(float(len(outside) / max(1, len(speeds))), 5),
            "non_tail_speed_scale_outside_default_range_count": len(non_tail_outside),
            "max_non_tail_speed_scale": round(float(max(non_tail_speeds or [1.0])), 6),
            "max_speed_scale": round(float(max(speeds or [1.0])), 6),
            "min_speed_scale": round(float(min(speeds or [1.0])), 6),
            "future_visibility_violations": sum(1 for item in decisions if not bool(dict(item.get("future_visibility_guard", {}) or {}).get("passed"))),
            "gap_count": len(gaps),
            "rhythm_hard_reject_count": rhythm_reject_count,
            "non_tail_speed_hard_reject_count": non_tail_speed_reject_count,
            "transition_hard_reject_count": transition_reject_count,
            "repeat_unit_hard_reject_count": repeat_reject_count,
            "repeat_unit_preferred_reject_count": repeat_preferred_reject_count,
            "max_consecutive_motion_unit_run": _max_consecutive_motion_unit_run(decisions),
            "cross_sequence_transition_count": cross_sequence_transition_count,
            "low_confidence_continuation_count": sum(
                1 for item in decisions if bool(dict(item.get("switch_reason", {}) or {}).get("low_confidence_continuation"))
            ),
            "initial_upright_hold_sec": round(float(initial_upright_hold_sec), 5),
            "first_dance_start_sec": round(float(min(first_dance_start_candidates) if first_dance_start_candidates else 0.0), 5),
        },
        "cohort_source_sequences": sorted({str(sequence_id) for decision in decisions for sequence_id in list(decision.get("cohort_source_sequences", []) or [])}),
        "gaps": gaps,
        "acceptance": {
            "no_future_visibility_violations": all(bool(dict(item.get("future_visibility_guard", {}) or {}).get("passed")) for item in decisions),
            "no_gaps": not gaps,
            "speed_outside_ratio_le_10pct": len(outside) / max(1, len(speeds)) <= 0.10,
            "non_tail_max_speed_le_1_25": max(non_tail_speeds or [1.0]) <= 1.25,
            "max_consecutive_motion_unit_run_le_2": _max_consecutive_motion_unit_run(decisions) <= M17_IDEAL_MAX_CONSECUTIVE_SAME_UNIT,
            "max_consecutive_motion_unit_run_le_3": _max_consecutive_motion_unit_run(decisions) <= M17_MAX_CONSECUTIVE_SAME_UNIT,
        },
    }


def build_motion_transition_report(mesh_report: dict[str, Any]) -> dict[str, Any]:
    transitions = [dict(item) for item in list(mesh_report.get("transitions", []) or [])]
    metrics = dict(mesh_report.get("metrics", {}) or {})
    derived_root_accel = max(
        [
            _safe_float(item.get("root_acceleration_discontinuity_proxy"), float(np.linalg.norm(np.asarray(item.get("root_offset_xyz", [0.0, 0.0, 0.0]), dtype=np.float32))))
            for item in transitions
        ]
        or [0.0]
    )
    derived_joint_jerk = max(
        [_safe_float(item.get("joint_jerk_proxy"), _safe_float(item.get("joint_delta_after_blend"))) for item in transitions] or [0.0]
    )
    derived_foot_slide = max(
        [_safe_float(item.get("foot_slide_proxy"), _safe_float(item.get("joint_delta_after_blend")) * 0.25) for item in transitions] or [0.0]
    )
    metrics.setdefault("max_root_acceleration_discontinuity_proxy", round(float(derived_root_accel), 6))
    metrics.setdefault("max_joint_jerk_proxy", round(float(derived_joint_jerk), 6))
    metrics.setdefault("max_foot_slide_proxy", round(float(derived_foot_slide), 6))
    return {
        "schema_version": 1,
        "report_id": f"{mesh_report.get('report_id', 'mesh')}_motion_transition_quality",
        "source_report_id": mesh_report.get("report_id"),
        "song_id": mesh_report.get("song_id"),
        "generated_at_utc": utc_now_iso(),
        "transition_count": len(transitions),
        "metrics": {
            "max_temporal_vertex_delta_after_smoothing": _safe_float(metrics.get("max_temporal_vertex_delta_after_smoothing")),
            "max_temporal_joint_delta_after_smoothing": _safe_float(metrics.get("max_temporal_joint_delta_after_smoothing")),
            "max_root_acceleration_discontinuity_proxy": _safe_float(metrics.get("max_root_acceleration_discontinuity_proxy")),
            "max_joint_jerk_proxy": _safe_float(metrics.get("max_joint_jerk_proxy")),
            "max_foot_slide_proxy": _safe_float(metrics.get("max_foot_slide_proxy")),
            "max_vertex_delta_after_blend": _safe_float(metrics.get("max_vertex_delta_after_blend")),
        },
        "transitions": [
            {
                "index": item.get("index"),
                "outgoing_step": item.get("outgoing_step"),
                "incoming_step": item.get("incoming_step"),
                "blend_frames": item.get("blend_frames"),
                "root_acceleration_discontinuity_proxy": item.get("root_acceleration_discontinuity_proxy"),
                "joint_jerk_proxy": item.get("joint_jerk_proxy"),
                "foot_slide_proxy": item.get("foot_slide_proxy"),
                "max_temporal_vertex_delta_after_smoothing": item.get("max_temporal_vertex_delta_after_smoothing"),
            }
            for item in transitions
        ],
        "acceptance": {
            "vertex_delta_le_0_05": _safe_float(metrics.get("max_temporal_vertex_delta_after_smoothing")) <= 0.05,
            "has_motion_level_metrics": True,
        },
        "notes": [
            "M13 report exposes motion-level transition proxies in addition to the existing visual vertex smoothing metrics.",
            "The proxies are review metrics; production foot locking still needs Unity/runtime IK validation.",
        ],
    }


def export_unity_streaming_runtime_bundle(
    output_dir: Path,
    annotated_library: dict[str, Any],
    stream_plan_records: list[dict[str, Any]] | None = None,
    stream_event_records: list[dict[str, Any]] | None = None,
    planner_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    units = []
    for unit in list(annotated_library.get("units", []) or []):
        source = dict(unit.get("reference_artifacts", {}) or {})
        units.append(
            {
                "unit_id": unit.get("unit_id"),
                "source_sequence": unit.get("source_sequence"),
                "source_motion_path": source.get("source_motion_path"),
                "frame_range": dict(unit.get("frame_range", {}) or {}),
                "duration_beats": unit.get("duration_beats"),
                "duration_sec": unit.get("duration_sec"),
                "energy": unit.get("energy"),
                "style_tags": list(unit.get("style_tags", []) or []),
                "safe_retime_range": dict(unit.get("safe_retime_range", {}) or {}),
                "entry_pose_anchor": dict(unit.get("entry_pose_anchor", {}) or {}),
                "exit_pose_anchor": dict(unit.get("exit_pose_anchor", {}) or {}),
                "accent_lock_frames": list(unit.get("accent_lock_frames", []) or []),
                "foot_contact_windows": list(unit.get("foot_contact_windows", []) or []),
                "movement_quality": dict(unit.get("movement_quality", {}) or {}),
            }
        )
    compact_library = {
        "schema_version": 1,
        "library_id": annotated_library.get("library_id"),
        "unit_count": len(units),
        "units": units,
        "generated_at_utc": utc_now_iso(),
    }
    planner_payload = {
        "schema_version": 1,
        "planner": "streaming_retrieval_v2_phrase_aware",
        "initial_buffer_sec": DEFAULT_INITIAL_BUFFER_SEC,
        "lookahead_sec": DEFAULT_INITIAL_BUFFER_SEC,
        "safe_retime_range": {"min": DEFAULT_SAFE_RETIME_MIN, "max": DEFAULT_SAFE_RETIME_MAX},
        "score_weights": {"rhythm_lock": 0.45, "transition_smoothness": 0.25, "style_energy_bpm": 0.20, "diversity": 0.10},
        **dict(planner_config or {}),
    }
    manifest = {
        "schema_version": 1,
        "bundle_id": "unity_streaming_smplx_bundle",
        "library_id": annotated_library.get("library_id"),
        "generated_at_utc": utc_now_iso(),
        "files": {
            "motion_units": "motion_units_compact.json",
            "planner_config": "planner_config.json",
            "stream_plan_sample": "stream_plan_sample.jsonl" if stream_plan_records else None,
            "stream_events_sample": "stream_events_sample.jsonl" if stream_event_records else None,
        },
        "runtime_contract": {
            "mesh_generation": "precomputed_or_cached; endpoint runtime should not invoke Python SMPL-X generation",
            "streaming_audio": "planner consumes rolling event ticks with at most 2 seconds lookahead",
            "motion_playback": "Unity should retime units inside safe_retime_range and use root continuity plus runtime IK/contact handling",
        },
    }
    write_json(output_dir / "motion_units_compact.json", compact_library)
    write_json(output_dir / "planner_config.json", planner_payload)
    write_json(output_dir / "bundle_manifest.json", manifest)
    if stream_plan_records is not None:
        write_jsonl(output_dir / "stream_plan_sample.jsonl", stream_plan_records[:200])
    if stream_event_records is not None:
        write_jsonl(output_dir / "stream_events_sample.jsonl", stream_event_records[:200])
    report = {
        "schema_version": 1,
        "report_id": "unity_streaming_smplx_bundle_report",
        "bundle_dir": str(output_dir),
        "generated_at_utc": utc_now_iso(),
        "metrics": {
            "unit_count": len(units),
            "stream_plan_sample_records": len(stream_plan_records or []),
            "stream_event_sample_records": len(stream_event_records or []),
            "motion_units_json_bytes": int((output_dir / "motion_units_compact.json").stat().st_size),
            "planner_config_json_bytes": int((output_dir / "planner_config.json").stat().st_size),
        },
        "acceptance": {
            "does_not_require_python_smplx_runtime": True,
            "has_motion_units": bool(units),
            "has_planner_config": True,
        },
        "files": manifest["files"],
    }
    write_json(output_dir / "bundle_report.json", report)
    return report
