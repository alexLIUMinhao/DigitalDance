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
DEFAULT_CHUNK_MS = 46.44
DEFAULT_ROLLING_WINDOW_SEC = 8.0
DEFAULT_TEMPO_WINDOW_SEC = 16.0
DEFAULT_STREAM_FPS = 30
DEFAULT_BLEND_FRAMES = 10


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
            "rolling_window_sec": round(float(rolling_window_sec), 5),
            "beats_per_bar": int(beats_per_bar),
            "analysis_mode": "streaming_simulation",
            "generated_at_utc": utc_now_iso(),
        }
    ]

    tempo_cache: dict[str, Any] | None = None
    tempo_update_ticks = max(1, int(round(0.25 / chunk_sec)))
    for tick_index in range(tick_count):
        playhead_sec = min(duration_sec, tick_index * chunk_sec)
        available_until = min(duration_sec, playhead_sec + max(0.0, float(initial_buffer_sec)))
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
        tempo_state = dict(tempo_cache)
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
                "lookahead_sec": round(float(max(0.0, available_until - playhead_sec)), 5),
                "visible_window_sec": {"start": _round_time(window_start), "end": _round_time(available_until)},
                "tempo_hypotheses": [
                    {
                        "label": "stream_primary",
                        "bpm": tempo_state["bpm"],
                        "confidence": tempo_state["confidence"],
                    },
                    {
                        "label": "stream_half_time",
                        "bpm": round(_safe_float(tempo_state.get("bpm"), 120.0) / 2.0, 5),
                        "confidence": round(_safe_float(tempo_state.get("confidence")) * 0.55, 5),
                    },
                    {
                        "label": "stream_double_time",
                        "bpm": round(_safe_float(tempo_state.get("bpm"), 120.0) * 2.0, 5),
                        "confidence": round(_safe_float(tempo_state.get("confidence")) * 0.4, 5),
                    },
                ],
                "beat_phase": {
                    "bpm": tempo_state["bpm"],
                    "spacing_sec": tempo_state["spacing_sec"],
                    "offset_sec": tempo_state["offset_sec"],
                    "confidence": tempo_state["confidence"],
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
    downbeat_candidates = [beat for beat in beats if bool(beat.get("is_downbeat"))] or beats[:1]
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


def _root_speed_profile(segment: np.ndarray | None, fps: float = 30.0) -> dict[str, Any]:
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
    contact_windows: list[dict[str, int | str]] = []
    if speed.size:
        contact_floor = float(np.percentile(speed, 28.0))
        start: int | None = None
        for index, value in enumerate(speed.tolist()):
            if value <= contact_floor:
                if start is None:
                    start = index
            elif start is not None:
                if index - start >= 2:
                    contact_windows.append({"start_local_frame": start, "end_local_frame_exclusive": index + 1, "source": "root_speed_proxy"})
                start = None
        if start is not None and len(speed) - start >= 2:
            contact_windows.append({"start_local_frame": start, "end_local_frame_exclusive": len(speed) + 1, "source": "root_speed_proxy"})
    return {
        "mean": round(float(speed.mean()) if speed.size else 0.0, 5),
        "p90": round(float(np.percentile(speed, 90.0)) if speed.size else 0.0, 5),
        "entry": [round(float(value), 5) for value in velocity[0].tolist()] if velocity.size else [0.0, 0.0, 0.0],
        "exit": [round(float(value), 5) for value in velocity[-1].tolist()] if velocity.size else [0.0, 0.0, 0.0],
        "displacement": round(displacement, 5),
        "curve": curve,
        "motion_accent_local_frames": accent_frames,
        "foot_contact_windows": contact_windows[:8],
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


def annotate_finedance_motion_units(motion_library: dict[str, Any], project_root: Path) -> dict[str, Any]:
    motion_cache: dict[str, np.ndarray] = {}
    annotated_units: list[dict[str, Any]] = []
    for unit in list(motion_library.get("units", []) or []):
        annotated = dict(unit)
        segment = _motion_segment(project_root, annotated, motion_cache)
        root_profile = _root_speed_profile(segment)
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
                "safe_retime_range": {"min": 0.85, "max": 1.15},
            }
        )
        annotated_units.append(annotated)
    result = dict(motion_library)
    base_id = str(result.get("library_id", "finedance_rhythmic_smplx_library_v1"))
    result["library_id"] = base_id if base_id.endswith("_m9_annotated") else f"{base_id}_m9_annotated"
    result["units"] = annotated_units
    result["notes"] = [
        *list(result.get("notes", []) or []),
        "M9 annotations add count-grid locks, movement quality, transition anchors, and safe retime ranges for streaming retrieval.",
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


def _transition_score(previous_unit: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if previous_unit is None:
        return 1.0
    if candidate.get("unit_id") in previous_unit.get("compatible_next_units", []):
        return 1.0
    previous_exit = dict(previous_unit.get("exit_pose_anchor") or previous_unit.get("exit_anchor") or {})
    entry = dict(candidate.get("entry_pose_anchor") or candidate.get("entry_anchor") or {})
    speed_delta = abs(_safe_float(previous_exit.get("planar_speed")) - _safe_float(entry.get("planar_speed")))
    yaw_delta = abs((_safe_float(previous_exit.get("root_yaw_deg")) - _safe_float(entry.get("root_yaw_deg")) + 180.0) % 360.0 - 180.0)
    same_sequence = 0.1 if previous_unit.get("source_sequence") == candidate.get("source_sequence") else 0.0
    return round(float(_clamp(0.55 - speed_delta * 0.12 + 0.35 - yaw_delta / 180.0 + same_sequence, 0.0, 1.0)), 5)


def _music_style_score(candidate: dict[str, Any], target_energy: str, target_bpm: float) -> float:
    energy = str(candidate.get("energy", "mid_energy") or "mid_energy")
    energy_score = 1.0 if energy == target_energy else (0.72 if "mid" in {energy, target_energy} else 0.32)
    source_bpm = _safe_float(dict(candidate.get("rhythm_profile", {}) or {}).get("source_bpm"), target_bpm)
    bpm_score = 1.0 - min(1.0, abs(source_bpm - target_bpm) / 45.0)
    quality = dict(candidate.get("movement_quality", {}) or {})
    if target_energy == "high_energy":
        quality_score = 1.0 if quality.get("attack") == "sharp" or quality.get("size") == "large_motion" else 0.65
    elif target_energy == "low_energy":
        quality_score = 1.0 if quality.get("attack") == "smooth" or quality.get("size") == "small_motion" else 0.62
    else:
        quality_score = 0.82
    return round(float((energy_score * 0.45) + (bpm_score * 0.35) + (quality_score * 0.20)), 5)


def _rhythm_score(
    candidate: dict[str, Any],
    tick: dict[str, Any],
    target_start_sec: float,
    beat_spacing_sec: float,
    visible_until_sec: float,
    target_beats: int,
) -> tuple[float, list[dict[str, Any]], list[float]]:
    locks = list(candidate.get("accent_lock_frames", []) or [])
    if not locks:
        locks = [{"beat_offset": 0, "role": "count_1_downbeat", "count": 1}]
    locks = [
        lock
        for lock in locks
        if _safe_float(lock.get("beat_offset")) <= float(target_beats) + 1e-8
    ] or [{"beat_offset": 0, "role": "count_1_downbeat", "count": 1}]
    beats = list(tick.get("beats", []) or [])
    drums = list(tick.get("drum_hits", []) or [])
    accents = list(tick.get("accents", []) or [])
    lock_reports: list[dict[str, Any]] = []
    expected_hits: list[float] = []
    scores: list[float] = []
    for lock in locks[:6]:
        beat_offset = _safe_float(lock.get("beat_offset"))
        lock_time = target_start_sec + beat_offset * beat_spacing_sec
        expected_hits.append(_round_time(lock_time))
        visible_events = [*beats, *drums, *accents]
        visible_events = [event for event in visible_events if _safe_float(event.get("time_sec"), 99999.0) <= visible_until_sec + 1e-8]
        nearest_delta = min([abs(_safe_float(event.get("time_sec")) - lock_time) for event in visible_events] or [beat_spacing_sec])
        role = str(lock.get("role", "accent"))
        is_predicted = lock_time > visible_until_sec + 1e-8
        if role == "count_1_downbeat":
            downbeat_bonus = 0.18 if int(round(beat_offset)) % 4 == 0 else 0.0
        elif role == "count_3_backbeat":
            downbeat_bonus = 0.08
        else:
            downbeat_bonus = 0.0
        score = _clamp(1.0 - nearest_delta / max(beat_spacing_sec * 0.5, 1e-3) + downbeat_bonus, 0.0, 1.0)
        score *= 0.72 + 0.28 * _clamp(_safe_float(lock.get("strength"), 0.0), 0.0, 1.0)
        if is_predicted:
            score *= 0.82
        scores.append(score)
        lock_reports.append(
            {
                "kind": "stream_rhythm_lock",
                "role": role,
                "time_sec": _round_time(lock_time),
                "beat_offset": round(float(beat_offset), 5),
                "visible_at_decision": not is_predicted,
                "nearest_visible_event_delta_sec": round(float(nearest_delta), 5),
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


def _score_candidate(
    candidate: dict[str, Any],
    previous_unit: dict[str, Any] | None,
    recent_units: Counter[str],
    tick: dict[str, Any],
    target_start_sec: float,
    target_end_sec: float,
    target_beats: int,
    target_energy: str,
) -> tuple[float, dict[str, Any], list[dict[str, Any]], list[float]]:
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
    )
    transition = _transition_score(previous_unit, candidate)
    style = _music_style_score(candidate, target_energy=target_energy, target_bpm=_safe_float(beat_phase.get("bpm"), 120.0))
    diversity = max(0.0, 1.0 - recent_units[str(candidate.get("unit_id"))] * 0.35)
    total = rhythm_score * 0.45 + transition * 0.25 + style * 0.20 + diversity * 0.10
    source_duration = _candidate_source_duration(candidate)
    target_duration = max(1e-3, target_end_sec - target_start_sec)
    speed_scale = source_duration / target_duration
    safe_range = dict(candidate.get("safe_retime_range", {}) or {})
    min_retime = _safe_float(safe_range.get("min"), 0.85)
    max_retime = _safe_float(safe_range.get("max"), 1.15)
    if speed_scale < min_retime or speed_scale > max_retime:
        total -= min(1.0, abs(speed_scale - _clamp(speed_scale, min_retime, max_retime)) * 1.8)
    if abs(_safe_float(candidate.get("duration_beats"), target_beats) - target_beats) > 0.1:
        total -= 0.12
    breakdown = {
        "rhythm_lock": round(float(rhythm_score), 5),
        "transition_smoothness": round(float(transition), 5),
        "style_energy_bpm": round(float(style), 5),
        "diversity": round(float(diversity), 5),
        "weighted_total": round(float(total), 5),
        "speed_scale": round(float(speed_scale), 5),
    }
    return round(float(total), 5), breakdown, lock_reports, expected_hits


def simulate_streaming_smplx_plan_records(
    stream_event_records: list[dict[str, Any]],
    annotated_library: dict[str, Any],
    stream_events_path: str | None = None,
    max_steps: int = 0,
) -> list[dict[str, Any]]:
    header = _header(stream_event_records, "stream_header")
    ticks = _records_by_kind(stream_event_records, "tick")
    if not ticks:
        raise ValueError("stream_event_records contain no tick records")
    units = [dict(unit) for unit in list(annotated_library.get("units", []) or [])]
    if not units:
        raise ValueError("annotated motion library contains no units")
    song_id = str(header.get("song_id") or "stream_song")
    duration_sec = _safe_float(header.get("duration_sec"), _safe_float(ticks[-1].get("available_audio_until_sec")))
    initial_buffer_sec = _safe_float(header.get("initial_buffer_sec"), DEFAULT_INITIAL_BUFFER_SEC)
    counts = sorted({_safe_int(unit.get("duration_beats"), 0) for unit in units if _safe_int(unit.get("duration_beats"), 0) > 0})
    preferred_beats = 4 if 4 in counts else (2 if 2 in counts else (8 if 8 in counts else counts[0]))
    records: list[dict[str, Any]] = [
        {
            "schema_version": 1,
            "kind": "stream_plan_header",
            "song_id": song_id,
            "library_id": annotated_library.get("library_id"),
            "source_stream_events_path": stream_events_path,
            "duration_sec": round(float(duration_sec), 5),
            "initial_buffer_sec": round(float(initial_buffer_sec), 5),
            "lookahead_sec": round(float(initial_buffer_sec), 5),
            "planner": "streaming_retrieval_v1",
            "score_weights": {
                "rhythm_lock": 0.45,
                "transition_smoothness": 0.25,
                "style_energy_bpm": 0.20,
                "diversity": 0.10,
            },
            "generated_at_utc": utc_now_iso(),
        }
    ]
    next_start_sec = 0.0
    previous_unit: dict[str, Any] | None = None
    recent_units: Counter[str] = Counter()
    step_index = 0
    while next_start_sec < duration_sec - 0.20:
        if max_steps > 0 and step_index >= max_steps:
            break
        decision_time = max(0.0, next_start_sec - initial_buffer_sec)
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
        target_beats = (
            _choose_target_beats(units, spacing_sec=spacing, preferred_beats=preferred_beats)
            if tempo_confidence >= 0.28
            else min(preferred_beats, 4)
        )
        target_end_sec = min(duration_sec, next_start_sec + target_beats * spacing)
        if target_end_sec - next_start_sec < 0.35:
            break
        target_energy = _target_energy_for_tick(scoring_tick, next_start_sec, _safe_float(tick.get("available_audio_until_sec"), next_start_sec))
        candidates = [
            unit
            for unit in units
            if _safe_int(unit.get("duration_beats"), target_beats) == target_beats
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
                target_energy=target_energy,
            )
            scored.append((score, candidate, breakdown, lock_reports, expected_hits))
        scored.sort(key=lambda item: item[0], reverse=True)
        score, selected, breakdown, lock_reports, expected_hits = scored[0]
        source_artifacts = dict(selected.get("reference_artifacts", {}) or {})
        available_until = _safe_float(tick.get("available_audio_until_sec"), 0.0)
        playhead = _safe_float(tick.get("playhead_sec"), 0.0)
        guard_passed = available_until <= playhead + initial_buffer_sec + 1e-5 and next_start_sec <= available_until + 1e-5
        decision = {
            "schema_version": 1,
            "kind": "decision",
            "index": step_index,
            "decision_time_sec": _round_time(decision_time),
            "decision_tick_index": _safe_int(tick.get("tick_index")),
            "playhead_sec": _round_time(playhead),
            "available_audio_until_sec": _round_time(available_until),
            "future_visibility_guard": {
                "lookahead_sec": round(float(initial_buffer_sec), 5),
                "used_audio_until_sec": _round_time(available_until),
                "passed": bool(guard_passed),
            },
            "target_time_sec": {"start": _round_time(next_start_sec), "end": _round_time(target_end_sec)},
            "target_beats": int(target_beats),
            "target_bpm": round(float(effective_bpm), 5),
            "target_energy": target_energy,
            "selected_unit_id": selected.get("unit_id"),
            "source_sequence": selected.get("source_sequence"),
            "source_frame_range": dict(selected.get("frame_range", {}) or {}),
            "source_beat_range": dict(selected.get("beat_range", {}) or {}),
            "source_motion_path": source_artifacts.get("source_motion_path"),
            "speed_scale": breakdown["speed_scale"],
            "score": score,
            "score_breakdown": breakdown,
            "switch_reason": {
                "planner": "streaming_retrieval_v1",
                "fallback": bool(tempo_confidence < 0.28),
                "target_energy": target_energy,
                "candidate_energy": selected.get("energy"),
                "compatible_from_previous": bool(previous_unit and selected.get("unit_id") in previous_unit.get("compatible_next_units", [])),
            },
            "expected_accent_hits": expected_hits,
            "rhythm_locks": lock_reports,
            "reference_artifacts": source_artifacts,
        }
        records.append(decision)
        previous_unit = selected
        recent_units.update([str(selected.get("unit_id"))])
        next_start_sec = target_end_sec
        step_index += 1
    records.append(
        {
            "schema_version": 1,
            "kind": "stream_plan_summary",
            "song_id": song_id,
            "step_count": step_index,
            "planned_until_sec": _round_time(next_start_sec),
            "duration_sec": round(float(duration_sec), 5),
            "future_visibility_violations": sum(
                1
                for item in records
                if item.get("kind") == "decision" and not bool(dict(item.get("future_visibility_guard", {}) or {}).get("passed"))
            ),
            "generated_at_utc": utc_now_iso(),
        }
    )
    return records


def _dedupe_timed_events(records: list[dict[str, Any]], key: str, confidence_floor: float = 0.0) -> list[dict[str, Any]]:
    by_time: dict[float, dict[str, Any]] = {}
    for tick in _records_by_kind(records, "tick"):
        available_until = _safe_float(tick.get("available_audio_until_sec"), 0.0)
        for event in list(tick.get(key, []) or []):
            time_sec = _round_time(_safe_float(event.get("time_sec")))
            if time_sec > available_until + 1e-8:
                continue
            if _safe_float(event.get("confidence"), _safe_float(event.get("strength"))) < confidence_floor:
                continue
            slot = round(time_sec, 3)
            current = by_time.get(slot)
            if current is None or _safe_float(event.get("strength"), _safe_float(event.get("confidence"))) > _safe_float(current.get("strength"), _safe_float(current.get("confidence"))):
                by_time[slot] = dict(event, time_sec=time_sec)
    events = sorted(by_time.values(), key=lambda item: _safe_float(item.get("time_sec")))
    for index, event in enumerate(events):
        event["index"] = index
    return events


def stream_events_to_song_event_map(stream_records: list[dict[str, Any]]) -> dict[str, Any]:
    header = _header(stream_records, "stream_header")
    beats = _dedupe_timed_events(stream_records, "beats", confidence_floor=0.18)
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
        "beats_per_bar": _safe_int(header.get("beats_per_bar"), 4),
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
    plan_id = f"{header.get('song_id', 'stream_song')}_streaming_smplx_plan"
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
        source_motion_path = str(decision.get("source_motion_path") or dict(decision.get("reference_artifacts", {}) or {}).get("source_motion_path") or "")
        if not source_motion_path:
            raise FileNotFoundError(f"stream decision {index} has no source_motion_path")
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
            "source_motion_path": source_motion_path,
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
                    "selected_unit_id": item.get("selected_unit_id"),
                    "target_bpm": item.get("target_bpm"),
                    "target_energy": item.get("target_energy"),
                    "speed_scale": item.get("speed_scale"),
                    "score": item.get("score"),
                    "score_breakdown": item.get("score_breakdown"),
                    "future_visibility_guard": item.get("future_visibility_guard"),
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
    report["metrics"]["streaming_decision_count"] = len(decisions)
    report["metrics"]["speed_scale_outside_default_range_count"] = sum(1 for value in speed_values if value < 0.85 or value > 1.15)
    report["metrics"]["max_streaming_speed_scale"] = round(max(speed_values or [1.0]), 6)
    report["metrics"]["min_streaming_speed_scale"] = round(min(speed_values or [1.0]), 6)
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
