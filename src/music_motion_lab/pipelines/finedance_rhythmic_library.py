from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..contracts import MotionUnitLibrary
from ..utils import stable_digest, utc_now_iso

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("music-motion-lab FineDance rhythmic library requires numpy.") from exc


FINEDANCE_SOURCE_FPS = 30.0
DEFAULT_LIBRARY_ID = "finedance_rhythmic_smplx_library_v1"
DEFAULT_UNIT_BEATS = 8
DEFAULT_ACCENT_UNIT_BEATS = 4
DEFAULT_MAX_UNIT_BEATS = 16
DEFAULT_UNIT_BEAT_SET = (2, 4, 8, 16)
QUALITY_WEIGHT_BY_TIER = {
    "rhythmic_first": 1.0,
    "rhythmic_second": 0.9,
    "fallback": 0.72,
    "unknown": 0.65,
}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


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


def _motion_path(raw_root: Path, sequence_id: str) -> Path:
    return raw_root / "motion" / f"{sequence_id}.npy"


def _music_path(raw_root: Path, sequence_id: str) -> Path:
    return raw_root / "music_wav" / f"{sequence_id}.wav"


def _label_path(raw_root: Path, sequence_id: str) -> Path:
    return raw_root / "label_json" / f"{sequence_id}.json"


def _load_motion_array(raw_root: Path, sequence_id: str) -> np.ndarray:
    path = _motion_path(raw_root, sequence_id)
    if not path.exists():
        raise FileNotFoundError(f"FineDance motion not found for sequence {sequence_id}: {path}")
    payload = np.load(path, mmap_mode="r")
    if getattr(payload, "ndim", 0) != 2 or int(payload.shape[1]) < 9:
        raise ValueError(f"Unexpected FineDance motion shape for {sequence_id}: {getattr(payload, 'shape', None)}")
    return np.asarray(payload, dtype=np.float32)


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def _rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    payload = np.asarray(rot6d, dtype=np.float32)
    a1 = payload[..., 0:3]
    a2 = payload[..., 3:6]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), 1e-8)
    dot = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - dot * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-2).astype(np.float32)


def _yaw_from_rot6d(rot6d: np.ndarray) -> float:
    matrix = _rot6d_to_matrix(rot6d)
    forward = matrix @ np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    return math.degrees(math.atan2(float(forward[0]), float(forward[2])))


def _wrap_degrees(angle: float) -> float:
    return float((angle + 180.0) % 360.0 - 180.0)


def _median_beat_spacing(beats: list[dict[str, Any]]) -> float:
    times = [_safe_float(beat.get("time_sec")) for beat in beats]
    intervals = [times[index + 1] - times[index] for index in range(len(times) - 1) if times[index + 1] > times[index]]
    if not intervals:
        return 60.0 / 120.0
    return float(np.median(np.asarray(intervals, dtype=np.float32)))


def normalize_unit_beat_set(unit_beat_set: list[int] | tuple[int, ...] | None, fallback: int = DEFAULT_UNIT_BEATS) -> list[int]:
    values: list[int] = []
    for raw_value in list(unit_beat_set or [fallback]):
        value = int(raw_value)
        if value <= 0 or value in values:
            continue
        values.append(value)
    return sorted(values) or [int(fallback)]


def _beat_time(beats: list[dict[str, Any]], beat_index: int, spacing: float) -> float:
    if not beats:
        return 0.0
    if beat_index < len(beats):
        return _safe_float(beats[beat_index].get("time_sec"))
    return _safe_float(beats[-1].get("time_sec")) + float(beat_index - len(beats) + 1) * spacing


def _frame_for_time(time_sec: float, fps: float, frame_count: int) -> int:
    if frame_count <= 0:
        return 0
    return _clamp(int(round(time_sec * fps)), 0, frame_count - 1)


def _supporting_beat_events(
    beats: list[dict[str, Any]],
    accent_candidates: list[dict[str, Any]],
    start_beat: int,
    end_beat_exclusive: int,
    fps: float,
    frame_count: int,
) -> list[dict[str, Any]]:
    beat_events: list[dict[str, Any]] = []
    accent_by_beat: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for accent in accent_candidates:
        source_beat_index = _safe_int(accent.get("source_beat_index"), -1)
        if source_beat_index >= 0:
            accent_by_beat[source_beat_index].append(accent)

    for beat in beats:
        beat_index = _safe_int(beat.get("index"), -1)
        if beat_index < start_beat or beat_index >= end_beat_exclusive:
            continue
        time_sec = _safe_float(beat.get("time_sec"))
        accents = accent_by_beat.get(beat_index, [])
        event_kind = "downbeat" if bool(beat.get("is_downbeat", False)) else "beat"
        if accents:
            event_kind = "accent"
        beat_events.append(
            {
                "kind": event_kind,
                "beat_index": beat_index,
                "beat_offset": beat_index - start_beat,
                "time_sec": round(time_sec, 5),
                "frame_index": _frame_for_time(time_sec, fps=fps, frame_count=frame_count),
                "strength": round(max([_safe_float(beat.get("strength"))] + [_safe_float(item.get("strength")) for item in accents]), 5),
                "is_downbeat": bool(beat.get("is_downbeat", False)),
                "accent_bands": sorted({str(item.get("band", "unknown") or "unknown") for item in accents}),
                "accent_level": max([_safe_int(item.get("level"), 0) for item in accents] or [0]),
            }
        )
    return beat_events


def _unit_anchor(motion: np.ndarray, frame_index: int, fps: float) -> dict[str, Any]:
    frame_index = _clamp(frame_index, 0, int(motion.shape[0]) - 1)
    root_position = np.asarray(motion[frame_index, :3], dtype=np.float32)
    root_rot6d = np.asarray(motion[frame_index, 3:9], dtype=np.float32)
    next_index = _clamp(frame_index + 1, 0, int(motion.shape[0]) - 1)
    previous_index = _clamp(frame_index - 1, 0, int(motion.shape[0]) - 1)
    velocity = (np.asarray(motion[next_index, :3], dtype=np.float32) - np.asarray(motion[previous_index, :3], dtype=np.float32)) * (fps / max(next_index - previous_index, 1))
    planar_velocity = velocity[[0, 2]]
    return {
        "frame_index": int(frame_index),
        "root_translation": [round(float(value), 5) for value in root_position.tolist()],
        "root_velocity": [round(float(value), 5) for value in velocity.tolist()],
        "root_speed": round(float(np.linalg.norm(velocity)), 5),
        "planar_speed": round(float(np.linalg.norm(planar_velocity)), 5),
        "root_yaw_deg": round(_yaw_from_rot6d(root_rot6d), 5),
    }


def _unit_motion_profile(motion: np.ndarray, start_frame: int, end_frame_exclusive: int, fps: float) -> dict[str, Any]:
    segment = np.asarray(motion[start_frame:end_frame_exclusive, :3], dtype=np.float32)
    if segment.shape[0] < 2:
        return {
            "root_displacement": 0.0,
            "root_displacement_vector": [0.0, 0.0, 0.0],
            "mean_root_speed": 0.0,
            "p90_root_speed": 0.0,
        }
    displacement = segment[-1] - segment[0]
    velocities = np.diff(segment, axis=0) * fps
    speed = np.linalg.norm(velocities, axis=1)
    return {
        "root_displacement": round(float(np.linalg.norm(displacement)), 5),
        "root_displacement_vector": [round(float(value), 5) for value in displacement.tolist()],
        "mean_root_speed": round(float(speed.mean()), 5),
        "p90_root_speed": round(float(np.percentile(speed, 90)), 5),
    }


def _energy_label(entry: dict[str, Any], profile: dict[str, Any]) -> str:
    value = str(entry.get("audio_features", {}).get("energy_label") or "").strip()
    if value:
        return value
    mean_speed = _safe_float(profile.get("mean_root_speed"))
    if mean_speed >= 1.3:
        return "high_energy"
    if mean_speed <= 0.35:
        return "low_energy"
    return "mid_energy"


def _style_tags(entry: dict[str, Any]) -> list[str]:
    tags = dict(entry.get("style_tags", {}))
    values = [
        str(tags.get("coarse_style", "") or "").strip().lower(),
        str(tags.get("fine_style", "") or "").strip().lower(),
        str(tags.get("song_name", "") or "").strip().lower(),
    ]
    return [value for value in values if value][:5] or ["finedance"]


def _priority_tier(entry: dict[str, Any]) -> str:
    return str((entry.get("analysis_priority") or {}).get("tier", "unknown") or "unknown").strip().lower() or "unknown"


def _source_song_quality_weight(entry: dict[str, Any]) -> float:
    return float(QUALITY_WEIGHT_BY_TIER.get(_priority_tier(entry), QUALITY_WEIGHT_BY_TIER["unknown"]))


def _candidate_segments(
    beats: list[dict[str, Any]],
    accent_candidates: list[dict[str, Any]],
    unit_beats: int,
    accent_unit_beats: int,
    max_unit_beats: int,
    unit_beat_set: list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    if not beats:
        return []

    segments: dict[tuple[int, int, str], dict[str, Any]] = {}
    beat_lengths = [
        value
        for value in normalize_unit_beat_set(unit_beat_set, fallback=unit_beats)
        if value <= max_unit_beats
    ] or [min(int(unit_beats), int(max_unit_beats))]
    downbeat_indices = [
        _safe_int(beat.get("index"), index)
        for index, beat in enumerate(beats)
        if bool(beat.get("is_downbeat", False))
    ] or [0]
    for beat_index in downbeat_indices:
        for beat_length in beat_lengths:
            end_beat = beat_index + beat_length
            if end_beat <= len(beats):
                segments[(beat_index, end_beat, f"downbeat_phrase_{beat_length}b")] = {
                    "start_beat": beat_index,
                    "end_beat_exclusive": end_beat,
                    "source": "downbeat_phrase",
                }

    for accent in accent_candidates:
        if _safe_int(accent.get("level"), 0) < 3 and _safe_float(accent.get("confidence")) < 0.78:
            continue
        source_beat = _safe_int(accent.get("source_beat_index"), -1)
        if source_beat < 0:
            continue
        for beat_length in [value for value in beat_lengths if value <= max(1, accent_unit_beats)] or [accent_unit_beats]:
            start_beat = max(0, source_beat - (source_beat % max(1, beat_length)))
            end_beat = start_beat + beat_length
            if end_beat <= len(beats):
                segments[(start_beat, end_beat, f"accent_window_{beat_length}b")] = {
                    "start_beat": start_beat,
                    "end_beat_exclusive": end_beat,
                    "source": "accent_window",
                }

    return sorted(segments.values(), key=lambda item: (item["start_beat"], item["end_beat_exclusive"], item["source"]))


def _transition_score(exit_unit: dict[str, Any], entry_unit: dict[str, Any]) -> float:
    exit_anchor = dict(exit_unit["exit_anchor"])
    entry_anchor = dict(entry_unit["entry_anchor"])
    yaw_delta = abs(_wrap_degrees(_safe_float(exit_anchor.get("root_yaw_deg")) - _safe_float(entry_anchor.get("root_yaw_deg"))))
    speed_delta = abs(_safe_float(exit_anchor.get("planar_speed")) - _safe_float(entry_anchor.get("planar_speed")))
    energy_match = 1.0 if exit_unit.get("energy") == entry_unit.get("energy") else 0.35
    style_overlap = len(set(exit_unit.get("style_tags", [])) & set(entry_unit.get("style_tags", [])))
    beat_match = 1.0 if exit_unit.get("duration_beats") == entry_unit.get("duration_beats") else 0.55
    score = 4.0 * energy_match + min(2.0, style_overlap * 0.7) + 2.0 * beat_match
    score += max(0.0, 2.0 - yaw_delta / 45.0)
    score += max(0.0, 1.5 - speed_delta * 0.5)
    return round(float(score), 5)


def _attach_compatibility(units: list[dict[str, Any]]) -> None:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_energy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        by_sequence[str(unit["source_sequence"])].append(unit)
        by_energy[str(unit["energy"])].append(unit)

    for sequence_units in by_sequence.values():
        sequence_units.sort(key=lambda item: (int(item["frame_range"]["start"]), int(item["frame_range"]["end_exclusive"])))

    for unit in units:
        scores: list[tuple[float, str]] = []
        same_sequence = by_sequence.get(str(unit["source_sequence"]), [])
        for index, sequence_unit in enumerate(same_sequence):
            if sequence_unit["unit_id"] != unit["unit_id"]:
                continue
            if index + 1 < len(same_sequence):
                scores.append((999.0, same_sequence[index + 1]["unit_id"]))
            break

        pool = by_energy.get(str(unit["energy"]), []) + by_energy.get("mid_energy", [])
        seen = {unit["unit_id"]}
        for candidate in pool:
            candidate_id = str(candidate["unit_id"])
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            score = _transition_score(unit, candidate)
            if score >= 5.0:
                scores.append((score, candidate_id))
        scores.sort(key=lambda item: item[0], reverse=True)
        compatible_ids: list[str] = []
        compatible_seen: set[str] = set()
        for _, candidate_id in scores:
            if candidate_id in compatible_seen:
                continue
            compatible_seen.add(candidate_id)
            compatible_ids.append(candidate_id)
            if len(compatible_ids) >= 8:
                break
        unit["compatible_next_units"] = compatible_ids


def build_finedance_rhythmic_library_showcase(library: MotionUnitLibrary, sample_count: int = 12) -> dict[str, Any]:
    units = list(library.units)
    sequence_counts = Counter(str(unit.get("source_sequence", "unknown")) for unit in units)
    energy_counts = Counter(str(unit.get("energy", "unknown")) for unit in units)
    duration_counts = Counter(str(unit.get("duration_beats", "unknown")) for unit in units)
    segment_source_counts = Counter(str(unit.get("segment_source", "unknown")) for unit in units)
    tier_counts = Counter(str(unit.get("priority_tier", "unknown")) for unit in units)
    return {
        "schema_version": library.schema_version,
        "library_id": library.library_id,
        "generated_at_utc": library.generated_at_utc,
        "counts": {
            "unit_count": len(units),
            "sequence_count": len(sequence_counts),
            "energy": dict(energy_counts),
            "duration_beats": dict(duration_counts),
            "segment_source": dict(segment_source_counts),
            "priority_tier": dict(tier_counts),
        },
        "top_sequences": [{"sequence_id": sequence_id, "unit_count": count} for sequence_id, count in sequence_counts.most_common(12)],
        "sampled_units": [
            {
                "unit_id": unit["unit_id"],
                "sequence": unit["source_sequence"],
                "duration_beats": unit["duration_beats"],
                "energy": unit["energy"],
                "segment_source": unit["segment_source"],
                "priority_tier": unit.get("priority_tier"),
                "source_song_quality_weight": unit.get("source_song_quality_weight"),
                "frame_range": unit["frame_range"],
                "rhythm_profile": unit["rhythm_profile"],
                "compatible_next_units": unit.get("compatible_next_units", [])[:4],
            }
            for unit in units[:sample_count]
        ],
        "notes": [
            "Showcase for the FineDance rhythmic-first SMPL-X source motion library.",
            "Units are metadata references to raw FineDance SMPL-X motion and event-rail keyframes, not embedded mesh payloads.",
        ],
    }


def build_finedance_rhythmic_smplx_library(
    audio_feature_report: dict[str, Any],
    raw_root: Path,
    rhythmic_only: bool = True,
    fps: float = FINEDANCE_SOURCE_FPS,
    unit_beats: int = DEFAULT_UNIT_BEATS,
    accent_unit_beats: int = DEFAULT_ACCENT_UNIT_BEATS,
    max_unit_beats: int = DEFAULT_MAX_UNIT_BEATS,
    unit_beat_set: list[int] | tuple[int, ...] | None = None,
    max_sequences: int = 0,
    min_source_sec: float = 0.0,
) -> MotionUnitLibrary:
    selected_entries: list[dict[str, Any]] = []
    for entry in list(audio_feature_report.get("entries", [])):
        tier = str((entry.get("analysis_priority") or {}).get("tier", "") or "")
        if rhythmic_only and tier != "rhythmic_first":
            continue
        selected_entries.append(dict(entry))
        if max_sequences > 0 and len(selected_entries) >= max_sequences:
            break

    units: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    min_source_frame = max(0, int(round(float(min_source_sec) * max(float(fps), 1.0))))
    for entry in selected_entries:
        sequence_id = str(entry.get("sequence_id", "") or "")
        if not sequence_id:
            skipped.append({"sequence_id": "", "reason": "missing_sequence_id"})
            continue
        try:
            motion = _load_motion_array(raw_root, sequence_id)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"sequence_id": sequence_id, "reason": str(exc)})
            continue

        frame_count = int(motion.shape[0])
        audio_features = dict(entry.get("audio_features", {}))
        priority_tier = _priority_tier(entry)
        source_song_bpm = round(_safe_float(audio_features.get("bpm") or audio_features.get("beat_tracking", {}).get("global_bpm")), 5)
        source_song_energy = str(audio_features.get("energy_label") or "").strip() or "unknown"
        source_song_style_tags = _style_tags(entry)
        source_song_quality_weight = round(_source_song_quality_weight(entry), 5)
        beats = sorted(list(audio_features.get("beats", [])), key=lambda item: _safe_int(item.get("index")))
        accent_candidates = sorted(list(audio_features.get("accent_candidates", [])), key=lambda item: (_safe_int(item.get("source_beat_index")), _safe_float(item.get("time_sec"))))
        if len(beats) < max(2, accent_unit_beats + 1):
            skipped.append({"sequence_id": sequence_id, "reason": "not_enough_beats"})
            continue

        spacing = _median_beat_spacing(beats)
        for segment in _candidate_segments(
            beats=beats,
            accent_candidates=accent_candidates,
            unit_beats=int(unit_beats),
            accent_unit_beats=int(accent_unit_beats),
            max_unit_beats=int(max_unit_beats),
            unit_beat_set=unit_beat_set,
        ):
            start_beat = int(segment["start_beat"])
            end_beat = int(segment["end_beat_exclusive"])
            start_time = _beat_time(beats, start_beat, spacing)
            end_time = _beat_time(beats, end_beat, spacing)
            start_frame = _frame_for_time(start_time, fps=fps, frame_count=frame_count)
            end_frame = min(frame_count, max(start_frame + 2, int(math.ceil(end_time * fps))))
            if start_frame < min_source_frame:
                continue
            if end_frame - start_frame < max(8, int(round(0.35 * fps))):
                continue

            keyframes = _supporting_beat_events(
                beats=beats,
                accent_candidates=accent_candidates,
                start_beat=start_beat,
                end_beat_exclusive=end_beat,
                fps=fps,
                frame_count=frame_count,
            )
            motion_profile = _unit_motion_profile(motion, start_frame, end_frame, fps=fps)
            duration_beats = end_beat - start_beat
            unit_id = f"finedance_{sequence_id}_{start_beat:04d}b_{end_beat:04d}b_{start_frame:05d}_{end_frame - 1:05d}"
            entry_anchor = _unit_anchor(motion, start_frame, fps=fps)
            exit_anchor = _unit_anchor(motion, end_frame - 1, fps=fps)
            downbeat_count = sum(1 for item in keyframes if item.get("is_downbeat"))
            accent_count = sum(1 for item in keyframes if item.get("kind") == "accent")
            unit = {
                "unit_id": unit_id,
                "dataset": "finedance",
                "source_sequence": sequence_id,
                "skeleton_id": "finedance_smplx_source_v1",
                "mesh_backend": "smplx_neutral",
                "frame_range": {"start": start_frame, "end_exclusive": end_frame},
                "beat_range": {"start": start_beat, "end_exclusive": end_beat},
                "time_range_sec": {"start": round(float(start_time), 5), "end": round(float(end_time), 5)},
                "duration_sec": round(float((end_frame - start_frame) / max(fps, 1.0)), 5),
                "duration_beats": duration_beats,
                "duration_beats_estimate": float(duration_beats),
                "segment_source": str(segment["source"]),
                "energy": _energy_label(entry, motion_profile),
                "style_tags": source_song_style_tags,
                "priority_tier": priority_tier,
                "source_song_bpm": source_song_bpm,
                "source_song_energy": source_song_energy,
                "source_song_style_tags": source_song_style_tags,
                "source_song_quality_weight": source_song_quality_weight,
                "keyframes": keyframes,
                "rhythm_profile": {
                    "beat_count": duration_beats,
                    "downbeat_count": downbeat_count,
                    "accent_count": accent_count,
                    "mean_beat_spacing_sec": round(float(spacing), 5),
                    "source_profile": str(audio_features.get("beat_tracking", {}).get("profile", "unknown") or "unknown"),
                    "source_bpm": round(_safe_float(audio_features.get("bpm") or audio_features.get("beat_tracking", {}).get("global_bpm")), 5),
                },
                "entry_anchor": entry_anchor,
                "exit_anchor": exit_anchor,
                "transition_profile": {
                    **motion_profile,
                    "yaw_delta_deg": round(_wrap_degrees(_safe_float(exit_anchor["root_yaw_deg"]) - _safe_float(entry_anchor["root_yaw_deg"])), 5),
                },
                "reference_artifacts": {
                    "source_motion_path": str(_motion_path(raw_root, sequence_id)),
                    "source_music_path": str(_music_path(raw_root, sequence_id)),
                    "source_label_path": str(_label_path(raw_root, sequence_id)),
                    "event_report": "dataset_truth_finedance_audio_features_all",
                },
                "entry_pose_fingerprint": stable_digest({"sequence": sequence_id, "frame": start_frame, "anchor": entry_anchor}),
                "exit_pose_fingerprint": stable_digest({"sequence": sequence_id, "frame": end_frame - 1, "anchor": exit_anchor}),
                "compatible_next_units": [],
            }
            units.append(unit)

    _attach_compatibility(units)
    notes = [
        "FineDance rhythmic-first SMPL-X source motion library generated from M2-2 Event Rail audio feature reports.",
        "Units reference raw FineDance motion frames and beat/accent keyframes; mesh vertices are generated later by the SMPL-X stitch preview stage.",
        f"rhythmic_only={rhythmic_only}; selected_sequences={len(selected_entries)}; skipped_sequences={len(skipped)}; unit_beat_set={normalize_unit_beat_set(unit_beat_set, unit_beats)}; min_source_sec={float(min_source_sec):.2f}.",
    ]
    if skipped:
        notes.append(f"Skipped sequence samples: {skipped[:8]}.")

    return MotionUnitLibrary(
        schema_version=2,
        library_id=DEFAULT_LIBRARY_ID,
        source_roots={"finedance_raw_root": str(raw_root)},
        units=units,
        notes=notes,
        generated_at_utc=utc_now_iso(),
    )


def build_motion_library_coverage_report(
    motion_library: dict[str, Any] | MotionUnitLibrary,
    required_unit_beats: list[int] | tuple[int, ...] = DEFAULT_UNIT_BEAT_SET,
) -> dict[str, Any]:
    payload = motion_library.to_dict() if hasattr(motion_library, "to_dict") else dict(motion_library)
    units = [dict(unit) for unit in list(payload.get("units", []) or [])]
    required = normalize_unit_beat_set(required_unit_beats, fallback=DEFAULT_UNIT_BEATS)
    duration_counts = Counter(_safe_int(unit.get("duration_beats"), 0) for unit in units)
    sequence_counts = Counter(str(unit.get("source_sequence", "unknown")) for unit in units)
    energy_counts = Counter(str(unit.get("energy", "unknown")) for unit in units)
    tier_counts = Counter(str(unit.get("priority_tier", "unknown")) for unit in units)
    style_counts: Counter[str] = Counter()
    segment_source_counts = Counter(str(unit.get("segment_source", "unknown")) for unit in units)
    contact_count = 0
    accent_lock_count = 0
    root_speeds: list[float] = []
    yaw_values: list[float] = []
    retime_mins: list[float] = []
    retime_maxs: list[float] = []
    for unit in units:
        style_counts.update(str(tag) for tag in list(unit.get("style_tags", []) or []))
        contact_count += 1 if unit.get("foot_contact_windows") else 0
        accent_lock_count += len(list(unit.get("accent_lock_frames", []) or []))
        root_velocity = dict(unit.get("root_velocity", {}) or {})
        transition_profile = dict(unit.get("transition_profile", {}) or {})
        root_speeds.append(
            _safe_float(root_velocity.get("mean_speed"), _safe_float(transition_profile.get("mean_root_speed")))
        )
        yaw_values.append(abs(_safe_float(unit.get("yaw_delta"), _safe_float(transition_profile.get("yaw_delta_deg")))))
        safe_range = dict(unit.get("safe_retime_range", {}) or {})
        retime_mins.append(_safe_float(safe_range.get("min"), 0.85))
        retime_maxs.append(_safe_float(safe_range.get("max"), 1.15))
    missing_durations = [value for value in required if duration_counts.get(value, 0) == 0]
    speed_values = np.asarray(root_speeds or [0.0], dtype=np.float32)
    yaw_array = np.asarray(yaw_values or [0.0], dtype=np.float32)
    coverage_gaps: list[str] = []
    if missing_durations:
        coverage_gaps.append(f"missing_unit_beats:{','.join(str(value) for value in missing_durations)}")
    if len(sequence_counts) < 3:
        coverage_gaps.append("sequence_count_below_3")
    if units and contact_count / max(1, len(units)) < 0.5:
        coverage_gaps.append("foot_contact_coverage_below_50pct")
    if len(energy_counts) < 2:
        coverage_gaps.append("energy_diversity_low")
    return {
        "schema_version": 1,
        "report_id": f"{payload.get('library_id', 'motion_library')}_coverage_report",
        "library_id": payload.get("library_id"),
        "generated_at_utc": utc_now_iso(),
        "counts": {
            "unit_count": len(units),
            "sequence_count": len(sequence_counts),
            "duration_beats": {str(key): int(value) for key, value in sorted(duration_counts.items()) if key > 0},
            "energy": dict(energy_counts),
            "priority_tier": dict(tier_counts),
            "style_tags": dict(style_counts.most_common(20)),
            "segment_source": dict(segment_source_counts),
        },
        "coverage": {
            "required_unit_beats": required,
            "missing_unit_beats": missing_durations,
            "foot_contact_unit_ratio": round(float(contact_count / max(1, len(units))), 5),
            "accent_locks_per_unit": round(float(accent_lock_count / max(1, len(units))), 5),
            "root_speed_mean": round(float(speed_values.mean()), 5),
            "root_speed_p90": round(float(np.percentile(speed_values, 90.0)), 5),
            "yaw_delta_p90": round(float(np.percentile(yaw_array, 90.0)), 5),
            "safe_retime_min": round(float(min(retime_mins or [0.85])), 5),
            "safe_retime_max": round(float(max(retime_maxs or [1.15])), 5),
        },
        "top_sequences": [{"sequence_id": key, "unit_count": int(value)} for key, value in sequence_counts.most_common(20)],
        "coverage_gaps": coverage_gaps,
        "acceptance": {
            "has_required_2_4_8_16_units": not missing_durations,
            "has_multi_sequence_coverage": len(sequence_counts) >= 3,
            "has_contact_annotations": contact_count > 0,
        },
        "notes": [
            "M10 coverage report for FineDance rhythmic-first action-library breadth.",
            "Speed-scale acceptance is evaluated by streaming planner reports because it depends on target song tempo.",
        ],
    }
