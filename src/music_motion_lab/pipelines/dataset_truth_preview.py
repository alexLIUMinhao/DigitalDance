from __future__ import annotations

import base64
import html
import json
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np

from ..audio_analysis import (
    DEFAULT_FRAME_SIZE,
    DEFAULT_HOP_SIZE,
    best_beat_offset as _best_beat_offset,
    build_beat_frames as _build_beat_frames,
    build_frame_analysis as _build_frame_analysis,
    compute_mel_spectrogram as _compute_mel_spectrogram,
    estimate_bpm as _estimate_bpm,
    load_audio as _load_audio,
    local_peak_indices as _local_peak_indices,
    normalize as _normalize,
)
from ..config import AppConfig
from .music_analysis import analyze_song
from ..utils import load_json

RHYTHMIC_FIRST_COARSE_STYLES = {"Street"}
RHYTHMIC_FIRST_FINE_STYLES = {"Korean", "Popping", "Breaking", "Hiphop", "Locking", "Urban", "Jazz", "Choreography"}
RHYTHMIC_FIRST_CONFIDENCE_THRESHOLD = 0.68
RHYTHMIC_FIRST_LOCAL_TEMPO_WINDOW_SECONDS = 8.0
RHYTHMIC_FIRST_LOCAL_TEMPO_STEP_SECONDS = 1.0
RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO = 0.12
RHYTHMIC_FIRST_BAND_WINDOW_SECONDS = 2.0
RHYTHMIC_FIRST_BAND_TOP_K = 4


def _catalog_path(config: AppConfig) -> Path:
    return config.shared_roots.unity_reference_root / "motion_base" / "intake" / "dataset_catalog.json"


def _dataset_raw_root(config: AppConfig, dataset_name: str) -> Path:
    return config.shared_roots.motion_base_assets_root / "datasets" / dataset_name / "raw"


def list_dataset_sequence_ids(config: AppConfig, dataset_name: str) -> list[str]:
    catalog = load_json(_catalog_path(config))
    sequence_ids = {
        str(entry.get("sequenceId", ""))
        for entry in catalog.get("entries", [])
        if str(entry.get("datasetName", "")).lower() == dataset_name.lower() and str(entry.get("sequenceId", "")).strip()
    }
    return sorted(sequence_ids)


def _load_catalog_entry(config: AppConfig, dataset_name: str, sequence_id: str) -> dict:
    catalog = load_json(_catalog_path(config))
    for entry in catalog.get("entries", []):
        if str(entry.get("datasetName", "")).lower() == dataset_name.lower() and str(entry.get("sequenceId", "")) == str(sequence_id):
            return dict(entry)
    raise FileNotFoundError(f"Could not find {dataset_name}:{sequence_id} in dataset_catalog.json")


def _resolve_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve required asset from candidates: {candidates}")


def _resolve_finedance_paths(config: AppConfig, entry: dict, sequence_id: str) -> tuple[Path, Path, Path | None]:
    raw_root = _dataset_raw_root(config, "finedance")
    motion_rel = str(entry.get("motionPath", "") or "").strip()
    music_rel = str(entry.get("musicPath", "") or "").strip()
    label_rel = str(entry.get("labelPath", "") or "").strip()

    motion_path = _resolve_existing(
        (raw_root / motion_rel) if motion_rel else raw_root / "extracted" / "finedance" / "motion" / f"{sequence_id}.npy",
        raw_root / "extracted" / "finedance" / "motion" / f"{sequence_id}.npy",
    )
    music_path = _resolve_existing(
        (raw_root / music_rel) if music_rel else raw_root / "extracted" / "finedance" / "music_wav" / f"{sequence_id}.wav",
        raw_root / "extracted" / "finedance" / "music_wav" / f"{sequence_id}.wav",
        raw_root / "music_wav" / f"{sequence_id}.wav",
        raw_root / "music" / f"{sequence_id}.wav",
    )
    label_path = None
    if label_rel:
        candidate = raw_root / label_rel
        if candidate.exists():
            label_path = candidate
    if label_path is None:
        for candidate in (
            raw_root / "extracted" / "finedance" / "label_json" / f"{sequence_id}.json",
            raw_root / "label_json" / f"{sequence_id}.json",
            raw_root / "labels" / f"{sequence_id}.json",
        ):
            if candidate.exists():
                label_path = candidate
                break
    return motion_path, music_path, label_path


def _resolve_finedance_preview(config: AppConfig, sequence_id: str) -> tuple[Path | None, Path | None]:
    summary_path = (
        config.shared_roots.motion_base_assets_root
        / "previewCache"
        / "upstream_baselines"
        / "finedance"
        / str(sequence_id)
        / f"finedance_{sequence_id}_official_mesh_summary.json"
    )
    preview_path = summary_path.with_name(f"finedance_{sequence_id}_official_mesh_preview.mp4")
    if not preview_path.exists():
        return None, summary_path if summary_path.exists() else None
    return preview_path, summary_path if summary_path.exists() else None


def _wav_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frame_rate = handle.getframerate()
        return handle.getnframes() / float(frame_rate) if frame_rate > 0 else 0.0


def _motion_frame_count(path: Path) -> int:
    payload = np.load(path, mmap_mode="r")
    return int(payload.shape[0]) if getattr(payload, "shape", ()) else 0


def _load_label_summary(path: Path | None) -> dict:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _downsample_series(
    times_sec: np.ndarray,
    values: np.ndarray,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    max_points: int = 64,
) -> list[dict[str, float]]:
    if times_sec.size == 0 or values.size == 0:
        return []
    clip_end_seconds = clip_start_seconds + clip_duration_seconds
    mask = (times_sec >= clip_start_seconds - 1e-8) & (times_sec <= clip_end_seconds + 1e-8)
    clip_times = times_sec[mask] - clip_start_seconds
    clip_values = values[mask]
    if clip_times.size == 0:
        return []
    if clip_times.size > max_points:
        indices = np.linspace(0, clip_times.size - 1, max_points, dtype=np.float32).round().astype(np.int32)
        clip_times = clip_times[indices]
        clip_values = clip_values[indices]
    return [
        {
            "time_sec": round(float(time_sec), 5),
            "value": round(float(value), 5),
        }
        for time_sec, value in zip(clip_times.tolist(), clip_values.tolist())
    ]


def _assign_strong_beat_levels(
    beats: list[dict[str, Any]],
    downbeat_bonus: float = 0.28,
) -> list[dict[str, Any]]:
    if not beats:
        return []
    scores = np.asarray(
        [
            float(item.get("strength", 0.0) or 0.0) + (downbeat_bonus if bool(item.get("is_downbeat", False)) else 0.0)
            for item in beats
        ],
        dtype=np.float32,
    )
    q1, q2, q3 = [float(np.percentile(scores, quantile)) for quantile in (25.0, 50.0, 75.0)]
    strong_beats: list[dict[str, Any]] = []
    for beat, score in zip(beats, scores.tolist()):
        level = 1
        if score >= q1:
            level = 2
        if score >= q2:
            level = 3
        if score >= q3:
            level = 4
        # Keep only upper tiers for "strong beats" visualization.
        if level < 3:
            continue
        if bool(beat.get("is_downbeat", False)):
            level = max(level, 3)
        strong_beats.append(
            {
                "index": len(strong_beats),
                "beat_index": int(beat.get("index", 0) or 0),
                "time_sec": round(float(beat.get("time_sec", 0.0) or 0.0), 5),
                "strength": round(float(beat.get("strength", 0.0) or 0.0), 5),
                "score": round(float(score), 5),
                "level": int(level),
                "is_downbeat": bool(beat.get("is_downbeat", False)),
            }
        )
    return strong_beats


def _drum_hits_from_beats(
    beats: list[dict[str, Any]],
    low_band_envelope: np.ndarray,
    sample_rate: int,
    hop_size: int,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    min_gap_seconds: float = 0.18,
) -> list[dict[str, Any]]:
    if not beats or low_band_envelope.size == 0:
        return []
    scores: list[float] = []
    lows: list[float] = []
    candidates: list[dict[str, Any]] = []
    for beat in beats:
        beat_time = float(beat.get("time_sec", 0.0) or 0.0)
        if beat_time > clip_duration_seconds + 1e-8:
            continue
        absolute_time = clip_start_seconds + beat_time
        frame_index = min(
            len(low_band_envelope) - 1,
            max(0, int(round((absolute_time * sample_rate) / max(hop_size, 1)))),
        )
        low_strength = float(low_band_envelope[frame_index])
        beat_strength = float(beat.get("strength", 0.0) or 0.0)
        score = 0.6 * low_strength + 0.4 * beat_strength
        lows.append(low_strength)
        scores.append(score)
        candidates.append(
            {
                "time_sec": beat_time,
                "beat_strength": beat_strength,
                "low_strength": low_strength,
                "score": score,
                "is_downbeat": bool(beat.get("is_downbeat", False)),
            }
        )
    if not candidates:
        return []

    low_threshold = float(np.percentile(np.asarray(lows, dtype=np.float32), 55.0))
    score_threshold = float(np.percentile(np.asarray(scores, dtype=np.float32), 70.0))
    hits: list[dict[str, Any]] = []
    last_time = -1e9
    for item in candidates:
        qualifies = (
            item["low_strength"] >= low_threshold and item["score"] >= score_threshold
        ) or (item["is_downbeat"] and item["score"] >= score_threshold * 0.92)
        if not qualifies:
            continue
        if item["time_sec"] - last_time < min_gap_seconds:
            continue
        last_time = item["time_sec"]
        hits.append(
            {
                "index": len(hits),
                "time_sec": round(float(item["time_sec"]), 5),
                "strength": round(float(item["score"]), 5),
            }
        )
    return hits


def _classify_energy_labels(energy_scores: list[float]) -> list[str]:
    if not energy_scores:
        return []
    scores = np.asarray(energy_scores, dtype=np.float32)
    p33 = float(np.percentile(scores, 33.333))
    p66 = float(np.percentile(scores, 66.666))
    if abs(p66 - p33) <= 1e-8:
        return ["mid_energy" for _ in energy_scores]
    labels: list[str] = []
    for score in energy_scores:
        if score <= p33:
            labels.append("low_energy")
        elif score <= p66:
            labels.append("mid_energy")
        else:
            labels.append("high_energy")
    return labels


def _slug_tag(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value or ""))
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _classify_analysis_priority(
    coarse_style: str,
    fine_style: str,
    energy_label: str,
    beat_confidence: float,
) -> dict[str, Any]:
    coarse_style = str(coarse_style or "")
    fine_style = str(fine_style or "")
    energy_label = str(energy_label or "mid_energy")
    beat_confidence = float(beat_confidence or 0.0)

    style_seed = coarse_style in RHYTHMIC_FIRST_COARSE_STYLES or fine_style in RHYTHMIC_FIRST_FINE_STYLES
    tier = "fallback"
    if style_seed and (energy_label in {"mid_energy", "high_energy"} or beat_confidence >= RHYTHMIC_FIRST_CONFIDENCE_THRESHOLD):
        tier = "rhythmic_first"

    reason_tags: list[str] = []
    if coarse_style in RHYTHMIC_FIRST_COARSE_STYLES:
        reason_tags.append(f"{_slug_tag(coarse_style)}_seed")
    if fine_style in RHYTHMIC_FIRST_FINE_STYLES:
        reason_tags.append(f"fine_seed_{_slug_tag(fine_style)}")
    if energy_label in {"mid_energy", "high_energy"}:
        reason_tags.append(energy_label)
    if beat_confidence >= RHYTHMIC_FIRST_CONFIDENCE_THRESHOLD:
        reason_tags.append("high_confidence")
    if tier == "fallback":
        if coarse_style:
            reason_tags.append(f"fallback_{_slug_tag(coarse_style)}")
        else:
            reason_tags.append("fallback_unknown")
        if not style_seed:
            reason_tags.append("non_seed_style")
    return {
        "tier": tier,
        "style_seed": bool(style_seed),
        "reason_tags": reason_tags,
    }


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _envelope_value_at_time(
    values: np.ndarray,
    sample_rate: int,
    hop_size: int,
    time_sec: float,
) -> float:
    if values.size == 0:
        return 0.0
    frame_index = min(
        len(values) - 1,
        max(0, int(round((time_sec * sample_rate) / max(hop_size, 1)))),
    )
    return float(values[frame_index])


def _peak_shape_score(onset_envelope: np.ndarray, peak_index: int) -> tuple[float, dict[str, float]]:
    peak_value = float(onset_envelope[peak_index]) if onset_envelope.size else 0.0
    if peak_value <= 1e-8:
        return 0.0, {"peak_height": 0.0, "prominence": 0.0, "slope": 0.0, "width": 0.0}

    left_index = max(0, peak_index - 1)
    right_index = min(len(onset_envelope) - 1, peak_index + 1)
    left_slope = max(0.0, peak_value - float(onset_envelope[left_index]))
    right_slope = max(0.0, peak_value - float(onset_envelope[right_index]))
    slope_score = _clamp01((left_slope + right_slope) / max(peak_value * 1.5, 1e-6))

    window_start = max(0, peak_index - 4)
    window_end = min(len(onset_envelope), peak_index + 5)
    local_window = onset_envelope[window_start:window_end]
    neighborhood = np.delete(local_window, min(peak_index - window_start, max(len(local_window) - 1, 0)))
    baseline = float(np.median(neighborhood)) if neighborhood.size else 0.0
    prominence = max(0.0, peak_value - baseline)
    prominence_score = _clamp01(prominence / max(peak_value, 1e-6))

    width_floor = max(peak_value * 0.55, baseline)
    left_width = 0
    cursor = peak_index - 1
    while cursor >= 0 and onset_envelope[cursor] >= width_floor and left_width < 8:
        left_width += 1
        cursor -= 1
    right_width = 0
    cursor = peak_index + 1
    while cursor < len(onset_envelope) and onset_envelope[cursor] >= width_floor and right_width < 8:
        right_width += 1
        cursor += 1
    width_frames = 1 + left_width + right_width
    width_score = _clamp01(1.0 - abs(width_frames - 3.0) / 6.0)

    shape_score = (
        0.42 * peak_value
        + 0.24 * prominence_score
        + 0.18 * slope_score
        + 0.16 * width_score
    )
    return _clamp01(shape_score), {
        "peak_height": round(float(peak_value), 5),
        "prominence": round(float(prominence_score), 5),
        "slope": round(float(slope_score), 5),
        "width": round(float(width_score), 5),
    }


def _periodic_prior_values(beats: list[dict[str, Any]], beats_per_bar: int) -> tuple[int, list[float]]:
    if not beats or beats_per_bar <= 0:
        return 0, [0.0 for _ in beats]
    phase_scores = np.zeros((beats_per_bar,), dtype=np.float32)
    for beat in beats:
        phase = int(beat.get("index", 0) or 0) % beats_per_bar
        phase_scores[phase] += float(beat.get("strength", 0.0) or 0.0) * (0.5 + 0.5 * float(beat.get("confidence", 0.0) or 0.0))
    best_phase = int(phase_scores.argmax()) if phase_scores.size else 0

    priors: list[float] = []
    for beat in beats:
        phase = int(beat.get("index", 0) or 0) % beats_per_bar
        distance = min((phase - best_phase) % beats_per_bar, (best_phase - phase) % beats_per_bar)
        if distance == 0:
            prior = 1.0
        elif beats_per_bar % 2 == 0 and distance == beats_per_bar // 2:
            prior = 0.62
        else:
            prior = max(0.22, 0.48 - 0.08 * distance)
        priors.append(round(float(prior), 5))
    return best_phase, priors


def _structural_periodic_prior(beat_index: int, beats_per_bar: int) -> float:
    if beats_per_bar <= 0:
        return 0.0
    phase = int(beat_index) % beats_per_bar
    if phase == 0:
        return 1.0
    if beats_per_bar % 2 == 0 and phase == beats_per_bar // 2:
        return 0.62
    return max(0.24, 0.46 - 0.08 * min(phase, beats_per_bar - phase))


def _estimate_local_tempo_segments(
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_size: int,
    spacing_frames: float,
    window_seconds: float,
    step_seconds: float,
    max_delta_ratio: float,
) -> list[dict[str, Any]]:
    if onset_envelope.size == 0:
        return []
    total_duration_seconds = float((len(onset_envelope) * hop_size) / max(sample_rate, 1))
    if total_duration_seconds <= 0:
        return []

    window_frames = max(8, int(round((window_seconds * sample_rate) / max(hop_size, 1))))
    step_frames = max(1, int(round((step_seconds * sample_rate) / max(hop_size, 1))))
    previous_spacing = float(max(1.0, spacing_frames))
    global_spacing = float(max(1.0, spacing_frames))

    segments: list[dict[str, Any]] = []
    for segment_start_frame in range(0, max(len(onset_envelope) - 1, 1), step_frames):
        segment_end_frame = min(len(onset_envelope), segment_start_frame + window_frames)
        if segment_end_frame - segment_start_frame < 8:
            continue
        segment_values = onset_envelope[segment_start_frame:segment_end_frame]
        candidate_min = max(1, int(round(previous_spacing * (1.0 - max_delta_ratio))))
        candidate_max = max(candidate_min + 1, int(round(previous_spacing * (1.0 + max_delta_ratio))))
        candidate_min = max(candidate_min, int(round(global_spacing * 0.72)))
        candidate_max = min(candidate_max, max(candidate_min + 1, int(round(global_spacing * 1.28))))
        best_spacing = float(previous_spacing)
        best_score = -1.0
        for lag in range(candidate_min, candidate_max + 1):
            if lag >= len(segment_values):
                break
            lhs = segment_values[:-lag]
            rhs = segment_values[lag:]
            if lhs.size == 0 or rhs.size == 0:
                continue
            correlation = float(np.mean(lhs * rhs))
            consistency = _clamp01(1.0 - abs(lag - previous_spacing) / max(previous_spacing * max_delta_ratio, 1.0))
            score = 0.78 * correlation + 0.22 * consistency
            if score > best_score:
                best_score = score
                best_spacing = float(lag)

        lower_bound = max(1.0, previous_spacing * (1.0 - max_delta_ratio))
        upper_bound = max(lower_bound + 1.0, previous_spacing * (1.0 + max_delta_ratio))
        best_spacing = float(np.clip(best_spacing, lower_bound, upper_bound))
        previous_spacing = best_spacing

        segment_start_sec = float((segment_start_frame * hop_size) / max(sample_rate, 1))
        segment_end_sec = min(total_duration_seconds, float((segment_end_frame * hop_size) / max(sample_rate, 1)))
        spacing_seconds = float((best_spacing * hop_size) / max(sample_rate, 1))
        bpm = float(60.0 / max(spacing_seconds, 1e-6))
        confidence = _clamp01(0.28 + 0.58 * max(best_score, 0.0))
        segments.append(
            {
                "start_time_sec": round(segment_start_sec, 5),
                "end_time_sec": round(segment_end_sec, 5),
                "spacing_sec": round(spacing_seconds, 5),
                "bpm": round(bpm, 5),
                "confidence": round(float(confidence), 5),
            }
        )
        if segment_end_frame >= len(onset_envelope):
            break

    return segments


def _spacing_from_local_tempo_segments(
    time_sec: float,
    local_tempo_segments: list[dict[str, Any]],
    default_spacing_sec: float,
) -> float:
    if not local_tempo_segments:
        return float(default_spacing_sec)
    matching_segments = [
        segment
        for segment in local_tempo_segments
        if float(segment.get("start_time_sec", 0.0) or 0.0) - 1e-8 <= time_sec <= float(segment.get("end_time_sec", 0.0) or 0.0) + 1e-8
    ]
    if not matching_segments:
        return float(default_spacing_sec)
    matching_segments.sort(
        key=lambda segment: abs(
            time_sec
            - (
                float(segment.get("start_time_sec", 0.0) or 0.0)
                + float(segment.get("end_time_sec", 0.0) or 0.0)
            )
            / 2.0
        )
    )
    return float(matching_segments[0].get("spacing_sec", default_spacing_sec) or default_spacing_sec)


def _build_elastic_beats(
    onset_envelope: np.ndarray,
    sample_rate: int,
    hop_size: int,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    beats_per_bar: int,
    profile: str = "fallback",
) -> dict[str, Any]:
    profile = "rhythmic_first" if str(profile or "") == "rhythmic_first" else "fallback"
    if onset_envelope.size == 0:
        return {
            "bpm": 0.0,
            "tempo_confidence": 0.0,
            "tracking_mode": "elastic_grid",
            "profile": profile,
            "beats": [],
            "periodic_phase": 0,
            "beat_confidence_mean": 0.0,
            "local_tempo_segments": [],
        }

    bpm, spacing_frames, tempo_confidence = _estimate_bpm(onset_envelope, sample_rate=sample_rate, hop_size=hop_size)
    global_spacing_seconds = float((spacing_frames * hop_size) / max(sample_rate, 1))
    full_tempo_segments = _estimate_local_tempo_segments(
        onset_envelope=onset_envelope,
        sample_rate=sample_rate,
        hop_size=hop_size,
        spacing_frames=spacing_frames,
        window_seconds=RHYTHMIC_FIRST_LOCAL_TEMPO_WINDOW_SECONDS,
        step_seconds=RHYTHMIC_FIRST_LOCAL_TEMPO_STEP_SECONDS,
        max_delta_ratio=RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO,
    )
    peak_floor = float(np.percentile(onset_envelope, 62.0)) if onset_envelope.size else 0.0
    peak_indices = _local_peak_indices(onset_envelope, floor=peak_floor)
    peak_array = np.asarray(peak_indices, dtype=np.int32)
    peak_scores: dict[int, float] = {}
    peak_features: dict[int, dict[str, float]] = {}
    for peak_index in peak_indices:
        shape_score, features = _peak_shape_score(onset_envelope, peak_index)
        peak_scores[int(peak_index)] = float(shape_score)
        peak_features[int(peak_index)] = features

    start_frame = _best_beat_offset(onset_envelope=onset_envelope, spacing_frames=spacing_frames)
    if peak_array.size:
        start_tolerance = max(2, int(round(spacing_frames * 0.45)))
        candidate_mask = (peak_array >= start_frame - start_tolerance) & (peak_array <= start_frame + start_tolerance)
        candidate_indices = peak_array[candidate_mask]
        if candidate_indices.size:
            anchor_frame = int(
                max(
                    candidate_indices.tolist(),
                    key=lambda idx: peak_scores.get(int(idx), float(onset_envelope[int(idx)]))
                    - 0.18 * (abs(int(idx) - start_frame) / max(start_tolerance, 1)),
                )
            )
        else:
            anchor_frame = int(start_frame)
    else:
        anchor_frame = int(start_frame)

    all_beats: list[dict[str, Any]] = []
    current_frame = max(0, anchor_frame)
    local_spacing = float(max(1, spacing_frames))
    max_frame = len(onset_envelope)
    guard = 0
    while current_frame < max_frame and guard < max_frame + 8:
        beat_time_sec = float((current_frame * hop_size) / max(sample_rate, 1))
        shape_score = peak_scores.get(current_frame, float(onset_envelope[current_frame]))
        feature_payload = peak_features.get(
            current_frame,
            {
                "peak_height": round(float(onset_envelope[current_frame]), 5),
                "prominence": round(float(onset_envelope[current_frame]), 5),
                "slope": 0.0,
                "width": 0.0,
            },
        )
        beat_confidence = _clamp01(0.2 + 0.55 * shape_score + 0.25 * tempo_confidence)
        all_beats.append(
            {
                "index": len(all_beats),
                "absolute_time_sec": round(float(beat_time_sec), 5),
                "strength": round(float(onset_envelope[current_frame]), 5),
                "confidence": round(float(beat_confidence), 5),
                "local_spacing_sec": round(float((local_spacing * hop_size) / max(sample_rate, 1)), 5),
                "local_bpm": round(float(60.0 * sample_rate / max(local_spacing * hop_size, 1.0)), 5),
                "peak_height": feature_payload["peak_height"],
                "peak_prominence": feature_payload["prominence"],
                "peak_slope": feature_payload["slope"],
                "peak_width": feature_payload["width"],
                "profile": profile,
            }
        )

        predicted_spacing = local_spacing
        if profile == "rhythmic_first":
            predicted_spacing = float(
                _spacing_from_local_tempo_segments(
                    time_sec=beat_time_sec,
                    local_tempo_segments=full_tempo_segments,
                    default_spacing_sec=global_spacing_seconds,
                )
                * sample_rate
                / max(hop_size, 1)
            )
            predicted_spacing = float(
                np.clip(
                    predicted_spacing,
                    max(1.0, local_spacing * (1.0 - RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO)),
                    max(
                        max(1.0, local_spacing * (1.0 - RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO)) + 1.0,
                        local_spacing * (1.0 + RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO),
                    ),
                )
            )
        predicted = current_frame + predicted_spacing
        if profile == "rhythmic_first":
            search_start = int(max(current_frame + 1, round(predicted - (0.08 * sample_rate / max(hop_size, 1)))))
            search_end = int(min(max_frame - 1, round(predicted + (0.12 * sample_rate / max(hop_size, 1)))))
            tolerance = max(2, search_end - search_start)
        else:
            tolerance = max(2, int(round(local_spacing * 0.42)))
            search_start = int(max(current_frame + 1, round(predicted - tolerance)))
            search_end = int(min(max_frame - 1, round(predicted + tolerance)))
        if search_end <= search_start:
            break

        next_frame = None
        next_confidence = 0.0
        if peak_array.size:
            candidate_mask = (peak_array >= search_start) & (peak_array <= search_end) & (peak_array > current_frame)
            candidate_indices = peak_array[candidate_mask]
            if candidate_indices.size:
                best_score = -1.0
                for candidate in candidate_indices.tolist():
                    candidate = int(candidate)
                    interval = candidate - current_frame
                    peak_score = peak_scores.get(candidate, float(onset_envelope[candidate]))
                    distance_score = _clamp01(1.0 - abs(candidate - predicted) / max(tolerance, 1))
                    interval_score = _clamp01(1.0 - abs(interval - local_spacing) / max(local_spacing * 0.45, 1.0))
                    global_score = _clamp01(1.0 - abs(interval - spacing_frames) / max(spacing_frames * 0.55, 1.0))
                    feature_payload = peak_features.get(
                        candidate,
                        {
                            "peak_height": round(float(onset_envelope[candidate]), 5),
                            "prominence": round(float(onset_envelope[candidate]), 5),
                            "slope": 0.0,
                            "width": 0.0,
                        },
                    )
                    if profile == "rhythmic_first":
                        spacing_consistency = _clamp01(1.0 - abs(interval - predicted_spacing) / max(predicted_spacing * 0.18, 1.0))
                        periodic_prior = _structural_periodic_prior(len(all_beats) + 1, beats_per_bar)
                        candidate_score = (
                            float(feature_payload.get("peak_height", 0.0) or 0.0)
                            + float(feature_payload.get("prominence", 0.0) or 0.0)
                            + float(feature_payload.get("slope", 0.0) or 0.0)
                            + float(spacing_consistency)
                            + float(periodic_prior)
                        ) / 5.0
                    else:
                        candidate_score = (
                            0.42 * peak_score
                            + 0.22 * distance_score
                            + 0.20 * interval_score
                            + 0.10 * global_score
                            + 0.06 * tempo_confidence
                        )
                    if candidate_score > best_score:
                        best_score = candidate_score
                        next_frame = candidate
                        next_confidence = _clamp01(candidate_score)

        if next_frame is None:
            next_frame = int(min(max_frame - 1, max(current_frame + 1, round(predicted))))
            next_confidence = _clamp01(0.12 + 0.35 * float(onset_envelope[next_frame]) + 0.15 * tempo_confidence)

        observed_interval = max(1.0, float(next_frame - current_frame))
        if profile == "rhythmic_first":
            lower_bound = max(1.0, predicted_spacing * (1.0 - RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO))
            upper_bound = max(lower_bound + 1.0, predicted_spacing * (1.0 + RHYTHMIC_FIRST_LOCAL_TEMPO_MAX_DELTA_RATIO))
            clipped_interval = float(np.clip(observed_interval, lower_bound, upper_bound))
            local_spacing = float(0.74 * predicted_spacing + 0.26 * clipped_interval)
        else:
            lower_bound = max(1.0, spacing_frames * 0.72)
            upper_bound = max(lower_bound + 1.0, spacing_frames * 1.28)
            local_spacing = float(np.clip(0.68 * local_spacing + 0.32 * observed_interval, lower_bound, upper_bound))
        if next_frame <= current_frame:
            break
        current_frame = next_frame
        guard += 1

    periodic_phase, periodic_priors = _periodic_prior_values(all_beats, beats_per_bar=beats_per_bar)
    for beat, periodic_prior in zip(all_beats, periodic_priors):
        beat["periodic_prior"] = round(float(periodic_prior), 5)
        beat["is_downbeat"] = bool(periodic_prior >= 0.95)
        beat["tracking_mode"] = "elastic_grid"
        beat["profile"] = profile

    clip_end_seconds = clip_start_seconds + clip_duration_seconds
    clip_beats = [
        {
            **beat,
            "time_sec": round(float(beat["absolute_time_sec"] - clip_start_seconds), 5),
        }
        for beat in all_beats
        if clip_start_seconds - 1e-8 <= float(beat["absolute_time_sec"]) <= clip_end_seconds + 1e-8
    ]
    for clip_index, beat in enumerate(clip_beats):
        beat["index"] = clip_index

    _, clip_priors = _periodic_prior_values(clip_beats, beats_per_bar=beats_per_bar)
    for beat, periodic_prior in zip(clip_beats, clip_priors):
        beat["periodic_prior"] = round(float(periodic_prior), 5)
        beat["is_downbeat"] = bool(periodic_prior >= 0.95)
        beat["profile"] = profile

    confidence_mean = float(np.mean([float(beat.get("confidence", 0.0) or 0.0) for beat in clip_beats])) if clip_beats else 0.0
    clip_tempo_segments = [
        {
            "start_time_sec": round(
                max(0.0, float(segment.get("start_time_sec", 0.0) or 0.0) - clip_start_seconds),
                5,
            ),
            "end_time_sec": round(
                min(clip_duration_seconds, float(segment.get("end_time_sec", 0.0) or 0.0) - clip_start_seconds),
                5,
            ),
            "spacing_sec": round(float(segment.get("spacing_sec", 0.0) or 0.0), 5),
            "bpm": round(float(segment.get("bpm", 0.0) or 0.0), 5),
            "confidence": round(float(segment.get("confidence", 0.0) or 0.0), 5),
        }
        for segment in full_tempo_segments
        if float(segment.get("end_time_sec", 0.0) or 0.0) >= clip_start_seconds - 1e-8
        and float(segment.get("start_time_sec", 0.0) or 0.0) <= clip_end_seconds + 1e-8
    ]
    clip_tempo_segments = [
        segment
        for segment in clip_tempo_segments
        if float(segment.get("end_time_sec", 0.0) or 0.0) > float(segment.get("start_time_sec", 0.0) or 0.0)
    ]
    return {
        "bpm": round(float(bpm), 5),
        "tempo_confidence": round(float(tempo_confidence), 5),
        "tracking_mode": "elastic_grid",
        "profile": profile,
        "beats": clip_beats,
        "periodic_phase": int(periodic_phase),
        "beat_confidence_mean": round(confidence_mean, 5),
        "local_tempo_segments": clip_tempo_segments,
    }


def _section_importance_for_time(
    beat_time_sec: float,
    sections: list[dict[str, Any]],
    local_spacing_sec: float,
) -> float:
    if not sections:
        return 0.0
    window = max(0.24, min(1.1, local_spacing_sec * 1.45))
    boundary_times: list[float] = []
    for section in sections:
        boundary_times.append(float(section.get("start_time_sec", 0.0) or 0.0))
        boundary_times.append(float(section.get("end_time_sec", 0.0) or 0.0))
    importance = 0.0
    for boundary_time in boundary_times:
        delta = abs(float(beat_time_sec) - boundary_time)
        importance = max(importance, _clamp01(1.0 - delta / max(window, 1e-6)))
    return round(float(importance), 5)


def _local_contrast_for_beat(beats: list[dict[str, Any]], index: int) -> float:
    if not beats:
        return 0.0
    start = max(0, index - 2)
    end = min(len(beats), index + 3)
    neighborhood = [
        float(beats[neighbor_index].get("strength", 0.0) or 0.0)
        for neighbor_index in range(start, end)
        if neighbor_index != index
    ]
    current_strength = float(beats[index].get("strength", 0.0) or 0.0)
    if not neighborhood:
        return round(float(current_strength), 5)
    baseline = float(np.median(np.asarray(neighborhood, dtype=np.float32)))
    return round(float(_clamp01(current_strength - baseline)), 5)


def _sparsify_events(
    events: list[dict[str, Any]],
    score_key: str,
    min_gap_seconds: float,
) -> list[dict[str, Any]]:
    if not events:
        return []
    selected: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: float(item.get(score_key, 0.0) or 0.0), reverse=True):
        event_time = float(event.get("time_sec", 0.0) or 0.0)
        if any(abs(event_time - float(existing.get("time_sec", 0.0) or 0.0)) < min_gap_seconds for existing in selected):
            continue
        selected.append(event)
    return sorted(selected, key=lambda item: float(item.get("time_sec", 0.0) or 0.0))


def _assign_event_levels(
    events: list[dict[str, Any]],
    score_key: str,
) -> list[dict[str, Any]]:
    if not events:
        return []
    scores = np.asarray([float(item.get(score_key, 0.0) or 0.0) for item in events], dtype=np.float32)
    q1, q2, q3 = [float(np.percentile(scores, quantile)) for quantile in (25.0, 50.0, 75.0)]
    leveled: list[dict[str, Any]] = []
    for index, event in enumerate(events):
        score = float(event.get(score_key, 0.0) or 0.0)
        level = 1
        if score >= q1:
            level = 2
        if score >= q2:
            level = 3
        if score >= q3:
            level = 4
        next_event = dict(event)
        next_event["index"] = index
        next_event["level"] = int(level)
        leveled.append(next_event)
    return leveled


def _spacing_consistency_for_beat(beats: list[dict[str, Any]], index: int) -> float:
    if not beats:
        return 0.0
    beat = beats[index]
    target_spacing = float(beat.get("local_spacing_sec", 0.0) or 0.0)
    if target_spacing <= 0:
        return 0.0

    intervals: list[float] = []
    if index > 0:
        previous_time = float(beats[index - 1].get("time_sec", 0.0) or 0.0)
        intervals.append(abs(float(beat.get("time_sec", 0.0) or 0.0) - previous_time))
    if index + 1 < len(beats):
        next_time = float(beats[index + 1].get("time_sec", 0.0) or 0.0)
        intervals.append(abs(next_time - float(beat.get("time_sec", 0.0) or 0.0)))
    if not intervals:
        return 1.0
    deviations = [abs(interval - target_spacing) / max(target_spacing, 1e-6) for interval in intervals]
    return round(float(_clamp01(1.0 - float(np.mean(np.asarray(deviations, dtype=np.float32))) / 0.24)), 5)


def _limit_top_k_per_band_window(
    events: list[dict[str, Any]],
    score_key: str,
    window_seconds: float,
    top_k: int,
) -> list[dict[str, Any]]:
    if not events:
        return []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for event in events:
        band = str(event.get("band", "") or "")
        bucket = int(float(event.get("time_sec", 0.0) or 0.0) // max(window_seconds, 1e-6))
        grouped.setdefault((band, bucket), []).append(event)
    limited: list[dict[str, Any]] = []
    for (_band, _bucket), bucket_events in grouped.items():
        limited.extend(
            sorted(
                bucket_events,
                key=lambda item: float(item.get(score_key, 0.0) or 0.0),
                reverse=True,
            )[:top_k]
        )
    return sorted(limited, key=lambda item: (float(item.get("time_sec", 0.0) or 0.0), str(item.get("band", ""))))


def _build_motion_accents(
    beats: list[dict[str, Any]],
    low_band_envelope: np.ndarray,
    sample_rate: int,
    hop_size: int,
    clip_start_seconds: float,
    sections: list[dict[str, Any]],
    profile: str = "fallback",
) -> list[dict[str, Any]]:
    profile = "rhythmic_first" if str(profile or "") == "rhythmic_first" else "fallback"
    if not beats:
        return []
    spacing_values = np.asarray([float(item.get("local_spacing_sec", 0.0) or 0.0) for item in beats], dtype=np.float32)
    valid_spacing = spacing_values[spacing_values > 0]
    median_spacing = float(np.median(valid_spacing)) if valid_spacing.size else 0.5

    candidates: list[dict[str, Any]] = []
    scores: list[float] = []
    for beat_index, beat in enumerate(beats):
        absolute_time = clip_start_seconds + float(beat.get("time_sec", 0.0) or 0.0)
        onset_strength = float(beat.get("strength", 0.0) or 0.0)
        low_strength = _envelope_value_at_time(
            values=low_band_envelope,
            sample_rate=sample_rate,
            hop_size=hop_size,
            time_sec=absolute_time,
        )
        local_contrast = _local_contrast_for_beat(beats, beat_index)
        periodic_prior = float(beat.get("periodic_prior", 0.0) or 0.0)
        spacing_consistency = _spacing_consistency_for_beat(beats, beat_index)
        section_importance = _section_importance_for_time(
            beat_time_sec=float(beat.get("time_sec", 0.0) or 0.0),
            sections=sections,
            local_spacing_sec=float(beat.get("local_spacing_sec", median_spacing) or median_spacing),
        )
        confidence = float(beat.get("confidence", 0.0) or 0.0)
        if profile == "rhythmic_first":
            motion_score = (
                0.21 * onset_strength
                + 0.15 * low_strength
                + 0.17 * local_contrast
                + 0.17 * periodic_prior
                + 0.10 * section_importance
                + 0.14 * spacing_consistency
                + 0.06 * confidence
            )
        else:
            motion_score = (
                0.28 * onset_strength
                + 0.18 * low_strength
                + 0.22 * local_contrast
                + 0.14 * periodic_prior
                + 0.12 * section_importance
                + 0.06 * confidence
            )
        scores.append(float(motion_score))
        candidates.append(
            {
                "beat_index": int(beat_index),
                "time_sec": round(float(beat.get("time_sec", 0.0) or 0.0), 5),
                "strength": round(float(onset_strength), 5),
                "low_band_strength": round(float(low_strength), 5),
                "local_contrast": round(float(local_contrast), 5),
                "periodic_prior": round(float(periodic_prior), 5),
                "section_importance": round(float(section_importance), 5),
                "spacing_consistency": round(float(spacing_consistency), 5),
                "motion_score": round(float(_clamp01(motion_score)), 5),
                "confidence": round(float(_clamp01(0.55 * confidence + 0.45 * motion_score)), 5),
                "is_downbeat": bool(beat.get("is_downbeat", False)),
                "profile": profile,
            }
        )

    score_threshold = float(np.percentile(np.asarray(scores, dtype=np.float32), 58.0)) if scores else 0.0
    prelim = [
        candidate
        for candidate in candidates
        if (
            float(candidate["motion_score"]) >= score_threshold
            and float(candidate["confidence"]) >= 0.34
        )
        or float(candidate["section_importance"]) >= 0.45
        or float(candidate["periodic_prior"]) >= 0.95
    ]
    if profile == "rhythmic_first":
        prelim = [
            candidate
            for candidate in prelim
            if float(candidate["spacing_consistency"]) >= 0.46
            or float(candidate["periodic_prior"]) >= 0.95
            or float(candidate["section_importance"]) >= 0.48
        ]
    sparse = _sparsify_events(
        events=prelim,
        score_key="motion_score",
        min_gap_seconds=max(0.62 if profile == "rhythmic_first" else 0.58, median_spacing * (2.05 if profile == "rhythmic_first" else 1.9)),
    )
    return _assign_event_levels(sparse, score_key="motion_score")


def _build_multi_band_accents(
    beats: list[dict[str, Any]],
    band_envelopes: dict[str, np.ndarray],
    sample_rate: int,
    hop_size: int,
    clip_start_seconds: float,
    profile: str = "fallback",
) -> list[dict[str, Any]]:
    profile = "rhythmic_first" if str(profile or "") == "rhythmic_first" else "fallback"
    if not beats:
        return []
    spacing_values = np.asarray([float(item.get("local_spacing_sec", 0.0) or 0.0) for item in beats], dtype=np.float32)
    valid_spacing = spacing_values[spacing_values > 0]
    median_spacing = float(np.median(valid_spacing)) if valid_spacing.size else 0.5

    band_specs = [
        ("low", max(0.28, median_spacing * 0.8)),
        ("low_mid", max(0.24, median_spacing * 0.65)),
        ("high_attack", max(0.18, median_spacing * 0.5)),
    ]
    all_events: list[dict[str, Any]] = []
    high_confidence_threshold = max(
        RHYTHMIC_FIRST_CONFIDENCE_THRESHOLD,
        float(np.percentile(np.asarray([float(item.get("confidence", 0.0) or 0.0) for item in beats], dtype=np.float32), 60.0)),
    )
    for band_name, min_gap in band_specs:
        envelope = np.asarray(band_envelopes.get(band_name, np.zeros((0,), dtype=np.float32)), dtype=np.float32)
        if envelope.size == 0:
            continue
        candidates: list[dict[str, Any]] = []
        confidence_scores: list[float] = []
        for beat_index, beat in enumerate(beats):
            beat_time = float(beat.get("time_sec", 0.0) or 0.0)
            absolute_time = clip_start_seconds + beat_time
            start_frame = max(0, int(np.floor(((absolute_time - 0.08) * sample_rate) / max(hop_size, 1))))
            end_frame = min(len(envelope), int(np.ceil(((absolute_time + 0.12) * sample_rate) / max(hop_size, 1))) + 1)
            if end_frame <= start_frame:
                continue
            window = envelope[start_frame:end_frame]
            if window.size == 0:
                continue
            peak_offset = int(window.argmax())
            peak_frame = start_frame + peak_offset
            peak_value = float(envelope[peak_frame])
            context_start = max(0, peak_frame - 8)
            context_end = min(len(envelope), peak_frame + 9)
            context = envelope[context_start:context_end]
            baseline = float(np.median(context)) if context.size else 0.0
            contrast = _clamp01(peak_value - baseline)
            peak_time_sec = float((peak_frame * hop_size) / max(sample_rate, 1))
            alignment = _clamp01(1.0 - abs(peak_time_sec - absolute_time) / 0.12)
            confidence = (
                0.45 * peak_value
                + 0.25 * contrast
                + 0.18 * alignment
                + 0.12 * float(beat.get("confidence", 0.0) or 0.0)
            )
            if peak_time_sec < clip_start_seconds - 1e-8:
                continue
            if peak_time_sec > clip_start_seconds + float(beats[-1].get("time_sec", 0.0) or 0.0) + 1e-8:
                continue
            candidate = {
                "source_beat_index": int(beat_index),
                "time_sec": round(float(peak_time_sec - clip_start_seconds), 5),
                "band": band_name,
                "strength": round(float(peak_value), 5),
                "contrast": round(float(contrast), 5),
                "alignment": round(float(alignment), 5),
                "confidence": round(float(_clamp01(confidence)), 5),
                "profile": profile,
            }
            if profile == "rhythmic_first":
                if float(beat.get("confidence", 0.0) or 0.0) < high_confidence_threshold:
                    continue
                if float(candidate["alignment"]) < 0.38:
                    continue
            candidates.append(candidate)
            confidence_scores.append(float(confidence))

        if not candidates:
            continue
        confidence_threshold = float(np.percentile(np.asarray(confidence_scores, dtype=np.float32), 72.0))
        filtered = [candidate for candidate in candidates if float(candidate["confidence"]) >= confidence_threshold]
        sparse = _sparsify_events(filtered, score_key="confidence", min_gap_seconds=min_gap)
        leveled = _assign_event_levels(sparse, score_key="confidence")
        if profile == "rhythmic_first":
            leveled = _limit_top_k_per_band_window(
                leveled,
                score_key="confidence",
                window_seconds=RHYTHMIC_FIRST_BAND_WINDOW_SECONDS,
                top_k=RHYTHMIC_FIRST_BAND_TOP_K,
            )
        all_events.extend(leveled)

    return sorted(all_events, key=lambda item: (float(item.get("time_sec", 0.0) or 0.0), str(item.get("band", ""))))


def _clip_frame_series(
    times_sec: np.ndarray,
    values: np.ndarray,
    clip_start_seconds: float,
    clip_duration_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    if times_sec.size == 0 or values.size == 0 or clip_duration_seconds <= 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    clip_end_seconds = clip_start_seconds + clip_duration_seconds
    mask = (times_sec >= clip_start_seconds - 1e-8) & (times_sec <= clip_end_seconds + 1e-8)
    clip_times = times_sec[mask] - clip_start_seconds
    clip_values = values[mask]
    if clip_times.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    return np.asarray(clip_times, dtype=np.float32), np.asarray(clip_values, dtype=np.float32)


def _encode_uint8_payload(values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.uint8).reshape(-1)
    return base64.b64encode(flat.tobytes()).decode("ascii")


def _resample_uniform_curve(
    times_sec: np.ndarray,
    values: np.ndarray,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    point_count: int,
) -> np.ndarray:
    if point_count <= 0:
        return np.zeros((0,), dtype=np.uint8)
    if clip_duration_seconds <= 0:
        return np.zeros((point_count,), dtype=np.uint8)
    sample_times = np.linspace(
        clip_start_seconds,
        clip_start_seconds + clip_duration_seconds,
        point_count,
        dtype=np.float32,
    )
    if times_sec.size == 0 or values.size == 0:
        return np.zeros((point_count,), dtype=np.uint8)
    source_times = np.asarray(times_sec, dtype=np.float32)
    source_values = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    if source_times.size == 1:
        sampled = np.full((point_count,), float(source_values[0]), dtype=np.float32)
    else:
        sampled = np.interp(sample_times, source_times, source_values, left=float(source_values[0]), right=float(source_values[-1]))
    return np.clip(np.round(sampled * 255.0), 0.0, 255.0).astype(np.uint8)


def _build_waveform_panel(
    audio: np.ndarray,
    sample_rate: int,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    bin_count: int = 512,
) -> dict[str, Any]:
    if bin_count <= 0 or clip_duration_seconds <= 0 or audio.size == 0:
        zero_bins = np.zeros((max(bin_count, 0),), dtype=np.uint8)
        return {
            "bin_count": int(max(bin_count, 0)),
            "encoding": "uint8/base64",
            "min_b64": _encode_uint8_payload(zero_bins),
            "max_b64": _encode_uint8_payload(zero_bins),
        }

    sample_start = max(0, int(round(clip_start_seconds * sample_rate)))
    sample_end = min(audio.size, int(round((clip_start_seconds + clip_duration_seconds) * sample_rate)))
    clip_audio = np.asarray(audio[sample_start:sample_end], dtype=np.float32)
    if clip_audio.size == 0:
        clip_audio = np.zeros((1,), dtype=np.float32)
    peak = max(float(np.max(np.abs(clip_audio))), 1e-6)
    clip_audio = np.clip(clip_audio / peak, -1.0, 1.0)

    edges = np.linspace(0, clip_audio.size, bin_count + 1, dtype=np.int64)
    mins = np.zeros((bin_count,), dtype=np.float32)
    maxs = np.zeros((bin_count,), dtype=np.float32)
    for index in range(bin_count):
        start = int(edges[index])
        end = int(edges[index + 1])
        segment = clip_audio[start:end]
        if segment.size == 0:
            continue
        mins[index] = float(segment.min())
        maxs[index] = float(segment.max())

    mins_uint8 = np.clip(np.round(((mins + 1.0) * 0.5) * 255.0), 0.0, 255.0).astype(np.uint8)
    maxs_uint8 = np.clip(np.round(((maxs + 1.0) * 0.5) * 255.0), 0.0, 255.0).astype(np.uint8)
    return {
        "bin_count": int(bin_count),
        "encoding": "uint8/base64",
        "min_b64": _encode_uint8_payload(mins_uint8),
        "max_b64": _encode_uint8_payload(maxs_uint8),
    }


def _build_curve_panel(
    times_sec: np.ndarray,
    values: np.ndarray,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    point_count: int = 256,
) -> dict[str, Any]:
    return {
        "point_count": int(point_count),
        "encoding": "uint8/base64",
        "values_b64": _encode_uint8_payload(
            _resample_uniform_curve(
                times_sec=times_sec,
                values=values,
                clip_start_seconds=clip_start_seconds,
                clip_duration_seconds=clip_duration_seconds,
                point_count=point_count,
            )
        ),
    }


def _build_mel_spectrogram_panel(
    frame_times_sec: np.ndarray,
    mel_spectrogram: np.ndarray,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    mel_bin_count: int = 48,
    time_bin_count: int = 192,
) -> dict[str, Any]:
    if clip_duration_seconds <= 0 or mel_bin_count <= 0 or time_bin_count <= 0:
        zeros = np.zeros((max(mel_bin_count, 0), max(time_bin_count, 0)), dtype=np.uint8)
        return {
            "mel_bin_count": int(max(mel_bin_count, 0)),
            "time_bin_count": int(max(time_bin_count, 0)),
            "encoding": "uint8/base64",
            "values_b64": _encode_uint8_payload(zeros),
        }

    sample_times = np.linspace(
        clip_start_seconds,
        clip_start_seconds + clip_duration_seconds,
        time_bin_count,
        dtype=np.float32,
    )
    mel = np.asarray(mel_spectrogram, dtype=np.float32)
    if mel.size == 0 or frame_times_sec.size == 0:
        sampled = np.zeros((mel_bin_count, time_bin_count), dtype=np.float32)
    else:
        mel = mel[:mel_bin_count, :]
        sampled = np.zeros((mel.shape[0], time_bin_count), dtype=np.float32)
        if frame_times_sec.size == 1:
            sampled[:] = mel[:, [0]]
        else:
            for band_index in range(mel.shape[0]):
                sampled[band_index, :] = np.interp(
                    sample_times,
                    frame_times_sec,
                    mel[band_index, :],
                    left=float(mel[band_index, 0]),
                    right=float(mel[band_index, -1]),
                )
        if sampled.shape[0] < mel_bin_count:
            padding = np.zeros((mel_bin_count - sampled.shape[0], time_bin_count), dtype=np.float32)
            sampled = np.concatenate([sampled, padding], axis=0)

    mel_min = float(np.percentile(sampled, 8.0)) if sampled.size else 0.0
    mel_max = float(np.percentile(sampled, 99.0)) if sampled.size else 1.0
    if mel_max - mel_min <= 1e-8:
        normalized = np.zeros_like(sampled, dtype=np.float32)
    else:
        normalized = np.clip((sampled - mel_min) / (mel_max - mel_min), 0.0, 1.0)
    quantized = np.clip(np.round(normalized * 255.0), 0.0, 255.0).astype(np.uint8)
    return {
        "mel_bin_count": int(mel_bin_count),
        "time_bin_count": int(time_bin_count),
        "encoding": "uint8/base64",
        "values_b64": _encode_uint8_payload(quantized),
    }


def _clip_section_entries(
    sections: list[dict[str, Any]],
    clip_start_seconds: float,
    clip_duration_seconds: float,
) -> list[dict[str, Any]]:
    if clip_duration_seconds <= 0:
        return []
    clip_end_seconds = clip_start_seconds + clip_duration_seconds
    clipped: list[dict[str, Any]] = []
    for section in sections:
        start_time = float(section.get("start_time_sec", 0.0) or 0.0)
        end_time = float(section.get("end_time_sec", 0.0) or 0.0)
        if end_time < clip_start_seconds or start_time > clip_end_seconds:
            continue
        clipped_start = max(start_time, clip_start_seconds) - clip_start_seconds
        clipped_end = min(end_time, clip_end_seconds) - clip_start_seconds
        clipped.append(
            {
                "index": len(clipped),
                "label": str(section.get("label", "section") or "section"),
                "start_time_sec": round(float(clipped_start), 5),
                "end_time_sec": round(float(max(clipped_start, clipped_end)), 5),
                "confidence": round(float(section.get("confidence", 0.0) or 0.0), 5),
                "source": str(section.get("source", "automatic") or "automatic"),
            }
        )
    return clipped


def _build_audio_analysis_panel(
    music_path: Path,
    audio: np.ndarray,
    sample_rate: int,
    frame_analysis: dict[str, Any],
    clip_start_seconds: float,
    clip_duration_seconds: float,
    sections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    frame_times_sec = np.asarray(frame_analysis.get("frame_times_sec", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    onset_envelope = np.asarray(frame_analysis.get("onset_envelope", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    low_band_envelope = np.asarray(frame_analysis.get("low_band_envelope", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    mel_spectrogram = _compute_mel_spectrogram(
        spectrum=np.asarray(frame_analysis.get("spectrum", np.zeros((0, 0), dtype=np.float32)), dtype=np.float32),
        sample_rate=int(sample_rate),
        frame_size=int(frame_analysis.get("frame_size", DEFAULT_FRAME_SIZE) or DEFAULT_FRAME_SIZE),
        mel_bins=48,
    )
    if sections is None:
        song_event_map = analyze_song(
            input_path=music_path,
            song_id=music_path.stem,
            beats_per_bar=4,
        )
        sections = _clip_section_entries(
            sections=list(song_event_map.sections),
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
        )
    return {
        "clip_start_sec": round(float(clip_start_seconds), 5),
        "clip_end_sec": round(float(clip_start_seconds + clip_duration_seconds), 5),
        "waveform": _build_waveform_panel(
            audio=audio,
            sample_rate=sample_rate,
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            bin_count=512,
        ),
        "mel_spectrogram": _build_mel_spectrogram_panel(
            frame_times_sec=frame_times_sec,
            mel_spectrogram=mel_spectrogram,
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            mel_bin_count=48,
            time_bin_count=192,
        ),
        "onset_curve": _build_curve_panel(
            times_sec=frame_times_sec,
            values=onset_envelope,
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            point_count=256,
        ),
        "low_band_curve": _build_curve_panel(
            times_sec=frame_times_sec,
            values=low_band_envelope,
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            point_count=256,
        ),
        "sections": sections,
    }


def _empty_audio_bundle(clip_start_seconds: float, clip_duration_seconds: float, beats_per_bar: int) -> dict[str, Any]:
    return {
        "audio_features": {
            "bpm": 0.0,
            "beats_per_bar": int(beats_per_bar),
            "energy_score": 0.0,
            "energy_label": "mid_energy",
            "beat_tracking": {
                "mode": "elastic_grid",
                "profile": "fallback",
                "global_bpm": 0.0,
                "tempo_confidence": 0.0,
                "periodic_phase": 0,
                "average_confidence": 0.0,
                "local_tempo_segments": [],
            },
            "energy_curve": [],
            "beats": [],
            "strong_beats": [],
            "accent_candidates": [],
            "drum_hits": [],
        },
        "audio_analysis_panel": {
            "clip_start_sec": round(float(clip_start_seconds), 5),
            "clip_end_sec": round(float(clip_start_seconds + clip_duration_seconds), 5),
            "waveform": _build_waveform_panel(
                audio=np.zeros((1,), dtype=np.float32),
                sample_rate=1,
                clip_start_seconds=0.0,
                clip_duration_seconds=0.0,
                bin_count=512,
            ),
            "mel_spectrogram": _build_mel_spectrogram_panel(
                frame_times_sec=np.zeros((0,), dtype=np.float32),
                mel_spectrogram=np.zeros((48, 0), dtype=np.float32),
                clip_start_seconds=0.0,
                clip_duration_seconds=0.0,
                mel_bin_count=48,
                time_bin_count=192,
            ),
            "onset_curve": _build_curve_panel(
                times_sec=np.zeros((0,), dtype=np.float32),
                values=np.zeros((0,), dtype=np.float32),
                clip_start_seconds=0.0,
                clip_duration_seconds=0.0,
                point_count=256,
            ),
            "low_band_curve": _build_curve_panel(
                times_sec=np.zeros((0,), dtype=np.float32),
                values=np.zeros((0,), dtype=np.float32),
                clip_start_seconds=0.0,
                clip_duration_seconds=0.0,
                point_count=256,
            ),
            "sections": [],
        },
    }


def _extract_audio_bundle(
    music_path: Path,
    clip_start_seconds: float,
    clip_duration_seconds: float,
    beats_per_bar: int = 4,
    beat_profile: str = "fallback",
) -> dict[str, Any]:
    if clip_duration_seconds <= 0:
        return _empty_audio_bundle(
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            beats_per_bar=beats_per_bar,
        )

    audio, sample_rate = _load_audio(music_path)
    if audio.size == 0:
        return _empty_audio_bundle(
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            beats_per_bar=beats_per_bar,
        )

    frame_analysis = _build_frame_analysis(
        audio=audio,
        sample_rate=sample_rate,
        frame_size=DEFAULT_FRAME_SIZE,
        hop_size=DEFAULT_HOP_SIZE,
    )
    frame_times_sec = np.asarray(frame_analysis["frame_times_sec"], dtype=np.float32)
    rms_norm = np.asarray(frame_analysis["rms_norm"], dtype=np.float32)
    low_band_envelope = np.asarray(frame_analysis["low_band_envelope"], dtype=np.float32)
    band_envelopes = {
        name: np.asarray(values, dtype=np.float32)
        for name, values in dict(frame_analysis.get("band_envelopes", {})).items()
    }

    song_event_map = analyze_song(
        input_path=music_path,
        song_id=music_path.stem,
        beats_per_bar=beats_per_bar,
    )
    sections = _clip_section_entries(
        sections=list(song_event_map.sections),
        clip_start_seconds=clip_start_seconds,
        clip_duration_seconds=clip_duration_seconds,
    )
    beat_tracking = _build_elastic_beats(
        onset_envelope=np.asarray(frame_analysis["onset_envelope"], dtype=np.float32),
        sample_rate=sample_rate,
        hop_size=DEFAULT_HOP_SIZE,
        clip_start_seconds=clip_start_seconds,
        clip_duration_seconds=clip_duration_seconds,
        beats_per_bar=beats_per_bar,
        profile=beat_profile,
    )
    beats = list(beat_tracking.get("beats", []))
    strong_beats = _build_motion_accents(
        beats=beats,
        low_band_envelope=low_band_envelope,
        sample_rate=sample_rate,
        hop_size=DEFAULT_HOP_SIZE,
        clip_start_seconds=clip_start_seconds,
        sections=sections,
        profile=beat_profile,
    )
    accent_candidates = _build_multi_band_accents(
        beats=beats,
        band_envelopes=band_envelopes,
        sample_rate=sample_rate,
        hop_size=DEFAULT_HOP_SIZE,
        clip_start_seconds=clip_start_seconds,
        profile=beat_profile,
    )
    drum_hits = [
        {
            "index": len(
                [
                    existing
                    for existing in accent_candidates[:candidate_index]
                    if str(existing.get("band", "")) == "low"
                ]
            ),
            "time_sec": round(float(candidate.get("time_sec", 0.0) or 0.0), 5),
            "strength": round(float(candidate.get("strength", 0.0) or 0.0), 5),
            "confidence": round(float(candidate.get("confidence", 0.0) or 0.0), 5),
            "contrast": round(float(candidate.get("contrast", 0.0) or 0.0), 5),
            "level": int(candidate.get("level", 1) or 1),
            "band": "low",
        }
        for candidate_index, candidate in enumerate(accent_candidates)
        if str(candidate.get("band", "")) == "low"
    ]

    energy_curve = _downsample_series(
        times_sec=frame_times_sec,
        values=rms_norm,
        clip_start_seconds=clip_start_seconds,
        clip_duration_seconds=clip_duration_seconds,
        max_points=64,
    )
    _, clip_energy_values = _clip_frame_series(
        times_sec=frame_times_sec,
        values=rms_norm,
        clip_start_seconds=clip_start_seconds,
        clip_duration_seconds=clip_duration_seconds,
    )
    if clip_energy_values.size == 0:
        energy_score = 0.0
    else:
        energy_score = float(np.percentile(clip_energy_values, 85.0))

    return {
        "audio_features": {
            "bpm": round(float(beat_tracking.get("bpm", 0.0) or 0.0), 5),
            "beats_per_bar": int(beats_per_bar),
            "energy_score": round(float(energy_score), 5),
            "energy_label": "mid_energy",
            "beat_tracking": {
                "mode": str(beat_tracking.get("tracking_mode", "elastic_grid") or "elastic_grid"),
                "profile": str(beat_tracking.get("profile", beat_profile) or beat_profile),
                "global_bpm": round(float(beat_tracking.get("bpm", 0.0) or 0.0), 5),
                "tempo_confidence": round(float(beat_tracking.get("tempo_confidence", 0.0) or 0.0), 5),
                "periodic_phase": int(beat_tracking.get("periodic_phase", 0) or 0),
                "average_confidence": round(float(beat_tracking.get("beat_confidence_mean", 0.0) or 0.0), 5),
                "local_tempo_segments": list(beat_tracking.get("local_tempo_segments", [])),
            },
            "energy_curve": energy_curve,
            "beats": beats,
            "strong_beats": strong_beats,
            "accent_candidates": accent_candidates,
            "drum_hits": drum_hits,
        },
        "audio_analysis_panel": _build_audio_analysis_panel(
            music_path=music_path,
            audio=audio,
            sample_rate=sample_rate,
            frame_analysis=frame_analysis,
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            sections=sections,
        ),
    }


def resolve_dataset_truth_entries(config: AppConfig, dataset_name: str, sequence_ids: list[str]) -> list[dict]:
    if dataset_name != "finedance":
        raise ValueError(f"Unsupported dataset for truth preview: {dataset_name}")

    entries: list[dict] = []
    for sequence_id in sequence_ids:
        entry = _load_catalog_entry(config, dataset_name, sequence_id)
        motion_path, music_path, label_path = _resolve_finedance_paths(config, entry, sequence_id)
        preview_path, preview_summary_path = _resolve_finedance_preview(config, sequence_id)
        preview_with_audio_path = None
        if preview_path is not None:
            candidate = preview_path.with_name(f"finedance_{sequence_id}_official_mesh_preview_with_audio.mp4")
            if candidate.exists():
                preview_with_audio_path = candidate
        playback_preview_path = preview_with_audio_path or preview_path
        preview_summary = load_json(preview_summary_path) if preview_summary_path and preview_summary_path.exists() else {}
        frame_count = _motion_frame_count(motion_path)
        fps = float(preview_summary.get("fps", 30.0) or 30.0)
        motion_duration_seconds = frame_count / fps if fps > 0 else 0.0
        audio_duration_seconds = _wav_duration_seconds(music_path)
        clip_start_seconds = float(preview_summary.get("startSeconds", 0.0) or 0.0)
        preview_duration_seconds = float(preview_summary.get("maxSeconds", 0.0) or 0.0)
        clip_duration_seconds = min(
            [value for value in (preview_duration_seconds, motion_duration_seconds, audio_duration_seconds) if value > 0] or [0.0]
        )
        label_summary = _load_label_summary(label_path)
        entries.append(
            {
                "dataset_name": dataset_name,
                "sequence_id": str(sequence_id),
                "motion_path": str(motion_path),
                "music_path": str(music_path),
                "label_path": str(label_path) if label_path else None,
                "preview_video_path": str(playback_preview_path) if playback_preview_path else None,
                "preview_video_path_raw": str(preview_path) if preview_path else None,
                "preview_video_with_audio_path": str(preview_with_audio_path) if preview_with_audio_path else None,
                "preview_embedded_audio": preview_with_audio_path is not None,
                "preview_summary_path": str(preview_summary_path) if preview_summary_path else None,
                "preview_available": preview_path is not None,
                "motion_frame_count": frame_count,
                "fps": fps,
                "motion_duration_seconds": motion_duration_seconds,
                "audio_duration_seconds": audio_duration_seconds,
                "preview_duration_seconds": preview_duration_seconds,
                "clip_start_seconds": clip_start_seconds,
                "clip_duration_seconds": clip_duration_seconds,
                "duration_delta_seconds": abs(motion_duration_seconds - audio_duration_seconds),
                "label_summary": label_summary,
                "preview_summary": preview_summary,
            }
        )

    energy_scores: list[float] = []
    for entry in entries:
        clip_start_seconds = float(entry.get("clip_start_seconds", 0.0) or 0.0)
        clip_duration_seconds = float(entry.get("clip_duration_seconds", 0.0) or 0.0)
        audio_bundle = _extract_audio_bundle(
            music_path=Path(str(entry["music_path"])),
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            beats_per_bar=4,
            beat_profile="fallback",
        )
        entry["audio_features"] = dict(audio_bundle.get("audio_features", {}))
        entry["audio_analysis_panel"] = dict(audio_bundle.get("audio_analysis_panel", {}))
        energy_scores.append(float(entry["audio_features"].get("energy_score", 0.0) or 0.0))

    energy_labels = _classify_energy_labels(energy_scores)
    for entry, energy_label in zip(entries, energy_labels):
        entry["audio_features"]["energy_label"] = energy_label
        label_summary = dict(entry.get("label_summary", {}))
        coarse_style = str(label_summary.get("coarseStyle") or label_summary.get("style1") or "")
        fine_style = str(label_summary.get("fineStyle") or label_summary.get("style2") or "")
        average_confidence = float(entry["audio_features"].get("beat_tracking", {}).get("average_confidence", 0.0) or 0.0)
        entry["analysis_priority"] = _classify_analysis_priority(
            coarse_style=coarse_style,
            fine_style=fine_style,
            energy_label=energy_label,
            beat_confidence=average_confidence,
        )

    for entry in entries:
        if str((entry.get("analysis_priority") or {}).get("tier", "")) != "rhythmic_first":
            continue
        energy_label = str(entry["audio_features"].get("energy_label", "mid_energy") or "mid_energy")
        fallback_audio_features = dict(entry.get("audio_features", {}))
        fallback_audio_panel = dict(entry.get("audio_analysis_panel", {}))
        clip_start_seconds = float(entry.get("clip_start_seconds", 0.0) or 0.0)
        clip_duration_seconds = float(entry.get("clip_duration_seconds", 0.0) or 0.0)
        audio_bundle = _extract_audio_bundle(
            music_path=Path(str(entry["music_path"])),
            clip_start_seconds=clip_start_seconds,
            clip_duration_seconds=clip_duration_seconds,
            beats_per_bar=4,
            beat_profile="rhythmic_first",
        )
        entry["audio_features"] = dict(audio_bundle.get("audio_features", {}))
        entry["audio_analysis_panel"] = dict(audio_bundle.get("audio_analysis_panel", {}))
        entry["audio_features"]["energy_label"] = energy_label
        entry["analysis_priority"] = _classify_analysis_priority(
            coarse_style=str((entry.get("label_summary") or {}).get("coarseStyle") or (entry.get("label_summary") or {}).get("style1") or ""),
            fine_style=str((entry.get("label_summary") or {}).get("fineStyle") or (entry.get("label_summary") or {}).get("style2") or ""),
            energy_label=energy_label,
            beat_confidence=float(entry["audio_features"].get("beat_tracking", {}).get("average_confidence", 0.0) or 0.0),
        )
        if str((entry.get("analysis_priority") or {}).get("tier", "")) != "rhythmic_first":
            entry["audio_features"] = fallback_audio_features
            entry["audio_analysis_panel"] = fallback_audio_panel
            entry["analysis_priority"] = _classify_analysis_priority(
                coarse_style=str((entry.get("label_summary") or {}).get("coarseStyle") or (entry.get("label_summary") or {}).get("style1") or ""),
                fine_style=str((entry.get("label_summary") or {}).get("fineStyle") or (entry.get("label_summary") or {}).get("style2") or ""),
                energy_label=energy_label,
                beat_confidence=float(entry["audio_features"].get("beat_tracking", {}).get("average_confidence", 0.0) or 0.0),
            )

    return entries


def _asset_href(output_dir: Path, asset_path: str | None) -> str | None:
    if not asset_path:
        return None
    return os.path.relpath(Path(asset_path).resolve(), output_dir.resolve())


def _format_seconds(value: float) -> str:
    return f"{value:.2f}s"


def _format_bytes(value: int) -> str:
    kib = value / 1024.0
    if kib < 1024:
        return f"{kib:.0f} KB"
    return f"{kib / 1024.0:.2f} MB"


def build_dataset_truth_audio_feature_report(dataset_name: str, entries: list[dict]) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "entry_count": len(entries),
        "entries": [
            {
                "sequence_id": str(entry.get("sequence_id", "")),
                "style_tags": {
                    "song_name": str((entry.get("label_summary") or {}).get("songName") or (entry.get("label_summary") or {}).get("name") or ""),
                    "coarse_style": str((entry.get("label_summary") or {}).get("coarseStyle") or (entry.get("label_summary") or {}).get("style1") or ""),
                    "fine_style": str((entry.get("label_summary") or {}).get("fineStyle") or (entry.get("label_summary") or {}).get("style2") or ""),
                },
                "analysis_priority": dict(entry.get("analysis_priority", {})),
                "audio_features": dict(entry.get("audio_features", {})),
                "audio_analysis_panel": dict(entry.get("audio_analysis_panel", {})),
            }
            for entry in entries
        ],
    }


def build_dataset_truth_preview_document(dataset_name: str, entries: list[dict], output_dir: Path) -> str:
    if not entries:
        raise ValueError("At least one truth preview entry is required.")
    page_title = f"Milestone M2-2 · {dataset_name} dataset truth viewer"

    prepared_entries: list[dict] = []
    for entry in entries:
        label_summary = dict(entry.get("label_summary", {}))
        song_name = str(label_summary.get("songName") or label_summary.get("name") or "")
        coarse_style = str(label_summary.get("coarseStyle") or label_summary.get("style1") or "")
        fine_style = str(label_summary.get("fineStyle") or label_summary.get("style2") or "")
        audio_features = dict(entry.get("audio_features", {}))
        audio_analysis_panel = dict(entry.get("audio_analysis_panel", {}))
        prepared_entries.append(
            {
                "dataset_name": entry["dataset_name"],
                "sequence_id": str(entry["sequence_id"]),
                "song_name": song_name,
                "coarse_style": coarse_style,
                "fine_style": fine_style,
                "analysis_priority": dict(entry.get("analysis_priority", {})),
                "style_tags": {
                    "song_name": song_name,
                    "coarse_style": coarse_style,
                    "fine_style": fine_style,
                },
                "motion_path": str(entry["motion_path"]),
                "music_path": str(entry["music_path"]),
                "label_path": entry.get("label_path"),
                "preview_video_path": entry.get("preview_video_path"),
                "preview_video_path_raw": entry.get("preview_video_path_raw"),
                "preview_video_with_audio_path": entry.get("preview_video_with_audio_path"),
                "preview_video_href": _asset_href(output_dir, entry.get("preview_video_path")),
                "music_href": _asset_href(output_dir, str(entry["music_path"])),
                "motion_href": _asset_href(output_dir, str(entry["motion_path"])),
                "label_href": _asset_href(output_dir, entry.get("label_path")),
                "preview_available": bool(entry.get("preview_available", False)),
                "preview_embedded_audio": bool(entry.get("preview_embedded_audio", False)),
                "motion_frame_count": int(entry["motion_frame_count"]),
                "preview_frame_count": int(round(float(entry["preview_duration_seconds"]) * float(entry["fps"]))),
                "fps": float(entry["fps"]),
                "motion_duration_seconds": float(entry["motion_duration_seconds"]),
                "audio_duration_seconds": float(entry["audio_duration_seconds"]),
                "preview_duration_seconds": float(entry["preview_duration_seconds"]),
                "clip_start_seconds": float(entry.get("clip_start_seconds", 0.0) or 0.0),
                "clip_duration_seconds": float(entry["clip_duration_seconds"]),
                "duration_delta_seconds": float(entry["duration_delta_seconds"]),
                "audio_features": {
                    "bpm": round(float(audio_features.get("bpm", 0.0) or 0.0), 5),
                    "beats_per_bar": int(audio_features.get("beats_per_bar", 4) or 4),
                    "energy_score": round(float(audio_features.get("energy_score", 0.0) or 0.0), 5),
                    "energy_label": str(audio_features.get("energy_label", "mid_energy") or "mid_energy"),
                    "beat_tracking": dict(audio_features.get("beat_tracking", {})),
                    "energy_curve": list(audio_features.get("energy_curve", [])),
                    "beats": list(audio_features.get("beats", [])),
                    "strong_beats": list(audio_features.get("strong_beats", [])),
                    "accent_candidates": list(audio_features.get("accent_candidates", [])),
                    "drum_hits": list(audio_features.get("drum_hits", [])),
                },
                "audio_analysis_panel": {
                    "clip_start_sec": round(float(audio_analysis_panel.get("clip_start_sec", 0.0) or 0.0), 5),
                    "clip_end_sec": round(float(audio_analysis_panel.get("clip_end_sec", 0.0) or 0.0), 5),
                    "waveform": dict(audio_analysis_panel.get("waveform", {})),
                    "mel_spectrogram": dict(audio_analysis_panel.get("mel_spectrogram", {})),
                    "onset_curve": dict(audio_analysis_panel.get("onset_curve", {})),
                    "low_band_curve": dict(audio_analysis_panel.get("low_band_curve", {})),
                    "sections": list(audio_analysis_panel.get("sections", [])),
                },
                "preview_video_size_bytes": (
                    Path(entry["preview_video_path"]).stat().st_size
                    if entry.get("preview_video_path") and Path(entry["preview_video_path"]).exists()
                    else 0
                ),
            }
        )

    rhythmic_first_count = sum(
        1 for entry in prepared_entries if str((entry.get("analysis_priority") or {}).get("tier", "")) == "rhythmic_first"
    )
    fallback_count = len(prepared_entries) - rhythmic_first_count

    payload_json = json.dumps(
        {
            "dataset_name": dataset_name,
            "entry_count": len(prepared_entries),
            "entries": prepared_entries,
        },
        ensure_ascii=False,
    )
    # Keep JSON parseable inside <script type="application/json">.
    payload_script_json = payload_json.replace("</", "<\\/")
    initial_sequence_list = "".join(
        (
            f'<button class="sequence-card" data-sequence-id="{html.escape(entry["sequence_id"])}">'
            f'<span class="id">{html.escape(entry["sequence_id"])}</span>'
            f'<span class="style">{html.escape((entry["coarse_style"] or "unknown") + " / " + (entry["fine_style"] or "unknown"))}</span>'
            f'<span class="priority {"priority-rhythmic" if str((entry.get("analysis_priority") or {}).get("tier", "")) == "rhythmic_first" else "priority-fallback"}">{html.escape("rhythmic-first" if str((entry.get("analysis_priority") or {}).get("tier", "")) == "rhythmic_first" else "fallback")}</span>'
            f'<span class="energy">{html.escape(str(entry.get("audio_features", {}).get("energy_label", "mid_energy")))}</span>'
            "</button>"
        )
        for entry in prepared_entries
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #efe7d8;
      --paper: #fff9ef;
      --ink: #1f1a16;
      --accent: #b65d33;
      --accent-soft: rgba(182, 93, 51, 0.14);
      --muted: #6f655c;
      --line: rgba(31, 26, 22, 0.12);
      --shadow: 0 22px 60px rgba(60, 36, 18, 0.14);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "Helvetica Neue", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(182, 93, 51, 0.18), transparent 24%),
        linear-gradient(180deg, #f8f1e5, var(--bg));
    }}
    main {{ max-width: 1560px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 42px; }}
    .lede {{
      margin: 0 0 18px;
      color: var(--muted);
      line-height: 1.6;
      max-width: 1000px;
    }}
    .summary-bar {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 18px;
    }}
    .summary-pill {{
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      box-shadow: var(--shadow);
      font-weight: 600;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }}
    .sidebar, .viewer {{
      background: var(--paper);
      border-radius: 24px;
      border: 1px solid rgba(31, 26, 22, 0.08);
      box-shadow: var(--shadow);
    }}
    .sidebar {{
      position: sticky;
      top: 18px;
      padding: 16px;
      max-height: calc(100vh - 36px);
      overflow: hidden;
      display: flex;
      flex-direction: column;
    }}
    .viewer {{ padding: 20px; }}
    .search-box {{
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      padding: 12px 14px;
      font-size: 15px;
      margin-bottom: 12px;
      background: white;
    }}
    .sequence-list {{
      overflow: auto;
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      padding-right: 4px;
      align-content: start;
      min-height: 220px;
      flex: 1 1 auto;
    }}
    .sequence-card {{
      width: 100%;
      text-align: left;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.76);
      padding: 10px 9px;
      cursor: pointer;
    }}
    .sequence-card.active {{
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }}
    .sequence-card .id {{
      display: block;
      font-size: 16px;
      font-weight: 700;
      margin-bottom: 3px;
    }}
    .sequence-card .style {{
      display: block;
      font-size: 11px;
      color: var(--muted);
      line-height: 1.3;
      margin-bottom: 5px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .sequence-card .energy {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: rgba(182, 93, 51, 0.14);
      color: #7a3e20;
    }}
    .sequence-card .priority {{
      display: inline-block;
      border-radius: 999px;
      padding: 2px 8px;
      font-size: 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      margin: 0 6px 0 0;
    }}
    .priority-rhythmic {{
      background: rgba(58, 148, 110, 0.16);
      color: #205844;
    }}
    .priority-fallback {{
      background: rgba(90, 90, 105, 0.16);
      color: #4c4c58;
    }}
    .sequence-card.active .style {{
      color: rgba(255, 255, 255, 0.82);
    }}
    .sequence-card.active .energy {{
      background: rgba(255, 255, 255, 0.16);
      color: white;
    }}
    .sequence-card.active .priority-rhythmic,
    .sequence-card.active .priority-fallback {{
      background: rgba(255, 255, 255, 0.18);
      color: white;
    }}
    .sidebar-tools {{
      display: grid;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .priority-filters {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .priority-filter {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      background: rgba(255, 255, 255, 0.78);
      color: var(--ink);
      font-size: 12px;
      font-weight: 700;
      cursor: pointer;
    }}
    .priority-filter.active {{
      background: var(--accent);
      border-color: var(--accent);
      color: white;
    }}
    .viewer-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .viewer-title {{
      margin: 0;
      font-size: 34px;
    }}
    .nav-button {{
      border: none;
      border-radius: 999px;
      background: rgba(182, 93, 51, 0.12);
      color: #7a3e20;
      padding: 10px 14px;
      font-weight: 700;
      cursor: pointer;
    }}
    .picker-status {{
      font-size: 14px;
      color: var(--muted);
      font-weight: 600;
    }}
    .quick-nav {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin: 0 0 14px;
    }}
    .viewer-subtitle {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    .meta-badges {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-top: 10px;
    }}
    .meta-badge {{
      border-radius: 999px;
      background: var(--accent-soft);
      padding: 8px 12px;
      font-size: 13px;
      font-weight: 600;
      color: #6a371b;
    }}
    .media {{
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #130f0c;
      border-radius: 20px;
      display: block;
    }}
    .controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: center;
      margin: 14px 0 10px;
    }}
    .controls button {{
      border: none;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      padding: 10px 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    .lock-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .timeline {{ width: 100%; margin: 6px 0 10px; }}
    .music-timeline {{
      width: 100%;
      height: 140px;
      display: block;
      margin: 4px 0 10px;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(17, 27, 42, 0.98), rgba(14, 22, 35, 0.96));
    }}
    .timeline-legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 12px;
    }}
    .timeline-legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .timeline-legend i {{
      display: inline-block;
      width: 10px;
      height: 10px;
      border-radius: 50%;
    }}
    .analysis-panel {{
      margin: 14px 0 16px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: linear-gradient(180deg, rgba(15, 23, 35, 0.985), rgba(9, 15, 24, 0.98));
      padding: 14px 14px 12px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
    }}
    .analysis-header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .analysis-header h3 {{
      margin: 0;
      color: #f8f1e5;
      font-size: 18px;
    }}
    .analysis-header p {{
      margin: 0;
      color: rgba(248, 241, 229, 0.68);
      font-size: 13px;
      line-height: 1.5;
      max-width: 780px;
    }}
    .analysis-canvas {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: linear-gradient(180deg, rgba(14, 20, 30, 0.98), rgba(10, 15, 24, 0.98));
    }}
    .analysis-legend {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin: 10px 0 0;
      color: rgba(248, 241, 229, 0.72);
      font-size: 12px;
    }}
    .analysis-legend span {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }}
    .analysis-legend i {{
      display: inline-block;
      width: 12px;
      height: 12px;
      border-radius: 3px;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    .metric {{
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 12px;
    }}
    .metric .label {{
      display: block;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .metric .value {{
      display: block;
      font-size: 18px;
      font-weight: 700;
    }}
    .note {{
      margin: 0 0 16px;
      color: var(--muted);
      line-height: 1.55;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: start;
    }}
    .detail-card {{
      background: rgba(255, 255, 255, 0.72);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 16px;
    }}
    .detail-card h3 {{
      margin: 0 0 12px;
      font-size: 17px;
    }}
    .detail-card ul {{
      margin: 0;
      padding-left: 18px;
      line-height: 1.7;
      word-break: break-all;
    }}
    .detail-card pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.55;
    }}
    @media (max-width: 1080px) {{
      .layout {{ grid-template-columns: 1fr; }}
      .sidebar {{ position: static; max-height: none; }}
      .metrics {{ grid-template-columns: repeat(2, minmax(120px, 1fr)); }}
      .detail-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <h1>{html.escape(page_title)}</h1>
    <p class="lede">Milestone M2-2 rhythmic-first baseline: source-truth viewer for the original FineDance dataset. We keep the full dataset visible, but explicitly mark which clips belong to the current rhythmic-first benchmark set so beat-rail calibration can focus on the most rhythmically regular songs first.</p>
    <div class="summary-bar">
      <span class="summary-pill">milestone: M2-2</span>
      <span class="summary-pill">dataset: {html.escape(dataset_name)}</span>
      <span class="summary-pill">sequences in page: {len(prepared_entries)}</span>
      <span class="summary-pill">official preview clips: {sum(1 for entry in prepared_entries if entry["preview_available"])}</span>
      <span class="summary-pill">rhythmic-first: {rhythmic_first_count}</span>
      <span class="summary-pill">fallback: {fallback_count}</span>
    </div>
    <div class="layout">
      <aside class="sidebar">
        <div class="sidebar-tools">
          <input id="searchBox" class="search-box" type="search" placeholder="Search sequence, song, or style">
          <div id="priorityFilters" class="priority-filters">
            <button class="priority-filter active" data-priority-filter="all" type="button">All</button>
            <button class="priority-filter" data-priority-filter="rhythmic_first" type="button">Rhythmic-First</button>
            <button class="priority-filter" data-priority-filter="fallback" type="button">Fallback</button>
          </div>
        </div>
        <div id="sequenceList" class="sequence-list">{initial_sequence_list}</div>
      </aside>
      <section class="viewer">
        <div class="viewer-header">
          <div>
            <h2 id="viewerTitle" class="viewer-title">FineDance</h2>
            <p id="viewerSubtitle" class="viewer-subtitle"></p>
            <div id="metaBadges" class="meta-badges"></div>
          </div>
        </div>
        <div class="quick-nav">
          <button id="prevButton" class="nav-button" type="button">Previous</button>
          <button id="nextButton" class="nav-button" type="button">Next</button>
          <span id="pickerStatus" class="picker-status">0 / 0</span>
        </div>
        <video id="truthVideo" class="media" playsinline preload="none"></video>
        <section class="analysis-panel">
          <div class="analysis-header">
            <div>
              <h3>Audio Analysis Panel</h3>
              <p>Waveform, Mel spectrogram, onset / flux, low-band envelope, and event rail stay locked to the current proof clip window so we can inspect whether beats and section changes really match the audio.</p>
            </div>
          </div>
          <canvas id="audioAnalysisCanvas" class="analysis-canvas" width="1400" height="520"></canvas>
        <div class="analysis-legend">
            <span><i style="background: rgba(232, 203, 145, 0.9);"></i>waveform</span>
            <span><i style="background: linear-gradient(90deg, #241610, #7c3d21, #d67a3d, #f4d67b);"></i>Mel spectrogram</span>
            <span><i style="background: rgba(255, 255, 255, 0.92);"></i>onset / flux</span>
            <span><i style="background: rgba(104, 175, 255, 0.95);"></i>low-band envelope</span>
            <span><i style="background: rgba(82, 167, 255, 0.7);"></i>beats (confidence)</span>
            <span><i style="background: rgba(240, 116, 79, 0.95);"></i>motion accents (L1-L4)</span>
            <span><i style="background: rgba(244, 214, 113, 0.95);"></i>low accents</span>
            <span><i style="background: rgba(141, 216, 162, 0.95);"></i>mid accents</span>
            <span><i style="background: rgba(171, 217, 255, 0.95);"></i>high accents</span>
            <span><i style="background: rgba(122, 222, 184, 0.32);"></i>sections</span>
            <span><i style="background: rgba(255, 75, 75, 0.96);"></i>playhead</span>
          </div>
        </section>
        <div class="controls">
          <button id="playButton">Play</button>
          <button id="pauseButton">Pause</button>
          <button id="resetButton">Reset</button>
        </div>
        <input id="timeline" class="timeline" type="range" min="0" max="0" value="0" step="0.01">
        <canvas id="musicTimelineCanvas" class="music-timeline" width="1400" height="140"></canvas>
        <div class="timeline-legend">
          <span><i style="background: rgba(122, 200, 255, 0.9);"></i>beats (alpha = confidence)</span>
          <span><i style="background: rgba(240, 116, 79, 0.95);"></i>motion accents (deeper = stronger)</span>
          <span><i style="background: rgba(244, 214, 113, 0.95);"></i>low accents</span>
          <span><i style="background: rgba(255, 255, 255, 0.95);"></i>playhead</span>
        </div>
        <div class="metrics">
          <div class="metric"><span class="label">Proof Clip</span><span id="metricClip" class="value">-</span></div>
          <div class="metric"><span class="label">Preview Frames</span><span id="metricPreviewFrames" class="value">-</span></div>
          <div class="metric"><span class="label">Motion</span><span id="metricMotion" class="value">-</span></div>
          <div class="metric"><span class="label">Motion Frames</span><span id="metricMotionFrames" class="value">-</span></div>
          <div class="metric"><span class="label">BPM</span><span id="metricBpm" class="value">-</span></div>
          <div class="metric"><span class="label">Energy</span><span id="metricEnergy" class="value">-</span></div>
          <div class="metric"><span class="label">Audio</span><span id="metricAudio" class="value">-</span></div>
          <div class="metric"><span class="label">Delta</span><span id="metricDelta" class="value">-</span></div>
          <div class="metric"><span class="label">MP4 Size</span><span id="metricVideoSize" class="value">-</span></div>
        </div>
        <p class="note">The viewer uses the official upstream mesh preview clip as the visual proof lane. The raw dataset duration and raw motion frame count shown below are the original totals. The preview clip is only a short front slice for inspection.</p>
        <div class="detail-grid">
          <div class="detail-card">
            <h3>Resolved Assets</h3>
            <ul id="assetLinks"></ul>
          </div>
          <div class="detail-card">
            <h3>Metadata</h3>
            <pre id="metaJson"></pre>
          </div>
        </div>
      </section>
    </div>
  </main>
  <script id="datasetPayload" type="application/json">{payload_script_json}</script>
  <script>
    const payload = JSON.parse(document.getElementById('datasetPayload').textContent);
    const entries = payload.entries;
    const listRoot = document.getElementById('sequenceList');
    const searchBox = document.getElementById('searchBox');
    const priorityFilters = document.getElementById('priorityFilters');
    const viewerTitle = document.getElementById('viewerTitle');
    const viewerSubtitle = document.getElementById('viewerSubtitle');
    const metaBadges = document.getElementById('metaBadges');
    const prevButton = document.getElementById('prevButton');
    const nextButton = document.getElementById('nextButton');
    const pickerStatus = document.getElementById('pickerStatus');
    const truthVideo = document.getElementById('truthVideo');
    const playButton = document.getElementById('playButton');
    const pauseButton = document.getElementById('pauseButton');
    const resetButton = document.getElementById('resetButton');
    const timeline = document.getElementById('timeline');
    const musicTimelineCanvas = document.getElementById('musicTimelineCanvas');
    const musicTimelineContext = musicTimelineCanvas.getContext('2d');
    const audioAnalysisCanvas = document.getElementById('audioAnalysisCanvas');
    const audioAnalysisContext = audioAnalysisCanvas.getContext('2d');
    const metricClip = document.getElementById('metricClip');
    const metricPreviewFrames = document.getElementById('metricPreviewFrames');
    const metricMotion = document.getElementById('metricMotion');
    const metricMotionFrames = document.getElementById('metricMotionFrames');
    const metricBpm = document.getElementById('metricBpm');
    const metricEnergy = document.getElementById('metricEnergy');
    const metricAudio = document.getElementById('metricAudio');
    const metricDelta = document.getElementById('metricDelta');
    const metricVideoSize = document.getElementById('metricVideoSize');
    const assetLinks = document.getElementById('assetLinks');
    const metaJson = document.getElementById('metaJson');

    let currentEntry = null;
    let rafId = 0;
    let visibleEntries = entries.slice();
    let activePriorityFilter = 'all';
    const audioPanelCache = new Map();

    const AUDIO_PANEL_LAYOUT = {{
      marginLeft: 120,
      marginRight: 18,
      marginTop: 14,
      marginBottom: 32,
      rowGap: 10,
      rows: [
        {{ key: 'waveform', label: 'Waveform', height: 66 }},
        {{ key: 'mel', label: 'Mel Spectrogram', height: 146 }},
        {{ key: 'onset', label: 'Onset / Flux', height: 66 }},
        {{ key: 'lowBand', label: 'Low-band Envelope', height: 66 }},
        {{ key: 'events', label: 'Event Rail', height: 88 }},
      ],
    }};

    const formatSeconds = (value) => `${{Number(value || 0).toFixed(2)}}s`;
    const formatBytes = (value) => {{
      const kib = Number(value || 0) / 1024;
      if (kib < 1024) return `${{Math.round(kib)}} KB`;
      return `${{(kib / 1024).toFixed(2)}} MB`;
    }};
    const formatCompactDuration = (value) => {{
      const sec = Math.round(Number(value || 0));
      const min = Math.floor(sec / 60);
      const rem = sec % 60;
      return `${{min}}m${{String(rem).padStart(2, '0')}}s`;
    }};
    const clamp = (value, minValue, maxValue) => Math.max(minValue, Math.min(maxValue, value));
    const strongBeatColor = (level) => {{
      const alpha = clamp(0.35 + Number(level || 1) * 0.14, 0.35, 0.95);
      return `rgba(240, 116, 79, ${{alpha.toFixed(3)}})`;
    }};
    const beatAlpha = (confidence) => clamp(0.18 + Number(confidence || 0) * 0.72, 0.18, 0.95);
    const accentBandColor = (band, alpha = 0.92) => {{
      if (band === 'high_attack') return `rgba(171, 217, 255, ${{alpha}})`;
      if (band === 'low_mid') return `rgba(141, 216, 162, ${{alpha}})`;
      return `rgba(244, 214, 113, ${{alpha}})`;
    }};

    const stopTicker = () => {{
      if (rafId) {{
        window.cancelAnimationFrame(rafId);
        rafId = 0;
      }}
    }};

    const decodeBase64ToUint8 = (payloadString) => {{
      if (!payloadString) return new Uint8Array();
      const binary = window.atob(payloadString);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {{
        bytes[index] = binary.charCodeAt(index);
      }}
      return bytes;
    }};

    const decodeSignedByte = (value) => clamp((Number(value || 0) / 255) * 2 - 1, -1, 1);
    const decodeUnitByte = (value) => clamp(Number(value || 0) / 255, 0, 1);

    const melPalette = Array.from({{ length: 256 }}, (_value, index) => {{
      const t = index / 255;
      const blend = (a, b, mix) => a + (b - a) * mix;
      let r = 16;
      let g = 9;
      let b = 8;
      if (t < 0.32) {{
        const local = t / 0.32;
        r = blend(24, 124, local);
        g = blend(12, 61, local);
        b = blend(10, 33, local);
      }} else if (t < 0.7) {{
        const local = (t - 0.32) / 0.38;
        r = blend(124, 214, local);
        g = blend(61, 122, local);
        b = blend(33, 61, local);
      }} else {{
        const local = (t - 0.7) / 0.3;
        r = blend(214, 244, local);
        g = blend(122, 214, local);
        b = blend(61, 123, local);
      }}
      return `rgb(${{Math.round(r)}}, ${{Math.round(g)}}, ${{Math.round(b)}})`;
    }});

    const buildAudioPanelLayout = (width, height) => {{
      const chartLeft = AUDIO_PANEL_LAYOUT.marginLeft;
      const chartRight = width - AUDIO_PANEL_LAYOUT.marginRight;
      const chartWidth = Math.max(12, chartRight - chartLeft);
      let cursorY = AUDIO_PANEL_LAYOUT.marginTop;
      const rows = AUDIO_PANEL_LAYOUT.rows.map((row) => {{
        const top = cursorY;
        const bottom = top + row.height;
        cursorY = bottom + AUDIO_PANEL_LAYOUT.rowGap;
        return {{
          key: row.key,
          label: row.label,
          height: row.height,
          top,
          bottom,
          mid: (top + bottom) / 2,
        }};
      }});
      const axisY = height - 16;
      return {{
        chartLeft,
        chartRight,
        chartWidth,
        rows,
        axisY,
      }};
    }};

    const clearAudioAnalysisCanvas = (message = 'Select a sequence to inspect the audio analysis panel.') => {{
      if (!audioAnalysisContext) return;
      audioAnalysisContext.clearRect(0, 0, audioAnalysisCanvas.width, audioAnalysisCanvas.height);
      audioAnalysisContext.fillStyle = 'rgba(12, 18, 28, 0.98)';
      audioAnalysisContext.fillRect(0, 0, audioAnalysisCanvas.width, audioAnalysisCanvas.height);
      audioAnalysisContext.fillStyle = 'rgba(248, 241, 229, 0.78)';
      audioAnalysisContext.font = '16px Avenir Next';
      audioAnalysisContext.fillText(message, 24, 42);
    }};

    const renderMusicTimeline = (entry, currentTimeSec) => {{
      if (!entry || !musicTimelineContext) return;
      const ctx = musicTimelineContext;
      const width = musicTimelineCanvas.width;
      const height = musicTimelineCanvas.height;
      const duration = Math.max(Number(entry.clip_duration_seconds || 0), 0.001);
      const currentTime = clamp(Number(currentTimeSec || 0), 0, duration);
      const features = entry.audio_features || {{}};
      const energyCurve = Array.isArray(features.energy_curve) ? features.energy_curve : [];
      const beats = Array.isArray(features.beats) ? features.beats : [];
      const strongBeats = Array.isArray(features.strong_beats) ? features.strong_beats : [];
      const accentCandidates = Array.isArray(features.accent_candidates) ? features.accent_candidates : [];
      const drumHits = accentCandidates.filter((item) => item.band === 'low');

      const marginLeft = 16;
      const marginRight = 12;
      const chartWidth = Math.max(4, width - marginLeft - marginRight);
      const xForTime = (timeSec) => marginLeft + (clamp(Number(timeSec || 0), 0, duration) / duration) * chartWidth;

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'rgba(255,255,255,0.02)';
      ctx.fillRect(0, 0, width, height);

      const energyTop = 10;
      const energyBottom = 68;
      ctx.strokeStyle = 'rgba(126, 200, 255, 0.92)';
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      if (energyCurve.length) {{
        for (let i = 0; i < energyCurve.length; i += 1) {{
          const point = energyCurve[i];
          const x = xForTime(point.time_sec);
          const y = energyBottom - clamp(Number(point.value || 0), 0, 1) * (energyBottom - energyTop);
          if (i === 0) {{
            ctx.moveTo(x, y);
          }} else {{
            ctx.lineTo(x, y);
          }}
        }}
      }} else {{
        ctx.moveTo(marginLeft, energyBottom);
        ctx.lineTo(marginLeft + chartWidth, energyBottom);
      }}
      ctx.stroke();

      for (const beat of beats) {{
        const x = xForTime(beat.time_sec);
        ctx.fillStyle = `rgba(122, 200, 255, ${{beatAlpha(beat.confidence).toFixed(3)}})`;
        ctx.beginPath();
        ctx.arc(x, 88, 1.9 + Number(beat.confidence || 0) * 1.6, 0, Math.PI * 2);
        ctx.fill();
      }}

      for (const beat of strongBeats) {{
        const x = xForTime(beat.time_sec);
        const level = Number(beat.level || 1);
        const alpha = clamp(0.25 + 0.45 * Number(beat.confidence || 0) + 0.08 * level, 0.28, 0.95);
        ctx.fillStyle = `rgba(240, 116, 79, ${{alpha.toFixed(3)}})`;
        ctx.beginPath();
        ctx.arc(x, 106, 2.2 + level * 0.9, 0, Math.PI * 2);
        ctx.fill();
      }}

      for (const drum of drumHits) {{
        const x = xForTime(drum.time_sec);
        ctx.strokeStyle = accentBandColor('low', clamp(0.35 + 0.5 * Number(drum.confidence || 0), 0.35, 0.98).toFixed(3));
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(x, 118);
        ctx.lineTo(x, 132);
        ctx.stroke();
      }}

      const axisY = 136;
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(marginLeft, axisY);
      ctx.lineTo(marginLeft + chartWidth, axisY);
      ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.78)';
      ctx.font = '11px Avenir Next';
      for (let index = 0; index <= 4; index += 1) {{
        const ratio = index / 4;
        const timeSec = duration * ratio;
        const x = marginLeft + chartWidth * ratio;
        ctx.strokeStyle = 'rgba(255,255,255,0.24)';
        ctx.beginPath();
        ctx.moveTo(x, axisY - 5);
        ctx.lineTo(x, axisY + 1);
        ctx.stroke();
        ctx.fillText(`${{timeSec.toFixed(1)}}s`, x - 14, axisY - 8);
      }}

      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      ctx.font = '12px Avenir Next';
      ctx.fillText(`t = ${{currentTime.toFixed(2)}}s / ${{duration.toFixed(2)}}s`, width - 210, 18);

      ctx.strokeStyle = 'rgba(255,255,255,0.95)';
      ctx.lineWidth = 2;
      const playheadX = xForTime(currentTime);
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
    }};

    const drawPanelLine = (ctx, row, chartLeft, chartWidth, values, strokeStyle) => {{
      const safeValues = values instanceof Uint8Array ? values : new Uint8Array();
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = 1.9;
      ctx.beginPath();
      if (!safeValues.length) {{
        ctx.moveTo(chartLeft, row.bottom - 2);
        ctx.lineTo(chartLeft + chartWidth, row.bottom - 2);
      }} else {{
        for (let index = 0; index < safeValues.length; index += 1) {{
          const ratio = safeValues.length <= 1 ? 0 : index / (safeValues.length - 1);
          const x = chartLeft + ratio * chartWidth;
          const y = row.bottom - decodeUnitByte(safeValues[index]) * (row.height - 8) - 4;
          if (index === 0) {{
            ctx.moveTo(x, y);
          }} else {{
            ctx.lineTo(x, y);
          }}
        }}
      }}
      ctx.stroke();
    }};

    const drawPanelAxis = (ctx, duration, width, layout) => {{
      ctx.strokeStyle = 'rgba(255,255,255,0.28)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(layout.chartLeft, layout.axisY);
      ctx.lineTo(layout.chartRight, layout.axisY);
      ctx.stroke();
      ctx.fillStyle = 'rgba(248, 241, 229, 0.76)';
      ctx.font = '11px Avenir Next';
      for (let index = 0; index <= 6; index += 1) {{
        const ratio = index / 6;
        const x = layout.chartLeft + layout.chartWidth * ratio;
        ctx.strokeStyle = 'rgba(255,255,255,0.16)';
        ctx.beginPath();
        ctx.moveTo(x, AUDIO_PANEL_LAYOUT.marginTop);
        ctx.lineTo(x, layout.axisY);
        ctx.stroke();
        ctx.fillText(`${{(duration * ratio).toFixed(1)}}s`, x - 14, layout.axisY - 8);
      }}
      ctx.fillStyle = 'rgba(248, 241, 229, 0.82)';
      ctx.font = '12px Avenir Next';
      ctx.fillText(`clip window: 0.00s -> ${{duration.toFixed(2)}}s`, layout.chartLeft, 12);
      ctx.fillText(`duration: ${{duration.toFixed(2)}}s`, width - 150, 12);
    }};

    const renderAudioAnalysisStatic = (entry, decoded) => {{
      const canvas = decoded.staticCanvas;
      const ctx = decoded.staticContext;
      const width = canvas.width;
      const height = canvas.height;
      const duration = Math.max(Number(entry.clip_duration_seconds || 0), 0.001);
      const layout = buildAudioPanelLayout(width, height);
      decoded.layout = layout;
      decoded.duration = duration;
      const xForTime = (timeSec) => layout.chartLeft + (clamp(Number(timeSec || 0), 0, duration) / duration) * layout.chartWidth;
      const accentYForBand = (band) => {{
        if (band === 'high_attack') return eventsRow.top + 18;
        if (band === 'low_mid') return eventsRow.top + 40;
        return eventsRow.bottom - 16;
      }};

      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = 'rgba(12, 18, 28, 0.98)';
      ctx.fillRect(0, 0, width, height);
      drawPanelAxis(ctx, duration, width, layout);

      for (const row of layout.rows) {{
        ctx.fillStyle = 'rgba(255,255,255,0.03)';
        ctx.fillRect(layout.chartLeft, row.top, layout.chartWidth, row.height);
        ctx.fillStyle = 'rgba(248, 241, 229, 0.84)';
        ctx.font = '12px Avenir Next';
        ctx.fillText(row.label, 14, row.top + 18);
      }}

      const waveformRow = layout.rows.find((row) => row.key === 'waveform');
      if (waveformRow) {{
        const mins = decoded.waveform.mins;
        const maxs = decoded.waveform.maxs;
        ctx.strokeStyle = 'rgba(232, 203, 145, 0.88)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(layout.chartLeft, waveformRow.mid);
        ctx.lineTo(layout.chartRight, waveformRow.mid);
        ctx.stroke();
        for (let index = 0; index < Math.min(mins.length, maxs.length); index += 1) {{
          const ratio = mins.length <= 1 ? 0 : index / (mins.length - 1);
          const x = layout.chartLeft + ratio * layout.chartWidth;
          const minAmp = decodeSignedByte(mins[index]);
          const maxAmp = decodeSignedByte(maxs[index]);
          const minY = waveformRow.mid - minAmp * (waveformRow.height * 0.42);
          const maxY = waveformRow.mid - maxAmp * (waveformRow.height * 0.42);
          ctx.strokeStyle = 'rgba(232, 203, 145, 0.88)';
          ctx.beginPath();
          ctx.moveTo(x, maxY);
          ctx.lineTo(x, minY);
          ctx.stroke();
        }}
      }}

      const melRow = layout.rows.find((row) => row.key === 'mel');
      if (melRow && decoded.mel.values.length) {{
        const melBins = Math.max(decoded.mel.melBinCount, 1);
        const timeBins = Math.max(decoded.mel.timeBinCount, 1);
        const cellWidth = layout.chartWidth / timeBins;
        const cellHeight = melRow.height / melBins;
        for (let timeIndex = 0; timeIndex < timeBins; timeIndex += 1) {{
          for (let melIndex = 0; melIndex < melBins; melIndex += 1) {{
            const value = decoded.mel.values[melIndex * timeBins + timeIndex];
            const x = layout.chartLeft + timeIndex * cellWidth;
            const y = melRow.bottom - (melIndex + 1) * cellHeight;
            ctx.fillStyle = melPalette[value];
            ctx.fillRect(x, y, Math.ceil(cellWidth + 0.6), Math.ceil(cellHeight + 0.6));
          }}
        }}
      }}

      const onsetRow = layout.rows.find((row) => row.key === 'onset');
      if (onsetRow) {{
        drawPanelLine(ctx, onsetRow, layout.chartLeft, layout.chartWidth, decoded.onset.values, 'rgba(255, 255, 255, 0.92)');
      }}

      const lowBandRow = layout.rows.find((row) => row.key === 'lowBand');
      if (lowBandRow) {{
        drawPanelLine(ctx, lowBandRow, layout.chartLeft, layout.chartWidth, decoded.lowBand.values, 'rgba(104, 175, 255, 0.95)');
      }}

      const eventsRow = layout.rows.find((row) => row.key === 'events');
      if (eventsRow) {{
        const accentCandidates = Array.isArray(entry.audio_features?.accent_candidates) ? entry.audio_features.accent_candidates : [];
        for (const section of decoded.sections) {{
          const startX = xForTime(section.start_time_sec);
          const endX = xForTime(section.end_time_sec);
          const widthPx = Math.max(2, endX - startX);
          ctx.fillStyle = 'rgba(122, 222, 184, 0.18)';
          ctx.fillRect(startX, eventsRow.top, widthPx, eventsRow.height);
          ctx.strokeStyle = 'rgba(122, 222, 184, 0.42)';
          ctx.lineWidth = 1;
          ctx.strokeRect(startX, eventsRow.top, widthPx, eventsRow.height);
          ctx.fillStyle = 'rgba(122, 222, 184, 0.88)';
          ctx.font = '11px Avenir Next';
          ctx.fillText(String(section.label || 'section'), startX + 4, eventsRow.top + 16);
        }}

        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(layout.chartLeft, eventsRow.bottom - 16);
        ctx.lineTo(layout.chartRight, eventsRow.bottom - 16);
        ctx.stroke();

        for (const beat of (entry.audio_features?.beats || [])) {{
          const x = xForTime(beat.time_sec);
          ctx.strokeStyle = `rgba(82, 167, 255, ${{beatAlpha(beat.confidence).toFixed(3)}})`;
          ctx.lineWidth = 0.9 + Number(beat.confidence || 0) * 1.0;
          ctx.beginPath();
          ctx.moveTo(x, eventsRow.top + 8);
          ctx.lineTo(x, eventsRow.bottom - 20);
          ctx.stroke();
        }}

        for (const beat of (entry.audio_features?.strong_beats || [])) {{
          const x = xForTime(beat.time_sec);
          const level = Number(beat.level || 1);
          const alpha = clamp(0.24 + 0.44 * Number(beat.confidence || 0) + 0.08 * level, 0.28, 0.96);
          ctx.strokeStyle = `rgba(240, 116, 79, ${{alpha.toFixed(3)}})`;
          ctx.lineWidth = 1.5 + level * 0.55;
          ctx.beginPath();
          ctx.moveTo(x, eventsRow.top + 4);
          ctx.lineTo(x, eventsRow.bottom - 24);
          ctx.stroke();
        }}

        for (const accent of accentCandidates) {{
          const x = xForTime(accent.time_sec);
          const y = accentYForBand(accent.band);
          const alpha = clamp(0.28 + 0.5 * Number(accent.confidence || 0), 0.3, 0.96);
          ctx.fillStyle = accentBandColor(accent.band, alpha.toFixed(3));
          ctx.beginPath();
          ctx.arc(x, y, 1.8 + Number(accent.level || 1) * 0.65, 0, Math.PI * 2);
          ctx.fill();
        }}
      }}
    }};

    const ensureDecodedAudioPanel = (entry) => {{
      if (!entry) return null;
      if (audioPanelCache.has(entry.sequence_id)) {{
        return audioPanelCache.get(entry.sequence_id);
      }}
      const rawPanel = entry.audio_analysis_panel || {{}};
      const decoded = {{
        waveform: {{
          binCount: Number(rawPanel.waveform?.bin_count || 0),
          mins: decodeBase64ToUint8(rawPanel.waveform?.min_b64 || ''),
          maxs: decodeBase64ToUint8(rawPanel.waveform?.max_b64 || ''),
        }},
        mel: {{
          melBinCount: Number(rawPanel.mel_spectrogram?.mel_bin_count || 0),
          timeBinCount: Number(rawPanel.mel_spectrogram?.time_bin_count || 0),
          values: decodeBase64ToUint8(rawPanel.mel_spectrogram?.values_b64 || ''),
        }},
        onset: {{
          pointCount: Number(rawPanel.onset_curve?.point_count || 0),
          values: decodeBase64ToUint8(rawPanel.onset_curve?.values_b64 || ''),
        }},
        lowBand: {{
          pointCount: Number(rawPanel.low_band_curve?.point_count || 0),
          values: decodeBase64ToUint8(rawPanel.low_band_curve?.values_b64 || ''),
        }},
        sections: Array.isArray(rawPanel.sections) ? rawPanel.sections : [],
      }};
      decoded.staticCanvas = document.createElement('canvas');
      decoded.staticCanvas.width = audioAnalysisCanvas.width;
      decoded.staticCanvas.height = audioAnalysisCanvas.height;
      decoded.staticContext = decoded.staticCanvas.getContext('2d');
      renderAudioAnalysisStatic(entry, decoded);
      audioPanelCache.set(entry.sequence_id, decoded);
      return decoded;
    }};

    const renderAudioAnalysis = (entry, currentTimeSec) => {{
      if (!entry || !audioAnalysisContext) return;
      const decoded = ensureDecodedAudioPanel(entry);
      if (!decoded) {{
        clearAudioAnalysisCanvas();
        return;
      }}
      audioAnalysisContext.clearRect(0, 0, audioAnalysisCanvas.width, audioAnalysisCanvas.height);
      audioAnalysisContext.drawImage(decoded.staticCanvas, 0, 0);
      const duration = Math.max(decoded.duration || Number(entry.clip_duration_seconds || 0), 0.001);
      const currentTime = clamp(Number(currentTimeSec || 0), 0, duration);
      const x = decoded.layout.chartLeft + (currentTime / duration) * decoded.layout.chartWidth;
      audioAnalysisContext.strokeStyle = 'rgba(255, 75, 75, 0.96)';
      audioAnalysisContext.lineWidth = 2;
      audioAnalysisContext.beginPath();
      audioAnalysisContext.moveTo(x, AUDIO_PANEL_LAYOUT.marginTop);
      audioAnalysisContext.lineTo(x, decoded.layout.axisY);
      audioAnalysisContext.stroke();
      audioAnalysisContext.fillStyle = 'rgba(255, 75, 75, 0.96)';
      audioAnalysisContext.font = '12px Avenir Next';
      audioAnalysisContext.fillText(`t = ${{currentTime.toFixed(2)}}s / ${{duration.toFixed(2)}}s`, audioAnalysisCanvas.width - 190, decoded.layout.axisY + 18);
    }};

    const tick = () => {{
      if (!currentEntry) return;
      const clipDuration = Number(currentEntry.clip_duration_seconds || 0);
      if (truthVideo.currentTime >= clipDuration && clipDuration > 0) {{
        if (currentEntry.preview_available) {{
          truthVideo.pause();
          truthVideo.currentTime = clipDuration;
        }}
        timeline.value = String(clipDuration);
        renderMusicTimeline(currentEntry, clipDuration);
        renderAudioAnalysis(currentEntry, clipDuration);
        stopTicker();
        return;
      }}
      const nextTime = Math.min(truthVideo.currentTime, clipDuration);
      timeline.value = String(nextTime);
      renderMusicTimeline(currentEntry, nextTime);
      renderAudioAnalysis(currentEntry, nextTime);
      if (!truthVideo.paused) {{
        rafId = window.requestAnimationFrame(tick);
      }}
    }};

    const setBadges = (entry) => {{
      const audioFeatures = entry.audio_features || {{}};
      const analysisPriority = entry.analysis_priority || {{}};
      const badges = [
        `song: ${{entry.song_name || 'unknown'}}`,
        `style: ${{entry.coarse_style || 'unknown'}} / ${{entry.fine_style || 'unknown'}}`,
        `priority: ${{analysisPriority.tier === 'rhythmic_first' ? 'rhythmic-first' : 'fallback'}}`,
        `beat profile: ${{audioFeatures.beat_tracking?.profile || 'fallback'}}`,
        `fps: ${{entry.fps}}`,
        `frames: ${{entry.motion_frame_count}}`,
        `bpm: ${{Number(audioFeatures.bpm || 0).toFixed(2)}}`,
        `energy: ${{audioFeatures.energy_label || 'mid_energy'}}`,
        `preview: ${{entry.preview_available ? 'available' : 'missing'}}`,
      ];
      metaBadges.innerHTML = badges.map((label) => `<span class="meta-badge">${{label}}</span>`).join('');
    }};

    const setAssets = (entry) => {{
      const items = [
        ['motion npy', entry.motion_href],
        ['music wav', entry.music_href],
      ];
      if (entry.preview_video_href) {{
        items.push(['official mesh preview mp4', entry.preview_video_href]);
      }}
      if (entry.label_href) {{
        items.push(['label json', entry.label_href]);
      }}
      assetLinks.innerHTML = items
        .map(([label, href]) => `<li><a href="${{href}}">${{label}}</a></li>`)
        .join('');
    }};

    const setMetadata = (entry) => {{
      metaJson.textContent = JSON.stringify({{
        dataset: entry.dataset_name,
        sequence_id: entry.sequence_id,
        song_name: entry.song_name,
        coarse_style: entry.coarse_style,
        fine_style: entry.fine_style,
        style_tags: entry.style_tags,
        analysis_priority: entry.analysis_priority,
        motion_path: entry.motion_path,
        music_path: entry.music_path,
        preview_video_path: entry.preview_video_path,
        preview_available: entry.preview_available,
        motion_frame_count: entry.motion_frame_count,
        preview_frame_count: entry.preview_frame_count,
        motion_duration_seconds: Number(entry.motion_duration_seconds).toFixed(3),
        audio_duration_seconds: Number(entry.audio_duration_seconds).toFixed(3),
        preview_duration_seconds: Number(entry.preview_duration_seconds).toFixed(3),
        clip_start_seconds: Number(entry.clip_start_seconds || 0).toFixed(3),
        clip_duration_seconds: Number(entry.clip_duration_seconds || 0).toFixed(3),
        duration_delta_seconds: Number(entry.duration_delta_seconds).toFixed(3),
        preview_video_size_bytes: entry.preview_video_size_bytes,
        audio_features: {{
          bpm: entry.audio_features?.bpm,
          beats_per_bar: entry.audio_features?.beats_per_bar,
          beat_tracking: entry.audio_features?.beat_tracking,
          energy_score: entry.audio_features?.energy_score,
          energy_label: entry.audio_features?.energy_label,
          energy_curve_points: Array.isArray(entry.audio_features?.energy_curve) ? entry.audio_features.energy_curve.length : 0,
          beats_count: Array.isArray(entry.audio_features?.beats) ? entry.audio_features.beats.length : 0,
          local_tempo_segment_count: Array.isArray(entry.audio_features?.beat_tracking?.local_tempo_segments) ? entry.audio_features.beat_tracking.local_tempo_segments.length : 0,
          strong_beats_count: Array.isArray(entry.audio_features?.strong_beats) ? entry.audio_features.strong_beats.length : 0,
          accent_candidates_count: Array.isArray(entry.audio_features?.accent_candidates) ? entry.audio_features.accent_candidates.length : 0,
          low_accent_count: Array.isArray(entry.audio_features?.accent_candidates) ? entry.audio_features.accent_candidates.filter((item) => item.band === 'low').length : 0,
          low_mid_accent_count: Array.isArray(entry.audio_features?.accent_candidates) ? entry.audio_features.accent_candidates.filter((item) => item.band === 'low_mid').length : 0,
          high_accent_count: Array.isArray(entry.audio_features?.accent_candidates) ? entry.audio_features.accent_candidates.filter((item) => item.band === 'high_attack').length : 0,
          drum_hit_count: Array.isArray(entry.audio_features?.drum_hits) ? entry.audio_features.drum_hits.length : 0,
        }},
        audio_analysis_panel: {{
          clip_start_sec: entry.audio_analysis_panel?.clip_start_sec,
          clip_end_sec: entry.audio_analysis_panel?.clip_end_sec,
          waveform_bins: entry.audio_analysis_panel?.waveform?.bin_count,
          mel_bin_count: entry.audio_analysis_panel?.mel_spectrogram?.mel_bin_count,
          mel_time_bins: entry.audio_analysis_panel?.mel_spectrogram?.time_bin_count,
          onset_point_count: entry.audio_analysis_panel?.onset_curve?.point_count,
          low_band_point_count: entry.audio_analysis_panel?.low_band_curve?.point_count,
          section_count: Array.isArray(entry.audio_analysis_panel?.sections) ? entry.audio_analysis_panel.sections.length : 0,
        }},
      }}, null, 2);
    }};

    const loadEntry = (entry) => {{
      stopTicker();
      currentEntry = entry;
      if (currentEntry.preview_available) {{
        truthVideo.pause();
      }}
      if (entry.preview_video_href) {{
        truthVideo.src = entry.preview_video_href;
      }} else {{
        truthVideo.removeAttribute('src');
      }}
      truthVideo.load();
      try {{
        truthVideo.currentTime = 0;
      }} catch (_error) {{}}
      timeline.min = '0';
      timeline.max = String(entry.clip_duration_seconds || 0);
      timeline.value = '0';
      viewerTitle.textContent = `FineDance ${{entry.sequence_id}}`;
      viewerSubtitle.textContent = entry.song_name || 'unknown song';
      metricClip.textContent = formatSeconds(entry.clip_duration_seconds);
      metricPreviewFrames.textContent = String(entry.preview_frame_count || 0);
      metricMotion.textContent = formatSeconds(entry.motion_duration_seconds);
      metricMotionFrames.textContent = String(entry.motion_frame_count || 0);
      metricBpm.textContent = Number(entry.audio_features?.bpm || 0).toFixed(2);
      metricEnergy.textContent = String(entry.audio_features?.energy_label || 'mid_energy');
      metricAudio.textContent = formatSeconds(entry.audio_duration_seconds);
      metricDelta.textContent = formatSeconds(entry.duration_delta_seconds);
      metricVideoSize.textContent = entry.preview_available ? formatBytes(entry.preview_video_size_bytes) : 'no preview';
      setBadges(entry);
      setAssets(entry);
      setMetadata(entry);
      renderMusicTimeline(entry, 0.0);
      renderAudioAnalysis(entry, 0.0);
      const currentIndex = visibleEntries.findIndex((candidate) => candidate.sequence_id === entry.sequence_id);
      pickerStatus.textContent = currentIndex >= 0 ? `${{currentIndex + 1}} / ${{visibleEntries.length}}` : `1 / ${{visibleEntries.length}}`;
      prevButton.disabled = currentIndex <= 0;
      nextButton.disabled = currentIndex < 0 || currentIndex >= visibleEntries.length - 1;
      Array.from(document.querySelectorAll('.sequence-card')).forEach((node) => {{
        node.classList.toggle('active', node.dataset.sequenceId === entry.sequence_id);
        if (node.dataset.sequenceId === entry.sequence_id) {{
          try {{
            node.scrollIntoView({{ block: 'nearest' }});
          }} catch (_error) {{
            node.scrollIntoView();
          }}
        }}
      }});
    }};

    const loadByOffset = (delta) => {{
      if (!currentEntry || !visibleEntries.length) return;
      const currentIndex = visibleEntries.findIndex((entry) => entry.sequence_id === currentEntry.sequence_id);
      if (currentIndex < 0) return;
      const nextIndex = currentIndex + delta;
      if (nextIndex < 0 || nextIndex >= visibleEntries.length) return;
      loadEntry(visibleEntries[nextIndex]);
    }};

    const renderList = (query) => {{
      const q = query.trim().toLowerCase();
      const filtered = entries.filter((entry) => {{
        const tier = entry.analysis_priority?.tier || 'fallback';
        if (activePriorityFilter !== 'all' && tier !== activePriorityFilter) {{
          return false;
        }}
        const haystack = [
          entry.sequence_id,
          entry.song_name,
          entry.coarse_style,
          entry.fine_style,
          tier === 'rhythmic_first' ? 'rhythmic-first rhythmic first' : 'fallback',
        ].join(' ').toLowerCase();
        return !q || haystack.includes(q);
      }});
      visibleEntries = filtered;
      listRoot.innerHTML = filtered.map((entry) => `
        <button class="sequence-card" data-sequence-id="${{entry.sequence_id}}">
          <span class="id">${{entry.sequence_id}}</span>
          <span class="style">${{(entry.coarse_style || 'unknown')}} / ${{(entry.fine_style || 'unknown')}}</span>
          <span class="priority ${{entry.analysis_priority?.tier === 'rhythmic_first' ? 'priority-rhythmic' : 'priority-fallback'}}">${{entry.analysis_priority?.tier === 'rhythmic_first' ? 'rhythmic-first' : 'fallback'}}</span>
          <span class="energy">${{entry.audio_features?.energy_label || 'mid_energy'}}</span>
        </button>
      `).join('');
      if (filtered.length && (!currentEntry || !filtered.some((entry) => entry.sequence_id === currentEntry.sequence_id))) {{
        loadEntry(filtered[0]);
      }} else if (!filtered.length) {{
        pickerStatus.textContent = '0 / 0';
        prevButton.disabled = true;
        nextButton.disabled = true;
        if (musicTimelineContext) {{
          musicTimelineContext.clearRect(0, 0, musicTimelineCanvas.width, musicTimelineCanvas.height);
        }}
        clearAudioAnalysisCanvas('No sequence matches the current filter.');
      }}
    }};

    searchBox.addEventListener('input', () => renderList(searchBox.value));
    priorityFilters.addEventListener('click', (event) => {{
      const target = event.target instanceof Element ? event.target.closest('.priority-filter') : null;
      if (!target) return;
      activePriorityFilter = target.getAttribute('data-priority-filter') || 'all';
      Array.from(priorityFilters.querySelectorAll('.priority-filter')).forEach((node) => {{
        node.classList.toggle('active', node === target);
      }});
      renderList(searchBox.value);
    }});
    listRoot.addEventListener('click', (event) => {{
      const target = event.target instanceof Element ? event.target.closest('.sequence-card') : null;
      if (!target) return;
      const sequenceId = target.getAttribute('data-sequence-id') || '';
      const match = visibleEntries.find((entry) => entry.sequence_id === sequenceId);
      if (match) loadEntry(match);
    }});
    prevButton.addEventListener('click', () => loadByOffset(-1));
    nextButton.addEventListener('click', () => loadByOffset(1));

    playButton.addEventListener('click', async () => {{
      if (!currentEntry) return;
      stopTicker();
      try {{
        if (currentEntry.preview_available) {{
          await truthVideo.play();
        }}
      }} catch (error) {{
        console.warn('playback blocked', error);
      }}
      const now = truthVideo.currentTime || 0;
      renderMusicTimeline(currentEntry, now);
      renderAudioAnalysis(currentEntry, now);
      if (currentEntry.preview_available) {{
        rafId = window.requestAnimationFrame(tick);
      }}
    }});

    pauseButton.addEventListener('click', () => {{
      stopTicker();
      if (currentEntry && currentEntry.preview_available) {{
        truthVideo.pause();
      }}
      if (currentEntry) {{
        const now = truthVideo.currentTime || 0;
        renderMusicTimeline(currentEntry, now);
        renderAudioAnalysis(currentEntry, now);
      }}
    }});

    resetButton.addEventListener('click', () => {{
      stopTicker();
      if (currentEntry && currentEntry.preview_available) {{
        truthVideo.pause();
        truthVideo.currentTime = 0;
      }}
      timeline.value = '0';
      if (currentEntry) {{
        renderMusicTimeline(currentEntry, 0);
        renderAudioAnalysis(currentEntry, 0);
      }}
    }});

    timeline.addEventListener('input', () => {{
      if (!currentEntry) return;
      const clipDuration = Number(currentEntry.clip_duration_seconds || 0);
      const nextTime = Math.min(Number(timeline.value) || 0, clipDuration);
      if (currentEntry.preview_available) {{
        truthVideo.currentTime = nextTime;
      }}
      renderMusicTimeline(currentEntry, nextTime);
      renderAudioAnalysis(currentEntry, nextTime);
    }});

    clearAudioAnalysisCanvas();
    renderList('');
  </script>
</body>
</html>
"""
