from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any

from ..contracts import ChoreographyPlan
from ..utils import utc_now_iso


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


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _beat_time(song_event_map: dict[str, Any], beat_index: float) -> float:
    beats = list(song_event_map.get("beats", []))
    if not beats:
        return 0.0
    if beat_index <= 0:
        return _safe_float(beats[0].get("time_sec"))
    left_index = int(math.floor(beat_index))
    right_index = int(math.ceil(beat_index))
    if right_index >= len(beats):
        if len(beats) == 1:
            return _safe_float(beats[0].get("time_sec"))
        interval = _safe_float(beats[-1].get("time_sec")) - _safe_float(beats[-2].get("time_sec"))
        return _safe_float(beats[-1].get("time_sec")) + interval * float(beat_index - (len(beats) - 1))
    if left_index == right_index:
        return _safe_float(beats[left_index].get("time_sec"))
    mix = beat_index - left_index
    return _lerp(_safe_float(beats[left_index].get("time_sec")), _safe_float(beats[right_index].get("time_sec")), mix)


def _events_in_window(items: list[dict[str, Any]], start_time_sec: float, end_time_sec: float) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if _safe_float(item.get("time_sec"), -1.0) >= start_time_sec
        and _safe_float(item.get("time_sec"), 999999.0) <= end_time_sec
    ]


def _section_for_phrase(song_event_map: dict[str, Any], phrase: dict[str, Any]) -> dict[str, Any]:
    phrase_start = _safe_int(phrase.get("start_beat"))
    phrase_end = _safe_int(phrase.get("end_beat_exclusive"), phrase_start + 1)
    for section in song_event_map.get("sections", []):
        if phrase_start >= _safe_int(section.get("start_beat")) and phrase_end <= _safe_int(section.get("end_beat_exclusive"), phrase_end):
            return dict(section)
    return {"label": "full_song", "start_beat": phrase_start, "end_beat_exclusive": phrase_end}


def _fallback_phrases(song_event_map: dict[str, Any], phrase_beats: int = 8) -> list[dict[str, Any]]:
    beats = list(song_event_map.get("beats", []))
    beat_count = len(beats)
    if beat_count == 0:
        duration_sec = _safe_float(song_event_map.get("duration_sec"))
        return [{"index": 0, "label": "phrase", "start_beat": 0, "end_beat_exclusive": max(1, int(round(duration_sec * 2.0)))}]
    phrases: list[dict[str, Any]] = []
    for start_beat in range(0, beat_count - 1, phrase_beats):
        end_beat = min(beat_count - 1, start_beat + phrase_beats)
        if end_beat <= start_beat:
            continue
        phrases.append(
            {
                "index": len(phrases),
                "label": "phrase",
                "start_beat": start_beat,
                "end_beat_exclusive": end_beat,
            }
        )
    return phrases


def _target_energy(section_label: str, accent_density: float) -> str:
    if section_label in {"chorus", "instrumental"} or accent_density >= 0.65:
        return "high_energy"
    if section_label in {"intro", "outro"} and accent_density < 0.45:
        return "low_energy"
    return "mid_energy"


def _music_profile(song_event_map: dict[str, Any], phrase: dict[str, Any], section_label: str) -> dict[str, Any]:
    start_beat = _safe_int(phrase.get("start_beat"))
    end_beat = _safe_int(phrase.get("end_beat_exclusive"), start_beat + 1)
    target_beats = max(1.0, float(end_beat - start_beat))
    start_time = _safe_float(phrase.get("start_time_sec"), _beat_time(song_event_map, float(start_beat)))
    end_time = _safe_float(phrase.get("end_time_sec"), _beat_time(song_event_map, float(end_beat)))
    accents = _events_in_window(list(song_event_map.get("accents", [])), start_time, end_time)
    downbeats = _events_in_window(list(song_event_map.get("downbeats", [])), start_time, end_time)
    strengths = [_safe_float(item.get("strength")) for item in accents]
    accent_density = len(accents) / max(target_beats, 1.0)
    return {
        "start_time_sec": start_time,
        "end_time_sec": max(end_time, start_time + 1e-3),
        "target_beats": target_beats,
        "accent_count": len(accents),
        "downbeat_count": len(downbeats),
        "accent_density_per_beat": round(float(accent_density), 5),
        "mean_accent_strength": round(float(sum(strengths) / max(1, len(strengths))), 5),
        "max_accent_strength": round(float(max(strengths) if strengths else 0.0), 5),
        "target_energy": _target_energy(section_label, accent_density),
        "expected_accent_hits": [_safe_float(item.get("time_sec")) for item in accents[:12]],
    }


def _transition_score(previous_unit: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if previous_unit is None:
        return 0.0
    if candidate["unit_id"] in previous_unit.get("compatible_next_units", []):
        return 6.0
    previous_exit = dict(previous_unit.get("exit_anchor", {}))
    entry = dict(candidate.get("entry_anchor", {}))
    speed_delta = abs(_safe_float(previous_exit.get("planar_speed")) - _safe_float(entry.get("planar_speed")))
    yaw_delta = abs((_safe_float(previous_exit.get("root_yaw_deg")) - _safe_float(entry.get("root_yaw_deg")) + 180.0) % 360.0 - 180.0)
    score = max(0.0, 2.0 - speed_delta * 0.5) + max(0.0, 2.0 - yaw_delta / 45.0)
    if previous_unit.get("source_sequence") == candidate.get("source_sequence"):
        score += 1.25
    return round(float(score), 5)


def _candidate_score(
    candidate: dict[str, Any],
    previous_unit: dict[str, Any] | None,
    recent_units: Counter[str],
    music_profile: dict[str, Any],
    section_label: str,
    max_speed_adjustment: float,
) -> float:
    target_beats = float(music_profile["target_beats"])
    source_beats = max(1.0, _safe_float(candidate.get("duration_beats"), _safe_float(candidate.get("duration_beats_estimate"), target_beats)))
    speed_scale = source_beats / max(target_beats, 1.0)
    if speed_scale < 1.0 - max_speed_adjustment or speed_scale > 1.0 + max_speed_adjustment:
        return -9999.0

    score = 0.0
    if candidate.get("energy") == music_profile["target_energy"]:
        score += 5.0
    elif music_profile["target_energy"] == "mid_energy" or candidate.get("energy") == "mid_energy":
        score += 2.0
    score -= abs(source_beats - target_beats) * 0.8
    score += _transition_score(previous_unit, candidate)
    score += min(2.5, _safe_int(candidate.get("rhythm_profile", {}).get("accent_count")) * 0.25)
    if section_label in {"chorus", "instrumental"} and candidate.get("energy") == "high_energy":
        score += 1.0
    if section_label in {"intro", "outro"} and candidate.get("segment_source") == "downbeat_phrase":
        score += 0.5
    score -= recent_units[str(candidate["unit_id"])] * 2.2
    return round(float(score), 5)


def _top_candidates(
    units: list[dict[str, Any]],
    previous_unit: dict[str, Any] | None,
    recent_units: Counter[str],
    music_profile: dict[str, Any],
    section_label: str,
    max_speed_adjustment: float,
    limit: int,
) -> list[tuple[float, dict[str, Any]]]:
    scored = [
        (
            _candidate_score(
                candidate=unit,
                previous_unit=previous_unit,
                recent_units=recent_units,
                music_profile=music_profile,
                section_label=section_label,
                max_speed_adjustment=max_speed_adjustment,
            ),
            unit,
        )
        for unit in units
    ]
    viable = [(score, unit) for score, unit in scored if score > -1000.0]
    if not viable:
        viable = sorted(scored, key=lambda item: item[0], reverse=True)[:limit]
    return sorted(viable, key=lambda item: item[0], reverse=True)[:limit]


@dataclass
class _Beam:
    score: float
    steps: list[dict[str, Any]]
    previous_unit: dict[str, Any] | None
    recent_units: Counter[str]


def build_rhythmic_choreography_plan(
    song_event_map: dict[str, Any],
    motion_library: dict[str, Any],
    beam_width: int = 4,
    candidate_limit: int = 18,
    max_speed_adjustment: float = 0.10,
) -> ChoreographyPlan:
    song_id = str(song_event_map["song_id"])
    library_id = str(motion_library["library_id"])
    units = [dict(unit) for unit in motion_library.get("units", [])]
    if not units:
        raise ValueError("rhythmic motion library contains no units")

    phrases = list(song_event_map.get("phrases", [])) or _fallback_phrases(song_event_map)
    beams = [_Beam(score=0.0, steps=[], previous_unit=None, recent_units=Counter())]

    for phrase in phrases:
        section = _section_for_phrase(song_event_map, phrase)
        section_label = str(section.get("label", "full_song") or "full_song")
        profile = _music_profile(song_event_map, phrase, section_label)
        next_beams: list[_Beam] = []
        for beam in beams:
            candidates = _top_candidates(
                units=units,
                previous_unit=beam.previous_unit,
                recent_units=beam.recent_units,
                music_profile=profile,
                section_label=section_label,
                max_speed_adjustment=max_speed_adjustment,
                limit=candidate_limit,
            )
            for local_score, unit in candidates:
                source_beats = max(1.0, _safe_float(unit.get("duration_beats"), profile["target_beats"]))
                speed_scale = source_beats / max(float(profile["target_beats"]), 1.0)
                transition_score = _transition_score(beam.previous_unit, unit)
                step = {
                    "index": len(beam.steps),
                    "section_label": section_label,
                    "start_beat": _safe_int(phrase.get("start_beat")),
                    "target_beats": profile["target_beats"],
                    "target_time_sec": {
                        "start": round(float(profile["start_time_sec"]), 5),
                        "end": round(float(profile["end_time_sec"]), 5),
                    },
                    "selected_unit_id": unit["unit_id"],
                    "source_sequence": unit["source_sequence"],
                    "source_frame_range": dict(unit.get("frame_range", {})),
                    "source_beat_range": dict(unit.get("beat_range", {})),
                    "speed_scale": round(float(speed_scale), 5),
                    "transition_score": transition_score,
                    "switch_reason": {
                        "planner": "rhythmic_beam_search_v1",
                        "target_energy": profile["target_energy"],
                        "candidate_energy": unit.get("energy"),
                        "accent_count": profile["accent_count"],
                        "downbeat_count": profile["downbeat_count"],
                        "accent_density_per_beat": profile["accent_density_per_beat"],
                        "mean_accent_strength": profile["mean_accent_strength"],
                        "max_accent_strength": profile["max_accent_strength"],
                        "compatible_from_previous": bool(beam.previous_unit and unit["unit_id"] in beam.previous_unit.get("compatible_next_units", [])),
                        "same_source_sequence": bool(beam.previous_unit and beam.previous_unit.get("source_sequence") == unit.get("source_sequence")),
                    },
                    "expected_accent_hits": profile["expected_accent_hits"],
                    "reference_artifacts": dict(unit.get("reference_artifacts", {})),
                }
                recent_units = Counter(beam.recent_units)
                recent_units.update([str(unit["unit_id"])])
                next_beams.append(
                    _Beam(
                        score=beam.score + local_score,
                        steps=[*beam.steps, step],
                        previous_unit=unit,
                        recent_units=recent_units,
                    )
                )
        beams = sorted(next_beams, key=lambda item: item.score, reverse=True)[: max(1, beam_width)]

    if not beams:
        raise ValueError("no viable rhythmic choreography plan could be built")

    best = beams[0]
    return ChoreographyPlan(
        schema_version=2,
        plan_id=f"{song_id}_rhythmic_smplx_plan",
        song_id=song_id,
        library_id=library_id,
        steps=best.steps,
        notes=[
            "Rhythmic SMPL-X planner v1 uses target song beat/phrase windows and FineDance source-SMPL-X motion units.",
            f"beam_width={beam_width}; candidate_limit={candidate_limit}; max_speed_adjustment={max_speed_adjustment}.",
        ],
        generated_at_utc=utc_now_iso(),
    )
