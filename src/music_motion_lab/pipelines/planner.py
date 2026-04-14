from __future__ import annotations

from collections import Counter
from typing import Any

from ..contracts import ChoreographyPlan
from ..utils import utc_now_iso


def _base_section_energy(label: str) -> str:
    mapping = {
        "intro": "low_energy",
        "outro": "low_energy",
        "verse": "mid_energy",
        "bridge": "mid_energy",
        "chorus": "high_energy",
        "instrumental": "high_energy",
        "full_song": "mid_energy",
    }
    return mapping.get(label, "mid_energy")


def _target_travel(label: str, accent_density_per_beat: float) -> str:
    mapping = {
        "intro": "stationary",
        "outro": "stationary",
        "verse": "localized",
        "bridge": "localized",
        "chorus": "traveling",
        "instrumental": "traveling",
        "full_song": "localized",
    }
    if label in {"verse", "bridge"} and accent_density_per_beat >= 0.55:
        return "traveling"
    return mapping.get(label, "localized")


def _support_preference(label: str, target_energy: str) -> set[str]:
    if label in {"chorus", "instrumental"} or target_energy == "high_energy":
        return {"alternating_support", "airborne_bias"}
    return {"both_feet_planted", "left_support", "right_support", "alternating_support"}


def _target_anchor_label(label: str, target_energy: str, accent_density_per_beat: float) -> str:
    if label in {"intro", "outro"}:
        return "stable_support"
    if target_energy == "high_energy" or accent_density_per_beat >= 0.55:
        return "dynamic_transition"
    return "stable_support"


def _phrase_time_window(song_event_map: dict[str, Any], phrase: dict[str, Any]) -> tuple[float, float]:
    if "start_time_sec" in phrase and "end_time_sec" in phrase:
        return float(phrase["start_time_sec"]), float(phrase["end_time_sec"])

    beats = list(song_event_map.get("beats", []))
    if not beats:
        return 0.0, float(song_event_map.get("duration_sec", 0.0) or 0.0)

    start_beat = int(phrase.get("start_beat", 0) or 0)
    end_beat = int(phrase.get("end_beat_exclusive", start_beat + 1) or (start_beat + 1))
    start_beat = max(0, min(start_beat, len(beats) - 1))
    end_beat = max(start_beat, min(end_beat - 1, len(beats) - 1))
    return float(beats[start_beat]["time_sec"]), float(beats[end_beat]["time_sec"])


def _events_in_window(items: list[dict[str, Any]], start_time_sec: float, end_time_sec: float) -> list[dict[str, Any]]:
    return [
        item
        for item in items
        if float(item.get("time_sec", -1.0) or -1.0) >= start_time_sec
        and float(item.get("time_sec", 999999.0) or 999999.0) <= end_time_sec
    ]


def _phrase_music_profile(song_event_map: dict[str, Any], phrase: dict[str, Any], section_label: str) -> dict[str, Any]:
    start_time_sec, end_time_sec = _phrase_time_window(song_event_map, phrase)
    target_beats = max(1.0, float(phrase["end_beat_exclusive"] - phrase["start_beat"]))
    accents = _events_in_window(list(song_event_map.get("accents", [])), start_time_sec, end_time_sec)
    downbeats = _events_in_window(list(song_event_map.get("downbeats", [])), start_time_sec, end_time_sec)

    accent_strengths = [float(item.get("strength", 0.0) or 0.0) for item in accents]
    accent_count = len(accents)
    downbeat_count = len(downbeats)
    accent_density_per_beat = accent_count / max(target_beats, 1.0)
    mean_accent_strength = sum(accent_strengths) / max(len(accent_strengths), 1)
    max_accent_strength = max(accent_strengths) if accent_strengths else 0.0

    target_energy = _base_section_energy(section_label)
    if section_label in {"verse", "bridge"} and (accent_density_per_beat >= 0.60 or mean_accent_strength >= 0.34):
        target_energy = "high_energy"
    elif section_label in {"intro", "outro"} and accent_density_per_beat >= 0.45:
        target_energy = "mid_energy"
    elif section_label == "full_song" and mean_accent_strength < 0.2:
        target_energy = "low_energy"

    transition_mode = "compatible_chain"
    if target_energy == "high_energy" and accent_density_per_beat >= 0.50:
        transition_mode = "accent_switch"
    elif section_label in {"intro", "outro"}:
        transition_mode = "hold"

    return {
        "start_time_sec": start_time_sec,
        "end_time_sec": end_time_sec,
        "target_beats": target_beats,
        "accent_count": accent_count,
        "downbeat_count": downbeat_count,
        "accent_density_per_beat": round(accent_density_per_beat, 5),
        "mean_accent_strength": round(mean_accent_strength, 5),
        "max_accent_strength": round(max_accent_strength, 5),
        "target_energy": target_energy,
        "target_travel": _target_travel(section_label, accent_density_per_beat=accent_density_per_beat),
        "target_anchor_label": _target_anchor_label(section_label, target_energy, accent_density_per_beat=accent_density_per_beat),
        "transition_mode": transition_mode,
        "expected_accent_hits": [float(item["time_sec"]) for item in accents[:8]],
    }


def _sequence_penalty(previous_unit: dict[str, Any] | None, candidate: dict[str, Any]) -> float:
    if previous_unit is None:
        return 0.0
    if previous_unit.get("source_sequence") == candidate.get("source_sequence"):
        return 0.0
    if previous_unit.get("dataset") == candidate.get("dataset"):
        return -0.6
    return -1.15


def _compatibility_bonus(previous_unit: dict[str, Any] | None, candidate: dict[str, Any], transition_mode: str) -> float:
    if previous_unit is None:
        return 0.0
    compatible = candidate["unit_id"] in previous_unit.get("compatible_next_units", [])
    if compatible:
        return 5.2 if transition_mode == "compatible_chain" else 3.4
    return -4.0 if transition_mode == "compatible_chain" else -2.0


def _entry_anchor_bonus(candidate: dict[str, Any], target_anchor_label: str) -> float:
    return 1.4 if candidate.get("entry_anchor", {}).get("label") == target_anchor_label else -0.6


def _score_candidate(
    candidate: dict[str, Any],
    previous_unit: dict[str, Any] | None,
    recent_units: Counter[str],
    current_section_label: str,
    music_profile: dict[str, Any],
) -> float:
    score = 0.0
    target_energy = music_profile["target_energy"]
    target_travel = music_profile["target_travel"]
    target_beats = float(music_profile["target_beats"])

    if candidate.get("energy") == target_energy:
        score += 4.8
    elif target_energy == "mid_energy":
        score += 2.3
    elif {str(candidate.get("energy")), target_energy} == {"mid_energy", "high_energy"}:
        score += 0.9

    if current_section_label in candidate.get("preferred_section_labels", []):
        score += 3.2

    duration = float(candidate.get("duration_beats_estimate", target_beats) or target_beats)
    score -= abs(duration - target_beats) * 0.7

    candidate_travel = str(candidate.get("travel", "localized"))
    if candidate_travel == target_travel:
        score += 1.8
    elif {candidate_travel, target_travel} == {"stationary", "localized"}:
        score += 0.55

    candidate_support = (candidate.get("foot_contact") or ["unknown"])[0]
    if candidate_support in _support_preference(current_section_label, target_energy):
        score += 1.1

    body_focus = set(candidate.get("body_focus") or [])
    if current_section_label in {"verse", "bridge"} and "upper_body" in body_focus:
        score += 0.55
    if target_energy == "high_energy" and candidate_travel == "traveling":
        score += 0.9

    score += _compatibility_bonus(previous_unit, candidate, transition_mode=str(music_profile["transition_mode"]))
    score += _entry_anchor_bonus(candidate, target_anchor_label=str(music_profile["target_anchor_label"]))
    score += _sequence_penalty(previous_unit, candidate)

    if previous_unit is not None and previous_unit.get("source_sequence") == candidate.get("source_sequence"):
        score += 2.4

    if recent_units[candidate["unit_id"]]:
        score -= 2.0 * recent_units[candidate["unit_id"]]

    if candidate.get("bridge_quality") == "pass":
        score += 0.8
    if candidate.get("retarget_quality") == "pass":
        score += 0.6

    retarget_mean_joint_error = candidate.get("retarget_mean_joint_error")
    retarget_p95_joint_error = candidate.get("retarget_p95_joint_error")
    retarget_health = str(candidate.get("retarget_health", "unknown") or "unknown")
    if isinstance(retarget_mean_joint_error, (int, float)):
        score -= float(retarget_mean_joint_error) * 3.0
    if isinstance(retarget_p95_joint_error, (int, float)):
        score -= float(retarget_p95_joint_error) * 0.55

    health_bonus = {
        "preferred": 2.5,
        "usable": 0.75,
        "risky": -2.25,
        "avoid": -7.5,
        "unknown": -0.5,
    }
    score += health_bonus.get(retarget_health, -0.5)
    return score


def _candidate_pool(previous_unit: dict[str, Any] | None, units: list[dict[str, Any]], units_by_id: dict[str, dict[str, Any]], section_label: str) -> list[dict[str, Any]]:
    if previous_unit is None:
        return units

    compatible_ids = list(previous_unit.get("compatible_next_units", []))
    pool: list[dict[str, Any]] = []
    for unit_id in compatible_ids:
        candidate = units_by_id.get(unit_id)
        if candidate is not None:
            pool.append(candidate)

    same_sequence_units = [unit for unit in units if unit.get("source_sequence") == previous_unit.get("source_sequence")]
    if section_label not in {"chorus", "instrumental"}:
        pool.extend(same_sequence_units)

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in pool:
        unit_id = str(candidate["unit_id"])
        if unit_id in seen:
            continue
        seen.add(unit_id)
        deduped.append(candidate)
    return deduped or units


def build_choreography_plan(song_event_map: dict[str, Any], motion_library: dict[str, Any]) -> ChoreographyPlan:
    song_id = str(song_event_map["song_id"])
    library_id = str(motion_library["library_id"])
    phrases = list(song_event_map.get("phrases", []))
    sections = list(song_event_map.get("sections", []))
    units = list(motion_library.get("units", []))

    if not units:
        raise ValueError("motion unit library contains no units")

    units_by_id = {str(unit["unit_id"]): unit for unit in units}

    section_by_phrase = {}
    for section in sections:
        for phrase in phrases:
            if phrase["start_beat"] >= section["start_beat"] and phrase["end_beat_exclusive"] <= section["end_beat_exclusive"]:
                section_by_phrase[phrase["index"]] = section

    recent_units: Counter[str] = Counter()
    previous_unit: dict[str, Any] | None = None
    steps: list[dict[str, Any]] = []

    phrase_like_ranges = phrases or [
        {
            "index": 0,
            "label": "phrase",
            "start_beat": 0,
            "end_beat_exclusive": max(8, len(song_event_map.get("beats", []))),
        }
    ]

    for phrase in phrase_like_ranges:
        section = section_by_phrase.get(phrase["index"], sections[0] if sections else {"label": "full_song"})
        current_section_label = str(section.get("label", "full_song"))
        music_profile = _phrase_music_profile(song_event_map, phrase, current_section_label)
        pool = _candidate_pool(previous_unit=previous_unit, units=units, units_by_id=units_by_id, section_label=current_section_label)

        best_candidate = max(
            pool,
            key=lambda candidate: _score_candidate(
                candidate=candidate,
                previous_unit=previous_unit,
                recent_units=recent_units,
                current_section_label=current_section_label,
                music_profile=music_profile,
            ),
        )

        target_beats = float(music_profile["target_beats"])
        source_beats = max(0.25, float(best_candidate.get("duration_beats_estimate", target_beats) or target_beats))
        speed_scale = max(0.85, min(1.15, source_beats / target_beats))
        compatible_from_previous = bool(previous_unit and best_candidate["unit_id"] in previous_unit.get("compatible_next_units", []))

        steps.append(
            {
                "index": len(steps),
                "section_label": current_section_label,
                "start_beat": phrase["start_beat"],
                "target_beats": target_beats,
                "selected_unit_id": best_candidate["unit_id"],
                "source_sequence": best_candidate["source_sequence"],
                "speed_scale": round(speed_scale, 5),
                "switch_reason": {
                    "target_energy": music_profile["target_energy"],
                    "target_travel": music_profile["target_travel"],
                    "target_anchor_label": music_profile["target_anchor_label"],
                    "transition_mode": music_profile["transition_mode"],
                    "preferred_section_match": current_section_label in best_candidate.get("preferred_section_labels", []),
                    "compatible_from_previous": compatible_from_previous,
                    "same_source_sequence": bool(previous_unit and previous_unit.get("source_sequence") == best_candidate.get("source_sequence")),
                    "accent_count": music_profile["accent_count"],
                    "downbeat_count": music_profile["downbeat_count"],
                    "accent_density_per_beat": music_profile["accent_density_per_beat"],
                    "mean_accent_strength": music_profile["mean_accent_strength"],
                    "max_accent_strength": music_profile["max_accent_strength"],
                },
                "expected_accent_hits": music_profile["expected_accent_hits"],
                "reference_artifacts": best_candidate.get("reference_artifacts", {}),
            }
        )
        previous_unit = best_candidate
        recent_units.update([best_candidate["unit_id"]])

    notes = [
        "Planner is phrase-first in v1.1 and now strengthens section fit, musical accent profile, and transition compatibility when choosing clips.",
    ]
    if song_event_map.get("manual_review_required"):
        notes.append("Song analysis requested manual review; choreography should be treated as provisional.")

    return ChoreographyPlan(
        schema_version=1,
        plan_id=f"{song_id}_choreography_plan",
        song_id=song_id,
        library_id=library_id,
        steps=steps,
        notes=notes,
        generated_at_utc=utc_now_iso(),
    )
