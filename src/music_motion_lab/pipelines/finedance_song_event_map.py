from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils import slugify, utc_now_iso


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


def _event_time(item: dict[str, Any]) -> float:
    return _safe_float(item.get("time_sec", item.get("absolute_time_sec", 0.0)))


def _coerce_beat(item: dict[str, Any], index: int, beats_per_bar: int) -> dict[str, Any]:
    time_sec = _event_time(item)
    return {
        "index": _safe_int(item.get("index"), index),
        "time_sec": round(time_sec, 5),
        "strength": round(_safe_float(item.get("strength")), 5),
        "is_downbeat": bool(item.get("is_downbeat", index % max(1, beats_per_bar) == 0)),
        "confidence": round(_safe_float(item.get("confidence"), 1.0), 5),
    }


def _coerce_timed_event(item: dict[str, Any], index: int, kind: str) -> dict[str, Any]:
    event = {
        "index": _safe_int(item.get("index"), index),
        "time_sec": round(_event_time(item), 5),
        "strength": round(_safe_float(item.get("strength", item.get("confidence", 1.0)), 1.0), 5),
        "kind": kind,
    }
    for key in ("source_beat_index", "beat_index", "level", "band", "confidence", "motion_score"):
        if key in item:
            event[key] = item[key]
    return event


def _beat_time(beats: list[dict[str, Any]], beat_index: int) -> float:
    if not beats:
        return 0.0
    beat_index = max(0, beat_index)
    if beat_index < len(beats):
        return _safe_float(beats[beat_index].get("time_sec"))
    if len(beats) == 1:
        return _safe_float(beats[0].get("time_sec"))
    spacing = _safe_float(beats[-1].get("time_sec")) - _safe_float(beats[-2].get("time_sec"))
    return _safe_float(beats[-1].get("time_sec")) + spacing * float(beat_index - len(beats) + 1)


def _build_downbeats(audio_features: dict[str, Any], beats: list[dict[str, Any]], beats_per_bar: int) -> list[dict[str, Any]]:
    strong_downbeats = [
        item
        for item in audio_features.get("strong_beats", []) or []
        if bool(item.get("is_downbeat", False))
    ]
    source = strong_downbeats or [beat for beat in beats if bool(beat.get("is_downbeat", False))]
    if not source:
        source = [beat for beat in beats if _safe_int(beat.get("index")) % max(1, beats_per_bar) == 0]
    downbeats: list[dict[str, Any]] = []
    for item in source:
        source_beat_index = _safe_int(item.get("source_beat_index", item.get("beat_index", item.get("index"))))
        downbeats.append(
            {
                "index": len(downbeats),
                "time_sec": round(_event_time(item), 5),
                "source_beat_index": source_beat_index,
            }
        )
    return sorted(downbeats, key=lambda item: item["time_sec"])


def _build_phrases(beats: list[dict[str, Any]], phrase_beats: int) -> list[dict[str, Any]]:
    phrases: list[dict[str, Any]] = []
    beat_count = len(beats)
    for start in range(0, max(0, beat_count - 1), max(1, phrase_beats)):
        end = min(beat_count - 1, start + max(1, phrase_beats))
        if end <= start:
            continue
        phrases.append(
            {
                "index": len(phrases),
                "label": "phrase",
                "start_beat": start,
                "end_beat_exclusive": end,
                "start_time_sec": round(_beat_time(beats, start), 5),
                "end_time_sec": round(_beat_time(beats, end), 5),
            }
        )
    return phrases


def _build_sections(phrases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not phrases:
        return []
    labels = ["intro", "verse", "chorus", "verse", "chorus", "outro"]
    sections: list[dict[str, Any]] = []
    group_size = 4
    for start in range(0, len(phrases), group_size):
        group = phrases[start : start + group_size]
        if not group:
            continue
        label = labels[min(len(sections), len(labels) - 1)]
        sections.append(
            {
                "index": len(sections),
                "label": label,
                "start_beat": _safe_int(group[0].get("start_beat")),
                "end_beat_exclusive": _safe_int(group[-1].get("end_beat_exclusive")),
                "start_time_sec": group[0].get("start_time_sec"),
                "end_time_sec": group[-1].get("end_time_sec"),
                "source": "m2_2_finedance_audio_features",
            }
        )
    return sections


def _entry_by_sequence(audio_feature_report: dict[str, Any], sequence_id: str) -> dict[str, Any]:
    target = slugify(sequence_id)
    for entry in audio_feature_report.get("entries", []) or []:
        if slugify(str(entry.get("sequence_id", ""))) == target:
            return dict(entry)
    raise KeyError(f"FineDance audio feature entry not found: {sequence_id}")


def build_finedance_song_event_map_from_report(
    audio_feature_report: dict[str, Any],
    sequence_id: str,
    motion_base_assets_root: Path,
    phrase_beats: int = 8,
) -> dict[str, Any]:
    entry = _entry_by_sequence(audio_feature_report, sequence_id)
    audio_features = dict(entry.get("audio_features", {}) or {})
    beats_per_bar = max(1, _safe_int(audio_features.get("beats_per_bar"), 4))
    beats = [
        _coerce_beat(dict(item), index=index, beats_per_bar=beats_per_bar)
        for index, item in enumerate(audio_features.get("beats", []) or [])
    ]
    beats = sorted(beats, key=lambda item: (item["time_sec"], item["index"]))
    downbeats = _build_downbeats(audio_features, beats, beats_per_bar)
    accent_source = list(audio_features.get("accent_candidates", []) or [])
    if not accent_source:
        accent_source = [item for item in audio_features.get("strong_beats", []) or [] if _safe_int(item.get("level")) >= 3]
    accents = [
        _coerce_timed_event(dict(item), index=index, kind=str(item.get("band", "accent") or "accent"))
        for index, item in enumerate(accent_source)
    ]
    drum_hits = [
        _coerce_timed_event(dict(item), index=index, kind="drum_hit")
        for index, item in enumerate(audio_features.get("drum_hits", []) or [])
    ]
    phrases = _build_phrases(beats, phrase_beats=phrase_beats)
    sections = _build_sections(phrases)
    panel = dict(entry.get("audio_analysis_panel", {}) or {})
    duration_sec = _safe_float(panel.get("clip_end_sec"))
    if duration_sec <= 0.0 and beats:
        duration_sec = max(_safe_float(beats[-1].get("time_sec")), _safe_float(phrases[-1].get("end_time_sec")) if phrases else 0.0)
    source_audio_path = motion_base_assets_root / "datasets" / "finedance" / "raw" / "extracted" / "finedance" / "music_wav" / f"{sequence_id}.wav"
    style_tags = dict(entry.get("style_tags", {}) or {})
    song_name = style_tags.get("song_name") or f"finedance_{sequence_id}"
    return {
        "schema_version": 1,
        "song_id": f"finedance_{slugify(sequence_id)}_full_song",
        "source_audio_path": str(source_audio_path),
        "duration_sec": round(duration_sec, 5),
        "bpm": round(_safe_float(audio_features.get("bpm")), 5),
        "beats_per_bar": beats_per_bar,
        "beats": beats,
        "downbeats": downbeats,
        "accents": sorted(accents, key=lambda item: item["time_sec"]),
        "drum_hits": sorted(drum_hits, key=lambda item: item["time_sec"]),
        "phrases": phrases,
        "sections": sections,
        "confidence": {
            "source": "m2_2_finedance_audio_features",
            "energy_label": audio_features.get("energy_label"),
            "energy_score": audio_features.get("energy_score"),
            "analysis_priority": entry.get("analysis_priority", {}),
        },
        "metadata": {
            "dataset": "finedance",
            "sequence_id": sequence_id,
            "song_name": song_name,
            "style_tags": style_tags,
        },
        "generated_at_utc": utc_now_iso(),
        "notes": [
            "Converted from M2-2 FineDance rhythmic-first audio feature report.",
            f"phrase_beats={int(phrase_beats)}.",
        ],
    }
