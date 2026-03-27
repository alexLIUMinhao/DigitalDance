#!/usr/bin/env python3
"""Analyze a local song and emit Unity-friendly song_analysis.json."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    import librosa
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: install tools/dance_pipeline/requirements.txt before running analyze_song.py"
    ) from exc


DEFAULT_SEGMENT_LABELS = ["intro", "verse", "chorus", "instrumental", "outro"]
GROOVE_POINT_COUNT = 128


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to the local audio file.")
    parser.add_argument("--song-id", required=True, help="Stable song identifier.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--overrides", help="Optional override JSON path.")
    parser.add_argument(
        "--audio-resource-path",
        help="Unity Resources path without extension. Overrides the value from overrides JSON.",
    )
    return parser.parse_args()


def load_json(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def to_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, "item"):
        return float(value.item())
    if isinstance(value, (list, tuple)):
        return float(value[0]) if value else default
    return float(value)


def coerce_float_list(values: Any) -> List[float]:
    if not values:
        return []
    return [round(float(value), 5) for value in values]


def ensure_strictly_increasing(name: str, values: List[float]) -> None:
    for index in range(len(values) - 1):
        if values[index + 1] <= values[index]:
            raise SystemExit(f"{name} must be strictly increasing at index {index}: {values[index]} -> {values[index + 1]}")


def derive_bpm_from_beats(beat_times: List[float], fallback_bpm: float) -> float:
    if len(beat_times) < 2:
        return fallback_bpm

    intervals = [max(0.2, beat_times[index + 1] - beat_times[index]) for index in range(len(beat_times) - 1)]
    median_interval = float(np.median(np.asarray(intervals, dtype=np.float32)))
    return 60.0 / max(1e-3, median_interval)


def normalize_energy(envelope: np.ndarray, target_points: int = 64) -> List[float]:
    if envelope.size == 0:
        return []
    indices = np.linspace(0, envelope.size - 1, num=target_points)
    sampled = np.interp(indices, np.arange(envelope.size), envelope)
    peak = np.max(sampled) or 1.0
    return [round(float(value / peak), 5) for value in sampled]


def normalize_curve(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples
    peak = float(np.max(samples))
    if peak <= 1e-6:
        return np.zeros_like(samples, dtype=np.float32)
    return (samples / peak).astype(np.float32)


def build_beat_windows(
    beat_times: List[float],
    duration_sec: float,
    global_bpm: float,
    beats_per_bar: int,
    explicit_downbeats: List[float] | None = None,
) -> List[Dict[str, Any]]:
    if not beat_times:
        return []

    windows: List[Dict[str, Any]] = []
    fallback_spacing = 60.0 / max(global_bpm, 1e-3)
    downbeat_lookup = {round(float(value), 5) for value in (explicit_downbeats or [])}
    for index, start_sec in enumerate(beat_times):
        if index + 1 < len(beat_times):
            end_sec = beat_times[index + 1]
        else:
            previous_spacing = beat_times[index] - beat_times[index - 1] if index > 0 else fallback_spacing
            end_sec = min(duration_sec, start_sec + max(0.2, previous_spacing))

        duration = max(0.2, end_sec - start_sec)
        local_bpm = 60.0 / duration
        start_sec_rounded = round(float(start_sec), 5)
        windows.append(
            {
                "index": index,
                "startSec": start_sec_rounded,
                "endSec": round(float(min(duration_sec, end_sec)), 5),
                "durationSec": round(float(duration), 5),
                "localBpm": round(float(local_bpm), 5),
                "isDownbeat": start_sec_rounded in downbeat_lookup if downbeat_lookup else bool(index % max(1, beats_per_bar) == 0),
            }
        )

    return windows


def smooth_local_bpms(beat_windows: List[Dict[str, Any]], kernel_size: int = 5) -> None:
    if len(beat_windows) < 3:
        return
    local_bpms = np.array([window["localBpm"] for window in beat_windows], dtype=np.float32)
    pad = kernel_size // 2
    padded = np.pad(local_bpms, (pad, pad), mode="edge")
    kernel = np.ones(kernel_size, dtype=np.float32) / float(kernel_size)
    smoothed = np.convolve(padded, kernel, mode="valid")
    for index, value in enumerate(smoothed):
        beat_windows[index]["localBpm"] = round(float(value), 5)


def build_tempo_map(beat_windows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not beat_windows:
        return []

    tempo_map: List[Dict[str, Any]] = []
    active = {
        "startSec": beat_windows[0]["startSec"],
        "endSec": beat_windows[0]["endSec"],
        "bpm": beat_windows[0]["localBpm"],
    }

    for window in beat_windows[1:]:
        if abs(window["localBpm"] - active["bpm"]) <= 4.0:
            active["endSec"] = window["endSec"]
            active["bpm"] = round(float((active["bpm"] + window["localBpm"]) * 0.5), 5)
            continue

        tempo_map.append(active)
        active = {
            "startSec": window["startSec"],
            "endSec": window["endSec"],
            "bpm": window["localBpm"],
        }

    tempo_map.append(active)
    return tempo_map


def resample_curve(values: np.ndarray, times: np.ndarray, target_points: int) -> List[Dict[str, Any]]:
    if values.size == 0 or times.size == 0:
        return []
    sample_times = np.linspace(float(times[0]), float(times[-1]), num=target_points)
    sampled = np.interp(sample_times, times, values)
    return [
        {
            "timeSec": round(float(time_sec), 5),
            "value": round(float(value), 5),
        }
        for time_sec, value in zip(sample_times, sampled)
    ]


def build_auto_segments(total_beats: int, beats_per_bar: int) -> List[Dict[str, Any]]:
    if total_beats <= 0:
        return []

    if total_beats < beats_per_bar * 8:
        split_points = [0.0, 0.5, 1.0]
        labels = ["intro", "outro"]
    elif total_beats < beats_per_bar * 16:
        split_points = [0.0, 0.25, 0.65, 1.0]
        labels = ["intro", "verse", "outro"]
    else:
        split_points = [0.0, 0.12, 0.48, 0.78, 1.0]
        labels = DEFAULT_SEGMENT_LABELS[: len(split_points) - 1]

    boundaries = []
    for ratio in split_points:
        beat_index = int(round(total_beats * ratio))
        snapped = max(0, min(total_beats, beat_index))
        snapped = (snapped // beats_per_bar) * beats_per_bar
        boundaries.append(snapped)

    boundaries[0] = 0
    boundaries[-1] = total_beats

    segments: List[Dict[str, Any]] = []
    for index, label in enumerate(labels):
        start_beat = boundaries[index]
        end_beat = boundaries[index + 1]
        if end_beat <= start_beat:
            continue
        segments.append(
            {
                "segmentId": f"{label}_{index}",
                "label": label,
                "startBeat": start_beat,
                "endBeatExclusive": end_beat,
                "energy": infer_segment_energy(label),
            }
        )

    if not segments:
        segments.append(
            {
                "segmentId": "full_song_0",
                "label": "verse",
                "startBeat": 0,
                "endBeatExclusive": total_beats,
                "energy": infer_segment_energy("verse"),
            }
        )

    return segments


def infer_segment_energy(label: str) -> float:
    mapping = {
        "intro": 0.2,
        "verse": 0.5,
        "chorus": 0.9,
        "instrumental": 0.8,
        "bridge": 0.65,
        "outro": 0.25,
    }
    return mapping.get(label, 0.5)


def coerce_segments(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    segments = []
    for index, segment in enumerate(raw_segments):
        label = segment.get("label", "verse")
        segments.append(
            {
                "segmentId": segment.get("segmentId", f"{label}_{index}"),
                "label": label,
                "startBeat": int(segment["startBeat"]),
                "endBeatExclusive": int(segment["endBeatExclusive"]),
                "energy": float(segment.get("energy", infer_segment_energy(label))),
            }
        )
    return segments


def synthesize_beats_from_tempo(duration_sec: float, tempo_bpm: float) -> List[float]:
    if tempo_bpm <= 0.0:
        tempo_bpm = 90.0
    spacing = 60.0 / tempo_bpm
    beat_count = max(1, int(math.floor(duration_sec / spacing)))
    return [round(index * spacing, 5) for index in range(beat_count)]


def main() -> None:
    args = parse_args()
    overrides = load_json(args.overrides)

    audio_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples, sample_rate = librosa.load(str(audio_path), sr=None, mono=True)
    duration_sec = float(librosa.get_duration(y=samples, sr=sample_rate))
    hop_length = 512
    tempo_bpm, beat_frames = librosa.beat.beat_track(y=samples, sr=sample_rate, trim=False, hop_length=hop_length)
    tempo_bpm = to_float(tempo_bpm, 90.0)
    auto_beat_times = librosa.frames_to_time(beat_frames, sr=sample_rate, hop_length=hop_length).tolist()
    auto_beat_times = [round(float(value), 5) for value in auto_beat_times]
    if not auto_beat_times:
        auto_beat_times = synthesize_beats_from_tempo(duration_sec, tempo_bpm)

    override_beat_times = coerce_float_list(overrides.get("beats"))
    if override_beat_times:
        ensure_strictly_increasing("override beats", override_beat_times)
        beat_times = override_beat_times
    else:
        beat_times = auto_beat_times

    beats_per_bar = int(overrides.get("beats_per_bar", 4))
    if beats_per_bar <= 0:
        beats_per_bar = 4

    explicit_downbeats = coerce_float_list(overrides.get("downbeats"))
    if explicit_downbeats:
        ensure_strictly_increasing("override downbeats", explicit_downbeats)

    rms = librosa.feature.rms(y=samples, frame_length=2048, hop_length=hop_length)[0]
    energy_envelope = normalize_energy(rms)
    onset_env = librosa.onset.onset_strength(y=samples, sr=sample_rate, hop_length=hop_length)
    low_band = librosa.feature.rms(
        y=librosa.effects.preemphasis(samples, coef=0.82),
        frame_length=2048,
        hop_length=hop_length,
    )[0]
    low_band = np.power(np.maximum(low_band, 0.0), 0.8)
    onset_norm = normalize_curve(onset_env)
    low_norm = normalize_curve(low_band)
    groove_curve = normalize_curve((onset_norm * 0.65) + (low_norm * 0.35))
    groove_times = librosa.frames_to_time(np.arange(groove_curve.size), sr=sample_rate, hop_length=hop_length)

    auto_segments = build_auto_segments(len(beat_times), beats_per_bar)
    final_segments = overrides.get("segments", auto_segments)
    final_segments = coerce_segments(final_segments)

    if overrides.get("bpm") is not None:
        tempo_bpm = float(overrides["bpm"])
    elif override_beat_times:
        tempo_bpm = derive_bpm_from_beats(beat_times, tempo_bpm)

    beat_windows = build_beat_windows(beat_times, duration_sec, tempo_bpm, beats_per_bar, explicit_downbeats)
    smooth_local_bpms(beat_windows)
    downbeats = explicit_downbeats if explicit_downbeats else [window["startSec"] for window in beat_windows if window["isDownbeat"]]
    tempo_map = build_tempo_map(beat_windows)
    groove_envelope = resample_curve(groove_curve, groove_times, GROOVE_POINT_COUNT)

    display_name = overrides.get("display_name", audio_path.stem)
    audio_resource_path = args.audio_resource_path or overrides.get(
        "audio_resource_path",
        f"Dance/Audio/{audio_path.stem}",
    )

    payload = {
        "songId": args.song_id,
        "displayName": display_name,
        "audioResourcePath": audio_resource_path,
        "durationSec": round(duration_sec, 5),
        "bpm": round(tempo_bpm, 5),
        "beatsPerBar": beats_per_bar,
        "beats": beat_times,
        "beatWindows": beat_windows,
        "downbeats": [round(float(value), 5) for value in downbeats],
        "energyEnvelope": energy_envelope,
        "grooveEnvelope": groove_envelope,
        "tempoMap": tempo_map,
        "segments": final_segments,
        "generatedAtUtc": datetime.now(tz=timezone.utc).isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        json.dumps(
            {
                "songId": payload["songId"],
                "beats": len(payload["beats"]),
                "segments": len(payload["segments"]),
                "bpm": payload["bpm"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
