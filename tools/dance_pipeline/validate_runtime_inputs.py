#!/usr/bin/env python3
"""Validate runtime song/motion inputs for the dance demo."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_monotonic_increasing(values: list[float]) -> bool:
    return all(values[index] < values[index + 1] for index in range(len(values) - 1))


def summarize_floats(values: list[float]) -> str:
    if not values:
        return "n/a"

    return (
        f"min={min(values):0.2f}, "
        f"median={statistics.median(values):0.2f}, "
        f"mean={statistics.fmean(values):0.2f}, "
        f"max={max(values):0.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Unity project root (default: current project).",
    )
    parser.add_argument(
        "--song-id",
        default="audio1_mp3",
        help="Song id to validate from song_catalog.json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dance_data_root = args.project_root / "Assets" / "StreamingAssets" / "DanceData"
    catalog_path = dance_data_root / "song_catalog.json"
    manifest_path = dance_data_root / "motion_manifest.json"

    errors: list[str] = []
    warnings: list[str] = []

    if not catalog_path.exists():
        errors.append(f"Missing catalog: {catalog_path}")
        print_report(args.song_id, errors, warnings)
        return 1

    if not manifest_path.exists():
        errors.append(f"Missing motion manifest: {manifest_path}")
        print_report(args.song_id, errors, warnings)
        return 1

    catalog = load_json(catalog_path)
    manifest = load_json(manifest_path)

    songs = catalog.get("songs", [])
    song_entry = next((entry for entry in songs if entry.get("songId") == args.song_id), None)
    if song_entry is None:
        errors.append(f"Song id '{args.song_id}' not found in song_catalog.json")
        print_report(args.song_id, errors, warnings)
        return 1

    analysis_path = resolve_streaming_asset_path(dance_data_root, song_entry.get("analysisPath", ""))
    if not analysis_path.exists():
        errors.append(f"Missing analysis file for '{args.song_id}': {analysis_path}")
        print_report(args.song_id, errors, warnings)
        return 1

    analysis = load_json(analysis_path)
    playback_profile = song_entry.get("playbackProfile", {}) or {}
    tempo_scale = playback_profile.get("tempoScale", 1.0)
    beat_grouping = max(1, int(playback_profile.get("beatGrouping", 1) or 1))
    energy_mode = str(playback_profile.get("energyMode", "balanced") or "balanced").lower()

    beats = analysis.get("beats", [])
    beat_windows = analysis.get("beatWindows", [])
    local_bpms = [float(window.get("localBpm", 0.0) or 0.0) for window in beat_windows]
    beat_durations = [float(window.get("durationSec", 0.0) or 0.0) for window in beat_windows]
    groove_values = [float(sample.get("value", 0.0) or 0.0) for sample in analysis.get("grooveEnvelope", [])]

    if not beats:
        errors.append("Song analysis has no beats.")
    elif not is_monotonic_increasing([float(value) for value in beats]):
        errors.append("Song beats are not strictly increasing.")

    if not beat_windows:
        errors.append("Song analysis has no beatWindows.")
    else:
        non_positive_windows = sum(1 for duration in beat_durations if duration <= 0.0)
        if non_positive_windows:
            errors.append(f"Song analysis has {non_positive_windows} beat windows with non-positive duration.")

    if energy_mode not in {"gentle", "balanced", "driving"}:
        errors.append(f"Invalid playbackProfile.energyMode: {energy_mode}")

    if args.song_id == "audio1_mp3":
        if beat_grouping != 1:
            errors.append("audio1_mp3 should use beatGrouping = 1 after manual beat override.")
        if energy_mode != "gentle":
            errors.append("audio1_mp3 should use energyMode = gentle.")

    raw_bpm = float(analysis.get("bpm", 0.0) or 0.0)
    grouped_bpm = raw_bpm / beat_grouping if beat_grouping > 0 else raw_bpm
    dance_bpm = grouped_bpm * float(tempo_scale)
    if args.song_id == "audio1_mp3" and raw_bpm >= 110.0:
        errors.append("audio1_mp3 raw BPM is still too high. Manual beat override does not appear to be applied.")
    if args.song_id == "audio1_mp3" and dance_bpm >= 100.0:
        errors.append("audio1_mp3 Dance BPM is still too high for the intended slow-song profile.")

    clips = manifest.get("clips", [])
    energy_counts: dict[str, int] = {}
    slow_friendly = 0
    for clip in clips:
        energy_band = str(clip.get("energyBand", "unknown") or "unknown").lower()
        energy_counts[energy_band] = energy_counts.get(energy_band, 0) + 1
        native_bpm = float(clip.get("nativeBpm", 120.0) or 120.0)
        speed_min = float(clip.get("speedMin", 0.85) or 0.85)
        speed_max = float(clip.get("speedMax", 1.15) or 1.15)
        if energy_band in {"low_energy", "mid_energy"} and native_bpm <= 110.0 and speed_min <= 0.8 and speed_max <= 1.08:
            slow_friendly += 1

    if energy_counts.get("low_energy", 0) < 3:
        errors.append("Motion manifest has fewer than 3 low_energy clips.")
    if energy_counts.get("mid_energy", 0) < 2:
        warnings.append("Motion manifest has fewer than 2 mid_energy clips.")
    if slow_friendly < 3:
        warnings.append("Fewer than 3 clips look slow-song-friendly by nativeBpm/speed range.")

    print(f"Song: {args.song_id}")
    print(f"Catalog: {catalog_path}")
    print(f"Analysis: {analysis_path}")
    print(f"Motion manifest: {manifest_path}")
    print()
    print("Playback profile")
    print(f"  tempoScale: {tempo_scale:0.2f}")
    print(f"  beatGrouping: {beat_grouping}")
    print(f"  energyMode: {energy_mode}")
    print(f"  raw BPM: {raw_bpm:0.2f}")
    print(f"  grouped BPM: {grouped_bpm:0.2f}")
    print(f"  dance BPM: {dance_bpm:0.2f}")
    print()
    print("Song analysis")
    print(f"  beats: {len(beats)}")
    print(f"  beatWindows: {len(beat_windows)}")
    print(f"  segments: {len(analysis.get('segments', []))}")
    print(f"  beat duration stats: {summarize_floats(beat_durations)}")
    print(f"  local BPM stats: {summarize_floats(local_bpms)}")
    print(f"  groove stats: {summarize_floats(groove_values)}")
    print()
    print("Motion manifest")
    print(f"  total clips: {len(clips)}")
    for key in sorted(energy_counts):
        print(f"  {key}: {energy_counts[key]}")
    print(f"  slow-friendly clips: {slow_friendly}")
    print()

    print_report(args.song_id, errors, warnings)
    return 1 if errors else 0


def print_report(song_id: str, errors: list[str], warnings: list[str]) -> None:
    if errors:
        print("Validation: FAILED")
        for message in errors:
            print(f"  ERROR: {message}")
    else:
        print("Validation: PASSED")

    if warnings:
        for message in warnings:
            print(f"  WARN: {message}")

    if not errors and not warnings:
        print(f"  OK: {song_id} inputs look internally consistent.")


def resolve_streaming_asset_path(dance_data_root: Path, relative_path: str) -> Path:
    normalized = str(relative_path or "").replace("\\", "/").lstrip("/")
    if normalized.startswith("DanceData/"):
        normalized = normalized[len("DanceData/") :]
    return dance_data_root / normalized


if __name__ == "__main__":
    raise SystemExit(main())
