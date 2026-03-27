#!/usr/bin/env python3
"""Scan local FBX clips and emit Unity-friendly motion_manifest.json."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips-root", required=True, help="Directory containing local FBX clips.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--tags", help="Optional JSON file with per-clip tags and overrides.")
    parser.add_argument(
        "--resource-prefix",
        default="Dance/Clips",
        help="Unity Resources prefix used to build resourcePath values.",
    )
    return parser.parse_args()


def load_tags(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip().lower()).strip("_")
    return slug or "clip"


def derive_variety_group(clip_id: str) -> str:
    normalized = slugify(clip_id or "clip")
    return re.sub(r"_\d+$", "", normalized) or normalized


def build_tag_lookup(tags_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for entry in tags_payload.get("clips", []):
        key_candidates = [
            entry.get("file"),
            entry.get("relative_path"),
            entry.get("clip_id"),
            entry.get("display_name"),
        ]
        for key in key_candidates:
            if key:
                lookup[key] = entry
    return lookup


def normalize_path(path: Path) -> str:
    return path.as_posix()


def build_manifest(args: argparse.Namespace) -> Dict[str, Any]:
    tags_payload = load_tags(args.tags)
    tag_lookup = build_tag_lookup(tags_payload)
    clips_root = Path(args.clips_root).resolve()
    discovered_files = sorted(clips_root.rglob("*.fbx"))
    default_phrase_beats = int(tags_payload.get("default_phrase_beats", 8))
    default_native_phrase_beats = int(tags_payload.get("default_native_phrase_beats", default_phrase_beats))
    default_native_bpm = float(tags_payload.get("default_native_bpm", 120.0))
    default_speed_min = float(tags_payload.get("default_speed_min", 0.85))
    default_speed_max = float(tags_payload.get("default_speed_max", 1.15))
    default_retime_profile = str(tags_payload.get("default_retime_profile", "smooth"))
    default_role = str(tags_payload.get("default_role", "loop"))
    default_entry_offsets_beats = list(tags_payload.get("default_entry_offsets_beats", [0]))
    default_slice_beats_options = list(tags_payload.get("default_slice_beats_options", [default_phrase_beats]))
    default_max_consecutive_selections = int(tags_payload.get("default_max_consecutive_selections", 2))
    default_group_cooldown_beats = int(tags_payload.get("default_group_cooldown_beats", 8))
    default_accent_bias = float(tags_payload.get("default_accent_bias", 0.0))

    clips: List[Dict[str, Any]] = []
    for file_path in discovered_files:
        relative_path = file_path.relative_to(clips_root)
        relative_path_posix = normalize_path(relative_path)
        stem_posix = normalize_path(relative_path.with_suffix(""))
        stem_name = file_path.stem

        tag_data = (
            tag_lookup.get(relative_path_posix)
            or tag_lookup.get(file_path.name)
            or tag_lookup.get(stem_name)
            or {}
        )

        resource_path = tag_data.get(
            "resource_path",
            normalize_path(Path(args.resource_prefix) / Path(stem_posix)),
        )

        display_name = tag_data.get("display_name", stem_name)
        clip_id = tag_data.get("clip_id", slugify(stem_posix))
        variety_group = str(tag_data.get("variety_group", derive_variety_group(clip_id)))

        clips.append(
            {
                "clipId": clip_id,
                "displayName": display_name,
                "resourcePath": resource_path,
                "clipName": tag_data.get("clip_name", ""),
                "styleTags": tag_data.get("style_tags", ["dance"]),
                "energyBand": tag_data.get("energy_band", "mid_energy"),
                "preferredSegments": tag_data.get("preferred_segments", ["verse", "chorus"]),
                "phraseBeats": int(tag_data.get("phrase_beats", default_phrase_beats)),
                "loopable": bool(tag_data.get("loopable", True)),
                "cooldownBeats": int(tag_data.get("cooldown_beats", 8)),
                "nativeBpm": float(tag_data.get("native_bpm", default_native_bpm)),
                "nativePhraseBeats": int(tag_data.get("native_phrase_beats", default_native_phrase_beats)),
                "speedMin": float(tag_data.get("speed_min", default_speed_min)),
                "speedMax": float(tag_data.get("speed_max", default_speed_max)),
                "retimeProfile": str(tag_data.get("retime_profile", default_retime_profile)),
                "varietyGroup": variety_group,
                "role": str(tag_data.get("role", default_role)),
                "entryOffsetsBeats": list(tag_data.get("entry_offsets_beats", default_entry_offsets_beats)),
                "sliceBeatsOptions": list(tag_data.get("slice_beats_options", default_slice_beats_options)),
                "maxConsecutiveSelections": int(tag_data.get("max_consecutive_selections", default_max_consecutive_selections)),
                "groupCooldownBeats": int(tag_data.get("group_cooldown_beats", default_group_cooldown_beats)),
                "accentBias": float(tag_data.get("accent_bias", default_accent_bias)),
            }
        )

    return {
        "libraryId": tags_payload.get("library_id", "manual_mixamo_library"),
        "defaultPhraseBeats": default_phrase_beats,
        "clips": clips,
        "generatedAtUtc": datetime.now(tz=timezone.utc).isoformat(),
    }


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = build_manifest(args)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        json.dumps(
            {
                "clips": len(payload["clips"]),
                "libraryId": payload["libraryId"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
