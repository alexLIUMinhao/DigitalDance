#!/usr/bin/env python3
"""Create the initial isolated motion-base seed library."""

from __future__ import annotations

from argparse import ArgumentParser
from typing import Any, Dict, List, Tuple

from common import MOTION_INDEX_PATH, REVIEWS_PATH, write_json


STYLE_FAMILIES: List[Tuple[str, List[str], int]] = [
    ("chinese_classical", ["water_sleeve", "fan_dance", "court_classical", "dunhuang_inspired"], -2),
    ("ethnic_folk", ["mongolian", "dai", "tibetan", "uygur"], 0),
    ("contemporary", ["lyrical", "modern_release", "urban_contemporary", "theatrical_modern"], 2),
]

LOOP_DEFINITIONS = [
    ("basic_step", "loop", "style_locked", 8, "low_energy", ["intro", "verse"], 96, [0], [8]),
    ("basic_step", "loop", "style_locked", 8, "mid_energy", ["verse", "chorus"], 104, [0, 4], [8]),
    ("travel_step", "loop", "style_locked", 8, "mid_energy", ["verse", "instrumental"], 110, [0, 4], [8]),
    ("travel_step", "loop", "style_locked", 8, "mid_energy", ["chorus", "instrumental"], 114, [0, 4], [8]),
    ("turn_phrase", "loop", "style_locked", 8, "mid_energy", ["chorus", "instrumental"], 118, [0, 4], [8]),
    ("turn_phrase", "loop", "style_locked", 16, "high_energy", ["chorus", "instrumental"], 124, [0, 8], [8, 16]),
    ("arm_expression", "loop", "style_locked", 8, "low_energy", ["intro", "verse", "outro"], 92, [0], [8]),
    ("arm_expression", "loop", "style_locked", 8, "mid_energy", ["verse", "instrumental"], 100, [0, 4], [8]),
    ("pose_hold", "loop", "style_locked", 8, "low_energy", ["intro", "outro"], 88, [0], [8]),
    ("basic_step", "loop", "style_locked", 8, "mid_energy", ["verse", "chorus"], 108, [0, 4], [8]),
    ("travel_step", "loop", "style_locked", 8, "high_energy", ["chorus", "instrumental"], 122, [0, 4], [8]),
    ("arm_expression", "loop", "style_locked", 16, "mid_energy", ["verse", "outro"], 98, [0, 8], [8, 16]),
]

SUPPORTING_DEFINITIONS = [
    ("accent_hit", "accent", "accent", 4, "high_energy", ["chorus", "instrumental"], 126, [0], [4]),
    ("accent_hit", "accent", "accent", 4, "high_energy", ["chorus"], 132, [0], [4]),
    ("turn_phrase", "loop", "bridge", 8, "mid_energy", ["verse", "chorus", "instrumental"], 112, [0, 4], [4, 8]),
    ("pose_hold", "loop", "bridge", 8, "low_energy", ["outro", "intro"], 90, [0], [4, 8]),
]

LICENSE_CYCLE = ["replace_before_ship", "prototype_only", "replace_before_ship", "commercial_safe"]
SOURCE_CYCLE = ["curated_fbx", "generated_from_video"]
REVIEWER_CYCLE = ["motion_qa_li", "motion_qa_chen", "motion_qa_wu"]


def build_seed_payload() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    motions: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []

    for family, substyles, bpm_offset in STYLE_FAMILIES:
        definitions = LOOP_DEFINITIONS + SUPPORTING_DEFINITIONS
        for index, definition in enumerate(definitions, start=1):
            meta_action, role, transition_profile, phrase_beats, energy_band, segments, base_bpm, entry_offsets, slice_options = definition
            substyle = substyles[(index - 1) % len(substyles)]
            motion_id = f"{family}_{index:02d}"

            motions.append(
                {
                    "motionId": motion_id,
                    "displayName": f"{family.replace('_', ' ').title()} Phrase {index:02d}",
                    "approvedFbxRelPath": f"{family}/{substyle}/{motion_id}.fbx",
                    "sourceKind": SOURCE_CYCLE[(index - 1) % len(SOURCE_CYCLE)],
                    "sourceRef": f"source_sheet_v1/{family}/{substyle}/{motion_id}",
                    "licenseTier": LICENSE_CYCLE[(index - 1) % len(LICENSE_CYCLE)],
                    "qualityTier": "production",
                    "styleFamily": family,
                    "styleSubstyle": substyle,
                    "metaAction": meta_action,
                    "energyBand": energy_band,
                    "preferredSegments": list(segments),
                    "nativeBpm": float(base_bpm + bpm_offset),
                    "phraseBeats": phrase_beats,
                    "entryOffsetsBeats": list(entry_offsets),
                    "sliceBeatsOptions": list(slice_options),
                    "transitionProfile": transition_profile,
                    "varietyGroup": f"{family}_{meta_action}",
                    "role": role,
                    "notes": f"Seed motion record for {family} / {substyle} / {meta_action}.",
                }
            )

            review_issues = []
            if meta_action in {"turn_phrase", "accent_hit"}:
                review_issues = ["check landing consistency during final integration"]

            reviews.append(
                {
                    "motionId": motion_id,
                    "reviewStatus": "approved",
                    "loopSeam": "clean" if phrase_beats >= 8 else "accent_only",
                    "footStability": "stable" if meta_action not in {"turn_phrase", "accent_hit"} else "monitor",
                    "styleClarity": "clear",
                    "tempoTolerance": "medium" if energy_band == "mid_energy" else "narrow" if energy_band == "low_energy" else "wide",
                    "reviewer": REVIEWER_CYCLE[(index - 1) % len(REVIEWER_CYCLE)],
                    "issues": review_issues,
                    "reviewedAtUtc": "2026-03-29T00:00:00Z",
                }
            )

    motion_index = {
        "schemaVersion": 1,
        "batchId": "motion_base_seed_v1",
        "targetCount": 48,
        "motions": motions,
    }
    review_index = {
        "schemaVersion": 1,
        "reviews": reviews,
    }
    return motion_index, review_index


def main() -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Overwrite existing seed files.")
    args = parser.parse_args()

    if not args.force and (MOTION_INDEX_PATH.exists() or REVIEWS_PATH.exists()):
        raise SystemExit("Seed files already exist. Re-run with --force to overwrite.")

    motion_index, review_index = build_seed_payload()
    write_json(MOTION_INDEX_PATH, motion_index)
    write_json(REVIEWS_PATH, review_index)
    print(f"seeded motions={len(motion_index['motions'])} reviews={len(review_index['reviews'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
