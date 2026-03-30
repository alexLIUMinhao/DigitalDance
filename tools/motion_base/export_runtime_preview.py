#!/usr/bin/env python3
"""Export production-ready motion-base records into a runtime-compatible preview manifest."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from common import PREVIEW_ROOT, load_library, load_taxonomy, review_map, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PREVIEW_ROOT / "motion_manifest.preview.json",
        help="Output path for the preview runtime manifest.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PREVIEW_ROOT / "export_report.preview.json",
        help="Output path for the export summary report.",
    )
    return parser.parse_args()


def unique(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def resolve_speed_range(defaults: Dict[str, Any], energy_band: str) -> tuple[float, float]:
    value = defaults["speedRanges"].get(energy_band, defaults["speedRanges"]["mid_energy"])
    return float(value[0]), float(value[1])


def build_clip(record: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    role = str(record["role"])
    transition = str(record["transitionProfile"])
    speed_min, speed_max = resolve_speed_range(defaults, str(record["energyBand"]))
    accent_bias = defaults["accentBiasByTransition"].get(transition, defaults["accentBiasByRole"].get(role, 0.15))
    resource_prefix = str(defaults["resourcePrefix"]).strip("/")
    clip_stem = Path(str(record["approvedFbxRelPath"])).stem

    return {
        "clipId": record["motionId"],
        "displayName": record["displayName"],
        "resourcePath": f"{resource_prefix}/{record['styleFamily']}/{clip_stem}",
        "clipName": clip_stem,
        "styleTags": unique(
            [
                "motion_base",
                "mixamo_candidate",
                record["styleFamily"],
                record["styleSubstyle"],
                record["metaAction"],
                record["energyBand"],
                record["transitionProfile"],
                record["sourceKind"],
            ]
        ),
        "energyBand": record["energyBand"],
        "preferredSegments": record["preferredSegments"],
        "phraseBeats": int(record["phraseBeats"]),
        "loopable": role != "accent",
        "cooldownBeats": int(defaults["cooldownBeats"].get(role, defaults["cooldownBeats"]["loop"])),
        "nativeBpm": float(record["nativeBpm"]),
        "nativePhraseBeats": int(record["phraseBeats"]),
        "speedMin": speed_min,
        "speedMax": speed_max,
        "retimeProfile": defaults["retimeProfiles"].get(record["energyBand"], "smooth"),
        "varietyGroup": record["varietyGroup"],
        "role": role,
        "entryOffsetsBeats": record["entryOffsetsBeats"],
        "sliceBeatsOptions": record["sliceBeatsOptions"],
        "maxConsecutiveSelections": int(defaults["maxConsecutiveSelections"].get(role, 2)),
        "groupCooldownBeats": int(defaults["groupCooldownBeats"].get(transition, 8)),
        "accentBias": float(accent_bias),
    }


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy()
    motion_index, review_index = load_library()
    reviews = review_map(review_index)
    defaults = taxonomy["runtimePreviewDefaults"]

    exported: List[Dict[str, Any]] = []
    skipped: List[Dict[str, str]] = []
    for record in motion_index.get("motions", []):
        motion_id = record["motionId"]
        review = reviews.get(motion_id)
        if record["qualityTier"] != "production":
            skipped.append({"motionId": motion_id, "reason": "qualityTier"})
            continue
        if review is None or review.get("reviewStatus") != "approved":
            skipped.append({"motionId": motion_id, "reason": "reviewStatus"})
            continue
        exported.append(build_clip(record, defaults))

    exported.sort(key=lambda item: item["clipId"])
    manifest = {
        "libraryId": defaults["libraryId"],
        "defaultPhraseBeats": int(defaults["defaultPhraseBeats"]),
        "clips": exported,
        "generatedAtUtc": datetime.now(tz=timezone.utc).isoformat(),
    }
    report = {
        "exportedClipCount": len(exported),
        "skippedClipCount": len(skipped),
        "skipped": skipped,
        "output": str(args.output),
    }

    write_json(args.output, manifest)
    write_json(args.report_output, report)
    print(f"exported clips={len(exported)} skipped={len(skipped)}")
    print(f"manifest={args.output}")
    print(f"report={args.report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
