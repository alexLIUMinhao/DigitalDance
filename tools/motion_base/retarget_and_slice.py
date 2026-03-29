#!/usr/bin/env python3
"""Run Blender in batch mode to normalize and slice candidate motion clips."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from common import (
    INTAKE_QUEUE_PATH,
    command_exists,
    infer_energy_band,
    infer_role,
    infer_transition_profile,
    load_intake_queue,
    make_candidate_id,
    resolve_asset_path,
    resolve_asset_root,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTAKE_QUEUE_PATH, help="Input intake queue.")
    parser.add_argument("--output", type=Path, default=INTAKE_QUEUE_PATH, help="Output intake queue.")
    parser.add_argument("--job-id", action="append", help="Optional specific job id to process.")
    parser.add_argument("--blender", default="blender", help="Blender executable.")
    parser.add_argument("--dry-run", action="store_true", help="Print Blender commands without executing them.")
    return parser.parse_args()


def helper_script_path() -> Path:
    return Path(__file__).with_name("blender_retarget_slice.py")


def resolve_input_motion_path(job: Dict[str, Any]) -> Path | None:
    if job.get("sourceLane") == "video_rokoko":
        provider_rel_path = job.get("artifactRelPaths", {}).get("providerExportRelPath", "")
        if not provider_rel_path:
            return None
        return resolve_asset_path("extractedMotion", provider_rel_path)
    return resolve_asset_path("rawFbx", job.get("sourceAssetRelPath", ""))


def candidate_phrase_beats(job: Dict[str, Any]) -> List[int]:
    meta_action = job.get("expectedMetaAction", "")
    if meta_action == "accent_hit":
        return [4]
    if meta_action in {"pose_hold", "arm_expression"}:
        return [8, 16]
    return [4, 8, 16]


def candidate_entry_offsets(job: Dict[str, Any]) -> List[int]:
    meta_action = job.get("expectedMetaAction", "")
    if meta_action in {"pose_hold", "arm_expression"}:
        return [0]
    return [0, 4]


def run_blender(blender_exe: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    command = [
        blender_exe,
        "--background",
        "--python",
        str(helper_script_path()),
        "--",
        json.dumps(payload),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError("Blender helper did not return JSON output")
    return json.loads(stdout[-1])


def main() -> int:
    args = parse_args()
    queue = load_intake_queue() if not args.input.exists() else json.loads(args.input.read_text(encoding="utf-8"))
    approved_root = resolve_asset_root("approvedFbx")
    if approved_root is None:
        print("ERROR: approvedFbx root is not configured in asset_roots.local.json")
        return 1

    if not args.dry_run and not command_exists(args.blender):
        print(f"ERROR: Blender executable not found: {args.blender}")
        return 1

    updated = 0
    requested_ids = set(args.job_id or [])
    for job in queue.get("jobs", []):
        if requested_ids and job.get("jobId") not in requested_ids:
            continue
        if job.get("stage") not in {"precheck_passed", "extracted", "retarget_ready", "hold"}:
            continue

        source_path = resolve_input_motion_path(job)
        if source_path is None:
            if job.get("sourceLane") == "video_rokoko":
                job["stage"] = "extraction_pending"
                job["updatedAtUtc"] = utc_now_iso()
                updated += 1
            continue
        if not source_path.exists():
            job["stage"] = "hold"
            job["operatorNotes"] = (job.get("operatorNotes", "") + "\nmissing input motion for retarget_and_slice").strip()
            job["updatedAtUtc"] = utc_now_iso()
            updated += 1
            continue

        payload = {
            "inputPath": str(source_path),
            "outputDir": str(approved_root / (job.get("targetStyleFamily", "") or "unclassified") / (job.get("targetStyleSubstyle", "") or "misc")),
            "motionPrefix": job.get("proposedMotionId", source_path.stem),
            "nativeBpm": float(job.get("nativeBpm", 120.0) or 120.0),
            "phraseBeats": candidate_phrase_beats(job),
            "entryOffsetsBeats": candidate_entry_offsets(job),
        }

        if args.dry_run:
            print(json.dumps(payload, ensure_ascii=False))
            continue

        helper_output = run_blender(args.blender, payload)
        candidates = []
        for index, item in enumerate(helper_output.get("candidates", []), start=1):
            candidate_relative_path = Path(item["filePath"]).relative_to(approved_root).as_posix()
            phrase_beats = int(item["phraseBeats"])
            entry_offset = int(item["entryOffsetBeats"])
            candidate_id = item.get("candidateId") or make_candidate_id(job["proposedMotionId"], phrase_beats, entry_offset, index)
            transition_profile = infer_transition_profile(job.get("expectedMetaAction", ""))
            candidates.append(
                {
                    "candidateId": candidate_id,
                    "candidateFbxRelPath": candidate_relative_path,
                    "displayName": item.get("displayName", candidate_id.replace("_", " ").title()),
                    "phraseBeats": phrase_beats,
                    "entryOffsetBeats": entry_offset,
                    "sliceIndex": index,
                    "transitionProfile": transition_profile,
                    "energyBand": infer_energy_band(job.get("expectedMetaAction", "")),
                    "role": infer_role(job.get("expectedMetaAction", ""), transition_profile),
                    "nativeBpm": float(payload["nativeBpm"]),
                }
            )

        artifact_paths = dict(job.get("artifactRelPaths", {}))
        artifact_paths["candidateSlices"] = candidates
        job["artifactRelPaths"] = artifact_paths
        job["stage"] = "sliced" if candidates else "hold"
        job["updatedAtUtc"] = utc_now_iso()
        updated += 1

    if not args.dry_run:
        write_json(args.output, queue)
    print(f"retarget_and_slice updated_jobs={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
