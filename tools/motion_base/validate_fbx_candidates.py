#!/usr/bin/env python3
"""Validate sliced candidate FBX clips and emit static metrics."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from common import (
    CANDIDATE_METRICS_PATH,
    INTAKE_QUEUE_PATH,
    command_exists,
    load_intake_queue,
    resolve_asset_path,
    resolve_asset_root,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTAKE_QUEUE_PATH, help="Input intake queue.")
    parser.add_argument("--output", type=Path, default=CANDIDATE_METRICS_PATH, help="Candidate metrics output JSON.")
    parser.add_argument("--queue-output", type=Path, default=INTAKE_QUEUE_PATH, help="Updated intake queue path.")
    parser.add_argument("--blender", default="blender", help="Blender executable.")
    return parser.parse_args()


def helper_script_path() -> Path:
    return Path(__file__).with_name("blender_candidate_metrics.py")


def inspect_candidate(blender_exe: str, candidate_path: Path) -> Dict[str, Any]:
    command = [
        blender_exe,
        "--background",
        "--python",
        str(helper_script_path()),
        "--",
        json.dumps({"inputPath": str(candidate_path)}),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    stdout = result.stdout.strip().splitlines()
    if not stdout:
        raise RuntimeError("Blender metrics helper did not return JSON output")
    return json.loads(stdout[-1])


def candidate_passes(candidate: Dict[str, Any], metrics: Dict[str, Any]) -> tuple[bool, List[str]]:
    issues = list(metrics.get("issues", []))
    if not metrics.get("valid", False):
        issues.append("invalid_candidate")
        return False, issues

    if int(metrics.get("frameCount", 0) or 0) < 2:
        issues.append("frame_count_too_low")
    if float(metrics.get("durationSec", 0.0) or 0.0) <= 0.0:
        issues.append("duration_zero")
    if not bool(metrics.get("mixamoCompatibleGuess", False)):
        issues.append("not_mixamo_compatible")

    phrase_beats = int(candidate.get("phraseBeats", 8) or 8)
    loop_threshold = 0.65 if phrase_beats <= 4 else 0.35
    if float(metrics.get("loopDeltaScore", 0.0) or 0.0) > loop_threshold:
        issues.append("loop_delta_too_high")

    foot_delta = max(float(metrics.get("leftFootDeltaMeters", 0.0) or 0.0), float(metrics.get("rightFootDeltaMeters", 0.0) or 0.0))
    if foot_delta > 0.15:
        issues.append("foot_drift_too_high")

    return len(issues) == 0, issues


def main() -> int:
    args = parse_args()
    queue = load_intake_queue() if not args.input.exists() else json.loads(args.input.read_text(encoding="utf-8"))
    approved_root = resolve_asset_root("approvedFbx")
    if approved_root is None:
        print("ERROR: approvedFbx root is not configured in asset_roots.local.json")
        return 1
    if not command_exists(args.blender):
        print(f"ERROR: Blender executable not found: {args.blender}")
        return 1

    metrics_output: List[Dict[str, Any]] = []
    updated_jobs = 0
    for job in queue.get("jobs", []):
        candidate_slices = list(job.get("artifactRelPaths", {}).get("candidateSlices", []))
        if not candidate_slices:
            continue

        any_passed = False
        for candidate in candidate_slices:
            candidate_rel_path = candidate["candidateFbxRelPath"]
            candidate_path = approved_root / candidate_rel_path
            metrics: Dict[str, Any]
            if not candidate_path.exists():
                metrics = {
                    "valid": False,
                    "frameCount": 0,
                    "durationSec": 0.0,
                    "mixamoCompatibleGuess": False,
                    "loopDeltaScore": 0.0,
                    "rootTranslationDeltaMeters": 0.0,
                    "leftFootDeltaMeters": 0.0,
                    "rightFootDeltaMeters": 0.0,
                    "issues": ["missing_candidate_file"],
                }
            else:
                metrics = inspect_candidate(args.blender, candidate_path)

            passed, issues = candidate_passes(candidate, metrics)
            metrics_record = {
                "candidateId": candidate["candidateId"],
                "jobId": job["jobId"],
                "candidateFbxRelPath": candidate_rel_path,
                "passed": passed,
                "issues": issues,
                "generatedAtUtc": utc_now_iso(),
                "metricsSummary": {
                    "frameCount": metrics.get("frameCount", 0),
                    "durationSec": metrics.get("durationSec", 0.0),
                    "fps": metrics.get("fps", 0.0),
                    "mixamoCompatibleGuess": metrics.get("mixamoCompatibleGuess", False),
                    "loopDeltaScore": metrics.get("loopDeltaScore", 0.0),
                    "rootTranslationDeltaMeters": metrics.get("rootTranslationDeltaMeters", 0.0),
                    "leftFootDeltaMeters": metrics.get("leftFootDeltaMeters", 0.0),
                    "rightFootDeltaMeters": metrics.get("rightFootDeltaMeters", 0.0),
                },
            }
            metrics_output.append(metrics_record)
            any_passed = any_passed or passed

        artifact_paths = dict(job.get("artifactRelPaths", {}))
        artifact_paths["candidateMetricsRelPath"] = str(args.output.relative_to(args.output.parents[2]).as_posix()) if args.output.is_absolute() else args.output.as_posix()
        job["artifactRelPaths"] = artifact_paths
        job["stage"] = "review_ready" if any_passed else "hold"
        job["updatedAtUtc"] = utc_now_iso()
        updated_jobs += 1

    payload = {"schemaVersion": 1, "generatedAtUtc": utc_now_iso(), "metrics": metrics_output}
    write_json(args.output, payload)
    write_json(args.queue_output, queue)
    print(f"validated candidates={len(metrics_output)} updated_jobs={updated_jobs}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
