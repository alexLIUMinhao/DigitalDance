#!/usr/bin/env python3
"""Register a Rokoko export file into the extracted-motion root and update the intake queue."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import INTAKE_QUEUE_PATH, load_intake_queue, resolve_asset_root, write_json, utc_now_iso


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True, help="Intake job id to update.")
    parser.add_argument("--input", type=Path, required=True, help="Exported Rokoko FBX/BVH file.")
    parser.add_argument("--output", type=Path, default=INTAKE_QUEUE_PATH, help="Updated intake queue path.")
    parser.add_argument("--dry-run", action="store_true", help="Show the normalized destination without writing.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue = load_intake_queue()
    extracted_root = resolve_asset_root("extractedMotion")
    if extracted_root is None:
        print("ERROR: extractedMotion root is not configured in asset_roots.local.json")
        return 1

    if not args.input.exists():
        print(f"ERROR: input export does not exist: {args.input}")
        return 1

    if args.input.suffix.lower() not in {".fbx", ".bvh"}:
        print("ERROR: Rokoko export must be .fbx or .bvh")
        return 1

    job = next((item for item in queue.get("jobs", []) if item.get("jobId") == args.job_id), None)
    if job is None:
        print(f"ERROR: job not found: {args.job_id}")
        return 1

    source_review = dict(job.get("sourceReview", {}) or {})
    if job.get("sourceLane") == "video_rokoko" and str(source_review.get("decisionStatus", "pending") or "pending") != "approved":
        print(f"ERROR: source review is not approved for {args.job_id}")
        return 1

    destination_relative = Path("video_rokoko") / args.job_id / f"{job['proposedMotionId']}{args.input.suffix.lower()}"
    destination_path = extracted_root / destination_relative

    if args.dry_run:
        print(f"would register: {args.input} -> {destination_path}")
        return 0

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if args.input.resolve() != destination_path.resolve():
        shutil.copy2(args.input, destination_path)

    artifact_paths = dict(job.get("artifactRelPaths", {}))
    artifact_paths["providerExportRelPath"] = destination_relative.as_posix()
    job["artifactRelPaths"] = artifact_paths
    job["stage"] = "extracted"
    job["updatedAtUtc"] = utc_now_iso()
    write_json(args.output, queue)

    print(f"registered export for {args.job_id}: {destination_relative.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
