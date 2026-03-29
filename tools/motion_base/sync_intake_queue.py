#!/usr/bin/env python3
"""Discover source assets and upsert intake jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from common import (
    INTAKE_QUEUE_PATH,
    MOTION_EXTENSIONS,
    VIDEO_EXTENSIONS,
    discover_meta_action,
    discover_style_family,
    discover_style_substyle,
    job_sort_key,
    load_intake_queue,
    load_taxonomy,
    make_job_id,
    slugify,
    titleize_slug,
    upsert_records,
    write_json,
    resolve_asset_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=INTAKE_QUEUE_PATH, help="Target intake queue JSON path.")
    parser.add_argument("--dry-run", action="store_true", help="Discover jobs without writing.")
    return parser.parse_args()


def fingerprint_for_path(path: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "sizeBytes": stat.st_size,
        "modifiedAtUtc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
    }


def infer_substyle(path: Path, style_family: str) -> str:
    discovered = discover_style_substyle(path.as_posix(), style_family)
    if discovered:
        return discovered
    if path.parent.name and path.parent.name != ".":
        return slugify(path.parent.name)
    return ""


def infer_motion_id(path: Path, style_family: str, style_substyle: str) -> str:
    stem = slugify(path.stem)
    if style_family and style_substyle:
        return f"{style_family}_{style_substyle}_{stem}"
    if style_family:
        return f"{style_family}_{stem}"
    return stem


def build_job(relative_path: Path, source_lane: str, source_asset_kind: str, taxonomy: Dict[str, Any], existing: Dict[str, Any] | None) -> Dict[str, Any]:
    style_family = discover_style_family(relative_path.as_posix(), taxonomy)
    style_substyle = infer_substyle(relative_path, style_family)
    meta_action = discover_meta_action(relative_path.as_posix(), taxonomy)
    job_id = make_job_id(source_lane, relative_path.as_posix())
    fingerprint = fingerprint_for_path(resolve_asset_root("sourceVideo") / relative_path) if source_asset_kind == "video" and resolve_asset_root("sourceVideo") else None
    if source_asset_kind != "video":
        root_name = "rawFbx"
        root = resolve_asset_root(root_name)
        fingerprint = fingerprint_for_path(root / relative_path) if root else None

    default_stage = "discovered"
    if existing and existing.get("stage"):
        default_stage = existing["stage"]

    return {
        "jobId": job_id,
        "sourceLane": source_lane,
        "sourceAssetKind": source_asset_kind,
        "sourceAssetRelPath": relative_path.as_posix(),
        "targetStyleFamily": existing.get("targetStyleFamily", style_family) if existing else style_family,
        "targetStyleSubstyle": existing.get("targetStyleSubstyle", style_substyle) if existing else style_substyle,
        "expectedMetaAction": existing.get("expectedMetaAction", meta_action) if existing else meta_action,
        "proposedMotionId": existing.get("proposedMotionId", infer_motion_id(relative_path, style_family, style_substyle)) if existing else infer_motion_id(relative_path, style_family, style_substyle),
        "displayName": existing.get("displayName", titleize_slug(relative_path.stem)) if existing else titleize_slug(relative_path.stem),
        "stage": default_stage,
        "artifactRelPaths": dict(existing.get("artifactRelPaths", {})) if existing else {},
        "reviewFeedRelPath": existing.get("reviewFeedRelPath", "") if existing else "",
        "operatorNotes": existing.get("operatorNotes", "") if existing else "",
        "sourceFingerprint": fingerprint or existing.get("sourceFingerprint", {}) if existing else fingerprint or {},
        "discoveredAtUtc": existing.get("discoveredAtUtc", datetime.now(tz=timezone.utc).isoformat()) if existing else datetime.now(tz=timezone.utc).isoformat(),
        "updatedAtUtc": datetime.now(tz=timezone.utc).isoformat(),
    }


def discover_jobs(queue: Dict[str, Any], taxonomy: Dict[str, Any]) -> List[Dict[str, Any]]:
    existing_by_id = {job["jobId"]: job for job in queue.get("jobs", [])}
    discovered: List[Dict[str, Any]] = []

    scans = [
        ("sourceVideo", "video_rokoko", "video", VIDEO_EXTENSIONS),
        ("rawFbx", "curated_fbx", "fbx", MOTION_EXTENSIONS),
    ]

    for root_name, source_lane, source_asset_kind, extensions in scans:
        root = resolve_asset_root(root_name)
        if root is None or not root.exists():
            continue

        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() in extensions):
            relative_path = path.relative_to(root)
            job_id = make_job_id(source_lane, relative_path.as_posix())
            discovered.append(build_job(relative_path, source_lane, source_asset_kind, taxonomy, existing_by_id.get(job_id)))

    preserved = []
    discovered_ids = {job["jobId"] for job in discovered}
    for job in queue.get("jobs", []):
        if job["jobId"] not in discovered_ids:
            preserved.append(job)

    return preserved + discovered


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy()
    queue = load_intake_queue()
    discovered_jobs = discover_jobs(queue, taxonomy)
    merged_jobs = upsert_records(queue.get("jobs", []), discovered_jobs, "jobId", job_sort_key)
    payload = {"schemaVersion": 1, "jobs": merged_jobs}

    if args.dry_run:
        print(f"discovered jobs={len(merged_jobs)}")
        return 0

    write_json(args.output, payload)
    print(f"written intake jobs={len(merged_jobs)} to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
