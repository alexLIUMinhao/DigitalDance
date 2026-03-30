#!/usr/bin/env python3
"""Discover source assets and upsert intake jobs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from common import (
    DATASET_CATALOG_PATH,
    MOTION_EXTENSIONS,
    VIDEO_EXTENSIONS,
    INTAKE_QUEUE_PATH,
    SOURCE_CATALOG_PATH,
    discover_meta_action,
    discover_style_family,
    discover_style_substyle,
    job_sort_key,
    load_dataset_catalog,
    load_intake_queue,
    load_source_catalog,
    load_taxonomy,
    make_job_id,
    resolve_asset_root,
    slugify,
    titleize_slug,
    upsert_records,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=INTAKE_QUEUE_PATH, help="Target intake queue JSON path.")
    parser.add_argument("--catalog", type=Path, default=SOURCE_CATALOG_PATH, help="Source catalog JSON path used for source-video jobs.")
    parser.add_argument("--dataset-catalog", type=Path, default=DATASET_CATALOG_PATH, help="Dataset catalog JSON path used for dataset-first jobs.")
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


def source_review_defaults(existing: Dict[str, Any] | None) -> Dict[str, Any]:
    review = dict((existing or {}).get("sourceReview", {}) or {})
    return {
        "decisionStatus": str(review.get("decisionStatus", "pending") or "pending"),
        "reviewer": str(review.get("reviewer", "") or ""),
        "notes": str(review.get("notes", "") or ""),
    }


def source_provenance_defaults(existing: Dict[str, Any] | None, incoming: Dict[str, Any] | None) -> Dict[str, Any]:
    provenance = dict((existing or {}).get("sourceProvenance", {}) or {})
    provenance.update({key: value for key, value in (incoming or {}).items() if value not in (None, "")})
    return {
        "sourceId": str(provenance.get("sourceId", "") or ""),
        "provider": str(provenance.get("provider", "") or ""),
        "remoteAssetId": str(provenance.get("remoteAssetId", "") or ""),
        "sourcePageUrl": str(provenance.get("sourcePageUrl", "") or ""),
        "downloadUrl": str(provenance.get("downloadUrl", "") or ""),
        "creatorName": str(provenance.get("creatorName", "") or ""),
        "licenseName": str(provenance.get("licenseName", "") or ""),
        "query": str(provenance.get("query", "") or ""),
    }


def build_job(
    relative_path: Path,
    source_lane: str,
    source_asset_kind: str,
    taxonomy: Dict[str, Any],
    existing: Dict[str, Any] | None,
    source_path: Path | None = None,
    overrides: Dict[str, Any] | None = None,
    source_provenance: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    overrides = overrides or {}
    style_family = str(overrides.get("targetStyleFamily") or discover_style_family(relative_path.as_posix(), taxonomy))
    style_substyle = str(overrides.get("targetStyleSubstyle") or infer_substyle(relative_path, style_family))
    meta_action = str(overrides.get("expectedMetaAction") or discover_meta_action(relative_path.as_posix(), taxonomy))
    job_id = make_job_id(source_lane, relative_path.as_posix())
    fingerprint = fingerprint_for_path(source_path) if source_path is not None and source_path.exists() else None
    existing = existing or {}

    default_stage = "discovered"
    if existing.get("stage"):
        default_stage = existing["stage"]

    target_style_family = str(overrides.get("targetStyleFamily") or existing.get("targetStyleFamily", style_family) or style_family)
    target_style_substyle = str(overrides.get("targetStyleSubstyle") or existing.get("targetStyleSubstyle", style_substyle) or style_substyle)
    expected_meta_action = str(overrides.get("expectedMetaAction") or existing.get("expectedMetaAction", meta_action) or meta_action)
    proposed_motion_id = str(overrides.get("proposedMotionId") or existing.get("proposedMotionId", infer_motion_id(relative_path, style_family, style_substyle)) or infer_motion_id(relative_path, style_family, style_substyle))
    display_name = str(overrides.get("displayName") or existing.get("displayName", titleize_slug(relative_path.stem)) or titleize_slug(relative_path.stem))
    operator_notes = str(overrides.get("operatorNotes") or existing.get("operatorNotes", "") or "")

    return {
        "jobId": job_id,
        "sourceLane": source_lane,
        "sourceAssetKind": source_asset_kind,
        "sourceAssetRelPath": relative_path.as_posix(),
        "targetStyleFamily": target_style_family,
        "targetStyleSubstyle": target_style_substyle,
        "expectedMetaAction": expected_meta_action,
        "proposedMotionId": proposed_motion_id,
        "displayName": display_name,
        "stage": default_stage,
        "artifactRelPaths": dict(existing.get("artifactRelPaths", {})),
        "reviewFeedRelPath": str(existing.get("reviewFeedRelPath", "")),
        "operatorNotes": operator_notes,
        "sourceFingerprint": fingerprint or existing.get("sourceFingerprint", {}) or {},
        "sourceProvenance": source_provenance_defaults(existing, source_provenance),
        "sourceReview": source_review_defaults(existing),
        "discoveredAtUtc": existing.get("discoveredAtUtc", datetime.now(tz=timezone.utc).isoformat()),
        "updatedAtUtc": datetime.now(tz=timezone.utc).isoformat(),
    }


def dataset_source_lane(entry: Dict[str, Any]) -> str:
    source_format = str(entry.get("sourceFormat", "") or "")
    if source_format.startswith("json"):
        return "dataset_json"
    return "dataset_smpl"


def discover_dataset_jobs(queue: Dict[str, Any], taxonomy: Dict[str, Any], dataset_catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_fbx_root = resolve_asset_root("rawFbx")
    if raw_fbx_root is None or not raw_fbx_root.exists():
        return []

    existing_by_id = {job["jobId"]: job for job in queue.get("jobs", [])}
    discovered: List[Dict[str, Any]] = []
    for entry in sorted(
        dataset_catalog.get("entries", []),
        key=lambda item: f"{item.get('datasetName', '')}::{item.get('sequenceId', '')}",
    ):
        converted_rel_path = str(entry.get("convertedMotionRelPath", "") or "")
        if not converted_rel_path:
            continue

        converted_path = raw_fbx_root / converted_rel_path
        if not converted_path.exists() or converted_path.suffix.lower() not in MOTION_EXTENSIONS:
            continue

        rel_path = Path(converted_rel_path)
        source_lane = dataset_source_lane(entry)
        source_asset_kind = converted_path.suffix.lower().lstrip(".") or "fbx"
        provenance = {
            "sourceId": str(entry.get("entryId", "") or ""),
            "provider": str(entry.get("datasetName", "") or ""),
            "remoteAssetId": str(entry.get("sequenceId", "") or ""),
            "sourcePageUrl": str(entry.get("sourcePageUrl", "") or ""),
            "downloadUrl": "",
            "creatorName": "",
            "licenseName": str(entry.get("license", "") or ""),
            "query": str(entry.get("datasetName", "") or ""),
        }
        label_summary = dict(entry.get("labelSummary", {}) or {})
        note_parts = [
            f"dataset={entry.get('datasetName', '')}",
            f"sequence={entry.get('sequenceId', '')}",
        ]
        if label_summary:
            note_parts.append(f"labels={label_summary}")
        overrides = {
            "targetStyleFamily": str(entry.get("targetStyleFamily", "") or ""),
            "targetStyleSubstyle": str(entry.get("targetStyleSubstyle", "") or ""),
            "expectedMetaAction": str(entry.get("expectedMetaAction", "") or ""),
            "displayName": titleize_slug(f"{entry.get('datasetName', '')} {entry.get('sequenceId', '')}"),
            "operatorNotes": "\n".join(part for part in note_parts if part),
        }
        job_id = make_job_id(source_lane, rel_path.as_posix())
        job = build_job(
            rel_path,
            source_lane,
            source_asset_kind,
            taxonomy,
            existing_by_id.get(job_id),
            source_path=converted_path,
            overrides=overrides,
            source_provenance=provenance,
        )
        if job.get("stage") in {"discovered", "precheck_passed"}:
            job["stage"] = "retarget_ready"
        discovered.append(job)
    return discovered


def discover_catalog_jobs(queue: Dict[str, Any], taxonomy: Dict[str, Any], source_catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    source_video_root = resolve_asset_root("sourceVideo")
    if source_video_root is None or not source_video_root.exists():
        return []

    existing_by_id = {job["jobId"]: job for job in queue.get("jobs", [])}
    discovered: List[Dict[str, Any]] = []
    for source_entry in sorted(source_catalog.get("sources", []), key=lambda item: str(item.get("sourceId", ""))):
        rel_path = Path(str(source_entry.get("localVideoRelPath", "") or ""))
        if not rel_path.parts:
            continue
        source_path = source_video_root / rel_path
        if not source_path.exists() or source_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        provenance = {
            "sourceId": str(source_entry.get("sourceId", "") or ""),
            "provider": str(source_entry.get("provider", "") or ""),
            "remoteAssetId": str(source_entry.get("remoteAssetId", "") or ""),
            "sourcePageUrl": str(source_entry.get("sourcePageUrl", "") or ""),
            "downloadUrl": str(source_entry.get("downloadUrl", "") or ""),
            "creatorName": str(source_entry.get("creatorName", "") or ""),
            "licenseName": str(source_entry.get("licenseName", "") or ""),
            "query": str(source_entry.get("query", "") or ""),
        }
        overrides = {
            "targetStyleFamily": str(source_entry.get("targetStyleFamily", "") or ""),
            "targetStyleSubstyle": str(source_entry.get("targetStyleSubstyle", "") or ""),
            "expectedMetaAction": str(source_entry.get("expectedMetaAction", "") or ""),
            "displayName": titleize_slug(source_path.stem),
        }
        job_id = make_job_id("video_rokoko", rel_path.as_posix())
        discovered.append(
            build_job(
                rel_path,
                "video_rokoko",
                "video",
                taxonomy,
                existing_by_id.get(job_id),
                source_path=source_path,
                overrides=overrides,
                source_provenance=provenance,
            )
        )
    return discovered


def discover_root_jobs(queue: Dict[str, Any], taxonomy: Dict[str, Any], include_videos: bool) -> List[Dict[str, Any]]:
    existing_by_id = {job["jobId"]: job for job in queue.get("jobs", [])}
    discovered: List[Dict[str, Any]] = []
    scans = [("rawFbx", "curated_fbx", "fbx", MOTION_EXTENSIONS)]
    if include_videos:
        scans.append(("sourceVideo", "video_rokoko", "video", VIDEO_EXTENSIONS))
    for root_name, source_lane, source_asset_kind, extensions in scans:
        root = resolve_asset_root(root_name)
        if root is None or not root.exists():
            continue

        for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file() and candidate.suffix.lower() in extensions):
            relative_path = path.relative_to(root)
            if root_name == "rawFbx" and relative_path.parts and relative_path.parts[0] == "datasets":
                continue
            job_id = make_job_id(source_lane, relative_path.as_posix())
            discovered.append(
                build_job(
                    relative_path,
                    source_lane,
                    source_asset_kind,
                    taxonomy,
                    existing_by_id.get(job_id),
                    source_path=path,
                )
            )
    return discovered


def discover_jobs(queue: Dict[str, Any], taxonomy: Dict[str, Any], source_catalog: Dict[str, Any], dataset_catalog: Dict[str, Any]) -> List[Dict[str, Any]]:
    discovered = discover_dataset_jobs(queue, taxonomy, dataset_catalog)
    discovered.extend(discover_catalog_jobs(queue, taxonomy, source_catalog))
    include_video_scan = len([job for job in discovered if job.get("sourceLane") == "video_rokoko"]) == 0
    discovered.extend(discover_root_jobs(queue, taxonomy, include_video_scan))

    source_video_root = resolve_asset_root("sourceVideo")
    raw_fbx_root = resolve_asset_root("rawFbx")
    preserved = []
    discovered_ids = {job["jobId"] for job in discovered}
    for job in queue.get("jobs", []):
        if job["jobId"] not in discovered_ids:
            if (
                job.get("sourceAssetKind") == "video"
                and source_video_root is not None
                and job.get("stage") in {"discovered", "precheck_passed", "hold", "rejected", "extraction_pending"}
            ):
                source_path = source_video_root / Path(str(job.get("sourceAssetRelPath", "") or ""))
                has_downstream_artifacts = bool((job.get("artifactRelPaths", {}) or {}).get("providerExportRelPath")) or bool((job.get("artifactRelPaths", {}) or {}).get("candidateSlices"))
                if not source_path.exists() and not has_downstream_artifacts:
                    continue
            if (
                job.get("sourceLane") in {"dataset_smpl", "dataset_json"}
                and raw_fbx_root is not None
                and job.get("stage") in {"discovered", "precheck_passed", "hold", "rejected", "extraction_pending"}
            ):
                source_path = raw_fbx_root / Path(str(job.get("sourceAssetRelPath", "") or ""))
                has_downstream_artifacts = bool((job.get("artifactRelPaths", {}) or {}).get("providerExportRelPath")) or bool((job.get("artifactRelPaths", {}) or {}).get("candidateSlices"))
                if not source_path.exists() and not has_downstream_artifacts:
                    continue
            preserved.append(job)
    return preserved + discovered


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy()
    queue = load_intake_queue()
    source_catalog = load_source_catalog() if args.catalog.exists() else {"schemaVersion": 1, "sources": []}
    dataset_catalog = load_dataset_catalog() if args.dataset_catalog.exists() else {"schemaVersion": 1, "entries": []}
    discovered_jobs = discover_jobs(queue, taxonomy, source_catalog, dataset_catalog)
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
