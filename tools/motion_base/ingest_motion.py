#!/usr/bin/env python3
"""Upsert motion-base records into the isolated library."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from common import (
    CANDIDATE_REVIEW_PATH,
    INTAKE_QUEUE_PATH,
    MOTION_INDEX_PATH,
    REVIEWS_PATH,
    build_motion_record_from_candidate,
    build_review_record_from_candidate,
    candidate_review_map,
    candidate_sort_key,
    intake_job_map,
    load_json,
    load_candidate_review,
    load_intake_queue,
    load_library,
    load_taxonomy,
    motion_sort_key,
    upsert_records,
    validate_motion_record,
    validate_review_record,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-file", type=Path, help="JSON file with one motion record or a motions[] batch.")
    parser.add_argument("--review-file", type=Path, help="JSON file with one review record or a reviews[] batch.")
    parser.add_argument("--sync-approved", action="store_true", help="Ingest approved candidate review entries into the formal motion library.")
    parser.add_argument("--candidate-review-file", type=Path, default=CANDIDATE_REVIEW_PATH, help="Candidate review JSON used with --sync-approved.")
    parser.add_argument("--intake-queue-file", type=Path, default=INTAKE_QUEUE_PATH, help="Intake queue JSON used with --sync-approved.")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing.")
    return parser.parse_args()


def normalize_records(payload: Dict[str, Any], key: str) -> List[Dict[str, Any]]:
    if key in payload:
        records = payload[key]
        if not isinstance(records, list):
            raise ValueError(f"{key} must be a list")
        return [dict(record) for record in records]
    return [dict(payload)]


def load_manual_records(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not args.motion_file or not args.review_file:
        raise ValueError("--motion-file and --review-file are required unless --sync-approved is used")

    motion_payload = load_json(args.motion_file)
    review_payload = load_json(args.review_file)
    return normalize_records(motion_payload, "motions"), normalize_records(review_payload, "reviews")


def load_approved_candidate_records(args: argparse.Namespace) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidate_review = load_json(args.candidate_review_file) if args.candidate_review_file.exists() else load_candidate_review()
    intake_queue = load_json(args.intake_queue_file) if args.intake_queue_file.exists() else load_intake_queue()
    jobs_by_id = intake_job_map(intake_queue)

    motions: List[Dict[str, Any]] = []
    reviews: List[Dict[str, Any]] = []
    for entry in sorted(candidate_review_map(candidate_review).values(), key=candidate_sort_key):
        if entry.get("decisionStatus") != "approved" or entry.get("reviewStatus") != "approved":
            continue

        job = jobs_by_id.get(entry.get("jobId", ""))
        if job is not None and job.get("stage") not in {"approved", "review_ready"}:
            continue

        motions.append(build_motion_record_from_candidate(entry))
        reviews.append(build_review_record_from_candidate(entry))

    return motions, reviews


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy()
    if args.sync_approved:
        incoming_motions, incoming_reviews = load_approved_candidate_records(args)
    else:
        incoming_motions, incoming_reviews = load_manual_records(args)

    errors: List[str] = []
    warnings: List[str] = []
    for record in incoming_motions:
        report = validate_motion_record(record, taxonomy)
        errors.extend(report.errors)
        warnings.extend(report.warnings)

    for record in incoming_reviews:
        report = validate_review_record(record, taxonomy)
        errors.extend(report.errors)
        warnings.extend(report.warnings)

    if errors:
        for message in errors:
            print(f"ERROR: {message}")
        for message in warnings:
            print(f"WARN: {message}")
        return 1

    motion_index, review_index = load_library() if MOTION_INDEX_PATH.exists() and REVIEWS_PATH.exists() else (
        {"schemaVersion": 1, "batchId": "manual_ingest", "motions": []},
        {"schemaVersion": 1, "reviews": []},
    )

    merged_motions = upsert_records(motion_index.get("motions", []), incoming_motions, "motionId", motion_sort_key)
    merged_reviews = upsert_records(review_index.get("reviews", []), incoming_reviews, "motionId", motion_sort_key)

    if args.dry_run:
        print(f"dry-run motions={len(merged_motions)} reviews={len(merged_reviews)}")
        for message in warnings:
            print(f"WARN: {message}")
        return 0

    motion_index["motions"] = merged_motions
    review_index["reviews"] = merged_reviews
    write_json(MOTION_INDEX_PATH, motion_index)
    write_json(REVIEWS_PATH, review_index)

    print(f"written motions={len(merged_motions)} reviews={len(merged_reviews)}")
    for message in warnings:
        print(f"WARN: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
