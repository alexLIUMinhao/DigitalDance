#!/usr/bin/env python3
"""Build the Unity review feed and candidate review working records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from common import (
    CANDIDATE_METRICS_PATH,
    CANDIDATE_REVIEW_PATH,
    INTAKE_QUEUE_PATH,
    REVIEW_FEED_PATH,
    candidate_metrics_map,
    candidate_review_map,
    candidate_sort_key,
    infer_energy_band,
    infer_preferred_segments,
    infer_role,
    infer_transition_profile,
    load_candidate_metrics,
    load_candidate_review,
    load_intake_queue,
    load_taxonomy,
    titleize_slug,
    upsert_records,
    utc_now_iso,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTAKE_QUEUE_PATH, help="Input intake queue.")
    parser.add_argument("--metrics", type=Path, default=CANDIDATE_METRICS_PATH, help="Candidate metrics JSON.")
    parser.add_argument("--review-feed-output", type=Path, default=REVIEW_FEED_PATH, help="Output review feed.")
    parser.add_argument("--candidate-review-output", type=Path, default=CANDIDATE_REVIEW_PATH, help="Output candidate review work file.")
    return parser.parse_args()


def build_feed_entry(job: Dict[str, Any], candidate: Dict[str, Any], metrics_record: Dict[str, Any]) -> Dict[str, Any]:
    motion_id = candidate["candidateId"]
    return {
        "candidateId": candidate["candidateId"],
        "jobId": job["jobId"],
        "motionId": motion_id,
        "displayName": candidate.get("displayName", titleize_slug(motion_id)),
        "sourceLane": job["sourceLane"],
        "stage": job["stage"],
        "sourceVideoRelPath": job["sourceAssetRelPath"] if job.get("sourceAssetKind") == "video" else "",
        "candidateFbxRelPath": candidate["candidateFbxRelPath"],
        "targetStyleFamily": job.get("targetStyleFamily", ""),
        "targetStyleSubstyle": job.get("targetStyleSubstyle", ""),
        "expectedMetaAction": job.get("expectedMetaAction", ""),
        "transitionProfile": candidate.get("transitionProfile", infer_transition_profile(job.get("expectedMetaAction", ""))),
        "operatorNotes": job.get("operatorNotes", ""),
        "metricsSummary": dict(metrics_record.get("metricsSummary", {})),
    }


def build_candidate_review_entry(feed_entry: Dict[str, Any], existing: Dict[str, Any] | None, taxonomy: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    defaults = taxonomy["candidateReviewDefaults"]
    meta_action = feed_entry.get("expectedMetaAction", "")
    transition_profile = feed_entry["transitionProfile"]
    energy_band = candidate.get("energyBand", infer_energy_band(meta_action))
    role = candidate.get("role", infer_role(meta_action, transition_profile))
    phrase_beats = int(candidate.get("phraseBeats", 8) or 8)
    display_name = feed_entry["displayName"]

    base = {
        "candidateId": feed_entry["candidateId"],
        "jobId": feed_entry["jobId"],
        "motionId": feed_entry["motionId"],
        "displayName": display_name,
        "sourceLane": feed_entry["sourceLane"],
        "sourceVideoRelPath": feed_entry["sourceVideoRelPath"],
        "candidateFbxRelPath": feed_entry["candidateFbxRelPath"],
        "targetStyleFamily": feed_entry["targetStyleFamily"],
        "targetStyleSubstyle": feed_entry["targetStyleSubstyle"],
        "expectedMetaAction": meta_action,
        "transitionProfile": transition_profile,
        "energyBand": energy_band,
        "preferredSegments": infer_preferred_segments(meta_action, energy_band),
        "nativeBpm": float(candidate.get("nativeBpm", 120.0) or 120.0),
        "phraseBeats": phrase_beats,
        "entryOffsetsBeats": [int(candidate.get("entryOffsetBeats", 0) or 0)],
        "sliceBeatsOptions": [phrase_beats],
        "varietyGroup": f"{feed_entry['targetStyleFamily'] or 'unclassified'}_{meta_action or 'phrase'}",
        "role": role,
        "qualityTier": defaults["qualityTier"],
        "licenseTier": "prototype_only" if feed_entry["sourceLane"] == "video_rokoko" else "replace_before_ship",
        "decisionStatus": defaults["decisionStatus"],
        "reviewStatus": "",
        "loopSeam": defaults["loopSeam"],
        "footStability": defaults["footStability"],
        "styleClarity": defaults["styleClarity"],
        "tempoTolerance": defaults["tempoTolerance"],
        "reviewer": "",
        "issues": list(defaults["issues"]),
        "notes": feed_entry.get("operatorNotes", ""),
        "metricsSummary": feed_entry["metricsSummary"],
        "savedAtUtc": "",
    }

    if not existing:
        return base

    merged = dict(base)
    merged.update({key: value for key, value in existing.items() if value not in (None, "") or key in {"issues", "metricsSummary"}})
    merged["metricsSummary"] = feed_entry["metricsSummary"]
    return merged


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy()
    queue = load_intake_queue() if not args.input.exists() else json.loads(args.input.read_text(encoding="utf-8"))
    metrics = load_candidate_metrics() if not args.metrics.exists() else json.loads(args.metrics.read_text(encoding="utf-8"))
    existing_candidate_review = load_candidate_review() if not args.candidate_review_output.exists() else json.loads(args.candidate_review_output.read_text(encoding="utf-8"))

    metrics_by_candidate = candidate_metrics_map(metrics)
    existing_by_candidate = candidate_review_map(existing_candidate_review)

    feed_entries: List[Dict[str, Any]] = []
    candidate_entries: List[Dict[str, Any]] = []

    for job in queue.get("jobs", []):
        if job.get("stage") not in {"review_ready", "approved", "hold", "rejected"}:
            continue
        candidate_slices = job.get("artifactRelPaths", {}).get("candidateSlices", [])
        for candidate in candidate_slices:
            metrics_record = metrics_by_candidate.get(candidate["candidateId"])
            if metrics_record is None:
                continue
            if not metrics_record.get("passed", False) and job.get("stage") == "review_ready":
                continue
            feed_entry = build_feed_entry(job, candidate, metrics_record)
            feed_entries.append(feed_entry)
            candidate_entries.append(
                build_candidate_review_entry(feed_entry, existing_by_candidate.get(candidate["candidateId"]), taxonomy, candidate)
            )

    feed_payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "entries": sorted(feed_entries, key=candidate_sort_key),
    }
    merged_candidate_entries = upsert_records(existing_candidate_review.get("entries", []), candidate_entries, "candidateId", candidate_sort_key)
    candidate_payload = {
        "schemaVersion": 1,
        "entries": merged_candidate_entries,
    }

    write_json(args.review_feed_output, feed_payload)
    write_json(args.candidate_review_output, candidate_payload)
    print(f"review feed entries={len(feed_entries)} candidate review entries={len(merged_candidate_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
