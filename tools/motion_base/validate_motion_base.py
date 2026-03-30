#!/usr/bin/env python3
"""Validate the isolated motion-base database and coverage targets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Dict, List

from common import (
    MOTION_INDEX_PATH,
    REVIEWS_PATH,
    asset_root_configured,
    load_library,
    load_taxonomy,
    motion_map,
    review_map,
    validate_motion_record,
    validate_review_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict-assets", action="store_true", help="Treat missing external FBX files as errors when local asset roots are configured.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not MOTION_INDEX_PATH.exists() or not REVIEWS_PATH.exists():
        print("ERROR: motion_base library files are missing. Seed or ingest records first.")
        return 1

    taxonomy = load_taxonomy()
    motion_index, review_index = load_library()
    motions = motion_index.get("motions", [])
    reviews = review_index.get("reviews", [])
    review_lookup = review_map(review_index)
    motion_lookup = motion_map(motion_index)

    errors: List[str] = []
    warnings: List[str] = []

    for record in motions:
        report = validate_motion_record(record, taxonomy)
        errors.extend(report.errors)
        if args.strict_assets:
            errors.extend(report.warnings)
        else:
            warnings.extend(report.warnings)

    for record in reviews:
        report = validate_review_record(record, taxonomy)
        errors.extend(report.errors)
        warnings.extend(report.warnings)

    for motion_id in motion_lookup:
        review = review_lookup.get(motion_id)
        if review is None:
            errors.append(f"{motion_id}: missing review record")
            continue
        if not str(review.get("reviewStatus", "")).strip():
            errors.append(f"{motion_id}: reviewStatus must be non-empty")

    for motion_id in review_lookup:
        if motion_id not in motion_lookup:
            errors.append(f"{motion_id}: review record exists without a motion record")

    style_counts = Counter()
    style_loop_counts = Counter()
    style_support_counts = Counter()
    style_meta_coverage: Dict[str, set[str]] = defaultdict(set)

    for record in motions:
        family = record["styleFamily"]
        style_counts[family] += 1
        style_meta_coverage[family].add(record["metaAction"])
        if record["role"] == "loop":
            style_loop_counts[family] += 1
        if record["transitionProfile"] in {"accent", "bridge"}:
            style_support_counts[family] += 1

    expected_styles = set(taxonomy.get("styleFamilies", []))
    expected_meta = set(taxonomy.get("metaActions", []))

    target_count = int(motion_index.get("targetCount", 48) or 48)
    if len(motions) < target_count:
        errors.append(f"Expected at least {target_count} motions, found {len(motions)}")

    for family in expected_styles:
        if style_counts[family] < 16:
            errors.append(f"{family}: expected at least 16 motions, found {style_counts[family]}")
        if style_loop_counts[family] < 12:
            errors.append(f"{family}: expected at least 12 loop phrases, found {style_loop_counts[family]}")
        if style_support_counts[family] < 4:
            errors.append(f"{family}: expected at least 4 accent/bridge phrases, found {style_support_counts[family]}")
        missing_meta = sorted(expected_meta - style_meta_coverage[family])
        if missing_meta:
            errors.append(f"{family}: missing metaAction coverage for {', '.join(missing_meta)}")

    approved_reviews = sum(1 for record in reviews if record.get("reviewStatus") == "approved")
    required_approved_reviews = min(target_count, len(motions))
    if approved_reviews < required_approved_reviews:
        errors.append(f"Expected at least {required_approved_reviews} approved reviews, found {approved_reviews}")

    print(f"motions={len(motions)} reviews={len(reviews)} approved_reviews={approved_reviews}")
    print("style counts:")
    for family in sorted(expected_styles):
        print(
            f"  {family}: total={style_counts[family]} "
            f"loops={style_loop_counts[family]} support={style_support_counts[family]} "
            f"meta={sorted(style_meta_coverage[family])}"
        )

    if errors:
        print("Validation: FAILED")
        for message in errors:
            print(f"  ERROR: {message}")
    else:
        print("Validation: PASSED")

    if not asset_root_configured():
        print("  WARN: local asset root not configured; external FBX existence checks were skipped.")

    for message in warnings:
        print(f"  WARN: {message}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
