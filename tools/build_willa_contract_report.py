#!/usr/bin/env python3
"""Build a coordinate/rotation contract report for source BVH -> canonical BVH -> Willa."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.willa_contract_diagnostic import build_willa_contract_report  # noqa: E402
from music_motion_lab.willa_retarget import load_willa_retarget_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    project_root = REPO_ROOT / "music-motion-lab"
    profile = load_willa_retarget_profile(project_root=project_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(str(profile["profilePath"])),
    )
    parser.add_argument(
        "--source-bvh",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source.bvh",
    )
    parser.add_argument(
        "--canonical-bvh",
        type=Path,
        default=Path(str(profile["defaultInputBvh"])),
    )
    parser.add_argument(
        "--canonical-report",
        type=Path,
        default=Path(str(profile["defaultCanonicalReport"])),
    )
    parser.add_argument(
        "--willa-rest-dump",
        type=Path,
        default=project_root / "outputs" / "reports" / "willa_rest_contract_dump.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "reports" / "finedance_168_willa_contract_report.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_willa_contract_report(
        project_root=REPO_ROOT / "music-motion-lab",
        source_bvh=args.source_bvh.expanduser().resolve(),
        canonical_bvh=args.canonical_bvh.expanduser().resolve(),
        canonical_report_path=args.canonical_report.expanduser().resolve(),
        rest_dump_path=args.willa_rest_dump.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        profile_path=args.profile.expanduser().resolve(),
    )
    print(args.output.expanduser().resolve())
    print(payload["diagnosis"]["issues"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

