#!/usr/bin/env python3
"""Export an AIST++-style SMPL motion pickle to BVH using CharacterAnimationTools."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.external_smpl_bvh_baselines import run_external_character_animation_tools  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--poses-path", type=Path, required=True)
    parser.add_argument("--smpl-path", type=Path, required=True)
    parser.add_argument("--output-bvh", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_external_character_animation_tools(
        repo_root=args.repo_root.expanduser().resolve(),
        poses_path=args.poses_path.expanduser().resolve(),
        smpl_path=args.smpl_path.expanduser().resolve(),
        output_bvh=args.output_bvh.expanduser().resolve(),
        fps=int(args.fps),
        scale=float(args.scale),
    )
    print(args.output_bvh.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
