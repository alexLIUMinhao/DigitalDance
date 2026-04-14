#!/usr/bin/env python3
"""Dump Willa rest skeleton contract data from willa.blend."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import bpy  # type: ignore
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.willa_retarget import load_willa_retarget_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def choose_target_armature(profile: dict[str, Any]) -> bpy.types.Object:
    requested_name = str(profile.get("targetArmatureName", "") or "").strip()
    target_motion_root = str(profile["targetMotionRootBone"])
    mapped_target_bones = set(dict(profile.get("boneMap", {}) or {}).values())
    if requested_name:
        candidate = bpy.data.objects.get(requested_name)
        if candidate is not None and candidate.type == "ARMATURE":
            return candidate

    candidates = []
    for obj in bpy.data.objects:
        if obj.type != "ARMATURE":
            continue
        if target_motion_root not in obj.data.bones:
            continue
        score = sum(1 for name in mapped_target_bones if name in obj.data.bones)
        candidates.append((score, obj))
    if not candidates:
        raise RuntimeError(f"No target armature found containing motion root bone {target_motion_root}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def matrix_to_list(matrix: Any) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).round(6).tolist()


def vector_to_list(vector: Any) -> list[float]:
    return np.asarray(vector, dtype=np.float64).round(6).tolist()


def main() -> int:
    args = parse_args()
    profile = load_willa_retarget_profile(profile_path=Path(args.profile))
    armature = choose_target_armature(profile)
    target_names = set(list(profile.get("targetOrder", []) or []))
    target_names.add(str(profile.get("targetRootBone", "Root") or "Root"))
    target_names.add(str(profile.get("targetMotionRootBone", "") or ""))

    bones_payload: dict[str, Any] = {}
    for bone_name in sorted(target_names):
        bone = armature.data.bones.get(bone_name)
        if bone is None:
            continue
        local_matrix = bone.matrix_local.copy()
        if bone.parent is not None:
            parent_local = bone.parent.matrix_local.copy()
            relative = parent_local.inverted() @ local_matrix
        else:
            relative = local_matrix.copy()
        bones_payload[bone_name] = {
            "parent": bone.parent.name if bone.parent else None,
            "headLocal": vector_to_list(bone.head_local),
            "tailLocal": vector_to_list(bone.tail_local),
            "matrixLocal": matrix_to_list(local_matrix),
            "matrixRelativeToParent": matrix_to_list(relative),
            "rotationRelativeToParent": matrix_to_list(relative.to_3x3()),
            "length": float(bone.length),
        }

    payload = {
        "schemaVersion": 1,
        "blendPath": bpy.data.filepath,
        "armatureName": armature.name,
        "targetRootBone": str(profile.get("targetRootBone", "Root") or "Root"),
        "targetMotionRootBone": str(profile.get("targetMotionRootBone", "") or ""),
        "armatureMatrixWorld": matrix_to_list(armature.matrix_world),
        "bones": bones_payload,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

