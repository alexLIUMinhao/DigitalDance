#!/usr/bin/env python3
"""Blender helper to inspect candidate FBX clips and emit lightweight metrics."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore
from mathutils import Vector  # type: ignore


ROOT_BONE_NAMES = {"hips", "mixamorig:hips", "pelvis"}
LEFT_FOOT_NAMES = {"leftfoot", "mixamorig:leftfoot", "foot_l"}
RIGHT_FOOT_NAMES = {"rightfoot", "mixamorig:rightfoot", "foot_r"}


def parse_payload() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Missing '--' payload separator")
    payload = argv[argv.index("--") + 1]
    return json.loads(payload)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_fbx(path: Path) -> bpy.types.Object | None:
    bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=True)
    for obj in bpy.data.objects:
        if obj.type == "ARMATURE":
            return obj
    return None


def resolve_action(armature: bpy.types.Object | None):
    if armature and armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    for action in bpy.data.actions:
        return action
    return None


def pose_bone_world_location(armature: bpy.types.Object, bone_names: set[str]) -> Vector | None:
    for pose_bone in armature.pose.bones:
        if pose_bone.name.lower() in bone_names:
            return armature.matrix_world @ pose_bone.matrix.translation
    return None


def root_delta(armature: bpy.types.Object) -> float:
    bone = pose_bone_world_location(armature, ROOT_BONE_NAMES)
    return bone.length if bone is not None else 0.0


def main() -> None:
    payload = parse_payload()
    input_path = Path(payload["inputPath"])
    reset_scene()

    armature = import_fbx(input_path)
    action = resolve_action(armature)
    if armature is None or action is None:
        print(json.dumps({"valid": False, "issues": ["missing_armature_or_action"]}))
        return

    frame_start = int(math.floor(action.frame_range[0]))
    frame_end = int(math.ceil(action.frame_range[1]))
    fps = float(bpy.context.scene.render.fps or 30.0)
    duration = max(0.0, (frame_end - frame_start) / fps)

    def sample_world_location(frame: int, names: set[str]) -> Vector | None:
        bpy.context.scene.frame_set(frame)
        return pose_bone_world_location(armature, names)

    start_root = sample_world_location(frame_start, ROOT_BONE_NAMES)
    end_root = sample_world_location(frame_end, ROOT_BONE_NAMES)
    start_left = sample_world_location(frame_start, LEFT_FOOT_NAMES)
    end_left = sample_world_location(frame_end, LEFT_FOOT_NAMES)
    start_right = sample_world_location(frame_start, RIGHT_FOOT_NAMES)
    end_right = sample_world_location(frame_end, RIGHT_FOOT_NAMES)

    loop_delta = 0.0
    if start_root is not None and end_root is not None:
        loop_delta = (end_root - start_root).length

    left_delta = (end_left - start_left).length if start_left is not None and end_left is not None else 0.0
    right_delta = (end_right - start_right).length if start_right is not None and end_right is not None else 0.0

    bone_names = {bone.name.lower() for bone in armature.pose.bones}
    mixamo_compatible = any(name in bone_names for name in ROOT_BONE_NAMES)

    payload = {
        "valid": True,
        "frameStart": frame_start,
        "frameEnd": frame_end,
        "frameCount": max(0, frame_end - frame_start + 1),
        "fps": fps,
        "durationSec": duration,
        "mixamoCompatibleGuess": mixamo_compatible,
        "loopDeltaScore": loop_delta,
        "rootTranslationDeltaMeters": loop_delta,
        "leftFootDeltaMeters": left_delta,
        "rightFootDeltaMeters": right_delta,
        "issues": [],
    }
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
