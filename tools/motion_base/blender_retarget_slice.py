#!/usr/bin/env python3
"""Blender helper to slice Mixamo-compatible motion files into phrase candidates."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import bpy  # type: ignore


def parse_payload() -> dict:
    argv = sys.argv
    if "--" not in argv:
        raise SystemExit("Missing '--' payload separator")
    payload = argv[argv.index("--") + 1]
    return json.loads(payload)


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_motion(path: Path) -> tuple[list[bpy.types.Object], bpy.types.Object | None]:
    before = set(bpy.data.objects.keys())
    if path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=True)
    elif path.suffix.lower() == ".bvh":
        bpy.ops.import_anim.bvh(filepath=str(path), rotate_mode="NATIVE")
    else:
        raise SystemExit(f"Unsupported motion input: {path.suffix}")

    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    imported_objects = [bpy.data.objects[name] for name in imported_names]
    armature = next((obj for obj in imported_objects if obj.type == "ARMATURE"), None)
    return imported_objects, armature


def resolve_action(armature: bpy.types.Object | None):
    if armature and armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    for action in bpy.data.actions:
        return action
    return None


def export_slice(objects: list[bpy.types.Object], file_path: Path, frame_start: int, frame_end: int) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None
    bpy.ops.export_scene.fbx(
        filepath=str(file_path),
        use_selection=True,
        bake_anim=True,
        bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False,
        bake_anim_simplify_factor=0.0,
        bake_anim_step=1.0,
        bake_anim_force_startend_keying=True,
        bake_anim_start=frame_start,
        bake_anim_end=frame_end,
        add_leaf_bones=False,
    )


def main() -> None:
    payload = parse_payload()
    input_path = Path(payload["inputPath"])
    output_dir = Path(payload["outputDir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    reset_scene()
    objects, armature = import_motion(input_path)
    action = resolve_action(armature)
    if action is None:
        print(json.dumps({"candidates": [], "warnings": ["no_action_found"]}))
        return

    frame_start = int(math.floor(action.frame_range[0]))
    frame_end = int(math.ceil(action.frame_range[1]))
    fps = float(bpy.context.scene.render.fps or 30.0)
    native_bpm = float(payload.get("nativeBpm", 120.0) or 120.0)
    frames_per_beat = max(1.0, fps * 60.0 / native_bpm)

    candidates = []
    slice_index = 0
    for phrase_beats in payload.get("phraseBeats", [8]):
        for entry_offset in payload.get("entryOffsetsBeats", [0]):
            start_frame = int(round(frame_start + (float(entry_offset) * frames_per_beat)))
            end_frame = int(round(start_frame + (float(phrase_beats) * frames_per_beat) - 1))
            if end_frame > frame_end:
                continue

            slice_index += 1
            candidate_id = f"{payload['motionPrefix']}_p{int(phrase_beats):02d}_o{int(entry_offset):02d}_{slice_index:02d}"
            output_path = output_dir / f"{candidate_id}.fbx"
            export_slice(objects, output_path, start_frame, end_frame)
            candidates.append(
                {
                    "candidateId": candidate_id,
                    "displayName": candidate_id.replace("_", " ").title(),
                    "filePath": str(output_path),
                    "phraseBeats": int(phrase_beats),
                    "entryOffsetBeats": int(entry_offset),
                }
            )

    print(json.dumps({"candidates": candidates, "warnings": []}, ensure_ascii=False))


if __name__ == "__main__":
    main()
