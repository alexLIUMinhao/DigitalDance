#!/usr/bin/env python3
"""Minimal Blender smoke test for importing a BVH clip."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import bpy  # type: ignore


def parse_args() -> tuple[Path, Path | None]:
    if "--" not in sys.argv:
        raise SystemExit("Missing '--' separator")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if not args:
        raise SystemExit("Usage: -- <input-bvh> [output-json]")
    input_path = Path(args[0]).expanduser().resolve()
    output_path = Path(args[1]).expanduser().resolve() if len(args) > 1 else None
    return input_path, output_path


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_bvh(input_path: Path):
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(filepath=str(input_path), rotate_mode="NATIVE")
    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    imported_objects = [bpy.data.objects[name] for name in imported_names]
    armature = next((obj for obj in imported_objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise SystemExit(f"No armature found after importing {input_path}")
    return armature


def main() -> int:
    input_path, output_path = parse_args()
    reset_scene()
    armature = import_bvh(input_path)
    action = armature.animation_data.action if armature.animation_data else None
    payload = {
        "ok": True,
        "inputPath": str(input_path),
        "armatureName": str(armature.name),
        "boneCount": int(len(armature.data.bones)),
        "boneNames": [str(bone.name) for bone in armature.data.bones],
        "actionName": str(action.name) if action is not None else "",
        "actionFrameRange": [
            float(action.frame_range[0]),
            float(action.frame_range[1]),
        ] if action is not None else [],
    }
    text = json.dumps(payload, indent=2) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
