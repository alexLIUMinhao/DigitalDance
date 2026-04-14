#!/usr/bin/env python3
"""Import a BVH into Blender GUI and align its global orientation to Willa."""

from __future__ import annotations

import json
import math
import sys
import traceback
from pathlib import Path

import addon_utils  # type: ignore
import bpy  # type: ignore


# User-confirmed world contract in Blender:
# - Willa stands at the origin
# - Up axis is +Z
# - Forward/facing direction is -Y
# The imported source BVH currently arrives with the trunk along +Y and facing +Z,
# so a global +90 deg rotation around X aligns the two world conventions.
WILLA_ALIGNMENT_EULER_XYZ_DEG = (90.0, 0.0, 0.0)


def parse_args() -> tuple[Path, Path | None]:
    if "--" not in sys.argv:
        raise SystemExit("Missing '--' payload separator")
    args = sys.argv[sys.argv.index("--") + 1 :]
    if not args:
        raise SystemExit("Usage: -- <input-bvh> [output-json]")
    input_path = Path(args[0]).expanduser().resolve()
    output_path = Path(args[1]).expanduser().resolve() if len(args) > 1 else None
    return input_path, output_path


def reset_scene() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_bvh_import_addon() -> None:
    addon_utils.enable("io_anim_bvh", default_set=False, persistent=False)


def import_bvh(input_path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(filepath=str(input_path), rotate_mode="NATIVE")
    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    imported_objects = [bpy.data.objects[name] for name in imported_names]
    armature = next((obj for obj in imported_objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise SystemExit(f"No armature found after importing {input_path}")
    return armature


def align_to_willa_world(armature: bpy.types.Object) -> None:
    armature.rotation_mode = "XYZ"
    armature.location = (0.0, 0.0, 0.0)
    armature.rotation_euler = tuple(math.radians(value) for value in WILLA_ALIGNMENT_EULER_XYZ_DEG)
    armature.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)


def write_payload(output_path: Path | None, payload: dict[str, object]) -> None:
    text = json.dumps(payload, indent=2) + "\n"
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    print(text)


def main() -> int:
    input_path, output_path = parse_args()
    def run_import_once() -> None | float:
        if not bpy.context.window_manager.windows:
            return 0.25
        try:
            reset_scene()
            ensure_bvh_import_addon()
            armature = import_bvh(input_path)
            align_to_willa_world(armature)
            action = armature.animation_data.action if armature.animation_data else None
            write_payload(
                output_path,
                {
                    "ok": True,
                    "inputPath": str(input_path),
                    "armatureName": str(armature.name),
                    "location": [float(value) for value in armature.location],
                    "rotationEulerDegXYZ": list(WILLA_ALIGNMENT_EULER_XYZ_DEG),
                    "scale": [float(value) for value in armature.scale],
                    "boneCount": int(len(armature.data.bones)),
                    "actionName": str(action.name) if action is not None else "",
                    "actionFrameRange": [
                        float(action.frame_range[0]),
                        float(action.frame_range[1]),
                    ] if action is not None else [],
                },
            )
        except Exception as exc:  # pragma: no cover - Blender runtime only
            write_payload(
                output_path,
                {
                    "ok": False,
                    "inputPath": str(input_path),
                    "errorType": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(traceback.format_exc())
        return None

    bpy.app.timers.register(run_import_once, first_interval=0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
