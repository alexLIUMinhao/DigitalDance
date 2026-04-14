#!/usr/bin/env python3
"""Render a simple motion preview from a preview_job using Blender."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector


PARENT_BY_BONE = {
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
}


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in bpy.app.argv:
        argv = bpy.app.argv[bpy.app.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-job", required=True)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--output-video")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--render-animation", action="store_true")
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def unity_to_blender(coords: list[float]) -> tuple[float, float, float]:
    x, y, z = coords
    return (x, z, y)


def positions_to_frames(payload: dict, start: int, end_exclusive: int, frame_stride: int) -> tuple[list[str], list[list[tuple[float, float, float]]]]:
    bone_order = list(payload["boneOrder"])
    joint_count = int(payload["jointCount"])
    flat_positions = list(payload["positions"])
    frame_count = int(payload["frameCount"])

    start = max(0, min(start, frame_count - 1))
    end_exclusive = max(start + 1, min(end_exclusive, frame_count))
    frames: list[list[tuple[float, float, float]]] = []
    segment_start_pelvis = None

    for frame_index in range(start, end_exclusive, max(1, frame_stride)):
        offset = frame_index * joint_count * 3
        frame_points: list[tuple[float, float, float]] = []
        for joint_index in range(joint_count):
            base = offset + joint_index * 3
            coords = unity_to_blender(flat_positions[base : base + 3])
            frame_points.append(coords)
        if segment_start_pelvis is None:
            segment_start_pelvis = frame_points[0]
        pelvis_x, pelvis_y, _ = segment_start_pelvis
        normalized = []
        for x, y, z in frame_points:
            normalized.append((x - pelvis_x, y - pelvis_y, z))
        frames.append(normalized)
    return bone_order, frames


def collect_preview_frames(preview_job: dict, max_steps: int, frame_stride: int) -> tuple[list[str], list[list[tuple[float, float, float]]]]:
    bone_order: list[str] | None = None
    all_frames: list[list[tuple[float, float, float]]] = []
    selected_steps = list(preview_job.get("steps", []))
    if max_steps > 0:
        selected_steps = selected_steps[: max_steps]

    for step in selected_steps:
        pose_driver_path = Path(step["reference_artifacts"]["pose_driver_json"])
        frame_range = dict(step.get("frame_range", {}))
        payload = load_json(pose_driver_path)
        current_bone_order, frames = positions_to_frames(
            payload=payload,
            start=int(frame_range.get("start", 0)),
            end_exclusive=int(frame_range.get("end_exclusive", 1)),
            frame_stride=frame_stride,
        )
        if bone_order is None:
            bone_order = current_bone_order
        if current_bone_order != bone_order:
            raise ValueError(f"Incompatible bone order in {pose_driver_path}")
        all_frames.extend(frames)
    if not bone_order or not all_frames:
        raise ValueError("No preview frames could be collected from preview_job")
    return bone_order, all_frames


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def make_material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 0.45
    shader.inputs["Emission Color"].default_value = color
    shader.inputs["Emission Strength"].default_value = 0.25
    return material


def create_joint_objects(bone_order: list[str], material: bpy.types.Material) -> dict[str, bpy.types.Object]:
    objects: dict[str, bpy.types.Object] = {}
    for bone_name in bone_order:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.035, location=(0.0, 0.0, 0.0), segments=16, ring_count=8)
        obj = bpy.context.object
        obj.name = f"joint_{bone_name}"
        obj.data.materials.append(material)
        objects[bone_name] = obj
    return objects


def create_bone_objects(bone_order: list[str], material: bpy.types.Material) -> dict[str, bpy.types.Object]:
    objects: dict[str, bpy.types.Object] = {}
    for bone_name in bone_order:
        parent_name = PARENT_BY_BONE.get(bone_name)
        if not parent_name:
            continue
        bpy.ops.mesh.primitive_cylinder_add(radius=0.018, depth=1.0, location=(0.0, 0.0, 0.0), vertices=10)
        obj = bpy.context.object
        obj.name = f"bone_{parent_name}_{bone_name}"
        obj.data.materials.append(material)
        objects[bone_name] = obj
    return objects


def animate_objects(
    bone_order: list[str],
    frames: list[list[tuple[float, float, float]]],
    joint_objects: dict[str, bpy.types.Object],
    bone_objects: dict[str, bpy.types.Object],
) -> None:
    for frame_number, frame_points in enumerate(frames, start=1):
        positions = dict(zip(bone_order, frame_points))
        for bone_name, obj in joint_objects.items():
            obj.location = positions[bone_name]
            obj.keyframe_insert(data_path="location", frame=frame_number)

        for bone_name, obj in bone_objects.items():
            parent_name = PARENT_BY_BONE[bone_name]
            start = Vector(positions[parent_name])
            end = Vector(positions[bone_name])
            delta = end - start
            length = max(delta.length, 1e-5)
            obj.location = (start + end) / 2.0
            obj.scale = (1.0, 1.0, length / 2.0)
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = delta.to_track_quat("Z", "Y")
            obj.keyframe_insert(data_path="location", frame=frame_number)
            obj.keyframe_insert(data_path="scale", frame=frame_number)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_number)


def configure_scene(frame_count: int, fps: int, output_video: Path | None) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = 960
    scene.render.resolution_y = 540
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = frame_count
    if output_video is not None:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.filepath = str(output_video)
    scene.world.color = (0.02, 0.02, 0.03)


def add_camera_and_lights() -> None:
    bpy.ops.object.camera_add(location=(0.0, -7.5, 2.8), rotation=(math.radians(78.0), 0.0, 0.0))
    camera = bpy.context.object
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.0))
    key = bpy.context.object
    key.data.energy = 2500
    key.data.shape = "RECTANGLE"
    key.data.size = 6.0
    key.data.size_y = 6.0

    bpy.ops.object.light_add(type="POINT", location=(3.0, -1.5, 2.5))
    rim = bpy.context.object
    rim.data.energy = 600


def add_ground() -> None:
    bpy.ops.mesh.primitive_plane_add(size=8.0, location=(0.0, 0.0, 0.0))
    plane = bpy.context.object
    material = make_material("ground_material", (0.08, 0.08, 0.1, 1.0))
    plane.data.materials.append(material)


def main() -> None:
    args = parse_args()
    preview_job = load_json(Path(args.preview_job))
    bone_order, frames = collect_preview_frames(preview_job, max_steps=args.max_steps, frame_stride=args.frame_stride)

    clear_scene()
    add_camera_and_lights()
    add_ground()
    joint_material = make_material("joint_material", (0.96, 0.72, 0.33, 1.0))
    bone_material = make_material("bone_material", (0.19, 0.71, 0.91, 1.0))
    joint_objects = create_joint_objects(bone_order, joint_material)
    bone_objects = create_bone_objects(bone_order, bone_material)
    animate_objects(bone_order, frames, joint_objects, bone_objects)
    output_video = Path(args.output_video) if args.output_video else None
    configure_scene(frame_count=len(frames), fps=args.fps, output_video=output_video)

    output_blend = Path(args.output_blend)
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    if args.render_animation:
        if output_video is None:
            raise ValueError("--render-animation requires --output-video")
        bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
