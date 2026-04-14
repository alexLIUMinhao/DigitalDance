#!/usr/bin/env python3
"""Build a true skinned-mesh choreography preview scene in Blender."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import bpy  # type: ignore
from mathutils import Matrix, Quaternion, Vector  # type: ignore


SOURCE_TO_TARGET_BONES = {
    "pelvis": "mixamorig:Hips",
    "left_hip": "mixamorig:LeftUpLeg",
    "left_knee": "mixamorig:LeftLeg",
    "left_ankle": "mixamorig:LeftFoot",
    "left_foot": "mixamorig:LeftToeBase",
    "right_hip": "mixamorig:RightUpLeg",
    "right_knee": "mixamorig:RightLeg",
    "right_ankle": "mixamorig:RightFoot",
    "right_foot": "mixamorig:RightToeBase",
    "spine1": "mixamorig:Spine",
    "spine2": "mixamorig:Spine1",
    "spine3": "mixamorig:Spine2",
    "neck": "mixamorig:Neck",
    "head": "mixamorig:Head",
    "left_collar": "mixamorig:LeftShoulder",
    "left_shoulder": "mixamorig:LeftArm",
    "left_elbow": "mixamorig:LeftForeArm",
    "left_wrist": "mixamorig:LeftHand",
    "right_collar": "mixamorig:RightShoulder",
    "right_shoulder": "mixamorig:RightArm",
    "right_elbow": "mixamorig:RightForeArm",
    "right_wrist": "mixamorig:RightHand",
}
ROOT_SOURCE_BONE = "pelvis"
ROOT_TARGET_BONE = "mixamorig:Hips"
LEFT_LEG_SOURCE_BONES = ("left_hip", "left_knee", "left_ankle")
LEFT_LEG_TARGET_BONES = ("mixamorig:LeftUpLeg", "mixamorig:LeftLeg", "mixamorig:LeftFoot")
PRIMARY_CHILD_SOURCE_BONES = {
    "pelvis": "spine1",
    "left_hip": "left_knee",
    "left_knee": "left_ankle",
    "left_ankle": "left_foot",
    "right_hip": "right_knee",
    "right_knee": "right_ankle",
    "right_ankle": "right_foot",
    "spine1": "spine2",
    "spine2": "spine3",
    "spine3": "neck",
    "neck": "head",
    "left_collar": "left_shoulder",
    "left_shoulder": "left_elbow",
    "left_elbow": "left_wrist",
    "right_collar": "right_shoulder",
    "right_shoulder": "right_elbow",
    "right_elbow": "right_wrist",
}
FRAME_AUX_SOURCE_BONES = {
    "pelvis": ("left_hip", "right_hip"),
    "left_hip": (ROOT_SOURCE_BONE, "spine1"),
    "right_hip": (ROOT_SOURCE_BONE, "spine1"),
    "spine1": ("left_hip", "right_hip"),
    "left_knee": (ROOT_SOURCE_BONE, "spine1"),
    "right_knee": (ROOT_SOURCE_BONE, "spine1"),
    "spine2": ("left_collar", "right_collar"),
    "left_ankle": (ROOT_SOURCE_BONE, "spine1"),
    "right_ankle": (ROOT_SOURCE_BONE, "spine1"),
    "spine3": ("left_collar", "right_collar"),
    "neck": ("left_collar", "right_collar"),
    "head": ("left_collar", "right_collar"),
    "left_collar": ("neck", "head"),
    "right_collar": ("neck", "head"),
    "left_shoulder": ("left_shoulder", "left_wrist"),
    "right_shoulder": ("right_shoulder", "right_wrist"),
    "left_elbow": ("left_shoulder", "left_wrist"),
    "right_elbow": ("right_shoulder", "right_wrist"),
    "left_wrist": ("left_elbow", "left_wrist"),
    "right_wrist": ("right_elbow", "right_wrist"),
}


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-blend", required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def log_message(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def reset_scene() -> None:
    if bpy.app.background:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        return

    ensure_object_mode()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)

    for data_block in (
        bpy.data.meshes,
        bpy.data.armatures,
        bpy.data.materials,
        bpy.data.lights,
        bpy.data.cameras,
        bpy.data.actions,
    ):
        for item in list(data_block):
            if item.users == 0:
                data_block.remove(item)

    bpy.context.view_layer.update()


def import_motion(path: Path) -> tuple[list[bpy.types.Object], bpy.types.Object | None]:
    before = set(bpy.data.objects.keys())
    if path.suffix.lower() == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(path), automatic_bone_orientation=False)
    elif path.suffix.lower() == ".bvh":
        bpy.ops.import_anim.bvh(filepath=str(path), rotate_mode="NATIVE")
    else:
        raise SystemExit(f"Unsupported motion input: {path.suffix}")

    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    imported_objects = [bpy.data.objects[name] for name in imported_names]
    armature = next((obj for obj in imported_objects if obj.type == "ARMATURE"), None)
    return imported_objects, armature


def ensure_object_mode() -> None:
    active_object = bpy.context.view_layer.objects.active
    if active_object and active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def active_object() -> bpy.types.Object:
    obj = bpy.context.view_layer.objects.active
    if obj is None:
        raise RuntimeError("No active object is available in the current Blender context")
    return obj


def select_objects(objects: list[bpy.types.Object]) -> None:
    ensure_object_mode()
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0] if objects else None


def delete_objects(objects: list[bpy.types.Object]) -> None:
    if not objects:
        return
    select_objects(objects)
    bpy.ops.object.delete(use_global=False)


def resolve_action(armature: bpy.types.Object | None):
    if armature and armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    for action in bpy.data.actions:
        return action
    return None


def bone_local_rest_matrix(armature: bpy.types.Object, bone_name: str) -> Matrix:
    bone = armature.data.bones[bone_name]
    if bone.parent:
        return bone.parent.matrix_local.inverted() @ bone.matrix_local
    return bone.matrix_local.copy()


def bone_local_rest_rotation(armature: bpy.types.Object, bone_name: str) -> Matrix:
    return bone_local_rest_matrix(armature, bone_name).to_3x3().normalized()


def bone_world_rest_rotation(armature: bpy.types.Object, bone_name: str) -> Matrix:
    bone = armature.data.bones[bone_name]
    return bone.matrix_local.to_3x3().normalized()


def normalize_vector(vector: Vector) -> Vector:
    if vector.length <= 1e-8:
        return Vector((0.0, 0.0, 0.0))
    return vector.normalized()


def orthogonalize(primary: Vector, aux: Vector) -> Vector:
    primary_unit = normalize_vector(primary)
    if primary_unit.length <= 1e-8:
        return Vector((1.0, 0.0, 0.0))
    aux_proj = aux - primary_unit * aux.dot(primary_unit)
    if aux_proj.length <= 1e-8:
        for fallback_axis in (Vector((1.0, 0.0, 0.0)), Vector((0.0, 1.0, 0.0)), Vector((0.0, 0.0, 1.0))):
            candidate = fallback_axis - primary_unit * fallback_axis.dot(primary_unit)
            if candidate.length > 1e-8:
                aux_proj = candidate
                break
    return normalize_vector(aux_proj)


def build_frame(primary: Vector, aux: Vector) -> Matrix:
    z_axis = normalize_vector(primary)
    if z_axis.length <= 1e-8:
        return Matrix.Identity(3)
    x_axis = orthogonalize(z_axis, aux)
    y_axis = normalize_vector(z_axis.cross(x_axis))
    if y_axis.length <= 1e-8:
        return Matrix.Identity(3)
    x_axis = normalize_vector(y_axis.cross(z_axis))
    return Matrix((x_axis, y_axis, z_axis)).transposed()


def rest_bone_head(armature: bpy.types.Object, bone_name: str) -> Vector | None:
    bone = armature.data.bones.get(bone_name)
    if bone is None:
        return None
    return bone.head_local.copy()


def resolve_target_name(source_name: str) -> str | None:
    return SOURCE_TO_TARGET_BONES.get(source_name)


def rest_basis_frame(armature: bpy.types.Object, source_name: str, bone_name: str, *, target_space: bool) -> Matrix:
    child_source_name = PRIMARY_CHILD_SOURCE_BONES.get(source_name)
    child_bone_name = resolve_target_name(child_source_name) if (target_space and child_source_name) else child_source_name
    head = rest_bone_head(armature, bone_name)
    child_head = rest_bone_head(armature, child_bone_name) if child_bone_name else None
    if head is None or child_head is None:
        return bone_world_rest_rotation(armature, bone_name)

    aux_pair = FRAME_AUX_SOURCE_BONES.get(source_name)
    if aux_pair is None:
        return bone_world_rest_rotation(armature, bone_name)

    aux_start_name = resolve_target_name(aux_pair[0]) if target_space else aux_pair[0]
    aux_end_name = resolve_target_name(aux_pair[1]) if target_space else aux_pair[1]
    aux_start = rest_bone_head(armature, aux_start_name) if aux_start_name else None
    aux_end = rest_bone_head(armature, aux_end_name) if aux_end_name else None
    if aux_start is None or aux_end is None:
        return bone_world_rest_rotation(armature, bone_name)

    return build_frame(child_head - head, aux_end - aux_start)


def bone_length(armature: bpy.types.Object, bone_name: str) -> float:
    bone = armature.data.bones.get(bone_name)
    return float(bone.length) if bone is not None else 0.0


def compute_scale_ratio(source_armature: bpy.types.Object, target_armature: bpy.types.Object) -> float:
    source_length = sum(bone_length(source_armature, name) for name in LEFT_LEG_SOURCE_BONES)
    target_length = sum(bone_length(target_armature, name) for name in LEFT_LEG_TARGET_BONES)
    if source_length <= 1e-6 or target_length <= 1e-6:
        return 1.0
    return target_length / source_length


def create_target_action(target_armature: bpy.types.Object, action_name: str) -> bpy.types.Action:
    target_armature.animation_data_create()
    action = bpy.data.actions.new(name=action_name)
    target_armature.animation_data.action = action

    for pose_bone in target_armature.pose.bones:
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.location = Vector((0.0, 0.0, 0.0))
        pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        pose_bone.scale = Vector((1.0, 1.0, 1.0))

    bpy.context.view_layer.update()
    return action


def mapped_target_pairs(source_armature: bpy.types.Object, target_armature: bpy.types.Object) -> list[tuple[str, str]]:
    source_bones = source_armature.data.bones
    target_bones = target_armature.data.bones
    pairs: list[tuple[str, str]] = []
    for source_name, target_name in SOURCE_TO_TARGET_BONES.items():
        if source_bones.get(source_name) is not None and target_bones.get(target_name) is not None:
            pairs.append((source_name, target_name))
    return pairs


def basis_quaternion(source_armature: bpy.types.Object, target_armature: bpy.types.Object, source_name: str, target_name: str) -> Quaternion:
    source_rest = bone_local_rest_rotation(source_armature, source_name)
    target_rest = bone_local_rest_rotation(target_armature, target_name)
    return (target_rest.inverted() @ source_rest).to_quaternion().normalized()


def world_basis_quaternion(source_armature: bpy.types.Object, target_armature: bpy.types.Object, source_name: str, target_name: str) -> Quaternion:
    source_rest = bone_world_rest_rotation(source_armature, source_name)
    target_rest = bone_world_rest_rotation(target_armature, target_name)
    return (target_rest.inverted() @ source_rest).to_quaternion().normalized()


def frame_basis_quaternion(source_armature: bpy.types.Object, target_armature: bpy.types.Object, source_name: str, target_name: str) -> Quaternion:
    source_rest = rest_basis_frame(source_armature, source_name, source_name, target_space=False)
    target_rest = rest_basis_frame(target_armature, source_name, target_name, target_space=True)
    return (target_rest.inverted() @ source_rest).to_quaternion().normalized()


def constraint_spaces_for_mode(basis_mode: str) -> tuple[str, str]:
    suffix = basis_mode[len("constraint_") :] if basis_mode.startswith("constraint_") else basis_mode
    normalized = suffix.strip().lower()
    if normalized in {"world_hybrid", "world_pelvis_local", "world_pelvis_spine_local", "world_pelvis_local_plain"}:
        return "WORLD", "WORLD"
    if normalized == "local_with_parent":
        return "LOCAL_WITH_PARENT", "LOCAL_WITH_PARENT"
    if normalized == "world":
        return "WORLD", "WORLD"
    if normalized == "pose":
        return "POSE", "POSE"
    if normalized == "local":
        return "LOCAL", "LOCAL"
    return "LOCAL_OWNER_ORIENT", "LOCAL"


def scale_root_location_curves(action: bpy.types.Action, bone_name: str, scale_ratio: float) -> None:
    data_path = f'pose.bones["{bone_name}"].location'
    for fcurve in action.fcurves:
        if fcurve.data_path != data_path:
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.co[1] *= scale_ratio
            keyframe.handle_left[1] *= scale_ratio
            keyframe.handle_right[1] *= scale_ratio


def root_chain_override_sources(basis_mode: str) -> set[str]:
    normalized = basis_mode.strip().lower()
    if normalized == "constraint_world_pelvis_local":
        return {ROOT_SOURCE_BONE}
    if normalized == "constraint_world_pelvis_torso_local":
        return {ROOT_SOURCE_BONE, "spine1", "spine2", "spine3", "neck"}
    if normalized == "constraint_world_pelvis_upper_local":
        return {
            ROOT_SOURCE_BONE,
            "spine1",
            "spine2",
            "spine3",
            "neck",
            "head",
            "left_collar",
            "right_collar",
            "left_shoulder",
            "right_shoulder",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        }
    if normalized == "constraint_world_pelvis_spine_local":
        return {ROOT_SOURCE_BONE, "spine1"}
    if normalized == "constraint_world_hybrid":
        return {ROOT_SOURCE_BONE, "left_hip", "right_hip", "spine1"}
    return set()


def explicit_constraint_override_spaces(basis_mode: str, source_name: str) -> tuple[str, str] | None:
    normalized = basis_mode.strip().lower()
    if normalized in {"constraint_world_pelvis_local", "constraint_world_pelvis_torso_local", "constraint_world_pelvis_upper_local"}:
        if source_name in root_chain_override_sources(normalized):
            return "LOCAL_OWNER_ORIENT", "LOCAL"
    if normalized == "constraint_world_pelvis_local_plain" and source_name == ROOT_SOURCE_BONE:
        return "LOCAL", "LOCAL"
    return None


def retarget_action(
    source_armature: bpy.types.Object,
    source_action: bpy.types.Action,
    target_armature: bpy.types.Object,
    motion_prefix: str,
    basis_mode: str = "constraint_world_pelvis_local",
    frame_start_override: int | None = None,
    frame_end_override: int | None = None,
    log_path: Path | None = None,
) -> tuple[bpy.types.Action, list[str], float]:
    warnings: list[str] = []
    if log_path is not None:
        log_message(log_path, f"retarget_prepare {motion_prefix}")
    pairs = mapped_target_pairs(source_armature, target_armature)
    if not pairs:
        raise SystemExit("No source-to-target bone mapping could be resolved")
    if log_path is not None:
        log_message(log_path, f"retarget_pairs {motion_prefix} {len(pairs)}")

    use_frame_basis = "frame_" in basis_mode
    use_world_basis = basis_mode.startswith("world_")
    if use_frame_basis:
        basis_resolver = frame_basis_quaternion
    else:
        basis_resolver = world_basis_quaternion if use_world_basis else basis_quaternion
    basis_map = {
        (source_name, target_name): basis_resolver(source_armature, target_armature, source_name, target_name)
        for source_name, target_name in pairs
    }
    if log_path is not None:
        log_message(log_path, f"retarget_basis_ready {motion_prefix}")
    scale_ratio = compute_scale_ratio(source_armature, target_armature)
    if log_path is not None:
        log_message(log_path, f"retarget_scale_ratio {motion_prefix} {scale_ratio:.5f}")
    target_action = create_target_action(target_armature, f"{motion_prefix}_retarget")
    if log_path is not None:
        log_message(log_path, f"retarget_action_created {motion_prefix}")
    frame_start = int(frame_start_override if frame_start_override is not None else math.floor(source_action.frame_range[0]))
    frame_end = int(frame_end_override if frame_end_override is not None else math.ceil(source_action.frame_range[1]))

    if basis_mode.startswith("constraint_"):
        target_space, owner_space = constraint_spaces_for_mode(basis_mode)
        root_chain_sources = root_chain_override_sources(basis_mode)
        target_root = target_armature.pose.bones[ROOT_TARGET_BONE]
        for source_name, target_name in pairs:
            target_pose_bone = target_armature.pose.bones[target_name]
            rotation_constraint = target_pose_bone.constraints.new(type="COPY_ROTATION")
            rotation_constraint.name = "music_motion_lab_copy_rotation"
            rotation_constraint.target = source_armature
            rotation_constraint.subtarget = source_name
            explicit_override = explicit_constraint_override_spaces(basis_mode, source_name)
            if explicit_override is not None:
                rotation_constraint.target_space = explicit_override[0]
                rotation_constraint.owner_space = explicit_override[1]
            elif source_name in root_chain_sources:
                rotation_constraint.target_space = "LOCAL_OWNER_ORIENT"
                rotation_constraint.owner_space = "LOCAL"
            else:
                rotation_constraint.target_space = target_space
                rotation_constraint.owner_space = owner_space

        location_constraint = target_root.constraints.new(type="COPY_LOCATION")
        location_constraint.name = "music_motion_lab_copy_location"
        location_constraint.target = source_armature
        location_constraint.subtarget = ROOT_SOURCE_BONE
        location_constraint.target_space = "LOCAL"
        location_constraint.owner_space = "LOCAL"

        ensure_object_mode()
        bpy.ops.object.select_all(action="DESELECT")
        target_armature.select_set(True)
        bpy.context.view_layer.objects.active = target_armature
        bpy.ops.nla.bake(
            frame_start=frame_start,
            frame_end=frame_end,
            step=1,
            only_selected=True,
            visual_keying=True,
            clear_constraints=True,
            use_current_action=True,
            bake_types={"POSE"},
        )
        if log_path is not None:
            log_message(log_path, f"retarget_bake_done {motion_prefix} {frame_start}:{frame_end}")
        scale_root_location_curves(target_action, ROOT_TARGET_BONE, scale_ratio)
        for fcurve in target_action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = "LINEAR"
        return target_action, warnings, float(scale_ratio)

    scene = bpy.context.scene
    source_root = source_armature.pose.bones[ROOT_SOURCE_BONE]
    target_root = target_armature.pose.bones[ROOT_TARGET_BONE]
    root_basis = basis_map[(ROOT_SOURCE_BONE, ROOT_TARGET_BONE)]
    root_basis_matrix = root_basis.to_matrix()
    if log_path is not None:
        log_message(log_path, f"retarget_frame_loop {motion_prefix} {frame_start}:{frame_end}")

    for frame in range(frame_start, frame_end + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
        source_location = source_root.matrix_basis.to_translation()
        target_root.location = root_basis_matrix @ (source_location * scale_ratio)
        target_root.keyframe_insert(data_path="location", frame=frame, group=ROOT_TARGET_BONE)
        for source_name, target_name in pairs:
            source_pose_bone = source_armature.pose.bones[source_name]
            target_pose_bone = target_armature.pose.bones[target_name]
            target_pose_bone.rotation_mode = "QUATERNION"
            source_rotation = source_pose_bone.matrix_basis.to_quaternion()
            basis = basis_map[(source_name, target_name)]
            target_pose_bone.rotation_quaternion = (basis @ source_rotation).normalized()
            target_pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame, group=target_name)
        if log_path is not None and ((frame - frame_start) % 10 == 0 or frame == frame_end):
            log_message(log_path, f"retarget_frame_done {motion_prefix} {frame}")

    for fcurve in target_action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"

    return target_action, warnings, float(scale_ratio)


def prepare_template_armature(template_path: Path, *, keep_meshes: bool = False) -> tuple[list[bpy.types.Object], bpy.types.Object]:
    template_objects, template_armature = import_motion(template_path)
    if template_armature is None:
        raise SystemExit(f"Template file did not contain an armature: {template_path}")

    for obj in template_objects:
        obj.animation_data_clear()

    template_armature.animation_data_clear()
    if keep_meshes:
        kept_objects = [obj for obj in template_objects if obj.type in {"ARMATURE", "MESH"}]
        delete_objects([obj for obj in template_objects if obj not in kept_objects])
        return kept_objects, template_armature

    delete_objects([obj for obj in template_objects if obj.type != "ARMATURE"])
    return [template_armature], template_armature


def move_objects_to_collection(objects: list[bpy.types.Object], collection_name: str) -> bpy.types.Collection:
    collection = bpy.data.collections.new(collection_name)
    bpy.context.scene.collection.children.link(collection)
    for obj in objects:
        for existing in list(obj.users_collection):
            existing.objects.unlink(obj)
        collection.objects.link(obj)
    return collection


def configure_scene(frame_start: int, frame_end: int, fps: int) -> None:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.eevee.taa_render_samples = 16
    scene.render.resolution_x = 1280
    scene.render.resolution_y = 720
    scene.render.fps = fps
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    if scene.world is None:
        scene.world = bpy.data.worlds.new("World")
    scene.world.color = (0.03, 0.04, 0.055)


def add_camera_and_lights() -> None:
    bpy.ops.object.camera_add(location=(0.0, -7.2, 2.1), rotation=(math.radians(78.0), 0.0, 0.0))
    camera = active_object()
    bpy.context.scene.camera = camera

    bpy.ops.object.light_add(type="AREA", location=(0.0, -4.0, 5.5))
    key = active_object()
    key.data.energy = 2200
    key.data.shape = "RECTANGLE"
    key.data.size = 7.0
    key.data.size_y = 6.0

    bpy.ops.object.light_add(type="POINT", location=(3.0, -1.2, 2.6))
    rim = active_object()
    rim.data.energy = 450


def add_ground() -> None:
    bpy.ops.mesh.primitive_plane_add(size=10.0, location=(0.0, 0.0, 0.0))
    plane = active_object()
    material = bpy.data.materials.new(name="ground_material")
    material.use_nodes = True
    shader = material.node_tree.nodes["Principled BSDF"]
    shader.inputs["Base Color"].default_value = (0.08, 0.1, 0.12, 1.0)
    shader.inputs["Roughness"].default_value = 0.72
    plane.data.materials.append(material)


def key_visibility(objects: list[bpy.types.Object], windows: list[tuple[int, int]], frame_end: int) -> None:
    for obj in objects:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=1)
        obj.keyframe_insert(data_path="hide_render", frame=1)
        for start_frame, end_frame in windows:
            before = max(1, start_frame - 1)
            after = min(frame_end, end_frame + 1)
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_viewport", frame=before)
            obj.keyframe_insert(data_path="hide_render", frame=before)
            obj.hide_viewport = False
            obj.hide_render = False
            obj.keyframe_insert(data_path="hide_viewport", frame=start_frame)
            obj.keyframe_insert(data_path="hide_render", frame=start_frame)
            obj.hide_viewport = False
            obj.hide_render = False
            obj.keyframe_insert(data_path="hide_viewport", frame=end_frame)
            obj.keyframe_insert(data_path="hide_render", frame=end_frame)
            obj.hide_viewport = True
            obj.hide_render = True
            obj.keyframe_insert(data_path="hide_viewport", frame=after)
            obj.keyframe_insert(data_path="hide_render", frame=after)


def add_sequence_preview(
    sequence_manifest: dict,
    steps: list[dict],
    mesh_template_fbx: Path,
    basis_mode: str,
    log_path: Path,
) -> None:
    source_motion_fbx = Path(sequence_manifest["source_motion_fbx"])
    log_message(log_path, f"import_source {source_motion_fbx}")
    source_objects, source_armature = import_motion(source_motion_fbx)
    if source_armature is None:
        raise SystemExit(f"Source motion did not contain an armature: {source_motion_fbx}")
    source_action = resolve_action(source_armature)
    if source_action is None:
        raise SystemExit(f"No action found in source motion: {source_motion_fbx}")

    source_mesh_objects = [obj for obj in source_objects if obj.type == "MESH"]
    if source_mesh_objects:
        collection_name = f"preview_{sequence_manifest['dataset']}_{sequence_manifest['source_sequence']}"
        move_objects_to_collection(source_objects, collection_name)
        prefix = f"{sequence_manifest['dataset']}_{sequence_manifest['source_sequence']}"
        log_message(log_path, f"sequence_mode {prefix} direct_mesh")
        source_armature.animation_data_create()
        source_armature.animation_data.action = None
        track = source_armature.animation_data.nla_tracks.new()
        track.name = f"{prefix}_track"
        base_frame = int(round(source_action.frame_range[0]))
        for step in steps:
            action_frame_start = base_frame + int(step["source_frame_start"])
            action_frame_end = base_frame + int(step["source_frame_end"])
            strip = track.strips.new(f"step_{step['index']:03d}", int(step["scene_frame_start"]), source_action)
            strip.action_frame_start = float(action_frame_start)
            strip.action_frame_end = float(action_frame_end)
            strip.frame_end = float(step["scene_frame_end"])
            strip.blend_type = "REPLACE"
            strip.extrapolation = "NOTHING"
            strip.use_auto_blend = True

        windows = [(int(step["scene_frame_start"]), int(step["scene_frame_end"])) for step in steps]
        key_visibility(source_objects, windows, max(end for _, end in windows))
        log_message(log_path, f"sequence_scene_ready {prefix}")
        return

    log_message(log_path, f"import_template {mesh_template_fbx}")
    template_objects, template_armature = prepare_template_armature(mesh_template_fbx, keep_meshes=True)
    collection_name = f"preview_{sequence_manifest['dataset']}_{sequence_manifest['source_sequence']}"
    move_objects_to_collection(template_objects, collection_name)

    prefix = f"{sequence_manifest['dataset']}_{sequence_manifest['source_sequence']}"
    base_frame = int(round(source_action.frame_range[0]))
    log_message(log_path, f"retarget_start {prefix}")
    template_armature.animation_data_create()
    template_armature.animation_data.action = None
    track = template_armature.animation_data.nla_tracks.new()
    track.name = f"{prefix}_track"

    for step in steps:
        action_frame_start = base_frame + int(step["source_frame_start"])
        action_frame_end = base_frame + int(step["source_frame_end"])
        step_prefix = f"{prefix}_step_{int(step['index']):03d}"
        log_message(log_path, f"retarget_step_start {step_prefix} {action_frame_start}:{action_frame_end}")
        target_action, _, _ = retarget_action(
            source_armature=source_armature,
            source_action=source_action,
            target_armature=template_armature,
            motion_prefix=step_prefix,
            basis_mode=basis_mode,
            frame_start_override=action_frame_start,
            frame_end_override=action_frame_end,
            log_path=log_path,
        )
        log_message(log_path, f"retarget_step_done {step_prefix}")
        strip = track.strips.new(f"step_{step['index']:03d}", int(step["scene_frame_start"]), target_action)
        strip.action_frame_start = float(action_frame_start)
        strip.action_frame_end = float(action_frame_end)
        strip.frame_end = float(step["scene_frame_end"])
        strip.blend_type = "REPLACE"
        strip.extrapolation = "NOTHING"
        strip.use_auto_blend = True

    windows = [(int(step["scene_frame_start"]), int(step["scene_frame_end"])) for step in steps]
    key_visibility(template_objects, windows, max(end for _, end in windows))
    log_message(log_path, f"retarget_done {prefix}")
    log_message(log_path, f"sequence_scene_ready {prefix}")
    delete_objects(source_objects)
    log_message(log_path, f"source_deleted {prefix}")


def build_scene(args: argparse.Namespace) -> None:
    manifest = load_json(Path(args.manifest))
    mesh_template_fbx = Path(manifest["mesh_template_fbx"])
    output_blend = Path(args.output_blend)
    log_path = output_blend.with_suffix(".log")
    if log_path.exists():
        log_path.unlink()
    log_message(log_path, "mesh_preview_start")

    reset_scene()
    log_message(log_path, "scene_reset")
    configure_scene(
        frame_start=int(manifest.get("scene", {}).get("frame_start", 1)),
        frame_end=int(manifest.get("scene", {}).get("frame_end", 1)),
        fps=int(manifest.get("fps", 24)),
    )
    add_camera_and_lights()
    add_ground()
    log_message(log_path, "scene_layout_ready")

    steps_by_sequence: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for step in manifest.get("steps", []):
        key = (step["dataset"], step["source_sequence"], step["source_motion_fbx"])
        steps_by_sequence[key].append(step)

    for sequence_manifest in manifest.get("sequences", []):
        key = (
            sequence_manifest["dataset"],
            sequence_manifest["source_sequence"],
            sequence_manifest["source_motion_fbx"],
        )
        add_sequence_preview(
            sequence_manifest=sequence_manifest,
            steps=sorted(steps_by_sequence[key], key=lambda item: int(item["scene_frame_start"])),
            mesh_template_fbx=mesh_template_fbx,
            basis_mode=str(manifest.get("basis_mode", "constraint_world_pelvis_local")),
            log_path=log_path,
        )

    bpy.context.scene.frame_set(int(manifest.get("scene", {}).get("frame_start", 1)))
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    log_message(log_path, f"saving_blend {output_blend}")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    log_message(log_path, "mesh_preview_done")


def main() -> None:
    args = parse_args()
    if bpy.app.background:
        build_scene(args)
        return

    def run_when_ui_ready() -> float | None:
        windows = list(getattr(bpy.context.window_manager, "windows", []))
        if not windows:
            return 0.25
        try:
            build_scene(args)
        except Exception:
            output_candidate = None
            if "--" in sys.argv and "--output-blend" in sys.argv:
                try:
                    output_candidate = Path(sys.argv[sys.argv.index("--output-blend") + 1]).with_suffix(".log")
                except Exception:
                    output_candidate = None
            if output_candidate is not None:
                log_message(output_candidate, traceback.format_exc())
            raise
        return None

    bpy.app.timers.register(run_when_ui_ready, first_interval=0.5)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        output_candidate = None
        if "--" in sys.argv and "--output-blend" in sys.argv:
            try:
                output_candidate = Path(sys.argv[sys.argv.index("--output-blend") + 1]).with_suffix(".log")
            except Exception:
                output_candidate = None
        if output_candidate is not None:
            log_message(output_candidate, traceback.format_exc())
        raise
