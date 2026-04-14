#!/usr/bin/env python3
"""Retarget a source_willa_aligned BVH onto Willa inside Blender GUI."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import addon_utils  # type: ignore
import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Quaternion, Vector  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.willa_retarget import (  # noqa: E402
    angle_deg,
    build_basis,
    joint_angle,
    load_willa_retarget_profile,
    resolve_reference_vector,
    summarize_values,
    validate_willa_retarget_profile,
)


LEFT_LEG_SOURCE_BONES = ("left_hip", "left_knee", "left_ankle")
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


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    argv = []
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bvh", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--basis-mode", help="Optional constraint bake basis mode override.")
    parser.add_argument("--skip-basis-correction", action="store_true", help="Skip reference basis correction during constraint/bake retarget.")
    parser.add_argument("--quit-after-run", action="store_true", help="Quit Blender after the retarget run finishes.")
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--report-output", required=True)
    return parser.parse_args(argv)


def ensure_object_mode() -> None:
    active_object = bpy.context.view_layer.objects.active
    if active_object and active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def ensure_bvh_import_addon() -> None:
    addon_utils.enable("io_anim_bvh", default_set=False, persistent=False)


def import_bvh(path: Path) -> bpy.types.Object:
    before = set(bpy.data.objects.keys())
    bpy.ops.import_anim.bvh(filepath=str(path), rotate_mode="NATIVE")
    imported_names = [name for name in bpy.data.objects.keys() if name not in before]
    imported_objects = [bpy.data.objects[name] for name in imported_names]
    armature = next((obj for obj in imported_objects if obj.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError(f"No armature found after importing {path}")
    return armature


def resolve_action(armature: bpy.types.Object | None):
    if armature and armature.animation_data and armature.animation_data.action:
        return armature.animation_data.action
    for action in bpy.data.actions:
        return action
    return None


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


def create_target_action(target_armature: bpy.types.Object, action_name: str) -> bpy.types.Action:
    target_armature.animation_data_clear()
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


def bone_length(armature: bpy.types.Object, bone_name: str) -> float:
    bone = armature.data.bones.get(bone_name)
    return float(bone.length) if bone is not None else 0.0


def compute_scale_ratio(source_armature: bpy.types.Object, target_armature: bpy.types.Object, bone_map: dict[str, str]) -> float:
    source_length = sum(bone_length(source_armature, name) for name in LEFT_LEG_SOURCE_BONES)
    target_length = sum(
        bone_length(target_armature, bone_map[name])
        for name in LEFT_LEG_SOURCE_BONES
        if name in bone_map
    )
    if source_length <= 1e-6 or target_length <= 1e-6:
        return 1.0
    return target_length / source_length


def scale_root_location_curves(action: bpy.types.Action, bone_name: str, scale_ratio: float) -> None:
    data_path = f'pose.bones["{bone_name}"].location'
    for fcurve in action.fcurves:
        if fcurve.data_path != data_path:
            continue
        for keyframe in fcurve.keyframe_points:
            keyframe.co[1] *= scale_ratio
            keyframe.handle_left[1] *= scale_ratio
            keyframe.handle_right[1] *= scale_ratio


def recenter_root_location_curves(action: bpy.types.Action, bone_name: str) -> None:
    data_path = f'pose.bones["{bone_name}"].location'
    offsets: dict[int, float] = {}
    for fcurve in action.fcurves:
        if fcurve.data_path != data_path:
            continue
        if not fcurve.keyframe_points:
            continue
        offsets[int(fcurve.array_index)] = float(fcurve.keyframe_points[0].co[1])
    for fcurve in action.fcurves:
        if fcurve.data_path != data_path:
            continue
        offset = offsets.get(int(fcurve.array_index), 0.0)
        for keyframe in fcurve.keyframe_points:
            keyframe.co[1] -= offset
            keyframe.handle_left[1] -= offset
            keyframe.handle_right[1] -= offset


def clear_bone_animation(action: bpy.types.Action, bone_name: str) -> None:
    prefix = f'pose.bones["{bone_name}"].'
    for fcurve in list(action.fcurves):
        if fcurve.data_path.startswith(prefix):
            action.fcurves.remove(fcurve)


def clear_disabled_rotation_bones(
    target_armature: bpy.types.Object,
    action: bpy.types.Action,
    profile: dict[str, Any],
    trace_output: Path | None = None,
) -> list[str]:
    disabled_source_joints = [str(value) for value in list(profile.get("rotationDisabledSourceJoints", []) or [])]
    if not disabled_source_joints:
        append_trace(trace_output, "rotation_disable:none")
        return []

    bone_map = dict(profile.get("boneMap", {}) or {})
    cleared: list[str] = []
    for source_joint in disabled_source_joints:
        target_bone = bone_map.get(source_joint)
        if not target_bone:
            continue
        clear_bone_animation(action, target_bone)
        if target_bone in target_armature.pose.bones:
            pose_bone = target_armature.pose.bones[target_bone]
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
            pose_bone.location = Vector((0.0, 0.0, 0.0))
            pose_bone.scale = Vector((1.0, 1.0, 1.0))
        cleared.append(target_bone)
    append_trace(trace_output, f"rotation_disable:cleared={cleared}")
    return cleared


def apply_basis_corrections_to_action(
    target_armature: bpy.types.Object,
    action: bpy.types.Action,
    profile: dict[str, Any],
    frame_start: int,
    frame_end: int,
    trace_output: Path | None = None,
) -> list[str]:
    bone_map = dict(profile.get("boneMap", {}) or {})
    correction_payloads = dict(profile.get("basisCorrectionTransforms", {}) or {})
    if not correction_payloads:
        append_trace(trace_output, "basis_correction:none")
        return []

    corrections: list[tuple[str, Quaternion]] = []
    for source_joint, payload in correction_payloads.items():
        target_bone = bone_map.get(str(source_joint))
        if not target_bone or target_bone not in target_armature.pose.bones:
            continue
        matrix_values = np.asarray((payload or {}).get("matrix", np.eye(3)), dtype=np.float64).reshape((3, 3))
        correction_quaternion = Matrix(matrix_values.tolist()).to_quaternion()
        angle = abs(correction_quaternion.angle)
        if angle <= 1e-6:
            continue
        corrections.append((target_bone, correction_quaternion))

    if not corrections:
        append_trace(trace_output, "basis_correction:identity_only")
        return []

    applied_names = [name for name, _ in corrections]
    append_trace(trace_output, f"basis_correction:targets={applied_names}")
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for target_bone, correction_quaternion in corrections:
            pose_bone = target_armature.pose.bones[target_bone]
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.rotation_quaternion = pose_bone.rotation_quaternion @ correction_quaternion
            pose_bone.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    append_trace(trace_output, "basis_correction:done")
    return applied_names


def root_chain_override_sources(basis_mode: str) -> set[str]:
    normalized = basis_mode.strip().lower()
    if normalized == "constraint_world_pelvis_local":
        return {"pelvis"}
    if normalized == "constraint_world_pelvis_torso_local":
        return {"pelvis", "spine1", "spine2", "spine3", "neck"}
    if normalized == "constraint_world_pelvis_upper_local":
        return {
            "pelvis",
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
    return set()


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


def explicit_constraint_override_spaces(basis_mode: str, source_name: str) -> tuple[str, str] | None:
    normalized = basis_mode.strip().lower()
    if normalized in {"constraint_world_pelvis_local", "constraint_world_pelvis_torso_local", "constraint_world_pelvis_upper_local"}:
        if source_name in root_chain_override_sources(normalized):
            return "LOCAL_OWNER_ORIENT", "LOCAL"
    if normalized == "constraint_world_pelvis_local_plain" and source_name == "pelvis":
        return "LOCAL", "LOCAL"
    return None


def remove_music_motion_lab_constraints(target_armature: bpy.types.Object) -> None:
    for pose_bone in target_armature.pose.bones:
        for constraint in list(pose_bone.constraints):
            if constraint.name.startswith("music_motion_lab_"):
                pose_bone.constraints.remove(constraint)


def manual_visual_bake_action(
    *,
    target_armature: bpy.types.Object,
    mapped_pairs: list[tuple[str, str]],
    frame_start: int,
    frame_end: int,
    target_motion_root: str,
    trace_output: Path | None = None,
) -> None:
    ensure_object_mode()
    bpy.context.view_layer.objects.active = target_armature
    bpy.ops.object.mode_set(mode="POSE")
    append_trace(trace_output, "bake:manual_visual_keying_start")
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        for _, target_name in mapped_pairs:
            pose_bone = target_armature.pose.bones.get(target_name)
            if pose_bone is None:
                continue
            pose_bone.rotation_mode = "QUATERNION"
            pose_bone.keyframe_insert(
                data_path="rotation_quaternion",
                frame=frame,
                options={"INSERTKEY_VISUAL"},
            )
        motion_root_bone = target_armature.pose.bones.get(target_motion_root)
        if motion_root_bone is not None:
            motion_root_bone.keyframe_insert(
                data_path="location",
                frame=frame,
                options={"INSERTKEY_VISUAL"},
            )
    bpy.ops.object.mode_set(mode="OBJECT")
    append_trace(trace_output, "bake:manual_visual_keying_done")


def bake_retarget_action(
    source_armature: bpy.types.Object,
    source_action: bpy.types.Action,
    target_armature: bpy.types.Object,
    profile: dict[str, Any],
    motion_prefix: str,
    trace_output: Path | None = None,
) -> tuple[bpy.types.Action, float]:
    bone_map = dict(profile.get("boneMap", {}) or {})
    disabled_rotation_joints = {str(value) for value in list(profile.get("rotationDisabledSourceJoints", []) or [])}
    mapped_pairs = [
        (source_name, target_name)
        for source_name, target_name in bone_map.items()
        if source_name in source_armature.data.bones and target_name in target_armature.data.bones and source_name not in disabled_rotation_joints
    ]
    if not mapped_pairs:
        raise RuntimeError("No source-to-Willa bone mappings could be resolved")
    append_trace(trace_output, f"bake:resolved_pairs={len(mapped_pairs)}")

    target_action = create_target_action(target_armature, f"{motion_prefix}_willa_retarget")
    frame_start = int(math.floor(source_action.frame_range[0]))
    frame_end = int(math.ceil(source_action.frame_range[1]))
    basis_mode = str(profile.get("basisMode", "constraint_world_pelvis_local") or "constraint_world_pelvis_local")
    target_space, owner_space = constraint_spaces_for_mode(basis_mode)
    root_chain_sources = root_chain_override_sources(basis_mode)
    append_trace(trace_output, f"bake:frame_range={frame_start}-{frame_end}")

    for source_name, target_name in mapped_pairs:
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
    append_trace(trace_output, "bake:rotation_constraints_added")

    source_motion_root = str(profile["sourceMotionRootBone"])
    target_motion_root = str(profile["targetMotionRootBone"])
    location_constraint = target_armature.pose.bones[target_motion_root].constraints.new(type="COPY_LOCATION")
    location_constraint.name = "music_motion_lab_copy_location"
    location_constraint.target = source_armature
    location_constraint.subtarget = source_motion_root
    location_constraint.target_space = "LOCAL"
    location_constraint.owner_space = "LOCAL"
    append_trace(trace_output, f"bake:location_constraint_added target_motion_root={target_motion_root}")

    append_trace(trace_output, "bake:manual_bake_start")
    manual_visual_bake_action(
        target_armature=target_armature,
        mapped_pairs=mapped_pairs,
        frame_start=frame_start,
        frame_end=frame_end,
        target_motion_root=target_motion_root,
        trace_output=trace_output,
    )
    remove_music_motion_lab_constraints(target_armature)
    append_trace(trace_output, "bake:manual_bake_done")
    append_trace(
        trace_output,
        f"bake:fcurves={len(target_action.fcurves)} frame_range={tuple(float(value) for value in target_action.frame_range)}",
    )

    scale_ratio = compute_scale_ratio(source_armature, target_armature, bone_map)
    append_trace(trace_output, f"bake:scale_ratio={scale_ratio:.6f}")
    scale_root_location_curves(target_action, target_motion_root, scale_ratio)
    if str(profile.get("rootMotionMode", "origin_locked") or "origin_locked") == "origin_locked":
        recenter_root_location_curves(target_action, target_motion_root)
    cleared_rotation_bones = clear_disabled_rotation_bones(
        target_armature=target_armature,
        action=target_action,
        profile=profile,
        trace_output=trace_output,
    )
    applied_corrections = apply_basis_corrections_to_action(
        target_armature=target_armature,
        action=target_action,
        profile=profile,
        frame_start=frame_start,
        frame_end=frame_end,
        trace_output=trace_output,
    )
    target_root_bone = str(profile.get("targetRootBone", "Root") or "Root")
    clear_bone_animation(target_action, target_root_bone)
    if target_root_bone in target_armature.pose.bones:
        root_pose_bone = target_armature.pose.bones[target_root_bone]
        root_pose_bone.location = Vector((0.0, 0.0, 0.0))
        root_pose_bone.rotation_mode = "QUATERNION"
        root_pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        root_pose_bone.scale = Vector((1.0, 1.0, 1.0))
    append_trace(
        trace_output,
        f"bake:root_motion_processed corrections={applied_corrections} disabled_rotation={cleared_rotation_bones}",
    )

    for fcurve in target_action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"
    append_trace(trace_output, "bake:keyframes_linearized")

    return target_action, float(scale_ratio)


def pose_positions_for_frame(armature: bpy.types.Object, bone_names: list[str]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for bone_name in bone_names:
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        head_world = armature.matrix_world @ pose_bone.head
        positions[bone_name] = np.asarray([head_world.x, head_world.y, head_world.z], dtype=np.float64)
    return positions


def build_pose_payload(
    armature_name: str,
    bone_order: list[str],
    frame_indices: list[int],
    poses: list[list[list[float]]],
) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "armatureName": armature_name,
        "boneOrder": bone_order,
        "frameIndices": [int(value) for value in frame_indices],
        "poses": poses,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_trace(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{utc_now_iso()} {message}\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)


def relative_artifact_path(report_output: Path, artifact_path: Path) -> str:
    try:
        return artifact_path.relative_to(report_output.parent).as_posix()
    except ValueError:
        return artifact_path.as_posix()


def build_diagnostics(
    source_frames: list[dict[str, np.ndarray]],
    target_frames: list[dict[str, np.ndarray]],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_parents = {
        "pelvis": None,
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
    target_parents = dict(profile.get("targetParents", {}) or {})
    direction_errors: list[float] = []
    hinge_errors: list[float] = []
    basis_errors: list[float] = []
    per_joint: dict[str, dict[str, list[float]]] = {}

    for source_positions, target_positions in zip(source_frames, target_frames):
        for mapping in list(profile.get("jointMappings", []) or []):
            source_joint = str(mapping["sourceJoint"])
            target_joint = str(mapping["targetBoneName"])
            joint_stats = per_joint.setdefault(
                source_joint,
                {"boneDirectionErrorDeg": [], "hingeAngleErrorDeg": [], "localBasisErrorDeg": []},
            )

            primary_source = resolve_reference_vector(source_positions, source_parents, source_joint, dict(mapping.get("primarySourceReference", {}) or {}))
            primary_target = resolve_reference_vector(target_positions, target_parents, target_joint, dict(mapping.get("primaryTargetReference", {}) or {}))
            direction_error = angle_deg(primary_source, primary_target)
            direction_errors.append(direction_error)
            joint_stats["boneDirectionErrorDeg"].append(direction_error)

            joint_class = str(mapping.get("jointClass", "3dof") or "3dof")
            if joint_class == "hinge":
                source_child = str((mapping.get("primarySourceReference", {}) or {}).get("joint", "") or "")
                target_child = str((mapping.get("primaryTargetReference", {}) or {}).get("joint", "") or "")
                hinge_error = abs(
                    joint_angle(source_positions, source_parents, source_joint, source_child)
                    - joint_angle(target_positions, target_parents, target_joint, target_child)
                )
                hinge_errors.append(hinge_error)
                joint_stats["hingeAngleErrorDeg"].append(hinge_error)
            else:
                secondary_source = resolve_reference_vector(source_positions, source_parents, source_joint, dict(mapping.get("secondarySourceReference", {}) or {}))
                secondary_target = resolve_reference_vector(target_positions, target_parents, target_joint, dict(mapping.get("secondaryTargetReference", {}) or {}))
                source_basis = build_basis(primary_source, secondary_source)
                target_basis = build_basis(primary_target, secondary_target)
                delta = target_basis @ source_basis.T
                trace_value = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
                basis_error = float(np.degrees(np.arccos(trace_value)))
                basis_errors.append(basis_error)
                joint_stats["localBasisErrorDeg"].append(basis_error)

    summary = {
        "hingeAngleErrorDeg": summarize_values(hinge_errors),
        "boneDirectionErrorDeg": summarize_values(direction_errors),
        "localBasisErrorDeg": summarize_values(basis_errors),
    }
    per_joint_summary = {
        joint_name: {
            "boneDirectionErrorDeg": summarize_values(values["boneDirectionErrorDeg"]),
            "hingeAngleErrorDeg": summarize_values(values["hingeAngleErrorDeg"]),
            "localBasisErrorDeg": summarize_values(values["localBasisErrorDeg"]),
        }
        for joint_name, values in per_joint.items()
    }
    return summary, per_joint_summary


def report_issues(summary: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if float(summary["hingeAngleErrorDeg"]["mean"]) > float(thresholds.get("maxHingeAngleMeanDeg", 8.0) or 8.0):
        issues.append("hinge_angle_mean_too_high")
    if float(summary["hingeAngleErrorDeg"]["p95"]) > float(thresholds.get("maxHingeAngleP95Deg", 18.0) or 18.0):
        issues.append("hinge_angle_p95_too_high")
    if float(summary["boneDirectionErrorDeg"]["mean"]) > float(thresholds.get("maxBoneDirectionMeanDeg", 10.0) or 10.0):
        issues.append("bone_direction_mean_too_high")
    if float(summary["boneDirectionErrorDeg"]["p95"]) > float(thresholds.get("maxBoneDirectionP95Deg", 22.0) or 22.0):
        issues.append("bone_direction_p95_too_high")
    if float(summary["localBasisErrorDeg"]["mean"]) > float(thresholds.get("maxLocalBasisMeanDeg", 18.0) or 18.0):
        issues.append("local_basis_mean_too_high")
    if float(summary["localBasisErrorDeg"]["p95"]) > float(thresholds.get("maxLocalBasisP95Deg", 35.0) or 35.0):
        issues.append("local_basis_p95_too_high")
    return issues


def root_bone_is_animated(action: bpy.types.Action, bone_name: str) -> bool:
    prefix = f'pose.bones["{bone_name}"].'
    return any(fcurve.data_path.startswith(prefix) for fcurve in action.fcurves)


def motion_root_has_animation(action: bpy.types.Action, bone_name: str) -> dict[str, bool]:
    location_prefix = f'pose.bones["{bone_name}"].location'
    rotation_prefix = f'pose.bones["{bone_name}"].rotation_quaternion'
    return {
        "location": any(fcurve.data_path == location_prefix for fcurve in action.fcurves),
        "rotation": any(fcurve.data_path == rotation_prefix for fcurve in action.fcurves),
    }


def run_pipeline(args: argparse.Namespace) -> None:
    trace_output = Path(getattr(args, "trace_output", "")).resolve() if getattr(args, "trace_output", "") else None
    append_trace(trace_output, "run_pipeline:start")
    profile = load_willa_retarget_profile(profile_path=Path(args.profile))
    if getattr(args, "basis_mode", None):
        profile["basisMode"] = str(args.basis_mode)
    if getattr(args, "skip_basis_correction", False):
        profile["basisCorrectionTransforms"] = {}
    validate_willa_retarget_profile(profile)
    append_trace(trace_output, "profile:loaded")

    ensure_object_mode()
    ensure_bvh_import_addon()
    append_trace(trace_output, "blender:addon_enabled")

    target_armature = choose_target_armature(profile)
    append_trace(trace_output, f"target_armature:{target_armature.name}")
    source_armature = import_bvh(Path(args.input_bvh).resolve())
    append_trace(trace_output, f"source_armature:{source_armature.name}")
    source_action = resolve_action(source_armature)
    if source_action is None:
        raise RuntimeError("Imported source BVH did not create an action")
    append_trace(trace_output, f"source_action:{source_action.name}")

    target_action, scale_ratio = bake_retarget_action(
        source_armature=source_armature,
        source_action=source_action,
        target_armature=target_armature,
        profile=profile,
        motion_prefix=Path(args.input_bvh).stem,
        trace_output=trace_output,
    )
    append_trace(trace_output, f"bake:done scale_ratio={scale_ratio:.6f}")

    frame_start = int(math.floor(target_action.frame_range[0]))
    frame_end = int(math.ceil(target_action.frame_range[1]))
    bpy.context.scene.frame_start = frame_start
    bpy.context.scene.frame_end = frame_end
    bpy.context.scene.render.fps = 30
    target_armature.location = Vector((0.0, 0.0, 0.0))
    target_armature.rotation_mode = "QUATERNION"
    target_armature.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    target_armature.scale = Vector((1.0, 1.0, 1.0))

    source_bone_order = list(profile.get("sourceBoneOrder", []) or [])
    target_bone_order = [dict(profile["boneMap"])[name] for name in source_bone_order]
    frame_indices = list(range(frame_start, frame_end + 1))
    source_frames: list[dict[str, np.ndarray]] = []
    target_frames: list[dict[str, np.ndarray]] = []
    source_pose_payload: list[list[list[float]]] = []
    target_pose_payload: list[list[list[float]]] = []

    for frame in frame_indices:
        bpy.context.scene.frame_set(frame)
        bpy.context.view_layer.update()
        source_positions = pose_positions_for_frame(source_armature, source_bone_order)
        target_positions = pose_positions_for_frame(target_armature, target_bone_order)
        source_frames.append(source_positions)
        target_frames.append(target_positions)
        source_pose_payload.append([[float(value) for value in source_positions[name]] for name in source_bone_order])
        target_pose_payload.append([[float(value) for value in target_positions[name]] for name in target_bone_order])

    append_trace(trace_output, "diagnostics:build_summary_start")
    summary, per_joint_summary = build_diagnostics(source_frames, target_frames, profile)
    append_trace(trace_output, "diagnostics:build_summary_done")
    issues = report_issues(summary, dict(profile.get("thresholds", {}) or {}))

    report_output = Path(args.report_output).resolve()
    source_pose_output = report_output.with_name(report_output.stem.replace("_report", "_source_pose") + ".json")
    target_pose_output = report_output.with_name(report_output.stem.replace("_report", "_target_pose") + ".json")
    append_trace(trace_output, "diagnostics:write_source_pose_start")
    write_json(source_pose_output, build_pose_payload(source_armature.name, source_bone_order, frame_indices, source_pose_payload))
    append_trace(trace_output, "diagnostics:write_source_pose_done")
    append_trace(trace_output, "diagnostics:write_target_pose_start")
    write_json(target_pose_output, build_pose_payload(target_armature.name, target_bone_order, frame_indices, target_pose_payload))
    append_trace(trace_output, "diagnostics:write_target_pose_done")

    output_blend = Path(args.output_blend).resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    append_trace(trace_output, "blend:save_start")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    append_trace(trace_output, f"blend:saved {output_blend}")

    target_root_bone = str(profile["targetRootBone"])
    target_motion_root = str(profile["targetMotionRootBone"])
    root_bone_exists = target_root_bone in target_armature.data.bones
    target_root_bone_animated = root_bone_is_animated(target_action, target_root_bone) if root_bone_exists else False
    motion_root_animation = motion_root_has_animation(target_action, target_motion_root)
    report = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "status": "pass" if not issues else "warning",
        "pipelineStrategy": "constraint_bake_canonical",
        "inputBvh": str(Path(args.input_bvh).resolve()),
        "targetSceneBlend": str(Path(profile["targetSceneBlend"]).resolve()),
        "savedBlend": str(output_blend),
        "targetArmatureName": str(target_armature.name),
        "sourceArmatureName": str(source_armature.name),
        "targetRootBone": target_root_bone,
        "targetRootBoneExists": root_bone_exists,
        "targetRootBoneAnimated": target_root_bone_animated,
        "rootLocked": root_bone_exists and not target_root_bone_animated,
        "targetMotionRootBone": target_motion_root,
        "sourceMotionRootBone": str(profile["sourceMotionRootBone"]),
        "rootMotionMode": str(profile["rootMotionMode"]),
        "basisMode": str(profile["basisMode"]),
        "basisCorrectionSource": str(profile.get("referenceDatasetProfile", "unknown") or "unknown"),
        "skipBasisCorrection": bool(getattr(args, "skip_basis_correction", False)),
        "basisCorrectionTransforms": dict(profile.get("basisCorrectionTransforms", {}) or {}),
        "rotationDisabledSourceJoints": list(profile.get("rotationDisabledSourceJoints", []) or []),
        "scaleRatio": float(scale_ratio),
        "frameRange": [frame_start, frame_end],
        "frameCount": len(frame_indices),
        "fps": int(bpy.context.scene.render.fps),
        "mappedBoneCount": len(source_bone_order),
        "targetActionFcurveCount": len(target_action.fcurves),
        "motionRootHasAnimation": motion_root_animation,
        "armatureObjectTransform": {
            "location": [float(value) for value in target_armature.location],
            "rotationQuaternion": [float(value) for value in target_armature.rotation_quaternion],
            "scale": [float(value) for value in target_armature.scale],
        },
        "thresholds": dict(profile.get("thresholds", {}) or {}),
        "metricsSummary": summary,
        "perJointSummary": per_joint_summary,
        "issues": issues,
        "artifacts": {
            "sourcePoseJson": relative_artifact_path(report_output, source_pose_output),
            "targetPoseJson": relative_artifact_path(report_output, target_pose_output),
        },
    }
    append_trace(trace_output, "report:write_start")
    write_json(report_output, report)
    append_trace(trace_output, f"report:written {report_output}")


def write_error_report(args: argparse.Namespace, exc: Exception, trace_output: Path | None) -> None:
    report_output = Path(args.report_output).resolve()
    error_payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "status": "error",
        "inputBvh": str(Path(args.input_bvh).resolve()),
        "errorType": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    append_trace(trace_output, f"error:{type(exc).__name__}:{exc}")
    write_json(report_output, error_payload)
    print(traceback.format_exc())


def maybe_quit_blender(args: argparse.Namespace, trace_output: Path | None) -> None:
    if not getattr(args, "quit_after_run", False):
        return
    if bpy.app.background:
        return
    if not bpy.context.window_manager.windows:
        return
    append_trace(trace_output, "main:quit_after_run")
    bpy.ops.wm.quit_blender()


def main() -> int:
    args = parse_args()
    report_output = Path(args.report_output).resolve()
    trace_output = report_output.with_name(report_output.stem.replace("_report", "_trace") + ".log")
    setattr(args, "trace_output", str(trace_output))
    append_trace(trace_output, "main:loaded")

    def delayed_run() -> None | float:
        if not bpy.context.window_manager.windows:
            append_trace(trace_output, "timer:waiting_for_window")
            return 0.25
        try:
            append_trace(trace_output, "timer:run_pipeline")
            run_pipeline(args)
            maybe_quit_blender(args, trace_output)
        except Exception as exc:  # pragma: no cover - Blender runtime only
            write_error_report(args, exc, trace_output)
            maybe_quit_blender(args, trace_output)
        return None

    if bpy.app.background or bpy.context.window_manager.windows:
        try:
            append_trace(trace_output, "main:immediate_run")
            run_pipeline(args)
            maybe_quit_blender(args, trace_output)
        except Exception as exc:  # pragma: no cover - Blender runtime only
            write_error_report(args, exc, trace_output)
            maybe_quit_blender(args, trace_output)
    else:
        append_trace(trace_output, "main:register_timer")
        bpy.app.timers.register(delayed_run, first_interval=0.5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
