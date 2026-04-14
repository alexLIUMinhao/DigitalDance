#!/usr/bin/env python3
"""Retarget a canonical Willa BVH onto Willa using rest-pose local rotations."""

from __future__ import annotations

import argparse
import json
import math
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import bpy  # type: ignore
import numpy as np
from mathutils import Matrix, Quaternion, Vector  # type: ignore


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.bvh import frame_pose, parse_bvh, rest_pose  # noqa: E402
from music_motion_lab.willa_retarget import (  # noqa: E402
    BRIDGE_BONE_ORDER_22,
    angle_deg,
    build_basis,
    joint_angle,
    load_willa_retarget_profile,
    resolve_reference_vector,
    summarize_values,
    validate_willa_retarget_profile,
)


SOURCE_PARENT_BY_BONE = {
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

LEFT_LEG_SOURCE_BONES = ("left_hip", "left_knee", "left_ankle")
LEFT_LEG_HUMAN_BONES = ("leftUpperLeg", "leftLowerLeg", "leftFoot")

ROTATION_LIMITS_DEG_BY_HUMAN_BONE = {
    "spine": 25.0,
    "chest": 30.0,
    "upperChest": 35.0,
    "neck": 28.0,
    "head": 26.0,
    "leftUpperArm": 95.0,
    "rightUpperArm": 95.0,
    "leftLowerArm": 145.0,
    "rightLowerArm": 145.0,
    "leftUpperLeg": 85.0,
    "rightUpperLeg": 85.0,
    "leftLowerLeg": 145.0,
    "rightLowerLeg": 145.0,
}

PROXY_RADII_BY_HUMAN_BONE = {
    "hips": 0.16,
    "chest": 0.14,
    "upperChest": 0.12,
    "neck": 0.08,
    "head": 0.11,
    "leftUpperArm": 0.085,
    "rightUpperArm": 0.085,
    "leftLowerArm": 0.075,
    "rightLowerArm": 0.075,
    "leftHand": 0.065,
    "rightHand": 0.065,
    "leftUpperLeg": 0.11,
    "rightUpperLeg": 0.11,
    "leftLowerLeg": 0.09,
    "rightLowerLeg": 0.09,
}

COLLISION_PAIRS_BY_HUMAN_BONE = [
    ("head", "chest"),
    ("head", "upperChest"),
    ("neck", "chest"),
    ("leftHand", "chest"),
    ("rightHand", "chest"),
    ("leftLowerArm", "chest"),
    ("rightLowerArm", "chest"),
    ("leftHand", "hips"),
    ("rightHand", "hips"),
    ("leftUpperLeg", "chest"),
    ("rightUpperLeg", "chest"),
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-bvh", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--normalization-report", required=False)
    parser.add_argument("--output-blend", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--skip-penetration-pass", action="store_true")
    parser.add_argument("--skip-diagnostics", action="store_true")
    parser.add_argument("--quit-after-run", action="store_true")
    return parser.parse_args(argv)


def ensure_object_mode() -> None:
    active_object = bpy.context.view_layer.objects.active
    if active_object and active_object.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")


def append_trace(path: Path | None, message: str) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now_iso()} {message}\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def relative_artifact_path(report_output: Path, artifact_path: Path) -> str:
    try:
        return artifact_path.relative_to(report_output.parent).as_posix()
    except ValueError:
        return artifact_path.as_posix()


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


def source_leg_chain_length(source_rest_positions: dict[str, np.ndarray]) -> float:
    return float(
        np.linalg.norm(source_rest_positions["left_hip"] - source_rest_positions["pelvis"])
        + np.linalg.norm(source_rest_positions["left_knee"] - source_rest_positions["left_hip"])
        + np.linalg.norm(source_rest_positions["left_ankle"] - source_rest_positions["left_knee"])
    )


def target_leg_chain_length(target_armature: bpy.types.Object, human_to_target: dict[str, str]) -> float:
    return float(sum(bone_length(target_armature, human_to_target[name]) for name in LEFT_LEG_HUMAN_BONES))


def compute_scale_ratio(source_rest_positions: dict[str, np.ndarray], target_armature: bpy.types.Object, human_to_target: dict[str, str]) -> float:
    source_length = source_leg_chain_length(source_rest_positions)
    target_length = target_leg_chain_length(target_armature, human_to_target)
    if source_length <= 1e-6 or target_length <= 1e-6:
        return 1.0
    return target_length / source_length


def target_rest_positions(target_armature: bpy.types.Object, profile: dict[str, Any]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for bone_name in list(profile.get("targetParents", {}) or {}):
        bone = target_armature.data.bones.get(bone_name)
        if bone is None:
            continue
        positions[bone_name] = np.asarray([bone.head_local.x, bone.head_local.y, bone.head_local.z], dtype=np.float64)
    return positions


def compute_basis_map(
    profile: dict[str, Any],
    source_rest_positions: dict[str, np.ndarray],
    target_rest_positions_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    target_parents = dict(profile.get("targetParents", {}) or {})
    basis_map: dict[str, np.ndarray] = {}
    for mapping in list(profile.get("jointMappings", []) or []):
        source_joint = str(mapping["sourceJoint"])
        target_bone = str(mapping["targetBoneName"])
        primary_source = resolve_reference_vector(
            source_rest_positions,
            SOURCE_PARENT_BY_BONE,
            source_joint,
            dict(mapping.get("primarySourceReference", {}) or {}),
        )
        secondary_source = resolve_reference_vector(
            source_rest_positions,
            SOURCE_PARENT_BY_BONE,
            source_joint,
            dict(mapping.get("secondarySourceReference", {}) or {}),
        )
        primary_target = resolve_reference_vector(
            target_rest_positions_map,
            target_parents,
            target_bone,
            dict(mapping.get("primaryTargetReference", {}) or {}),
        )
        secondary_target = resolve_reference_vector(
            target_rest_positions_map,
            target_parents,
            target_bone,
            dict(mapping.get("secondaryTargetReference", {}) or {}),
        )
        source_basis = build_basis(primary_source, secondary_source)
        target_basis = build_basis(primary_target, secondary_target)
        basis_map[source_joint] = target_basis @ source_basis.T
    return basis_map


def basis_correction_quaternions(profile: dict[str, Any]) -> dict[str, Quaternion]:
    if bool(profile.get("useBasisMapOverrides", False)):
        return {}
    corrections: dict[str, Quaternion] = {}
    for source_joint, payload in dict(profile.get("basisCorrectionTransforms", {}) or {}).items():
        matrix_values = np.asarray((payload or {}).get("matrix", np.eye(3)), dtype=np.float64).reshape((3, 3))
        correction = Matrix(matrix_values.tolist()).to_quaternion().normalized()
        if abs(correction.angle) <= 1e-6:
            continue
        corrections[str(source_joint)] = correction
    return corrections


def pose_positions_for_frame(armature: bpy.types.Object, bone_names: list[str]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {}
    for bone_name in bone_names:
        pose_bone = armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        head_world = armature.matrix_world @ pose_bone.head
        positions[bone_name] = np.asarray([head_world.x, head_world.y, head_world.z], dtype=np.float64)
    return positions


def build_pose_payload(armature_name: str, bone_order: list[str], frame_indices: list[int], poses: list[list[list[float]]]) -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "armatureName": armature_name,
        "boneOrder": bone_order,
        "frameIndices": [int(value) for value in frame_indices],
        "poses": poses,
    }


def clamp_quaternion_angle(quaternion: Quaternion, max_degrees: float) -> Quaternion:
    current = quaternion.normalized()
    max_radians = math.radians(max_degrees)
    if current.angle <= max_radians + 1e-8:
        return current
    axis = current.axis if current.axis.length > 1e-8 else Vector((0.0, 0.0, 1.0))
    clamped = Quaternion(axis, max_radians)
    clamped.normalize()
    return clamped


def quaternion_delta_degrees(before: Quaternion, after: Quaternion) -> float:
    delta = before.rotation_difference(after)
    return float(math.degrees(abs(delta.angle)))


def evaluate_collisions(
    target_armature: bpy.types.Object,
    human_to_target: dict[str, str],
) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
    positions = pose_positions_for_frame(target_armature, [name for name in human_to_target.values() if name in target_armature.pose.bones])
    details: list[dict[str, Any]] = []
    pair_counts: dict[str, int] = defaultdict(int)
    for lhs_human, rhs_human in COLLISION_PAIRS_BY_HUMAN_BONE:
        lhs_target = human_to_target.get(lhs_human)
        rhs_target = human_to_target.get(rhs_human)
        if lhs_target not in positions or rhs_target not in positions:
            continue
        radius_sum = float(PROXY_RADII_BY_HUMAN_BONE[lhs_human] + PROXY_RADII_BY_HUMAN_BONE[rhs_human])
        distance = float(np.linalg.norm(positions[lhs_target] - positions[rhs_target]))
        penetration = radius_sum - distance
        if penetration > 0.0:
            key = f"{lhs_human}:{rhs_human}"
            pair_counts[key] += 1
            details.append(
                {
                    "pair": key,
                    "lhsTargetBone": lhs_target,
                    "rhsTargetBone": rhs_target,
                    "distance": distance,
                    "radiusSum": radius_sum,
                    "penetration": penetration,
                }
            )
    return len(details), details, dict(pair_counts)


def apply_pose_frame(
    target_armature: bpy.types.Object,
    target_order: list[str],
    rotation_buffers: dict[str, list[Quaternion]],
    motion_root_locations: np.ndarray,
    frame_index: int,
    target_motion_root: str,
    target_root_bone: str,
) -> None:
    for bone_name in target_order:
        pose_bone = target_armature.pose.bones.get(bone_name)
        if pose_bone is None:
            continue
        pose_bone.rotation_mode = "QUATERNION"
        pose_bone.rotation_quaternion = rotation_buffers[bone_name][frame_index].copy()
        if bone_name == target_motion_root:
            location = motion_root_locations[frame_index]
            pose_bone.location = Vector((float(location[0]), float(location[1]), float(location[2])))
        else:
            pose_bone.location = Vector((0.0, 0.0, 0.0))
        pose_bone.scale = Vector((1.0, 1.0, 1.0))
    if target_root_bone in target_armature.pose.bones:
        root_pose_bone = target_armature.pose.bones[target_root_bone]
        root_pose_bone.rotation_mode = "QUATERNION"
        root_pose_bone.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
        root_pose_bone.location = Vector((0.0, 0.0, 0.0))
        root_pose_bone.scale = Vector((1.0, 1.0, 1.0))
    bpy.context.view_layer.update()


def run_penetration_pass(
    target_armature: bpy.types.Object,
    target_order: list[str],
    rotation_buffers: dict[str, list[Quaternion]],
    motion_root_locations: np.ndarray,
    human_to_target: dict[str, str],
    target_motion_root: str,
    target_root_bone: str,
) -> dict[str, Any]:
    frame_count = int(motion_root_locations.shape[0])
    before_total = 0
    after_total = 0
    before_pair_counts: dict[str, int] = defaultdict(int)
    after_pair_counts: dict[str, int] = defaultdict(int)
    corrected_bones: set[str] = set()
    max_correction_angle = 0.0

    human_limits = {
        human_to_target[human]: degrees
        for human, degrees in ROTATION_LIMITS_DEG_BY_HUMAN_BONE.items()
        if human in human_to_target and human_to_target[human] in rotation_buffers
    }

    for frame_index in range(frame_count):
        apply_pose_frame(target_armature, target_order, rotation_buffers, motion_root_locations, frame_index, target_motion_root, target_root_bone)
        before_count, before_details, before_pairs = evaluate_collisions(target_armature, human_to_target)
        before_total += before_count
        for key, value in before_pairs.items():
            before_pair_counts[key] += value

        for bone_name, max_degrees in human_limits.items():
            current = rotation_buffers[bone_name][frame_index]
            clamped = clamp_quaternion_angle(current, max_degrees)
            max_correction_angle = max(max_correction_angle, quaternion_delta_degrees(current, clamped))
            if quaternion_delta_degrees(current, clamped) > 1e-4:
                corrected_bones.add(bone_name)
            rotation_buffers[bone_name][frame_index] = clamped

        for detail in before_details:
            pair = str(detail["pair"])
            penetration = float(detail["penetration"])
            factor = min(0.6, max(0.15, penetration * 2.0))
            if pair.startswith("head:") or pair.startswith("neck:"):
                for human in ("neck", "head"):
                    target_bone = human_to_target.get(human)
                    if target_bone in rotation_buffers:
                        before = rotation_buffers[target_bone][frame_index]
                        after = before.slerp(Quaternion((1.0, 0.0, 0.0, 0.0)), factor)
                        max_correction_angle = max(max_correction_angle, quaternion_delta_degrees(before, after))
                        rotation_buffers[target_bone][frame_index] = after
                        corrected_bones.add(target_bone)
            elif pair.startswith("leftHand:") or pair.startswith("leftLowerArm:"):
                for human in ("leftUpperArm", "leftLowerArm", "leftHand"):
                    target_bone = human_to_target.get(human)
                    if target_bone in rotation_buffers:
                        before = rotation_buffers[target_bone][frame_index]
                        after = before.slerp(Quaternion((1.0, 0.0, 0.0, 0.0)), factor * 0.6)
                        max_correction_angle = max(max_correction_angle, quaternion_delta_degrees(before, after))
                        rotation_buffers[target_bone][frame_index] = after
                        corrected_bones.add(target_bone)
            elif pair.startswith("rightHand:") or pair.startswith("rightLowerArm:"):
                for human in ("rightUpperArm", "rightLowerArm", "rightHand"):
                    target_bone = human_to_target.get(human)
                    if target_bone in rotation_buffers:
                        before = rotation_buffers[target_bone][frame_index]
                        after = before.slerp(Quaternion((1.0, 0.0, 0.0, 0.0)), factor * 0.6)
                        max_correction_angle = max(max_correction_angle, quaternion_delta_degrees(before, after))
                        rotation_buffers[target_bone][frame_index] = after
                        corrected_bones.add(target_bone)
            elif pair.startswith("leftUpperLeg:"):
                for human in ("leftUpperLeg", "leftLowerLeg"):
                    target_bone = human_to_target.get(human)
                    if target_bone in rotation_buffers:
                        before = rotation_buffers[target_bone][frame_index]
                        after = before.slerp(Quaternion((1.0, 0.0, 0.0, 0.0)), factor * 0.5)
                        max_correction_angle = max(max_correction_angle, quaternion_delta_degrees(before, after))
                        rotation_buffers[target_bone][frame_index] = after
                        corrected_bones.add(target_bone)
            elif pair.startswith("rightUpperLeg:"):
                for human in ("rightUpperLeg", "rightLowerLeg"):
                    target_bone = human_to_target.get(human)
                    if target_bone in rotation_buffers:
                        before = rotation_buffers[target_bone][frame_index]
                        after = before.slerp(Quaternion((1.0, 0.0, 0.0, 0.0)), factor * 0.5)
                        max_correction_angle = max(max_correction_angle, quaternion_delta_degrees(before, after))
                        rotation_buffers[target_bone][frame_index] = after
                        corrected_bones.add(target_bone)

    corrected_bone_list = sorted(corrected_bones)
    for bone_name in corrected_bone_list:
        if bone_name not in rotation_buffers:
            continue
        for frame_index in range(1, frame_count - 1):
            prev_quat = rotation_buffers[bone_name][frame_index - 1]
            next_quat = rotation_buffers[bone_name][frame_index + 1]
            blended = prev_quat.slerp(next_quat, 0.5)
            rotation_buffers[bone_name][frame_index] = rotation_buffers[bone_name][frame_index].slerp(blended, 0.2)

    for frame_index in range(frame_count):
        apply_pose_frame(target_armature, target_order, rotation_buffers, motion_root_locations, frame_index, target_motion_root, target_root_bone)
        after_count, _, after_pairs = evaluate_collisions(target_armature, human_to_target)
        after_total += after_count
        for key, value in after_pairs.items():
            after_pair_counts[key] += value

    return {
        "beforeCollisionCount": before_total,
        "afterCollisionCount": after_total,
        "correctedBones": corrected_bone_list,
        "maxSingleFrameCorrectionAngleDeg": float(max_correction_angle),
        "pairCountsBefore": dict(before_pair_counts),
        "pairCountsAfter": dict(after_pair_counts),
    }


def build_diagnostics(
    source_frames: list[dict[str, np.ndarray]],
    target_frames: list[dict[str, np.ndarray]],
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
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
            primary_source = resolve_reference_vector(source_positions, SOURCE_PARENT_BY_BONE, source_joint, dict(mapping.get("primarySourceReference", {}) or {}))
            primary_target = resolve_reference_vector(target_positions, target_parents, target_joint, dict(mapping.get("primaryTargetReference", {}) or {}))
            direction_error = angle_deg(primary_source, primary_target)
            direction_errors.append(direction_error)
            joint_stats["boneDirectionErrorDeg"].append(direction_error)

            joint_class = str(mapping.get("jointClass", "3dof") or "3dof")
            if joint_class == "hinge":
                source_child = str((mapping.get("primarySourceReference", {}) or {}).get("joint", "") or "")
                target_child = str((mapping.get("primaryTargetReference", {}) or {}).get("joint", "") or "")
                hinge_error = abs(
                    joint_angle(source_positions, SOURCE_PARENT_BY_BONE, source_joint, source_child)
                    - joint_angle(target_positions, target_parents, target_joint, target_child)
                )
                hinge_errors.append(hinge_error)
                joint_stats["hingeAngleErrorDeg"].append(hinge_error)
            else:
                secondary_source = resolve_reference_vector(source_positions, SOURCE_PARENT_BY_BONE, source_joint, dict(mapping.get("secondarySourceReference", {}) or {}))
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


def report_issues(summary: dict[str, Any], thresholds: dict[str, Any], penetration_report: dict[str, Any]) -> list[str]:
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
    if int(penetration_report.get("afterCollisionCount", 0)) >= int(penetration_report.get("beforeCollisionCount", 0)):
        issues.append("penetration_not_improved")
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


def write_target_action(
    target_armature: bpy.types.Object,
    target_action: bpy.types.Action,
    target_order: list[str],
    rotation_buffers: dict[str, list[Quaternion]],
    motion_root_locations: np.ndarray,
    frame_indices: list[int],
    target_motion_root: str,
    target_root_bone: str,
) -> None:
    ensure_object_mode()
    bpy.context.view_layer.objects.active = target_armature
    frame_values = [float(frame) for frame in frame_indices]

    def write_curve(data_path: str, array_index: int, values: list[float], action_group: str) -> None:
        fcurve = target_action.fcurves.new(data_path=data_path, index=array_index, action_group=action_group)
        fcurve.keyframe_points.add(len(frame_values))
        coords: list[float] = []
        for frame, value in zip(frame_values, values):
            coords.extend((frame, float(value)))
        fcurve.keyframe_points.foreach_set("co", coords)
        fcurve.update()

    for bone_name in target_order:
        if bone_name == target_root_bone:
            continue
        quaternions = rotation_buffers.get(bone_name, [])
        if not quaternions:
            continue
        components = [[], [], [], []]
        for quaternion in quaternions:
            normalized = quaternion.normalized()
            components[0].append(float(normalized.w))
            components[1].append(float(normalized.x))
            components[2].append(float(normalized.y))
            components[3].append(float(normalized.z))
        data_path = f'pose.bones["{bone_name}"].rotation_quaternion'
        for array_index, values in enumerate(components):
            write_curve(data_path, array_index, values, bone_name)
        if bone_name == target_motion_root:
            location_path = f'pose.bones["{bone_name}"].location'
            for array_index in range(3):
                write_curve(
                    location_path,
                    array_index,
                    [float(row[array_index]) for row in motion_root_locations],
                    bone_name,
                )

    if frame_indices:
        apply_pose_frame(target_armature, target_order, rotation_buffers, motion_root_locations, 0, target_motion_root, target_root_bone)
    for fcurve in target_action.fcurves:
        for keyframe in fcurve.keyframe_points:
            keyframe.interpolation = "LINEAR"


def run_pipeline(args: argparse.Namespace) -> None:
    report_output = Path(args.report_output).resolve()
    trace_output = report_output.with_name(report_output.stem.replace("_report", "_trace") + ".log")
    append_trace(trace_output, "run_pipeline:start")

    profile = load_willa_retarget_profile(profile_path=Path(args.profile).resolve())
    validate_willa_retarget_profile(profile)
    append_trace(trace_output, "profile:loaded")

    normalization_report_override = getattr(args, "normalization_report", None)
    normalization_report_path = (
        Path(normalization_report_override).resolve()
        if normalization_report_override
        else Path(profile.get("defaultCanonicalReport", report_output)).resolve()
    )
    normalization_report = json.loads(normalization_report_path.read_text(encoding="utf-8")) if normalization_report_path.exists() else {}

    parsed = parse_bvh(Path(args.input_bvh).resolve())
    source_rest = rest_pose(parsed.nodes)
    source_rest_positions = {name: np.asarray(source_rest.world_positions[name], dtype=np.float64) for name in BRIDGE_BONE_ORDER_22}
    frame_indices = list(range(1, int(parsed.motion.shape[0]) + 1))
    fps = int(round(1.0 / parsed.frame_time)) if parsed.frame_time > 0 else 30
    append_trace(trace_output, f"source:frames={len(frame_indices)} fps={fps}")

    ensure_object_mode()
    target_armature = choose_target_armature(profile)
    append_trace(trace_output, f"target_armature:{target_armature.name}")
    target_action = create_target_action(target_armature, f"{Path(args.input_bvh).stem}_willa_retarget")

    target_rest_positions_map = target_rest_positions(target_armature, profile)
    basis_map = compute_basis_map(profile, source_rest_positions, target_rest_positions_map)
    if bool(profile.get("useBasisMapOverrides", False)):
        for source_joint, payload in dict(profile.get("basisMapOverrides", {}) or {}).items():
            if source_joint not in basis_map:
                continue
            matrix_values = np.asarray((payload or {}).get("matrix", np.eye(3)), dtype=np.float64).reshape((3, 3))
            basis_map[source_joint] = matrix_values
    append_trace(trace_output, f"basis_map:count={len(basis_map)}")
    append_trace(
        trace_output,
        f"basis_map_overrides:count={len(dict(profile.get('basisMapOverrides', {}) or {})) if bool(profile.get('useBasisMapOverrides', False)) else 0}",
    )
    correction_quaternions = basis_correction_quaternions(profile)
    append_trace(trace_output, f"basis_corrections:count={len(correction_quaternions)}")

    human_to_target = dict(profile.get("humanToTarget", {}) or {})
    scale_ratio = compute_scale_ratio(source_rest_positions, target_armature, human_to_target)
    append_trace(trace_output, f"scale_ratio={scale_ratio:.6f}")

    target_order = [dict(profile["boneMap"])[name] for name in BRIDGE_BONE_ORDER_22]
    rotation_buffers = {
        bone_name: [Quaternion((1.0, 0.0, 0.0, 0.0)) for _ in frame_indices]
        for bone_name in target_order
    }
    motion_root_locations = np.zeros((len(frame_indices), 3), dtype=np.float64)
    source_frames: list[dict[str, np.ndarray]] = []
    source_pose_payload: list[list[list[float]]] = []

    source_motion_root = str(profile["sourceMotionRootBone"])
    append_trace(trace_output, "source_buffer_build:start")
    for frame_offset, frame_values in enumerate(parsed.motion):
        pose = frame_pose(parsed.nodes, frame_values)
        source_positions = {name: np.asarray(pose.world_positions[name], dtype=np.float64) for name in BRIDGE_BONE_ORDER_22}
        source_frames.append(source_positions)
        source_pose_payload.append([[float(value) for value in source_positions[name]] for name in BRIDGE_BONE_ORDER_22])
        motion_root_locations[frame_offset] = np.asarray(pose.local_translations[source_motion_root], dtype=np.float64) * scale_ratio
        for mapping in list(profile.get("jointMappings", []) or []):
            source_joint = str(mapping["sourceJoint"])
            target_bone = str(mapping["targetBoneName"])
            source_delta = pose.local_rotations[source_joint].as_matrix()
            basis = np.asarray(basis_map[source_joint], dtype=np.float64)
            mapped_local = basis @ source_delta @ basis.T
            target_quaternion = Matrix(mapped_local.tolist()).to_quaternion().normalized()
            correction_quaternion = correction_quaternions.get(source_joint)
            if correction_quaternion is not None:
                target_quaternion = (target_quaternion @ correction_quaternion).normalized()
            rotation_buffers[target_bone][frame_offset] = target_quaternion
    append_trace(trace_output, "source_buffer_build:done")

    target_motion_root = str(profile["targetMotionRootBone"])
    target_root_bone = str(profile["targetRootBone"])
    if args.skip_penetration_pass:
        penetration_report = {
            "beforeCollisionCount": 0,
            "afterCollisionCount": 0,
            "correctedBones": [],
            "maxSingleFrameCorrectionDeg": 0.0,
            "status": "skipped",
        }
        append_trace(trace_output, "penetration:skipped")
    else:
        append_trace(trace_output, "penetration:start")
        penetration_report = run_penetration_pass(
            target_armature=target_armature,
            target_order=target_order,
            rotation_buffers=rotation_buffers,
            motion_root_locations=motion_root_locations,
            human_to_target=human_to_target,
            target_motion_root=target_motion_root,
            target_root_bone=target_root_bone,
        )
        append_trace(
            trace_output,
            f"penetration:before={penetration_report['beforeCollisionCount']} after={penetration_report['afterCollisionCount']}",
        )

    append_trace(trace_output, "write_target_action:start")
    write_target_action(
        target_armature=target_armature,
        target_action=target_action,
        target_order=target_order,
        rotation_buffers=rotation_buffers,
        motion_root_locations=motion_root_locations,
        frame_indices=frame_indices,
        target_motion_root=target_motion_root,
        target_root_bone=target_root_bone,
    )
    append_trace(trace_output, f"target_action:fcurves={len(target_action.fcurves)}")

    bpy.context.scene.frame_start = frame_indices[0]
    bpy.context.scene.frame_end = frame_indices[-1]
    bpy.context.scene.render.fps = fps
    target_armature.location = Vector((0.0, 0.0, 0.0))
    target_armature.rotation_mode = "QUATERNION"
    target_armature.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    target_armature.scale = Vector((1.0, 1.0, 1.0))

    output_blend = Path(args.output_blend).resolve()
    output_blend.parent.mkdir(parents=True, exist_ok=True)
    append_trace(trace_output, "blend:save_start")
    bpy.ops.wm.save_as_mainfile(filepath=str(output_blend))
    append_trace(trace_output, f"blend:saved {output_blend}")

    root_bone_exists = target_root_bone in target_armature.data.bones
    target_root_bone_animated = root_bone_is_animated(target_action, target_root_bone) if root_bone_exists else False
    motion_root_animation = motion_root_has_animation(target_action, target_motion_root)

    if args.skip_diagnostics:
        report = {
            "schemaVersion": 1,
            "generatedAtUtc": utc_now_iso(),
            "status": "partial",
            "inputBvh": str(Path(args.input_bvh).resolve()),
            "targetSceneBlend": str(Path(profile["targetSceneBlend"]).resolve()),
            "savedBlend": str(output_blend),
            "targetArmatureName": str(target_armature.name),
            "targetRootBone": target_root_bone,
            "targetRootBoneExists": root_bone_exists,
            "targetRootBoneAnimated": target_root_bone_animated,
            "rootLocked": root_bone_exists and not target_root_bone_animated,
            "targetMotionRootBone": target_motion_root,
            "sourceMotionRootBone": source_motion_root,
            "rootMotionMode": str(profile["rootMotionMode"]),
            "basisMode": str(profile["basisMode"]),
            "scaleRatio": float(scale_ratio),
            "frameRange": [frame_indices[0], frame_indices[-1]],
            "frameCount": len(frame_indices),
            "fps": int(fps),
            "mappedBoneCount": len(BRIDGE_BONE_ORDER_22),
            "targetActionFcurveCount": len(target_action.fcurves),
            "motionRootHasAnimation": motion_root_animation,
            "normalization": normalization_report,
            "penetration_pass": penetration_report,
            "issues": ["diagnostics_skipped"],
        }
        write_json(report_output, report)
        append_trace(trace_output, f"report:written {report_output}")
        return

    target_frames: list[dict[str, np.ndarray]] = []
    target_pose_payload: list[list[list[float]]] = []
    append_trace(trace_output, "diagnostics:target_pose_build_start")
    for frame_offset, scene_frame in enumerate(frame_indices):
        bpy.context.scene.frame_set(scene_frame)
        apply_pose_frame(target_armature, target_order, rotation_buffers, motion_root_locations, frame_offset, target_motion_root, target_root_bone)
        target_positions = pose_positions_for_frame(target_armature, target_order)
        target_frames.append(target_positions)
        target_pose_payload.append([[float(value) for value in target_positions[name]] for name in target_order])
    append_trace(trace_output, "diagnostics:target_pose_build_done")

    append_trace(trace_output, "diagnostics:summary_start")
    summary, per_joint_summary = build_diagnostics(source_frames, target_frames, profile)
    append_trace(trace_output, "diagnostics:summary_done")
    issues = report_issues(summary, dict(profile.get("thresholds", {}) or {}), penetration_report)

    source_pose_output = report_output.with_name(report_output.stem.replace("_report", "_source_pose") + ".json")
    target_pose_output = report_output.with_name(report_output.stem.replace("_report", "_target_pose") + ".json")
    append_trace(trace_output, "diagnostics:write_source_pose_start")
    write_json(source_pose_output, build_pose_payload("canonical_source_bvh", list(BRIDGE_BONE_ORDER_22), frame_indices, source_pose_payload))
    append_trace(trace_output, "diagnostics:write_source_pose_done")
    append_trace(trace_output, "diagnostics:write_target_pose_start")
    write_json(target_pose_output, build_pose_payload(target_armature.name, target_order, frame_indices, target_pose_payload))
    append_trace(trace_output, "diagnostics:write_target_pose_done")
    report = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "status": "pass" if not issues else "warning",
        "inputBvh": str(Path(args.input_bvh).resolve()),
        "targetSceneBlend": str(Path(profile["targetSceneBlend"]).resolve()),
        "savedBlend": str(output_blend),
        "targetArmatureName": str(target_armature.name),
        "sourceArmatureName": None,
        "targetRootBone": target_root_bone,
        "targetRootBoneExists": root_bone_exists,
        "targetRootBoneAnimated": target_root_bone_animated,
        "rootLocked": root_bone_exists and not target_root_bone_animated,
        "targetMotionRootBone": target_motion_root,
        "sourceMotionRootBone": source_motion_root,
        "rootMotionMode": str(profile["rootMotionMode"]),
        "basisMode": str(profile["basisMode"]),
        "basisCorrectionSource": str(profile.get("referenceDatasetProfile", "unknown") or "unknown"),
        "basisCorrectionTransforms": dict(profile.get("basisCorrectionTransforms", {}) or {}),
        "rotationDisabledSourceJoints": list(profile.get("rotationDisabledSourceJoints", []) or []),
        "scaleRatio": float(scale_ratio),
        "frameRange": [frame_indices[0], frame_indices[-1]],
        "frameCount": len(frame_indices),
        "fps": int(fps),
        "mappedBoneCount": len(BRIDGE_BONE_ORDER_22),
        "targetActionFcurveCount": len(target_action.fcurves),
        "motionRootHasAnimation": motion_root_animation,
        "armatureObjectTransform": {
            "location": [float(value) for value in target_armature.location],
            "rotationQuaternion": [float(value) for value in target_armature.rotation_quaternion],
            "scale": [float(value) for value in target_armature.scale],
        },
        "thresholds": dict(profile.get("thresholds", {}) or {}),
        "normalization": normalization_report,
        "local_rotation_alignment": {
            "metricsSummary": summary,
            "perJointSummary": per_joint_summary,
        },
        "penetration_pass": penetration_report,
        "metricsSummary": summary,
        "perJointSummary": per_joint_summary,
        "issues": issues,
        "artifacts": {
            "sourcePoseJson": relative_artifact_path(report_output, source_pose_output),
            "targetPoseJson": relative_artifact_path(report_output, target_pose_output),
            "canonicalReport": relative_artifact_path(report_output, normalization_report_path) if normalization_report_path.exists() else None,
        },
    }
    write_json(report_output, report)
    append_trace(trace_output, f"report:written {report_output}")


def write_error_report(args: argparse.Namespace, trace_output: Path, exc: Exception) -> None:
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


def main() -> int:
    args = parse_args()
    report_output = Path(args.report_output).resolve()
    trace_output = report_output.with_name(report_output.stem.replace("_report", "_trace") + ".log")
    append_trace(trace_output, "main:loaded")
    try:
        run_pipeline(args)
    except Exception as exc:  # pragma: no cover - Blender runtime only
        write_error_report(args, trace_output, exc)
    finally:
        if args.quit_after_run:
            append_trace(trace_output, "main:quit_after_run")
            bpy.ops.wm.quit_blender()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
