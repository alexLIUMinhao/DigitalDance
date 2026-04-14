from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from .bvh import parse_bvh, rest_pose, rotation_channels
from .utils import utc_now_iso, write_json
from .willa_retarget import (
    BRIDGE_BONE_ORDER_22,
    angle_deg,
    build_basis,
    load_willa_retarget_profile,
    resolve_reference_vector,
    summarize_values,
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

AXIS_BY_LABEL = {
    "+X": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "-X": np.asarray([-1.0, 0.0, 0.0], dtype=np.float64),
    "+Y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "-Y": np.asarray([0.0, -1.0, 0.0], dtype=np.float64),
    "+Z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    "-Z": np.asarray([0.0, 0.0, -1.0], dtype=np.float64),
}


def safe_normalize(vector: np.ndarray) -> np.ndarray:
    payload = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(payload))
    if norm <= 1e-8:
        return np.zeros((3,), dtype=np.float64)
    return payload / norm


def vector_to_axis_label(vector: np.ndarray) -> dict[str, Any]:
    payload = safe_normalize(np.asarray(vector, dtype=np.float64))
    if np.linalg.norm(payload) <= 1e-8:
        return {"label": "zero", "angleDeg": 0.0}
    best_label = "unknown"
    best_angle = 180.0
    for label, axis in AXIS_BY_LABEL.items():
        angle = angle_deg(payload, axis)
        if angle < best_angle:
            best_label = label
            best_angle = angle
    return {
        "label": best_label,
        "angleDeg": float(best_angle),
        "vector": np.asarray(payload, dtype=np.float64).round(6).tolist(),
    }


def parse_space_descriptor(descriptor: str) -> dict[str, Any]:
    text = str(descriptor or "").strip()
    forward_match = re.search(r"forward=([+-][XYZ])", text, flags=re.IGNORECASE)
    up_match = re.search(r"([XYZ])-up", text, flags=re.IGNORECASE)
    if up_match is None:
        up_match = re.search(r"up=([+-][XYZ])", text, flags=re.IGNORECASE)

    forward_label = (forward_match.group(1).upper() if forward_match else "+Y")
    up_label = (f"+{up_match.group(1).upper()}" if up_match and len(up_match.group(1)) == 1 else (up_match.group(1).upper() if up_match else "+Z"))
    return {
        "descriptor": text,
        "forwardLabel": forward_label,
        "upLabel": up_label,
        "forward": AXIS_BY_LABEL[forward_label].copy(),
        "up": AXIS_BY_LABEL[up_label].copy(),
    }


def load_reference_alignment_profile(profile: dict[str, Any]) -> dict[str, Any]:
    reference_profile_path = Path(str(profile["referenceWillaAlignmentProfile"]))
    return __import__("json").loads(reference_profile_path.read_text(encoding="utf-8"))


def load_willa_rest_dump(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def describe_bvh_motion_contract(path: Path) -> dict[str, Any]:
    parsed = parse_bvh(path)
    nodes = parsed.nodes
    root_index = next((index for index, node in enumerate(nodes) if node.parent is None), None)
    if root_index is None:
        raise ValueError(f"missing root in {path}")
    root_node = nodes[root_index]
    rotation_orders: dict[str, str] = {}
    unique_orders: list[str] = []
    for node in nodes:
        order = "".join(channel[0].upper() for channel in node.channels if channel.endswith("rotation"))
        rotation_orders[node.name] = order
        if order and order not in unique_orders:
            unique_orders.append(order)
    return {
        "path": str(path.resolve()),
        "frameCount": int(parsed.motion.shape[0]),
        "frameTime": float(parsed.frame_time),
        "fps": float(round(1.0 / parsed.frame_time, 6)) if parsed.frame_time > 0 else 0.0,
        "rootName": root_node.name,
        "rootPositionChannels": [channel for channel in root_node.channels if channel.endswith("position")],
        "rootRotationChannels": rotation_channels(root_node),
        "uniqueRotationOrders": unique_orders,
        "perBoneRotationOrder": rotation_orders,
    }


def build_rest_basis(positions: dict[str, np.ndarray], parents: dict[str, str | None], joint_name: str, primary_reference: dict[str, Any], secondary_reference: dict[str, Any]) -> np.ndarray:
    primary = resolve_reference_vector(positions, parents, joint_name, primary_reference)
    secondary = resolve_reference_vector(positions, parents, joint_name, secondary_reference)
    return build_basis(primary, secondary)


def compute_rest_basis_deltas(
    *,
    source_positions: dict[str, np.ndarray],
    target_positions: dict[str, np.ndarray],
    profile: dict[str, Any],
) -> dict[str, Any]:
    source_parents = dict(SOURCE_PARENT_BY_BONE)
    target_parents = dict(profile.get("targetParents", {}) or {})
    per_bone: dict[str, Any] = {}
    delta_angles: list[float] = []

    for mapping in list(profile.get("jointMappings", []) or []):
        source_joint = str(mapping["sourceJoint"])
        target_bone = str(mapping["targetBoneName"])
        if source_joint not in source_positions or target_bone not in target_positions:
            continue
        source_basis = build_rest_basis(
            source_positions,
            source_parents,
            source_joint,
            dict(mapping.get("primarySourceReference", {}) or {}),
            dict(mapping.get("secondarySourceReference", {}) or {}),
        )
        target_basis = build_rest_basis(
            target_positions,
            target_parents,
            target_bone,
            dict(mapping.get("primaryTargetReference", {}) or {}),
            dict(mapping.get("secondaryTargetReference", {}) or {}),
        )
        delta = target_basis @ source_basis.T
        trace_value = float(np.clip((np.trace(delta) - 1.0) * 0.5, -1.0, 1.0))
        delta_angle = float(np.degrees(np.arccos(trace_value)))
        delta_angles.append(delta_angle)
        per_bone[source_joint] = {
            "targetBone": target_bone,
            "deltaAngleDeg": delta_angle,
            "sourcePrimaryAxis": vector_to_axis_label(source_basis[:, 1]),
            "sourceSecondaryAxis": vector_to_axis_label(source_basis[:, 0]),
            "targetPrimaryAxis": vector_to_axis_label(target_basis[:, 1]),
            "targetSecondaryAxis": vector_to_axis_label(target_basis[:, 0]),
            "deltaMatrix": np.asarray(delta, dtype=np.float64).round(6).tolist(),
        }

    worst = sorted(
        (
            {"sourceJoint": source_joint, "targetBone": payload["targetBone"], "deltaAngleDeg": float(payload["deltaAngleDeg"])}
            for source_joint, payload in per_bone.items()
        ),
        key=lambda item: item["deltaAngleDeg"],
        reverse=True,
    )
    return {
        "summary": summarize_values(delta_angles),
        "worstBones": worst[:8],
        "perBone": per_bone,
    }


def build_space_contract_report(
    *,
    canonical_report: dict[str, Any],
    reference_alignment_profile: dict[str, Any],
) -> dict[str, Any]:
    defaults = dict(reference_alignment_profile.get("defaults", {}) or {})
    declared_source = parse_space_descriptor(str(defaults.get("canonicalSourceSpace", "")))
    declared_target = parse_space_descriptor(str(defaults.get("unityTargetSpace", "")))
    transform_matrix = np.asarray((defaults.get("globalAxisTransform", {}) or {}).get("matrix", np.eye(3)), dtype=np.float64).reshape((3, 3))

    actual_forward = np.asarray(canonical_report["firstFrameOrientation"]["after"]["forward"], dtype=np.float64)
    actual_up = np.asarray(canonical_report["firstFrameOrientation"]["after"]["up"], dtype=np.float64)
    actual_side = safe_normalize(np.cross(actual_up, actual_forward))
    target_side = safe_normalize(np.cross(declared_target["up"], declared_target["forward"]))
    transformed_forward = transform_matrix @ actual_forward
    transformed_up = transform_matrix @ actual_up
    recommended_transform = np.stack((target_side, declared_target["up"], declared_target["forward"]), axis=1) @ np.stack(
        (actual_side, actual_up, actual_forward),
        axis=1,
    ).T

    forward_error_source = angle_deg(actual_forward, declared_source["forward"])
    up_error_source = angle_deg(actual_up, declared_source["up"])
    forward_error_target = angle_deg(transformed_forward, declared_target["forward"])
    up_error_target = angle_deg(transformed_up, declared_target["up"])

    issues: list[str] = []
    if forward_error_source > 45.0:
        issues.append("canonical_forward_mismatch_with_reference_profile")
    if up_error_source > 15.0:
        issues.append("canonical_up_mismatch_with_reference_profile")
    if forward_error_target > 45.0:
        issues.append("global_axis_transform_forward_mismatch")
    if up_error_target > 15.0:
        issues.append("global_axis_transform_up_mismatch")

    return {
        "declaredCanonicalSourceSpace": {
            "descriptor": declared_source["descriptor"],
            "forward": declared_source["forwardLabel"],
            "up": declared_source["upLabel"],
        },
        "actualCanonicalSpace": {
            "forward": vector_to_axis_label(actual_forward),
            "up": vector_to_axis_label(actual_up),
        },
        "declaredUnityTargetSpace": {
            "descriptor": declared_target["descriptor"],
            "forward": declared_target["forwardLabel"],
            "up": declared_target["upLabel"],
        },
        "referenceGlobalAxisTransform": {
            "mode": str((defaults.get("globalAxisTransform", {}) or {}).get("mode", "")),
            "matrix": np.asarray(transform_matrix, dtype=np.float64).round(6).tolist(),
            "actualForwardAfterTransform": vector_to_axis_label(transformed_forward),
            "actualUpAfterTransform": vector_to_axis_label(transformed_up),
        },
        "recommendedGlobalAxisTransform": {
            "matrix": np.asarray(recommended_transform, dtype=np.float64).round(6).tolist(),
            "actualForwardAfterRecommendedTransform": vector_to_axis_label(recommended_transform @ actual_forward),
            "actualUpAfterRecommendedTransform": vector_to_axis_label(recommended_transform @ actual_up),
        },
        "errorsDeg": {
            "canonicalForwardVsDeclared": float(forward_error_source),
            "canonicalUpVsDeclared": float(up_error_source),
            "targetForwardAfterTransformVsDeclared": float(forward_error_target),
            "targetUpAfterTransformVsDeclared": float(up_error_target),
        },
        "issues": issues,
    }


def build_root_contract_report(
    *,
    source_motion_contract: dict[str, Any],
    canonical_report: dict[str, Any],
    profile: dict[str, Any],
    rest_dump: dict[str, Any],
) -> dict[str, Any]:
    target_root = str(profile.get("targetRootBone", "Root") or "Root")
    target_motion_root = str(profile.get("targetMotionRootBone", "") or "")
    return {
        "sourceRoot": {
            "name": source_motion_contract["rootName"],
            "positionChannels": list(source_motion_contract["rootPositionChannels"]),
            "rotationChannels": list(source_motion_contract["rootRotationChannels"]),
        },
        "canonicalRoot": dict(canonical_report.get("canonicalRoot", {}) or {}),
        "targetRig": {
            "armatureName": str(rest_dump.get("armatureName", "")),
            "targetRootBone": target_root,
            "targetMotionRootBone": target_motion_root,
            "targetRootParent": str(((rest_dump.get("bones", {}) or {}).get(target_root, {}) or {}).get("parent", "")),
            "targetMotionRootParent": str(((rest_dump.get("bones", {}) or {}).get(target_motion_root, {}) or {}).get("parent", "")),
        },
        "status": {
            "sourceRootIsPelvis": bool(source_motion_contract["rootName"] == "pelvis"),
            "canonicalRootIsStatic": bool((canonical_report.get("canonicalRoot", {}) or {}).get("isStatic", False)),
            "targetRootDistinctFromMotionRoot": bool(target_root != target_motion_root),
        },
    }


def build_willa_contract_report(
    *,
    project_root: Path,
    source_bvh: Path,
    canonical_bvh: Path,
    canonical_report_path: Path,
    rest_dump_path: Path,
    output_path: Path,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    import json

    profile = load_willa_retarget_profile(project_root=project_root, profile_path=profile_path)
    reference_alignment_profile = load_reference_alignment_profile(profile)
    canonical_report = json.loads(canonical_report_path.read_text(encoding="utf-8"))
    rest_dump = load_willa_rest_dump(rest_dump_path)

    source_motion_contract = describe_bvh_motion_contract(source_bvh)
    canonical_motion_contract = describe_bvh_motion_contract(canonical_bvh)

    source_rest = rest_pose(parse_bvh(source_bvh).nodes)
    canonical_rest = rest_pose(parse_bvh(canonical_bvh).nodes)
    target_rest_positions = {
        name: np.asarray((payload or {}).get("headLocal", [0.0, 0.0, 0.0]), dtype=np.float64)
        for name, payload in dict(rest_dump.get("bones", {}) or {}).items()
    }

    raw_rest_basis = compute_rest_basis_deltas(
        source_positions={name: np.asarray(value, dtype=np.float64) for name, value in source_rest.world_positions.items()},
        target_positions=target_rest_positions,
        profile=profile,
    )
    canonical_rest_basis = compute_rest_basis_deltas(
        source_positions={name: np.asarray(value, dtype=np.float64) for name, value in canonical_rest.world_positions.items()},
        target_positions=target_rest_positions,
        profile=profile,
    )
    space_contract = build_space_contract_report(
        canonical_report=canonical_report,
        reference_alignment_profile=reference_alignment_profile,
    )
    root_contract = build_root_contract_report(
        source_motion_contract=source_motion_contract,
        canonical_report=canonical_report,
        profile=profile,
        rest_dump=rest_dump,
    )

    issues = list(space_contract.get("issues", []))
    if float(canonical_rest_basis["summary"].get("mean", 0.0) or 0.0) > 35.0:
        issues.append("canonical_rest_basis_mean_high")
    raw_mean = float(raw_rest_basis["summary"].get("mean", 0.0) or 0.0)
    canonical_mean = float(canonical_rest_basis["summary"].get("mean", 0.0) or 0.0)
    if canonical_mean - raw_mean > 90.0:
        issues.append("canonicalization_introduces_rest_basis_flip")

    payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "projectRoot": str(project_root.resolve()),
        "profilePath": str(Path(str(profile["profilePath"])).resolve()),
        "sourceBvh": str(source_bvh.resolve()),
        "canonicalBvh": str(canonical_bvh.resolve()),
        "canonicalReportPath": str(canonical_report_path.resolve()),
        "willaRestDumpPath": str(rest_dump_path.resolve()),
        "spaceContract": space_contract,
        "rootContract": root_contract,
        "rotationContract": {
            "source": source_motion_contract,
            "canonical": canonical_motion_contract,
            "targetRuntime": {
                "rotationRepresentation": "blender_pose_quaternion",
                "targetRootBone": str(profile["targetRootBone"]),
                "targetMotionRootBone": str(profile["targetMotionRootBone"]),
            },
        },
        "restBasisContract": {
            "rawSourceToWilla": raw_rest_basis,
            "canonicalSourceToWilla": canonical_rest_basis,
        },
        "diagnosis": {
            "issues": issues,
            "summary": [
                "current_source.bvh remains the trusted source-truth generator",
                "canonical BVH root semantics align with the target rig contract when Root stays static and hips carry motion",
                "reference alignment defaults should match the actual canonical source orientation before further retarget tuning",
            ],
            "metrics": {
                "rawRestBasisMeanDeg": raw_mean,
                "canonicalRestBasisMeanDeg": canonical_mean,
                "canonicalMinusRawRestBasisMeanDeg": canonical_mean - raw_mean,
            },
        },
    }
    write_json(output_path, payload)
    return payload
