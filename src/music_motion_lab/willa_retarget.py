from __future__ import annotations

import json
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np

from .config import default_project_root


BRIDGE_BONE_ORDER_22 = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


TARGET_PARENT_BY_HUMAN_BONE = {
    "hips": None,
    "spine": "hips",
    "chest": "spine",
    "upperChest": "chest",
    "neck": "upperChest",
    "head": "neck",
    "leftUpperLeg": "hips",
    "leftLowerLeg": "leftUpperLeg",
    "leftFoot": "leftLowerLeg",
    "leftToes": "leftFoot",
    "rightUpperLeg": "hips",
    "rightLowerLeg": "rightUpperLeg",
    "rightFoot": "rightLowerLeg",
    "rightToes": "rightFoot",
    "leftShoulder": "upperChest",
    "leftUpperArm": "leftShoulder",
    "leftLowerArm": "leftUpperArm",
    "leftHand": "leftLowerArm",
    "rightShoulder": "upperChest",
    "rightUpperArm": "rightShoulder",
    "rightLowerArm": "rightUpperArm",
    "rightHand": "rightLowerArm",
}

TARGET_ORDER_BY_HUMAN_BONE = [
    "hips",
    "leftUpperLeg",
    "rightUpperLeg",
    "spine",
    "leftLowerLeg",
    "rightLowerLeg",
    "chest",
    "leftFoot",
    "rightFoot",
    "upperChest",
    "leftToes",
    "rightToes",
    "neck",
    "leftShoulder",
    "rightShoulder",
    "head",
    "leftUpperArm",
    "rightUpperArm",
    "leftLowerArm",
    "rightLowerArm",
    "leftHand",
    "rightHand",
]


RETARGET_STRATEGY_ALIASES = {
    "constraint-bake": "constraint_bake",
    "constraint_bake": "constraint_bake",
    "constraint": "constraint_bake",
    "bake": "constraint_bake",
    "rest-pose-first": "rest_pose_first",
    "rest_pose_first": "rest_pose_first",
    "restpose": "rest_pose_first",
    "local-rest": "rest_pose_first",
}


def default_willa_retarget_profile_path(project_root: Path | None = None) -> Path:
    root = (project_root or default_project_root()).resolve()
    return root / "config" / "willa_retarget_profile.json"


def normalize_retarget_strategy(value: str | None) -> str:
    normalized = str(value or "constraint_bake").strip().lower()
    return RETARGET_STRATEGY_ALIASES.get(normalized, normalized.replace("-", "_"))


def slugify_result_tag(value: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    return cleaned.strip("_") or "run"


def next_versioned_retarget_paths(
    output_blend: Path,
    report_output: Path,
    result_tag: str,
) -> tuple[Path, Path, int]:
    blend_base = output_blend.resolve()
    report_base = report_output.resolve()
    tag = slugify_result_tag(result_tag)
    pattern = re.compile(rf"^(?P<stem>.+)_{re.escape(tag)}_r(?P<index>\d{{3}})$")

    max_index = 0
    for parent, suffix in ((blend_base.parent, blend_base.suffix), (report_base.parent, report_base.suffix)):
        if not parent.exists():
            continue
        for candidate in parent.iterdir():
            if candidate.suffix != suffix:
                continue
            match = pattern.match(candidate.stem)
            if not match:
                continue
            max_index = max(max_index, int(match.group("index")))

    run_index = max_index + 1
    version_suffix = f"{tag}_r{run_index:03d}"
    blend_path = blend_base.with_name(f"{blend_base.stem}_{version_suffix}{blend_base.suffix}")
    report_path = report_base.with_name(f"{report_base.stem}_{version_suffix}{report_base.suffix}")
    return blend_path, report_path, run_index


def resolve_project_path(project_root: Path, value: str) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (project_root / candidate).resolve()


def load_glb_json(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header = handle.read(12)
        if len(header) != 12:
            raise ValueError(f"invalid glb header: {path}")
        magic, version, _ = struct.unpack("<III", header)
        if magic != 0x46546C67:
            raise ValueError(f"unsupported glb magic in {path}")
        if version != 2:
            raise ValueError(f"unsupported glb version in {path}: {version}")
        chunk_length, chunk_type = struct.unpack("<II", handle.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError(f"first glb chunk is not json in {path}")
        payload = handle.read(chunk_length)
        if len(payload) != chunk_length:
            raise ValueError(f"truncated glb json chunk in {path}")
    return json.loads(payload.decode("utf-8"))


def resolve_willa_human_bone_names(vrm_path: Path) -> dict[str, str]:
    document = load_glb_json(vrm_path)
    human_bones = dict(document["extensions"]["VRMC_vrm"]["humanoid"]["humanBones"])
    nodes = list(document.get("nodes", []))
    resolved: dict[str, str] = {}
    for human_bone, payload in human_bones.items():
        node_index = int(payload.get("node"))
        node_name = str(nodes[node_index].get("name", human_bone) or human_bone)
        resolved[str(human_bone)] = node_name
    return resolved


def translate_reference_names(reference: dict[str, Any], human_to_target: dict[str, str]) -> dict[str, Any]:
    payload = dict(reference or {})
    mode = str(payload.get("mode", "") or "")
    if mode in {"child", "to_joint"}:
        joint_name = str(payload.get("joint", "") or "")
        payload["joint"] = human_to_target.get(joint_name, joint_name)
        return payload
    if mode == "pair":
        from_name = str(payload.get("from", "") or "")
        to_name = str(payload.get("to", "") or "")
        payload["from"] = human_to_target.get(from_name, from_name)
        payload["to"] = human_to_target.get(to_name, to_name)
        return payload
    if mode == "cross":
        payload["a"] = translate_reference_names(dict(payload.get("a", {}) or {}), human_to_target)
        payload["b"] = translate_reference_names(dict(payload.get("b", {}) or {}), human_to_target)
        return payload
    return payload


def translate_target_mode_by_bone(mode_by_human_bone: dict[str, Any], human_to_target: dict[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for human_bone, mode in dict(mode_by_human_bone or {}).items():
        resolved[human_to_target.get(str(human_bone), str(human_bone))] = str(mode or "")
    return resolved


def resolve_joint_mappings(
    reference_profile: dict[str, Any],
    human_to_target: dict[str, str],
) -> list[dict[str, Any]]:
    mappings: list[dict[str, Any]] = []
    for mapping in list((reference_profile.get("defaults", {}) or {}).get("jointMappings", []) or []):
        source_joint = str(mapping.get("sourceJoint", "") or "")
        target_human_bone = str(mapping.get("targetHumanBone", "") or "")
        target_bone_name = human_to_target[target_human_bone]
        mappings.append(
            {
                "sourceJoint": source_joint,
                "targetHumanBone": target_human_bone,
                "targetBoneName": target_bone_name,
                "jointClass": str(mapping.get("jointClass", "3dof") or "3dof"),
                "primarySourceReference": dict(mapping.get("primarySourceReference", {}) or {}),
                "secondarySourceReference": dict(mapping.get("secondarySourceReference", {}) or {}),
                "primaryTargetReference": translate_reference_names(
                    dict(mapping.get("primaryTargetReference", {}) or {}),
                    human_to_target,
                ),
                "secondaryTargetReference": translate_reference_names(
                    dict(mapping.get("secondaryTargetReference", {}) or {}),
                    human_to_target,
                ),
            }
        )
    return mappings


def load_willa_retarget_profile(
    project_root: Path | None = None,
    profile_path: Path | None = None,
) -> dict[str, Any]:
    root = (project_root or default_project_root()).resolve()
    config_path = (profile_path or default_willa_retarget_profile_path(root)).resolve()
    raw_profile = json.loads(config_path.read_text(encoding="utf-8"))

    reference_profile_path = resolve_project_path(root, str(raw_profile["referenceWillaAlignmentProfile"]))
    reference_vrm_path = resolve_project_path(root, str(raw_profile["referenceWillaVrm"]))
    target_scene_blend = resolve_project_path(root, str(raw_profile["targetSceneBlend"]))
    default_source_bvh = resolve_project_path(root, str(raw_profile.get("defaultSourceBvh", raw_profile["defaultInputBvh"])))
    default_input_bvh = resolve_project_path(root, str(raw_profile["defaultInputBvh"]))
    default_canonical_report = resolve_project_path(
        root,
        str(raw_profile.get("defaultCanonicalReport", "outputs/reports/willa_canonicalization_report.json")),
    )
    default_output_blend = resolve_project_path(root, str(raw_profile["defaultOutputBlend"]))
    default_report = resolve_project_path(root, str(raw_profile["defaultReport"]))

    reference_profile = json.loads(reference_profile_path.read_text(encoding="utf-8"))
    reference_dataset_profile = str(raw_profile.get("referenceDatasetProfile", "finedance") or "finedance")
    human_to_target = resolve_willa_human_bone_names(reference_vrm_path)
    joint_mappings = resolve_joint_mappings(reference_profile, human_to_target)
    target_mode_by_bone = translate_target_mode_by_bone(
        dict((reference_profile.get("defaults", {}) or {}).get("targetLocalRotationModeByHumanBone", {}) or {}),
        human_to_target,
    )

    bone_map = {mapping["sourceJoint"]: mapping["targetBoneName"] for mapping in joint_mappings}
    target_parents = {
        human_to_target[human_bone]: (human_to_target[parent] if parent else None)
        for human_bone, parent in TARGET_PARENT_BY_HUMAN_BONE.items()
    }
    target_order = [human_to_target[human_bone] for human_bone in TARGET_ORDER_BY_HUMAN_BONE]
    dataset_basis_transforms = dict(
        (((reference_profile.get("datasetProfiles", {}) or {}).get(reference_dataset_profile, {}) or {}).get("basisTransforms") or {})
    )
    reference_source_to_target_basis_joints = [
        str(value)
        for value in list(raw_profile.get("referenceSourceToTargetBasisJoints", []) or [])
    ]
    basis_map_overrides: dict[str, Any] = {}
    basis_corrections: dict[str, Any] = {}
    for source_joint in reference_source_to_target_basis_joints:
        transform_payload = dict(dataset_basis_transforms.get(source_joint, {}) or {})
        override_matrix = transform_payload.get("sourceToTargetBasis")
        if override_matrix:
            basis_map_overrides[source_joint] = {"matrix": override_matrix}
        correction_matrix = transform_payload.get("basisCorrectionMatrix")
        if correction_matrix:
            basis_corrections[source_joint] = {"matrix": correction_matrix}
    basis_map_overrides.update(dict(raw_profile.get("basisMapOverrides", {}) or {}))
    basis_corrections.update(dict(raw_profile.get("basisCorrectionTransforms", {}) or {}))

    return {
        "schemaVersion": int(raw_profile.get("schemaVersion", 1) or 1),
        "profileVersion": str(raw_profile.get("profileVersion", "willa_retarget_v1") or "willa_retarget_v1"),
        "projectRoot": str(root),
        "profilePath": str(config_path),
        "referenceWillaAlignmentProfile": str(reference_profile_path),
        "referenceWillaAlignmentProfileVersion": str(
            reference_profile.get("profileVersion")
            or (reference_profile.get("defaults", {}) or {}).get("profileVersion")
            or (reference_profile.get("defaults", {}) or {}).get("alignmentProfileVersion")
            or (reference_profile.get("defaults", {}) or {}).get("profile_version")
            or "unknown"
        ),
        "referenceDatasetProfile": reference_dataset_profile,
        "referenceWillaVrm": str(reference_vrm_path),
        "targetSceneBlend": str(target_scene_blend),
        "defaultSourceBvh": str(default_source_bvh),
        "defaultInputBvh": str(default_input_bvh),
        "defaultCanonicalReport": str(default_canonical_report),
        "defaultOutputBlend": str(default_output_blend),
        "defaultReport": str(default_report),
        "defaultRetargetStrategy": normalize_retarget_strategy(str(raw_profile.get("defaultRetargetStrategy", "constraint_bake"))),
        "defaultConstraintBakeBasisMode": str(
            raw_profile.get("defaultConstraintBakeBasisMode", "constraint_world_pelvis_torso_local")
            or "constraint_world_pelvis_torso_local"
        ),
        "targetRootBone": str(raw_profile.get("targetRootBone", "Root") or "Root"),
        "targetMotionRootBone": str(raw_profile.get("targetMotionRootBone", human_to_target["hips"]) or human_to_target["hips"]),
        "sourceMotionRootBone": str(raw_profile.get("sourceMotionRootBone", "pelvis") or "pelvis"),
        "targetArmatureName": str(raw_profile.get("targetArmatureName", "") or ""),
        "rootMotionMode": str(raw_profile.get("rootMotionMode", "origin_locked") or "origin_locked"),
        "basisMode": str(raw_profile.get("basisMode", "constraint_world_pelvis_local") or "constraint_world_pelvis_local"),
        "useBasisMapOverrides": bool(raw_profile.get("useBasisMapOverrides", False)),
        "rotationDisabledSourceJoints": list(raw_profile.get("rotationDisabledSourceJoints", []) or []),
        "referenceSourceToTargetBasisJoints": reference_source_to_target_basis_joints,
        "basisMapOverrides": basis_map_overrides,
        "sourceBoneOrder": list(BRIDGE_BONE_ORDER_22),
        "boneMap": bone_map,
        "jointMappings": joint_mappings,
        "targetParents": target_parents,
        "targetOrder": target_order,
        "humanToTarget": dict(human_to_target),
        "targetLocalRotationMode": str(
            (reference_profile.get("defaults", {}) or {}).get("targetLocalRotationMode", "absolute_local") or "absolute_local"
        ),
        "targetLocalRotationModeByBone": target_mode_by_bone,
        "thresholds": dict((reference_profile.get("defaults", {}) or {}).get("thresholds", {}) or {}),
        "basisCorrectionTransforms": basis_corrections,
    }


def validate_willa_retarget_profile(profile: dict[str, Any]) -> None:
    source_bone_order = list(profile.get("sourceBoneOrder", []) or [])
    bone_map = dict(profile.get("boneMap", {}) or {})
    if source_bone_order != list(BRIDGE_BONE_ORDER_22):
        raise ValueError("sourceBoneOrder must match the canonical 22-joint bridge order")
    missing = [name for name in BRIDGE_BONE_ORDER_22 if name not in bone_map]
    if missing:
        raise ValueError(f"missing Willa bone mappings for: {missing}")
    if str(profile.get("targetRootBone", "") or "") == str(profile.get("targetMotionRootBone", "") or ""):
        raise ValueError("targetRootBone and targetMotionRootBone must remain distinct")


def safe_normalize(vector: np.ndarray) -> np.ndarray:
    payload = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(payload))
    if norm <= 1e-8:
        return np.zeros(3, dtype=np.float64)
    return payload / norm


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = safe_normalize(a)
    b_unit = safe_normalize(b)
    if np.linalg.norm(a_unit) <= 1e-8 or np.linalg.norm(b_unit) <= 1e-8:
        return 0.0
    dot = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(dot)))


def summarize_values(values: list[float]) -> dict[str, float]:
    payload = np.asarray(list(values), dtype=np.float64).reshape((-1,))
    if payload.size == 0:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(payload)),
        "p95": float(np.percentile(payload, 95)),
        "max": float(np.max(payload)),
    }


def build_basis(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    y_axis = safe_normalize(primary)
    if np.linalg.norm(y_axis) <= 1e-8:
        return np.eye(3, dtype=np.float64)

    secondary_proj = np.asarray(secondary, dtype=np.float64) - np.dot(secondary, y_axis) * y_axis
    if np.linalg.norm(secondary_proj) <= 1e-8:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(fallback, y_axis))) > 0.9:
            fallback = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        secondary_proj = fallback - np.dot(fallback, y_axis) * y_axis
    x_axis = safe_normalize(secondary_proj)
    z_axis = safe_normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) <= 1e-8:
        z_axis = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    x_axis = safe_normalize(np.cross(y_axis, z_axis))
    return np.stack((x_axis, y_axis, z_axis), axis=1)


def resolve_reference_vector(
    positions: dict[str, np.ndarray],
    parents: dict[str, str | None],
    joint_name: str,
    reference: dict[str, Any],
) -> np.ndarray:
    mode = str(reference.get("mode", "") or "")
    if mode == "cross":
        lhs = resolve_reference_vector(positions, parents, joint_name, dict(reference.get("a", {}) or {}))
        rhs = resolve_reference_vector(positions, parents, joint_name, dict(reference.get("b", {}) or {}))
        return np.cross(np.asarray(lhs, dtype=np.float64), np.asarray(rhs, dtype=np.float64))
    if mode == "child":
        child_name = str(reference.get("joint", "") or "")
        return positions[child_name] - positions[joint_name]
    if mode == "parent":
        parent_name = parents.get(joint_name)
        if not parent_name:
            return np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        return positions[joint_name] - positions[parent_name]
    if mode == "pair":
        from_name = str(reference.get("from", "") or "")
        to_name = str(reference.get("to", "") or "")
        return positions[from_name] - positions[to_name]
    if mode == "to_joint":
        other_name = str(reference.get("joint", "") or "")
        return positions[other_name] - positions[joint_name]
    raise ValueError(f"unsupported reference mode: {mode}")


def joint_angle(positions: dict[str, np.ndarray], parents: dict[str, str | None], joint_name: str, child_name: str) -> float:
    parent_name = parents.get(joint_name)
    if not parent_name:
        return 0.0
    parent_vector = positions[parent_name] - positions[joint_name]
    child_vector = positions[child_name] - positions[joint_name]
    return angle_deg(parent_vector, child_vector)
