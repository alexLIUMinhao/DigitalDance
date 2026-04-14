from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


BVH_TO_TARGET_BASIS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


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

PARENT_BY_BONE_22 = {
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


@dataclass(frozen=True)
class ValidationThresholds:
    fk_small: float = 1e-5
    import_small: float = 1e-5
    import_large: float = 1e-3


def round_nested_array(values: np.ndarray, decimals: int = 8) -> list[Any]:
    return np.asarray(values, dtype=np.float64).round(decimals=decimals).tolist()


def transform_vectors(values: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    payload = np.asarray(values, dtype=np.float64)
    basis = np.asarray(basis_matrix, dtype=np.float64)
    if payload.ndim == 1:
        return basis @ payload
    return np.einsum("ij,...j->...i", basis, payload)


def transform_rotation_matrix(matrix: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    basis = np.asarray(basis_matrix, dtype=np.float64)
    return basis @ np.asarray(matrix, dtype=np.float64) @ basis


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    payload = np.asarray(values, dtype=np.float64).reshape(-1)
    if payload.size == 0:
        return {
            "count": 0,
            "min": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p95": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(payload.size),
        "min": float(np.min(payload)),
        "mean": float(np.mean(payload)),
        "median": float(np.median(payload)),
        "p95": float(np.percentile(payload, 95)),
        "max": float(np.max(payload)),
    }


def choose_sample_indices_from_summary(frame_count: int, summary: dict[str, Any]) -> np.ndarray:
    source_fps = int(summary.get("sourceFps", 30) or 30)
    preview_fps = int(summary.get("fps", 30) or 30)
    start_seconds = float(summary.get("startSeconds", 0.0) or 0.0)
    max_seconds = float(summary.get("maxSeconds", 0.0) or 0.0)
    max_frames = summary.get("maxFrames")
    frame_step = max(1, int(round(source_fps / max(preview_fps, 1))))
    start_index = max(0, int(round(start_seconds * source_fps)))
    if max_seconds <= 0:
        end_index = frame_count
    else:
        end_index = min(frame_count, start_index + int(round(max_seconds * source_fps)))
    if start_index >= end_index:
        start_index = 0
        end_index = frame_count
    indices = np.arange(start_index, end_index, frame_step, dtype=np.int64)
    if max_frames is not None and len(indices) > int(max_frames):
        indices = indices[: int(max_frames)]
    if len(indices) == 0:
        indices = np.asarray([0], dtype=np.int64)
    return indices


def canonicalize_preview_positions(poses: np.ndarray) -> np.ndarray:
    payload = np.asarray(poses, dtype=np.float64).copy()
    root = payload[:, :1, :].copy()
    payload[..., 0] -= root[..., 0]
    payload[..., 2] -= root[..., 2]
    return payload[..., [0, 2, 1]]


def build_pose_payload(
    *,
    input_path: str,
    bone_order: list[str],
    frame_indices: list[int],
    poses: np.ndarray,
    include_preview_space: bool = False,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schemaVersion": 1,
        "inputPath": input_path,
        "boneOrder": list(bone_order),
        "frameIndices": [int(value) for value in frame_indices],
        "poses": round_nested_array(poses),
    }
    if include_preview_space:
        payload["previewSpacePoses"] = round_nested_array(canonicalize_preview_positions(poses))
    if extra:
        payload.update(extra)
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _axis_index(channel_name: str) -> int:
    axis = channel_name[0].upper()
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def extract_bvh_joint_positions(
    input_path: Path,
    parse_bvh,
    *,
    bone_order: list[str] | None = None,
    basis_matrix: np.ndarray | None = None,
) -> tuple[list[str], list[int], np.ndarray, float]:
    nodes, motion, frame_time = parse_bvh(input_path)
    node_by_name = {node.name: index for index, node in enumerate(nodes)}
    selected_names = list(bone_order or [node.name for node in nodes])
    missing = [name for name in selected_names if name not in node_by_name]
    if missing:
        raise KeyError(f"BVH is missing required bones: {missing}")

    frame_count = int(motion.shape[0])
    world_positions = np.zeros((frame_count, len(nodes), 3), dtype=np.float64)

    for frame_index in range(frame_count):
        world_rotations: list[Rotation | None] = [None] * len(nodes)
        for node_index, node in enumerate(nodes):
            channel_values = motion[frame_index, node.channel_start : node.channel_start + len(node.channels)]
            local_translation = np.zeros((3,), dtype=np.float64)
            rotation_order: list[str] = []
            rotation_values: list[float] = []
            for channel_name, value in zip(node.channels, channel_values.tolist()):
                if channel_name.endswith("position"):
                    local_translation[_axis_index(channel_name)] = float(value)
                elif channel_name.endswith("rotation"):
                    rotation_order.append(channel_name[0].upper())
                    rotation_values.append(float(value))
            local_rotation = (
                Rotation.from_euler("".join(rotation_order), rotation_values, degrees=True)
                if rotation_order
                else Rotation.identity()
            )
            parent_index = node.parent
            offset = np.asarray(node.offset, dtype=np.float64)
            if basis_matrix is not None:
                local_translation = transform_vectors(local_translation, basis_matrix)
                offset = transform_vectors(offset, basis_matrix)
                local_rotation = Rotation.from_matrix(transform_rotation_matrix(local_rotation.as_matrix(), basis_matrix))
            if parent_index is None:
                world_positions[frame_index, node_index] = local_translation + offset
                world_rotations[node_index] = local_rotation
            else:
                parent_rotation = world_rotations[parent_index]
                if parent_rotation is None:
                    raise ValueError(f"parent rotation missing for node {node.name}")
                world_positions[frame_index, node_index] = (
                    world_positions[frame_index, parent_index] + parent_rotation.apply(offset)
                )
                world_rotations[node_index] = parent_rotation * local_rotation

    selected_positions = np.stack(
        [world_positions[:, node_by_name[name], :] for name in selected_names],
        axis=1,
    )
    return selected_names, list(range(frame_count)), selected_positions, float(frame_time)


def compare_pose_payloads(
    truth_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
    compare_bridge_positions,
) -> dict[str, Any]:
    truth = np.asarray(truth_payload["poses"], dtype=np.float64)
    candidate = np.asarray(candidate_payload["poses"], dtype=np.float64)
    return compare_bridge_positions(truth, candidate, compare_joint_count=min(len(truth_payload["boneOrder"]), len(candidate_payload["boneOrder"]), 22))


def remap_per_joint_error_names(summary: dict[str, Any], bone_order: list[str]) -> dict[str, Any]:
    remapped = {}
    raw_stats = dict(summary.get("perJointPositionError", {}) or {})
    for key, stats in raw_stats.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            remapped[str(key)] = stats
            continue
        if 0 <= index < len(bone_order):
            remapped[str(bone_order[index])] = stats
        else:
            remapped[str(index)] = stats
    return remapped


def compute_bone_vector_diagnostics(
    truth_payload: dict[str, Any],
    candidate_payload: dict[str, Any],
) -> dict[str, Any]:
    truth_positions = np.asarray(truth_payload["poses"], dtype=np.float64)
    candidate_positions = np.asarray(candidate_payload["poses"], dtype=np.float64)
    truth_order = list(truth_payload.get("boneOrder", []) or BRIDGE_BONE_ORDER_22)
    candidate_order = list(candidate_payload.get("boneOrder", []) or BRIDGE_BONE_ORDER_22)
    truth_index = {name: index for index, name in enumerate(truth_order)}
    candidate_index = {name: index for index, name in enumerate(candidate_order)}
    frame_count = min(int(truth_positions.shape[0]), int(candidate_positions.shape[0]))
    per_bone: dict[str, Any] = {}

    for child_name in BRIDGE_BONE_ORDER_22:
        parent_name = PARENT_BY_BONE_22.get(child_name)
        if parent_name is None:
            continue
        if child_name not in truth_index or child_name not in candidate_index:
            continue
        if parent_name not in truth_index or parent_name not in candidate_index:
            continue
        truth_child = truth_positions[:frame_count, truth_index[child_name], :]
        truth_parent = truth_positions[:frame_count, truth_index[parent_name], :]
        candidate_child = candidate_positions[:frame_count, candidate_index[child_name], :]
        candidate_parent = candidate_positions[:frame_count, candidate_index[parent_name], :]
        truth_vector = truth_child - truth_parent
        candidate_vector = candidate_child - candidate_parent
        truth_length = np.linalg.norm(truth_vector, axis=-1)
        candidate_length = np.linalg.norm(candidate_vector, axis=-1)
        length_error = np.abs(truth_length - candidate_length)

        truth_denominator = np.where(truth_length > 1e-8, truth_length, 1.0)
        candidate_denominator = np.where(candidate_length > 1e-8, candidate_length, 1.0)
        truth_unit = truth_vector / truth_denominator[:, None]
        candidate_unit = candidate_vector / candidate_denominator[:, None]
        cosine = np.clip(np.sum(truth_unit * candidate_unit, axis=-1), -1.0, 1.0)
        direction_angle_deg = np.degrees(np.arccos(cosine))

        per_bone[child_name] = {
            "parent": parent_name,
            "truthLength": summarize_values(truth_length),
            "candidateLength": summarize_values(candidate_length),
            "lengthError": summarize_values(length_error),
            "directionAngleDeg": summarize_values(direction_angle_deg),
        }

    def _rank(metric_path: tuple[str, ...], limit: int = 8) -> list[dict[str, Any]]:
        ranked = []
        for bone_name, stats in per_bone.items():
            value: Any = stats
            for key in metric_path:
                value = value[key]
            ranked.append(
                {
                    "bone": bone_name,
                    "parent": str(stats["parent"]),
                    "metric": ".".join(metric_path),
                    "mean": float(stats[metric_path[0]]["mean"]),
                    "p95": float(stats[metric_path[0]]["p95"]),
                    "max": float(stats[metric_path[0]]["max"]),
                    "sortValue": float(value),
                }
            )
        ranked.sort(key=lambda item: item["sortValue"], reverse=True)
        for item in ranked:
            item.pop("sortValue", None)
        return ranked[:limit]

    return {
        "frameCountCompared": frame_count,
        "boneCountCompared": int(len(per_bone)),
        "perBone": per_bone,
        "topByDirectionAngleP95": _rank(("directionAngleDeg", "p95")),
        "topByLengthErrorMean": _rank(("lengthError", "mean")),
    }


def diagnose_validation(
    fk_summary: dict[str, Any] | None,
    import_summary: dict[str, Any] | None,
    *,
    thresholds: ValidationThresholds | None = None,
) -> dict[str, Any]:
    active = thresholds or ValidationThresholds()
    fk_mean = float((((fk_summary or {}).get("jointPositionError") or {}).get("mean", 0.0) or 0.0)
        if fk_summary
        else float("inf"))
    import_mean = float((((import_summary or {}).get("jointPositionError") or {}).get("mean", 0.0) or 0.0)
        if import_summary
        else float("inf"))

    if fk_summary is None and import_summary is None:
        return {
            "likely_root_cause": "validation_not_run",
            "reason": "Neither FK nor Blender import comparison produced a summary.",
        }
    if fk_summary is not None and import_summary is None:
        if fk_mean <= active.fk_small:
            return {
                "likely_root_cause": "blender_import_runtime_failure",
                "reason": "Direct FK matches the official joints closely, but Blender import did not produce comparison data.",
            }
        return {
            "likely_root_cause": "source_bvh_export_content",
            "reason": "Direct FK already diverges from the official joints, so the BVH content is still inconsistent before import.",
        }
    if fk_mean <= active.fk_small and import_mean > active.import_large:
        return {
            "likely_root_cause": "bvh_import_or_viewer_convention",
            "reason": "Direct FK is close to truth, but Blender import diverges materially.",
        }
    if fk_mean > active.import_large and import_mean > active.import_large:
        return {
            "likely_root_cause": "source_bvh_export_content",
            "reason": "Both direct FK and Blender import diverge materially from the official joints.",
        }
    if fk_mean <= active.fk_small and import_mean <= active.import_small:
        return {
            "likely_root_cause": "mesh_viewer_or_camera_interpretation",
            "reason": "Both direct FK and Blender import are numerically close to the official joints.",
        }
    return {
        "likely_root_cause": "mixed_or_partial_mismatch",
        "reason": "The validation signals do not point to a single failure mode yet.",
    }
