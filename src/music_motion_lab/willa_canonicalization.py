from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bvh import BvhNode, MotionSummary, frame_pose, parse_bvh, rotation_channels, rotation_matrix_to_channels, write_bvh
from .utils import utc_now_iso, write_json
from .willa_retarget import BRIDGE_BONE_ORDER_22


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

AXIS_LABEL_TO_VECTOR = {
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


def _orthogonalize(primary: np.ndarray, aux: np.ndarray) -> np.ndarray:
    primary_unit = safe_normalize(primary)
    aux_proj = np.asarray(aux, dtype=np.float64) - np.dot(aux, primary_unit) * primary_unit
    if np.linalg.norm(aux_proj) <= 1e-8:
        for fallback in (
            np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        ):
            candidate = fallback - np.dot(fallback, primary_unit) * primary_unit
            if np.linalg.norm(candidate) > 1e-8:
                aux_proj = candidate
                break
    return safe_normalize(aux_proj)


def _orientation_basis(positions: dict[str, np.ndarray]) -> np.ndarray:
    pelvis = np.asarray(positions["pelvis"], dtype=np.float64)
    spine = np.asarray(positions.get("spine3", positions.get("neck", pelvis + np.asarray([0.0, 1.0, 0.0]))), dtype=np.float64)
    up = safe_normalize(spine - pelvis)
    side_reference = np.asarray(positions.get("right_collar", positions["right_hip"]), dtype=np.float64) - np.asarray(
        positions.get("left_collar", positions["left_hip"]),
        dtype=np.float64,
    )
    side = _orthogonalize(up, side_reference)
    forward = safe_normalize(np.cross(side, up))
    side = safe_normalize(np.cross(up, forward))
    return np.stack((side, up, forward), axis=1)


def _support_center(pose_positions: dict[str, np.ndarray], end_sites: dict[str, np.ndarray]) -> np.ndarray:
    left_foot = np.asarray(pose_positions["left_foot"], dtype=np.float64)
    right_foot = np.asarray(pose_positions["right_foot"], dtype=np.float64)
    left_tip = np.asarray(end_sites.get("left_foot", left_foot), dtype=np.float64)
    right_tip = np.asarray(end_sites.get("right_foot", right_foot), dtype=np.float64)
    left_center = 0.5 * (left_foot + left_tip)
    right_center = 0.5 * (right_foot + right_tip)
    foot_heights = [left_foot[2], right_foot[2], left_tip[2], right_tip[2]]
    if "left_ankle" in pose_positions:
        foot_heights.append(float(pose_positions["left_ankle"][2]))
    if "right_ankle" in pose_positions:
        foot_heights.append(float(pose_positions["right_ankle"][2]))
    return np.asarray(
        [
            float((left_center[0] + right_center[0]) * 0.5),
            float((left_center[1] + right_center[1]) * 0.5),
            float(min(foot_heights)),
        ],
        dtype=np.float64,
    )


def _matrix_to_dict(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=np.float64).round(6).tolist()


def _vector_to_list(vector: np.ndarray) -> list[float]:
    return np.asarray(vector, dtype=np.float64).round(6).tolist()


def axis_label_to_vector(label: str) -> np.ndarray:
    normalized = str(label or "").strip().upper()
    if normalized not in AXIS_LABEL_TO_VECTOR:
        raise ValueError(f"unsupported axis label: {label}")
    return AXIS_LABEL_TO_VECTOR[normalized].copy()


def target_world_basis(*, target_up_label: str = "+Z", target_forward_label: str = "+Y") -> np.ndarray:
    up = safe_normalize(axis_label_to_vector(target_up_label))
    forward = safe_normalize(axis_label_to_vector(target_forward_label))
    if np.linalg.norm(np.cross(up, forward)) <= 1e-8:
        raise ValueError(f"target up {target_up_label} and forward {target_forward_label} must be orthogonal")
    side = safe_normalize(np.cross(up, forward))
    return np.stack((side, up, forward), axis=1)


def default_canonical_bvh_path(input_bvh: Path) -> Path:
    stem = input_bvh.stem.replace("_source_willa_aligned", "_source_willa_canonical")
    return input_bvh.with_name(f"{stem}.bvh")


def default_canonical_report_path(project_root: Path, sequence_stem: str) -> Path:
    return (project_root / "outputs" / "reports" / f"{sequence_stem}_canonicalization_report.json").resolve()


def canonicalize_source_bvh_for_willa(
    *,
    input_bvh: Path,
    output_bvh: Path,
    report_output: Path | None = None,
    target_up_label: str = "+Z",
    target_forward_label: str = "+Y",
) -> dict[str, Any]:
    parsed: MotionSummary = parse_bvh(input_bvh)
    nodes = parsed.nodes
    motion = np.asarray(parsed.motion, dtype=np.float64)
    frame_time = float(parsed.frame_time)
    frame_count = int(motion.shape[0])
    root_index = next((index for index, node in enumerate(nodes) if node.parent is None), None)
    if root_index is None:
        raise ValueError("source BVH is missing a root node")
    source_root = nodes[root_index]
    if source_root.name != "pelvis":
        raise ValueError(f"expected source root pelvis, got {source_root.name}")

    pose0 = frame_pose(nodes, motion[0])
    source_basis = _orientation_basis(pose0.world_positions)
    target_basis = target_world_basis(target_up_label=target_up_label, target_forward_label=target_forward_label)
    world_alignment = target_basis @ source_basis.T

    transformed_offsets: dict[str, np.ndarray] = {}
    transformed_end_sites: dict[str, np.ndarray | None] = {}
    for node in nodes:
        transformed_offsets[node.name] = world_alignment @ np.asarray(node.offset, dtype=np.float64)
        transformed_end_sites[node.name] = (
            world_alignment @ np.asarray(node.end_site_offset, dtype=np.float64)
            if node.end_site_offset is not None
            else None
        )

    transformed_local_translations: dict[str, list[np.ndarray]] = {node.name: [] for node in nodes}
    transformed_local_rotations: dict[str, list[np.ndarray]] = {node.name: [] for node in nodes}
    transformed_root_positions: list[np.ndarray] = []
    for frame_index in range(frame_count):
        pose = frame_pose(nodes, motion[frame_index])
        root_world = world_alignment @ np.asarray(pose.world_positions[source_root.name], dtype=np.float64)
        transformed_root_positions.append(root_world)
        for node in nodes:
            local_translation = world_alignment @ np.asarray(pose.local_translations[node.name], dtype=np.float64)
            local_rotation = world_alignment @ pose.local_rotations[node.name].as_matrix() @ world_alignment.T
            transformed_local_translations[node.name].append(local_translation)
            transformed_local_rotations[node.name].append(local_rotation)

    transformed_pose0_positions = {
        name: world_alignment @ np.asarray(value, dtype=np.float64)
        for name, value in pose0.world_positions.items()
    }
    transformed_pose0_end_sites = {
        name: world_alignment @ np.asarray(value, dtype=np.float64)
        for name, value in pose0.end_sites.items()
    }
    support_center = _support_center(transformed_pose0_positions, transformed_pose0_end_sites)
    first_root_position = np.asarray(transformed_root_positions[0], dtype=np.float64)
    pelvis_offset = first_root_position - support_center

    new_nodes: list[BvhNode] = [
        BvhNode(
            name="Root",
            parent=None,
            offset=np.zeros((3,), dtype=np.float64),
            channels=["Xposition", "Yposition", "Zposition", "Zrotation", "Xrotation", "Yrotation"],
            channel_start=0,
            end_site_offset=None,
        )
    ]
    old_to_new: dict[int, int] = {}
    channel_cursor = 6
    for old_index, node in enumerate(nodes):
        new_parent = 0 if old_index == root_index else old_to_new[node.parent]  # type: ignore[index]
        next_node = BvhNode(
            name=node.name,
            parent=new_parent,
            offset=pelvis_offset.copy() if old_index == root_index else transformed_offsets[node.name].copy(),
            channels=list(node.channels),
            channel_start=channel_cursor,
            end_site_offset=(transformed_end_sites[node.name].copy() if transformed_end_sites[node.name] is not None else None),
        )
        old_to_new[old_index] = len(new_nodes)
        new_nodes.append(next_node)
        channel_cursor += len(next_node.channels)

    new_motion = np.zeros((frame_count, channel_cursor), dtype=np.float64)
    for frame_index in range(frame_count):
        row_cursor = 0
        new_motion[frame_index, row_cursor : row_cursor + 6] = 0.0
        row_cursor += 6
        for old_index, node in enumerate(nodes):
            local_translation = np.asarray(transformed_local_translations[node.name][frame_index], dtype=np.float64)
            if old_index == root_index:
                local_translation = np.asarray(transformed_root_positions[frame_index], dtype=np.float64) - first_root_position
            local_rotation = np.asarray(transformed_local_rotations[node.name][frame_index], dtype=np.float64)
            values: list[float] = []
            rotation_values = rotation_matrix_to_channels(node, local_rotation)
            rotation_cursor = 0
            for channel_name in node.channels:
                if channel_name.endswith("position"):
                    values.append(float(local_translation[{"X": 0, "Y": 1, "Z": 2}[channel_name[0].upper()]]))
                elif channel_name.endswith("rotation"):
                    values.append(float(rotation_values[rotation_cursor]))
                    rotation_cursor += 1
            new_motion[frame_index, row_cursor : row_cursor + len(values)] = np.asarray(values, dtype=np.float64)
            row_cursor += len(values)

    write_bvh(output_bvh, new_nodes, new_motion, frame_time)
    canonical_pose0 = frame_pose(new_nodes, new_motion[0])
    canonical_basis = _orientation_basis(canonical_pose0.world_positions)
    report = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "inputBvh": str(input_bvh.resolve()),
        "outputBvh": str(output_bvh.resolve()),
        "frameCount": frame_count,
        "fps": float(round(1.0 / frame_time, 6)) if frame_time > 0 else 0.0,
        "frameTime": frame_time,
        "sourceBoneOrder": list(BRIDGE_BONE_ORDER_22),
        "originalRoot": {
            "name": source_root.name,
            "positionChannels": [channel for channel in source_root.channels if channel.endswith("position")],
            "rotationChannels": rotation_channels(source_root),
            "firstFrameWorldPosition": _vector_to_list(pose0.world_positions[source_root.name]),
        },
        "canonicalRoot": {
            "name": "Root",
            "isStatic": bool(np.allclose(new_motion[:, :6], 0.0, atol=1e-6)),
            "supportCenter": _vector_to_list(support_center),
            "pelvisOffset": _vector_to_list(pelvis_offset),
            "firstFrameFootHeight": float(support_center[2]),
        },
        "worldAlignmentMatrix": _matrix_to_dict(world_alignment),
        "targetWorldBasis": {
            "up": target_up_label,
            "forward": target_forward_label,
            "matrix": _matrix_to_dict(target_basis),
        },
        "firstFrameOrientation": {
            "before": {
                "side": _vector_to_list(source_basis[:, 0]),
                "up": _vector_to_list(source_basis[:, 1]),
                "forward": _vector_to_list(source_basis[:, 2]),
            },
            "after": {
                "side": _vector_to_list(canonical_basis[:, 0]),
                "up": _vector_to_list(canonical_basis[:, 1]),
                "forward": _vector_to_list(canonical_basis[:, 2]),
            },
        },
    }
    if report_output is not None:
        write_json(report_output, report)
    return report
