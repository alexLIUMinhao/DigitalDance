#!/usr/bin/env python3
"""Export simplified source BVH files from SMPL-X pose JSON."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


CHANNEL_ROTATION_ORDER = "XYZ"
DEFAULT_FRAME_TIME = 1.0 / 30.0
BVH_TO_TARGET_BASIS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class JointSpec:
    name: str
    source_name: str
    parent: str | None
    children: tuple[str, ...] = field(default_factory=tuple)
    primary_child: str | None = None
    secondary_child: str | None = None
    end_site_scale: float = 0.35


JOINT_SPECS: tuple[JointSpec, ...] = (
    JointSpec("Hips", "pelvis", None, ("Spine", "LeftUpLeg", "RightUpLeg"), primary_child="Spine", secondary_child="LeftUpLeg"),
    JointSpec("Spine", "spine1", "Hips", ("Chest",), primary_child="Chest"),
    JointSpec("Chest", "spine3", "Spine", ("Neck", "LeftShoulder", "RightShoulder"), primary_child="Neck", secondary_child="LeftShoulder"),
    JointSpec("Neck", "neck", "Chest", ("Head",), primary_child="Head"),
    JointSpec("Head", "head", "Neck"),
    JointSpec("LeftShoulder", "left_collar", "Chest", ("LeftArm",), primary_child="LeftArm"),
    JointSpec("LeftArm", "left_shoulder", "LeftShoulder", ("LeftForeArm",), primary_child="LeftForeArm"),
    JointSpec("LeftForeArm", "left_elbow", "LeftArm", ("LeftHand",), primary_child="LeftHand"),
    JointSpec("LeftHand", "left_wrist", "LeftForeArm"),
    JointSpec("RightShoulder", "right_collar", "Chest", ("RightArm",), primary_child="RightArm"),
    JointSpec("RightArm", "right_shoulder", "RightShoulder", ("RightForeArm",), primary_child="RightForeArm"),
    JointSpec("RightForeArm", "right_elbow", "RightArm", ("RightHand",), primary_child="RightHand"),
    JointSpec("RightHand", "right_wrist", "RightForeArm"),
    JointSpec("LeftUpLeg", "left_hip", "Hips", ("LeftLeg",), primary_child="LeftLeg"),
    JointSpec("LeftLeg", "left_knee", "LeftUpLeg", ("LeftFoot",), primary_child="LeftFoot"),
    JointSpec("LeftFoot", "left_ankle", "LeftLeg"),
    JointSpec("RightUpLeg", "right_hip", "Hips", ("RightLeg",), primary_child="RightLeg"),
    JointSpec("RightLeg", "right_knee", "RightUpLeg", ("RightFoot",), primary_child="RightFoot"),
    JointSpec("RightFoot", "right_ankle", "RightLeg"),
)


JOINT_BY_NAME = {joint.name: joint for joint in JOINT_SPECS}
CANONICAL_PRIMARY_DIRECTIONS = {
    "Hips": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Spine": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Chest": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Neck": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "Head": np.array([0.0, 1.0, 0.0], dtype=np.float64),
    "LeftShoulder": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "LeftArm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "LeftForeArm": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "LeftHand": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "RightShoulder": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "RightArm": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "RightForeArm": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "RightHand": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "LeftUpLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "LeftLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "LeftFoot": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightUpLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightLeg": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "RightFoot": np.array([0.0, -1.0, 0.0], dtype=np.float64),
}
CANONICAL_SECONDARY_DIRECTIONS = {
    "Hips": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "Chest": np.array([1.0, 0.0, 0.0], dtype=np.float64),
}


def _import_module_from_path(module_name: str, path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to import module from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)


def _load_motion_base_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "3d-digital-human" / "tools" / "motion_base" / "convert_dataset_motion.py"
    return _import_module_from_path("motion_base_convert_dataset_motion", module_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input SMPL-X pose json path.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output BVH path. Defaults to <input stem>_source.bvh in the same directory.",
    )
    parser.add_argument(
        "--frame-time",
        type=float,
        default=DEFAULT_FRAME_TIME,
        help="BVH frame time in seconds. Defaults to 1/30.",
    )
    parser.add_argument(
        "--match-official-mesh-preview",
        action="store_true",
        help="If available, use the upstream official mesh preview sampling window/fps instead of pose-json frameIndices.",
    )
    parser.add_argument(
        "--offset-profile",
        choices=["raw_bridge", "bridge_truth"],
        default="raw_bridge",
        help="BVH rest-offset profile. Use raw_bridge for viewer/import fidelity; bridge_truth is compare-space aligned.",
    )
    return parser.parse_args()


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    suffix = "_official_smplx_poses"
    if stem.endswith(suffix):
        stem = stem[: -len(suffix)] + "_source"
    else:
        stem = stem + "_source"
    return input_path.with_name(f"{stem}.bvh")


def _effective_frame_time(base_frame_time: float, frame_indices: list[int]) -> float:
    if len(frame_indices) < 2:
        return base_frame_time
    diffs = [max(1, int(b) - int(a)) for a, b in zip(frame_indices[:-1], frame_indices[1:])]
    if not diffs:
        return base_frame_time
    stride = int(round(float(np.median(np.asarray(diffs, dtype=np.float64)))))
    return base_frame_time * max(1, stride)


def _choose_sample_indices(
    frame_count: int,
    source_fps: int,
    preview_fps: int,
    start_seconds: float,
    max_seconds: float,
    max_frames: int | None,
) -> np.ndarray:
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
    if max_frames is not None and len(indices) > max_frames:
        indices = indices[:max_frames]
    if len(indices) == 0:
        indices = np.asarray([0], dtype=np.int64)
    return indices


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros((3,), dtype=np.float64)
    return vector / norm


def _transform_vectors(values: np.ndarray, basis_matrix: np.ndarray) -> np.ndarray:
    payload = np.asarray(values, dtype=np.float64)
    basis = np.asarray(basis_matrix, dtype=np.float64)
    if payload.ndim == 1:
        return basis @ payload
    return np.einsum("ij,...j->...i", basis, payload)


def _convert_motion_to_bvh_basis(
    root_positions: np.ndarray,
    local_quats: np.ndarray,
    offsets: dict[str, tuple[float, float, float]],
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[float, float, float]]]:
    root_bvh = _transform_vectors(root_positions, BVH_TO_TARGET_BASIS)
    local_matrices = Rotation.from_quat(local_quats.reshape((-1, 4))).as_matrix()
    local_bvh_matrices = BVH_TO_TARGET_BASIS @ local_matrices @ BVH_TO_TARGET_BASIS
    local_bvh = Rotation.from_matrix(local_bvh_matrices).as_quat().reshape(local_quats.shape)
    offsets_bvh = {
        joint_name: tuple(float(value) for value in _transform_vectors(np.asarray(offset, dtype=np.float64), BVH_TO_TARGET_BASIS))
        for joint_name, offset in offsets.items()
    }
    return root_bvh, local_bvh, offsets_bvh


def _bridge_child_indices(motion_module: ModuleType, joint_index: int) -> list[int]:
    return [index for index, parent in enumerate(motion_module.PARENTS) if parent == joint_index]


def _bridge_hierarchy_order(motion_module: ModuleType) -> list[int]:
    order: list[int] = []

    def walk(joint_index: int) -> None:
        order.append(joint_index)
        for child_index in _bridge_child_indices(motion_module, joint_index):
            walk(child_index)

    walk(0)
    return order


def _bridge_hierarchy_lines(
    motion_module: ModuleType,
    joint_index: int,
    offsets: dict[str, tuple[float, float, float]],
    *,
    indent: int = 0,
) -> list[str]:
    joint_name = motion_module.JOINT_NAMES[joint_index]
    prefix = "  " * indent
    label = "ROOT" if joint_index == 0 else "JOINT"
    lines = [f"{prefix}{label} {joint_name}", f"{prefix}{{"]
    lines.append(f"{prefix}  OFFSET {motion_module.format_offset(offsets[joint_name])}")
    if joint_index == 0:
        lines.append(f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation")

    child_indices = _bridge_child_indices(motion_module, joint_index)
    if not child_indices:
        lines.append(f"{prefix}  End Site")
        lines.append(f"{prefix}  {{")
        lines.append(f"{prefix}    OFFSET {motion_module.format_offset(motion_module.end_site_offset(joint_name, offsets))}")
        lines.append(f"{prefix}  }}")
    else:
        for child_index in child_indices:
            lines.extend(_bridge_hierarchy_lines(motion_module, child_index, offsets, indent=indent + 1))
    lines.append(f"{prefix}}}")
    return lines


def _write_bridge_bvh(
    motion_module: ModuleType,
    output_path: Path,
    root_positions: np.ndarray,
    local_quats: np.ndarray,
    fps: float,
    offsets: dict[str, tuple[float, float, float]],
) -> None:
    hierarchy = ["HIERARCHY", *_bridge_hierarchy_lines(motion_module, 0, offsets)]
    frame_count = int(root_positions.shape[0])
    local_eulers = Rotation.from_quat(local_quats.reshape((-1, 4))).as_euler(motion_module.ROTATION_ORDER, degrees=True)
    local_eulers = local_eulers.reshape((frame_count, len(motion_module.JOINT_NAMES), 3))
    hierarchy_order = _bridge_hierarchy_order(motion_module)

    motion_lines = ["MOTION", f"Frames: {frame_count}", f"Frame Time: {1.0 / fps:.8f}"]
    for frame_index in range(frame_count):
        channels: list[float] = [
            float(root_positions[frame_index, 0]),
            float(root_positions[frame_index, 1]),
            float(root_positions[frame_index, 2]),
            float(local_eulers[frame_index, 0, 0]),
            float(local_eulers[frame_index, 0, 1]),
            float(local_eulers[frame_index, 0, 2]),
        ]
        for joint_index in hierarchy_order[1:]:
            channels.extend(float(value) for value in local_eulers[frame_index, joint_index])
        motion_lines.append(" ".join(f"{value:.6f}" for value in channels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(hierarchy + motion_lines) + "\n", encoding="utf-8")


def _fallback_perpendicular(vector: np.ndarray) -> np.ndarray:
    if abs(float(vector[1])) < 0.95:
        candidate = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        candidate = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    perpendicular = np.cross(vector, candidate)
    if np.linalg.norm(perpendicular) <= 1e-8:
        perpendicular = np.cross(vector, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return _normalize(perpendicular)


def _rotation_matrix_from_axes(primary: np.ndarray, secondary: np.ndarray | None = None) -> np.ndarray:
    y_axis = _normalize(primary)
    if np.linalg.norm(y_axis) <= 1e-8:
        return np.identity(3, dtype=np.float64)

    if secondary is None or np.linalg.norm(secondary) <= 1e-8:
        x_axis = _fallback_perpendicular(y_axis)
    else:
        side = secondary - np.dot(secondary, y_axis) * y_axis
        if np.linalg.norm(side) <= 1e-8:
            x_axis = _fallback_perpendicular(y_axis)
        else:
            x_axis = _normalize(side)
    z_axis = _normalize(np.cross(x_axis, y_axis))
    if np.linalg.norm(z_axis) <= 1e-8:
        z_axis = _fallback_perpendicular(y_axis)
    x_axis = _normalize(np.cross(y_axis, z_axis))
    return np.column_stack((x_axis, y_axis, z_axis))


def _rotation_matrix_from_canonical(primary: np.ndarray, canonical_primary: np.ndarray, secondary: np.ndarray | None = None) -> np.ndarray:
    current_primary = _normalize(primary)
    rest_primary = _normalize(canonical_primary)
    if np.linalg.norm(current_primary) <= 1e-8 or np.linalg.norm(rest_primary) <= 1e-8:
        return np.identity(3, dtype=np.float64)

    current_basis = _rotation_matrix_from_axes(current_primary, secondary)
    rest_basis = _rotation_matrix_from_axes(rest_primary, None)
    return current_basis @ rest_basis.T


def _matrix_to_euler_xyz_degrees(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    sy = float(np.clip(m[0, 2], -1.0, 1.0))
    if abs(sy) < 0.999999:
        y_angle = math.asin(sy)
        x_angle = math.atan2(-m[1, 2], m[2, 2])
        z_angle = math.atan2(-m[0, 1], m[0, 0])
    else:
        y_angle = math.copysign(math.pi / 2.0, sy)
        x_angle = math.atan2(m[2, 1], m[1, 1])
        z_angle = 0.0
    return np.degrees(np.asarray([x_angle, y_angle, z_angle], dtype=np.float64))


def _joint_world_positions_from_pose_json(payload: dict[str, Any]) -> tuple[list[int], dict[str, np.ndarray]]:
    if "boneOrder" in payload and "poses" in payload:
        bone_order = list(payload["boneOrder"])
        source_index = {name: index for index, name in enumerate(bone_order)}
        missing = [joint.source_name for joint in JOINT_SPECS if joint.source_name not in source_index]
        if missing:
            raise KeyError(f"Input pose json is missing required source joints: {missing}")

        frames = np.asarray(payload["poses"], dtype=np.float64)
        if frames.ndim != 3 or frames.shape[2] != 3:
            raise ValueError("Expected `poses` to have shape [frame, joint, xyz].")
        frame_indices = list(payload.get("frameIndices") or range(int(frames.shape[0])))
        positions = {
            joint.name: frames[:, source_index[joint.source_name], :]
            for joint in JOINT_SPECS
        }
        return frame_indices, positions

    required_keys = {"transl", "global_orient", "body_pose"}
    if required_keys.issubset(payload.keys()):
        raise NotImplementedError(
            "This script currently supports pose json files with `boneOrder + poses`. "
            "For raw `transl/global_orient/body_pose` inputs, add the SMPL-X forward pass first."
        )

    raise ValueError("Unsupported input json structure.")


def _load_finedance_bridge_truth_offsets(
    motion_module: ModuleType,
) -> dict[str, tuple[float, float, float]] | None:
    repo_root = Path(__file__).resolve().parents[2]
    profile_path = repo_root / "3d-digital-human" / "motion_base" / "config" / "bridge_profile.json"
    if not profile_path.exists():
        return None
    payload = json.loads(profile_path.read_text(encoding="utf-8"))
    offsets_payload = (
        payload.get("datasetProfiles", {})
        .get("finedance", {})
        .get("offsets", {})
    )
    if not isinstance(offsets_payload, dict):
        return None
    offsets = {}
    for joint_name in motion_module.JOINT_NAMES:
        values = offsets_payload.get(joint_name)
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return None
        offsets[joint_name] = (float(values[0]), float(values[1]), float(values[2]))
    return offsets


def _load_official_mesh_summary(payload: dict[str, Any]) -> dict[str, Any] | None:
    input_path = str(payload.get("inputPath", "") or "")
    if not input_path.endswith(".npy"):
        return None
    sequence_id = Path(input_path).stem
    repo_root = Path(__file__).resolve().parents[2]
    summary_path = (
        repo_root
        / "motion-base-assets"
        / "previewCache"
        / "upstream_baselines"
        / "finedance"
        / sequence_id
        / f"finedance_{sequence_id}_official_mesh_summary.json"
    )
    if not summary_path.exists():
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _try_build_raw_motion_bvh(
    payload: dict[str, Any],
    output_path: Path,
    frame_time: float,
    *,
    match_official_mesh_preview: bool = False,
    offset_profile: str = "raw_bridge",
) -> bool:
    input_path = str(payload.get("inputPath", "") or "")
    if not input_path.endswith(".npy"):
        return False

    raw_motion_path = Path(input_path).expanduser()
    if not raw_motion_path.exists():
        return False

    motion_module = _load_motion_base_module()
    root_positions, full_quats, _ = motion_module.load_finedance_motion_components(raw_motion_path)
    local_quats = motion_module.remap_finedance_quaternions(full_quats)

    frame_indices = [int(value) for value in payload.get("frameIndices", [])]
    effective_frame_time = _effective_frame_time(frame_time, frame_indices)
    if match_official_mesh_preview:
        summary = _load_official_mesh_summary(payload)
        if summary is None:
            raise FileNotFoundError("official mesh summary not found for requested match mode")
        sample_indices = _choose_sample_indices(
            frame_count=int(root_positions.shape[0]),
            source_fps=int(summary.get("sourceFps", round(1.0 / frame_time)) or round(1.0 / frame_time)),
            preview_fps=int(summary.get("fps", round(1.0 / frame_time)) or round(1.0 / frame_time)),
            start_seconds=float(summary.get("startSeconds", 0.0) or 0.0),
            max_seconds=float(summary.get("maxSeconds", 0.0) or 0.0),
            max_frames=(
                int(summary["maxFrames"])
                if summary.get("maxFrames") is not None
                else None
            ),
        )
        effective_frame_time = 1.0 / float(summary.get("fps", round(1.0 / frame_time)) or round(1.0 / frame_time))
    elif frame_indices:
        valid_indices = [index for index in frame_indices if 0 <= index < int(root_positions.shape[0])]
        if len(valid_indices) != len(frame_indices):
            raise ValueError("pose json frameIndices exceed the raw motion frame range")
        sample_indices = np.asarray(valid_indices, dtype=np.int64)
        root_positions = root_positions[sample_indices]
        local_quats = local_quats[sample_indices]
    else:
        sample_indices = np.arange(int(root_positions.shape[0]), dtype=np.int64)

    if match_official_mesh_preview:
        root_positions = root_positions[sample_indices]
        local_quats = local_quats[sample_indices]

    if offset_profile == "bridge_truth":
        offsets = _load_finedance_bridge_truth_offsets(motion_module) or dict(motion_module.OFFSETS)
    else:
        offsets = dict(motion_module.OFFSETS)
    root_positions, local_quats, offsets = _convert_motion_to_bvh_basis(root_positions, local_quats, offsets)
    effective_fps = 1.0 / effective_frame_time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_bridge_bvh(motion_module, output_path, root_positions, local_quats, effective_fps, offsets)
    return True


def _build_rest_offsets(rest_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    offsets: dict[str, np.ndarray] = {"Hips": np.zeros((3,), dtype=np.float64)}
    for joint in JOINT_SPECS:
        if joint.parent is None:
            continue
        measured = rest_positions[joint.name] - rest_positions[joint.parent]
        length = float(np.linalg.norm(measured))
        if length <= 1e-8:
            length = 0.1
        direction = CANONICAL_PRIMARY_DIRECTIONS.get(joint.name, np.array([0.0, 1.0, 0.0], dtype=np.float64))
        offsets[joint.name] = _normalize(direction) * length
    return offsets


def _estimate_end_site_offsets(offsets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    end_sites: dict[str, np.ndarray] = {}
    for joint in JOINT_SPECS:
        if joint.children:
            continue
        parent_offset = offsets.get(joint.name, np.array([0.0, 0.1, 0.0], dtype=np.float64))
        if np.linalg.norm(parent_offset) <= 1e-8:
            parent_offset = np.array([0.0, 0.1, 0.0], dtype=np.float64)
        end_sites[joint.name] = parent_offset * joint.end_site_scale
    return end_sites


def _build_world_rotation_matrices(frame_positions: dict[str, np.ndarray], use_canonical_rest_axes: bool = False) -> dict[str, np.ndarray]:
    world_rotations: dict[str, np.ndarray] = {}
    for joint in JOINT_SPECS:
        if joint.primary_child is None:
            world_rotations[joint.name] = np.identity(3, dtype=np.float64)
            continue
        primary = frame_positions[joint.primary_child] - frame_positions[joint.name]
        secondary = None
        if joint.secondary_child is not None:
            secondary = frame_positions[joint.secondary_child] - frame_positions[joint.name]
        if use_canonical_rest_axes:
            canonical_primary = CANONICAL_PRIMARY_DIRECTIONS.get(joint.name, np.array([0.0, 1.0, 0.0], dtype=np.float64))
            world_rotations[joint.name] = _rotation_matrix_from_canonical(primary, canonical_primary, secondary)
        else:
            world_rotations[joint.name] = _rotation_matrix_from_axes(primary, secondary)
    return world_rotations


def _build_rest_positions_from_offsets(offsets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    positions: dict[str, np.ndarray] = {"Hips": np.zeros((3,), dtype=np.float64)}
    pending = True
    while pending:
        pending = False
        for joint in JOINT_SPECS:
            if joint.name in positions:
                continue
            if joint.parent is None or joint.parent not in positions:
                pending = True
                continue
            positions[joint.name] = positions[joint.parent] + offsets[joint.name]
    return positions


def _build_local_rotation_channels(frame_positions: dict[str, np.ndarray], rest_positions: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    world_rotations = _build_world_rotation_matrices(frame_positions)
    rest_world_rotations = _build_world_rotation_matrices(rest_positions, use_canonical_rest_axes=True)
    local_channels: dict[str, np.ndarray] = {}
    for joint in JOINT_SPECS:
        parent_name = joint.parent
        if parent_name is None:
            current_local = world_rotations[joint.name]
            rest_local = rest_world_rotations[joint.name]
        else:
            current_local = world_rotations[parent_name].T @ world_rotations[joint.name]
            rest_local = rest_world_rotations[parent_name].T @ rest_world_rotations[joint.name]
        local_channels[joint.name] = rest_local.T @ current_local
    return local_channels


def _build_local_euler_channels(
    frame_positions: dict[str, np.ndarray],
    rest_positions: dict[str, np.ndarray],
    calibration_rotations: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    local_rotations = _build_local_rotation_channels(frame_positions, rest_positions)
    local_channels: dict[str, np.ndarray] = {}
    for joint in JOINT_SPECS:
        local_rotation = local_rotations[joint.name]
        if calibration_rotations is not None:
            local_rotation = calibration_rotations[joint.name].T @ local_rotation
        local_channels[joint.name] = _matrix_to_euler_xyz_degrees(local_rotation)
    return local_channels


def _format_offset(vector: np.ndarray) -> str:
    return f"{vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}"


def _write_joint_hierarchy(lines: list[str], joint_name: str, indent: int, offsets: dict[str, np.ndarray], end_sites: dict[str, np.ndarray]) -> None:
    joint = JOINT_BY_NAME[joint_name]
    prefix = "  " * indent
    block_type = "ROOT" if joint.parent is None else "JOINT"
    lines.append(f"{prefix}{block_type} {joint.name}")
    lines.append(f"{prefix}{{")
    offset = offsets[joint.name]
    lines.append(f"{prefix}  OFFSET {_format_offset(offset)}")
    if joint.parent is None:
        lines.append(
            f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Xrotation Yrotation Zrotation"
        )
    else:
        lines.append(f"{prefix}  CHANNELS 3 Xrotation Yrotation Zrotation")

    for child_name in joint.children:
        _write_joint_hierarchy(lines, child_name, indent + 1, offsets, end_sites)

    if not joint.children:
        end_offset = end_sites[joint.name]
        lines.append(f"{prefix}  End Site")
        lines.append(f"{prefix}  {{")
        lines.append(f"{prefix}    OFFSET {_format_offset(end_offset)}")
        lines.append(f"{prefix}  }}")
    lines.append(f"{prefix}}}")


def build_bvh_text(frame_indices: list[int], positions_by_joint: dict[str, np.ndarray], frame_time: float) -> str:
    if not frame_indices:
        raise ValueError("No frames found in input pose json.")

    rest_positions = {
        joint_name: np.asarray(frames[0], dtype=np.float64)
        for joint_name, frames in positions_by_joint.items()
    }
    offsets = _build_rest_offsets(rest_positions)
    canonical_rest_positions = _build_rest_positions_from_offsets(offsets)
    end_sites = _estimate_end_site_offsets(offsets)
    calibration_rotations = _build_local_rotation_channels(rest_positions, canonical_rest_positions)

    hierarchy_lines = ["HIERARCHY"]
    _write_joint_hierarchy(hierarchy_lines, "Hips", 0, offsets, end_sites)

    motion_lines = ["MOTION", f"Frames: {len(frame_indices)}", f"Frame Time: {frame_time:.8f}"]
    for frame_id in range(len(frame_indices)):
        frame_positions = {
            joint_name: np.asarray(frames[frame_id], dtype=np.float64)
            for joint_name, frames in positions_by_joint.items()
        }
        local_rotations = _build_local_euler_channels(
            frame_positions,
            canonical_rest_positions,
            calibration_rotations=calibration_rotations,
        )
        root_position = frame_positions["Hips"]
        channels: list[float] = [float(root_position[0]), float(root_position[1]), float(root_position[2])]
        for joint in JOINT_SPECS:
            euler = local_rotations[joint.name]
            channels.extend(float(value) for value in euler.tolist())
        # Root already contributed translation, so only append rotations here.
        root_rotation = channels[3:6]
        frame_values = [channels[0], channels[1], channels[2], *root_rotation]
        for joint in JOINT_SPECS[1:]:
            frame_values.extend(local_rotations[joint.name].tolist())
        motion_lines.append(" ".join(f"{value:.6f}" for value in frame_values))

    return "\n".join(hierarchy_lines + motion_lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else _default_output_path(input_path).resolve()
    )

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not _try_build_raw_motion_bvh(
        payload,
        output_path,
        float(args.frame_time),
        match_official_mesh_preview=bool(args.match_official_mesh_preview),
        offset_profile=str(args.offset_profile),
    ):
        frame_indices, positions_by_joint = _joint_world_positions_from_pose_json(payload)
        bvh_text = build_bvh_text(
            frame_indices=frame_indices,
            positions_by_joint=positions_by_joint,
            frame_time=_effective_frame_time(float(args.frame_time), frame_indices),
        )
        output_path.write_text(bvh_text, encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
