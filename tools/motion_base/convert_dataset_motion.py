#!/usr/bin/env python3
"""Convert dataset-native dance motion into rawFbx bridge assets for Motion Base."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from common import (
    DATASET_CATALOG_PATH,
    load_dataset_catalog,
    resolve_asset_root,
    resolve_dataset_storage_root,
    slugify,
    upsert_records,
    utc_now_iso,
    write_json,
)


ROTATION_ORDER = "ZXY"
JOINT_NAMES = [
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
    "left_hand",
    "right_hand",
]
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
OFFSETS = {
    "pelvis": (0.0, 0.0, 0.0),
    "left_hip": (0.09, -0.09, 0.0),
    "right_hip": (-0.09, -0.09, 0.0),
    "spine1": (0.0, 0.10, 0.0),
    "left_knee": (0.0, -0.42, 0.0),
    "right_knee": (0.0, -0.42, 0.0),
    "spine2": (0.0, 0.11, 0.0),
    "left_ankle": (0.0, -0.42, 0.0),
    "right_ankle": (0.0, -0.42, 0.0),
    "spine3": (0.0, 0.12, 0.0),
    "left_foot": (0.0, -0.05, 0.12),
    "right_foot": (0.0, -0.05, 0.12),
    "neck": (0.0, 0.13, 0.0),
    "left_collar": (0.07, 0.05, 0.0),
    "right_collar": (-0.07, 0.05, 0.0),
    "head": (0.0, 0.12, 0.0),
    "left_shoulder": (0.12, 0.0, 0.0),
    "right_shoulder": (-0.12, 0.0, 0.0),
    "left_elbow": (0.27, 0.0, 0.0),
    "right_elbow": (-0.27, 0.0, 0.0),
    "left_wrist": (0.25, 0.0, 0.0),
    "right_wrist": (-0.25, 0.0, 0.0),
    "left_hand": (0.09, 0.0, 0.0),
    "right_hand": (-0.09, 0.0, 0.0),
}
BONE_ALIASES = {
    "pelvis": ("pelvis", "hips", "root"),
    "left_hip": ("left_hip", "l_hip", "lefthip", "leftupleg", "left_up_leg"),
    "right_hip": ("right_hip", "r_hip", "righthip", "rightupleg", "right_up_leg"),
    "spine1": ("spine1", "spine_01", "spine"),
    "left_knee": ("left_knee", "l_knee", "leftleg", "left_low_leg"),
    "right_knee": ("right_knee", "r_knee", "rightleg", "right_low_leg"),
    "spine2": ("spine2", "spine_02"),
    "left_ankle": ("left_ankle", "l_ankle", "leftfoot"),
    "right_ankle": ("right_ankle", "r_ankle", "rightfoot"),
    "spine3": ("spine3", "spine_03", "chest"),
    "left_foot": ("left_foot", "l_foot", "lefttoe", "left_toe"),
    "right_foot": ("right_foot", "r_foot", "righttoe", "right_toe"),
    "neck": ("neck",),
    "left_collar": ("left_collar", "l_collar", "leftshoulder"),
    "right_collar": ("right_collar", "r_collar", "rightshoulder"),
    "head": ("head",),
    "left_shoulder": ("left_shoulder", "l_shoulder", "leftarm", "left_upper_arm"),
    "right_shoulder": ("right_shoulder", "r_shoulder", "rightarm", "right_upper_arm"),
    "left_elbow": ("left_elbow", "l_elbow", "leftforearm", "left_lower_arm"),
    "right_elbow": ("right_elbow", "r_elbow", "rightforearm", "right_lower_arm"),
    "left_wrist": ("left_wrist", "l_wrist", "lefthand"),
    "right_wrist": ("right_wrist", "r_wrist", "righthand"),
    "left_hand": ("left_hand", "l_hand", "leftindex1"),
    "right_hand": ("right_hand", "r_hand", "rightindex1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DATASET_CATALOG_PATH, help="Input dataset catalog.")
    parser.add_argument("--output-catalog", type=Path, default=DATASET_CATALOG_PATH, help="Output dataset catalog.")
    parser.add_argument("--dataset", action="append", help="Optional dataset filter.")
    parser.add_argument("--sequence-id", action="append", help="Optional sequence filter.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max entries to convert.")
    parser.add_argument("--force", action="store_true", help="Re-convert even if the output already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned conversions without writing.")
    return parser.parse_args()


def identity_quaternions(frame_count: int) -> np.ndarray:
    quats = np.zeros((frame_count, len(JOINT_NAMES), 4), dtype=np.float64)
    quats[..., 3] = 1.0
    return quats


def safe_quaternions(quaternions: np.ndarray) -> np.ndarray:
    quats = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return quats / norms


def resolve_dataset_raw_root(dataset_name: str) -> Path | None:
    return resolve_dataset_storage_root(dataset_name, "raw")


def choose_output_path(entry: Dict[str, Any], extension: str = ".bvh") -> Path:
    raw_fbx_root = resolve_asset_root("rawFbx")
    if raw_fbx_root is None:
        raise RuntimeError("rawFbx root is not configured in asset_roots.local.json")

    style_family = slugify(entry.get("targetStyleFamily", "") or "contemporary")
    style_substyle = slugify(entry.get("targetStyleSubstyle", "") or entry.get("datasetName", "") or "dataset")
    meta_action = slugify(entry.get("expectedMetaAction", "") or "basic_step")
    filename = f"{slugify(entry['datasetName'])}_{slugify(entry['sequenceId'])}{extension}"
    return raw_fbx_root / "datasets" / slugify(entry["datasetName"]) / style_family / style_substyle / meta_action / filename


def load_aist_or_aioz_motion(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = pickle.loads(path.read_bytes())
    if not isinstance(payload, dict):
        raise ValueError(f"unsupported pickle payload in {path}")

    poses = payload.get("smpl_poses")
    trans = payload.get("smpl_trans")
    if trans is None:
        trans = payload.get("root_trans")
    if trans is None:
        trans = payload.get("trans")
    if poses is None or trans is None:
        raise ValueError(f"missing smpl_poses/root translation in {path}")

    pose_array = np.asarray(poses, dtype=np.float64)
    trans_array = np.asarray(trans, dtype=np.float64)
    if pose_array.ndim == 3:
        pose_array = pose_array[0]
    if trans_array.ndim == 3:
        trans_array = trans_array[0]
    if pose_array.shape[-1] < 72:
        raise ValueError(f"expected at least 72 pose dims in {path}, got {pose_array.shape[-1]}")

    pose_array = pose_array[:, :72].reshape((-1, len(JOINT_NAMES), 3))
    quats = Rotation.from_rotvec(pose_array.reshape((-1, 3))).as_quat().reshape((-1, len(JOINT_NAMES), 4))
    return np.asarray(trans_array, dtype=np.float64), safe_quaternions(quats)


def load_finedance_motion(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    if payload.ndim != 2 or payload.shape[1] < 75:
        raise ValueError(f"unexpected FineDance motion shape for {path}: {payload.shape}")

    trans = payload[:, :3]
    poses = payload[:, 3:75].reshape((-1, len(JOINT_NAMES), 3))
    quats = Rotation.from_rotvec(poses.reshape((-1, 3))).as_quat().reshape((-1, len(JOINT_NAMES), 4))
    return np.asarray(trans, dtype=np.float64), safe_quaternions(quats)


def bone_index_map(bone_names: Iterable[str]) -> Dict[str, int]:
    normalized = {slugify(name): index for index, name in enumerate(bone_names)}
    mapping: Dict[str, int] = {}
    for joint_name in JOINT_NAMES:
        for alias in BONE_ALIASES.get(joint_name, (joint_name,)):
            alias_key = slugify(alias)
            if alias_key in normalized:
                mapping[joint_name] = normalized[alias_key]
                break
    if "pelvis" not in mapping:
        raise ValueError("phantomdance skeleton is missing a pelvis/root bone")
    return mapping


def world_to_local_quats(world_quats: np.ndarray) -> np.ndarray:
    frame_count = world_quats.shape[0]
    local_quats = identity_quaternions(frame_count)
    world_quats = safe_quaternions(world_quats)
    world_rots = Rotation.from_quat(world_quats.reshape((-1, 4))).as_quat().reshape((-1, len(JOINT_NAMES), 4))

    for frame_index in range(frame_count):
        root_rotation = Rotation.from_quat(world_rots[frame_index, 0])
        local_quats[frame_index, 0] = root_rotation.as_quat()
        for joint_index in range(1, len(JOINT_NAMES)):
            parent_index = PARENTS[joint_index]
            parent_rotation = Rotation.from_quat(world_rots[frame_index, parent_index])
            joint_rotation = Rotation.from_quat(world_rots[frame_index, joint_index])
            local_quats[frame_index, joint_index] = (parent_rotation.inv() * joint_rotation).as_quat()
    return safe_quaternions(local_quats)


def load_phantomdance_motion(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    bone_names = payload.get("bone_name", [])
    raw_rotations = payload.get("rotations", [])
    raw_root_positions = payload.get("root_positions", [])

    index_map = bone_index_map(bone_names)
    filtered_positions: List[np.ndarray] = []
    filtered_rotations: List[np.ndarray] = []
    for frame_index in range(min(len(raw_root_positions), len(raw_rotations))):
        root_position = raw_root_positions[frame_index]
        frame_rotations = raw_rotations[frame_index]
        if not isinstance(root_position, list) or len(root_position) != 3:
            continue
        if not isinstance(frame_rotations, list):
            continue
        try:
            position_array = np.asarray(root_position, dtype=np.float64)
            rotation_array = np.asarray(frame_rotations, dtype=np.float64)
        except ValueError:
            continue
        if rotation_array.ndim != 2 or rotation_array.shape[1] != 4:
            continue
        filtered_positions.append(position_array)
        filtered_rotations.append(rotation_array)

    if not filtered_positions or not filtered_rotations:
        raise ValueError(f"unexpected PhantomDance payload shape for {path}")

    rotations = np.stack(filtered_rotations, axis=0)
    root_positions = np.stack(filtered_positions, axis=0)
    frame_count = rotations.shape[0]
    ordered_world_quats = identity_quaternions(frame_count)
    for joint_index, joint_name in enumerate(JOINT_NAMES):
        source_index = index_map.get(joint_name)
        if source_index is None:
            continue
        ordered_world_quats[:, joint_index, :] = rotations[:, source_index, :]
    return root_positions, world_to_local_quats(ordered_world_quats)


def load_motion(entry: Dict[str, Any], motion_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    source_format = str(entry.get("sourceFormat", "") or "")
    if source_format == "smpl_pkl":
        return load_aist_or_aioz_motion(motion_path)
    if source_format == "smplh_npy":
        return load_finedance_motion(motion_path)
    if source_format == "json_quat_smpl24":
        return load_phantomdance_motion(motion_path)
    raise ValueError(f"unsupported source format: {source_format}")


def end_site_offset(joint_name: str) -> Tuple[float, float, float]:
    x, y, z = OFFSETS.get(joint_name, (0.0, 0.08, 0.0))
    if joint_name in {"left_foot", "right_foot"}:
        return (0.0, 0.0, 0.08)
    if abs(x) > 0.0:
        return (0.08 if x >= 0 else -0.08, 0.0, 0.0)
    return (0.0, 0.08, 0.0)


def format_offset(values: Tuple[float, float, float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def hierarchy_lines(joint_index: int, indent: int = 0) -> List[str]:
    joint_name = JOINT_NAMES[joint_index]
    label = "ROOT" if joint_index == 0 else "JOINT"
    prefix = "  " * indent
    lines = [f"{prefix}{label} {joint_name}", f"{prefix}{{"]
    lines.append(f"{prefix}  OFFSET {format_offset(OFFSETS[joint_name])}")
    if joint_index == 0:
        lines.append(f"{prefix}  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{prefix}  CHANNELS 3 Zrotation Xrotation Yrotation")

    child_indices = [index for index, parent in enumerate(PARENTS) if parent == joint_index]
    if not child_indices:
        lines.append(f"{prefix}  End Site")
        lines.append(f"{prefix}  {{")
        lines.append(f"{prefix}    OFFSET {format_offset(end_site_offset(joint_name))}")
        lines.append(f"{prefix}  }}")
    else:
        for child_index in child_indices:
            lines.extend(hierarchy_lines(child_index, indent + 1))

    lines.append(f"{prefix}}}")
    return lines


def bvh_motion_lines(root_positions: np.ndarray, local_quats: np.ndarray) -> List[str]:
    frame_count = root_positions.shape[0]
    local_eulers = Rotation.from_quat(local_quats.reshape((-1, 4))).as_euler(ROTATION_ORDER, degrees=True)
    local_eulers = local_eulers.reshape((frame_count, len(JOINT_NAMES), 3))

    lines = ["MOTION", f"Frames: {frame_count}", f"Frame Time: {1.0 / 30.0:.8f}"]
    for frame_index in range(frame_count):
        channels: List[float] = [
            float(root_positions[frame_index, 0]),
            float(root_positions[frame_index, 1]),
            float(root_positions[frame_index, 2]),
            float(local_eulers[frame_index, 0, 0]),
            float(local_eulers[frame_index, 0, 1]),
            float(local_eulers[frame_index, 0, 2]),
        ]
        for joint_index in range(1, len(JOINT_NAMES)):
            channels.extend(float(value) for value in local_eulers[frame_index, joint_index])
        lines.append(" ".join(f"{value:.6f}" for value in channels))
    return lines


def write_bvh(output_path: Path, root_positions: np.ndarray, local_quats: np.ndarray, fps: float) -> None:
    hierarchy = ["HIERARCHY", *hierarchy_lines(0)]
    frame_count = root_positions.shape[0]
    local_eulers = Rotation.from_quat(local_quats.reshape((-1, 4))).as_euler(ROTATION_ORDER, degrees=True)
    local_eulers = local_eulers.reshape((frame_count, len(JOINT_NAMES), 3))
    motion = ["MOTION", f"Frames: {frame_count}", f"Frame Time: {1.0 / fps:.8f}"]
    for frame_index in range(frame_count):
        channels: List[float] = [
            float(root_positions[frame_index, 0]),
            float(root_positions[frame_index, 1]),
            float(root_positions[frame_index, 2]),
            float(local_eulers[frame_index, 0, 0]),
            float(local_eulers[frame_index, 0, 1]),
            float(local_eulers[frame_index, 0, 2]),
        ]
        for joint_index in range(1, len(JOINT_NAMES)):
            channels.extend(float(value) for value in local_eulers[frame_index, joint_index])
        motion.append(" ".join(f"{value:.6f}" for value in channels))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(hierarchy + motion) + "\n", encoding="utf-8")


def select_entries(catalog: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    requested_datasets = {slugify(name) for name in (args.dataset or [])}
    requested_sequences = {slugify(name) for name in (args.sequence_id or [])}
    entries: List[Dict[str, Any]] = []
    for entry in catalog.get("entries", []):
        dataset_name = slugify(entry.get("datasetName", "") or "")
        sequence_id = slugify(entry.get("sequenceId", "") or "")
        if requested_datasets and dataset_name not in requested_datasets:
            continue
        if requested_sequences and sequence_id not in requested_sequences:
            continue
        entries.append(dict(entry))
        if args.limit and len(entries) >= args.limit:
            break
    return entries


def convert_entry(entry: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    updated = dict(entry)
    raw_root = resolve_dataset_raw_root(entry["datasetName"])
    if raw_root is None:
        updated["conversionStatus"] = "blocked_missing_dataset_root"
        updated["notes"] = "dataset raw root is not configured"
        return updated

    motion_path = raw_root / str(entry.get("motionPath", "") or "")
    if not motion_path.exists():
        updated["conversionStatus"] = "blocked_missing_motion"
        updated["notes"] = f"missing motion file: {motion_path}"
        return updated

    output_path = choose_output_path(entry)
    updated["convertedMotionRelPath"] = output_path.relative_to(resolve_asset_root("rawFbx")).as_posix()

    if output_path.exists() and not args.force:
        updated["conversionStatus"] = "converted"
        updated["convertedAtUtc"] = updated.get("convertedAtUtc") or utc_now_iso()
        updated["notes"] = updated.get("notes", "")
        return updated

    root_positions, local_quats = load_motion(entry, motion_path)
    if root_positions.shape[0] == 0:
        raise ValueError(f"no frames found in {motion_path}")

    if not args.dry_run:
        write_bvh(output_path, root_positions, local_quats, float(entry.get("fps", 30.0) or 30.0))

    updated["conversionStatus"] = "converted"
    updated["convertedAtUtc"] = utc_now_iso()
    updated["notes"] = ""
    return updated


def main() -> int:
    args = parse_args()
    catalog = load_dataset_catalog() if not args.catalog.exists() else json.loads(args.catalog.read_text(encoding="utf-8"))
    selected_entries = select_entries(catalog, args)
    updated_entries: List[Dict[str, Any]] = []
    converted = 0
    blocked = 0

    for entry in selected_entries:
        try:
            updated_entry = convert_entry(entry, args)
            if updated_entry.get("conversionStatus") == "converted":
                converted += 1
            else:
                blocked += 1
        except Exception as exc:  # noqa: BLE001 - keep catalog progress visible.
            updated_entry = dict(entry)
            updated_entry["conversionStatus"] = "failed"
            updated_entry["notes"] = f"{type(exc).__name__}: {exc}"
            blocked += 1
        updated_entries.append(updated_entry)

    merged_entries = upsert_records(
        catalog.get("entries", []),
        updated_entries,
        "entryId",
        lambda item: f"{item.get('datasetName', '')}::{item.get('sequenceId', '')}",
    )
    payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "entries": merged_entries,
    }

    if not args.dry_run:
        write_json(args.output_catalog, payload)

    print(f"dataset conversions converted={converted} blocked={blocked} selected={len(selected_entries)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
