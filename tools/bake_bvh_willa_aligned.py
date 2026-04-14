#!/usr/bin/env python3
"""Bake the Willa/Blender world alignment into an existing BVH file."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.source_bvh_validation import (  # noqa: E402
    BRIDGE_BONE_ORDER_22,
    extract_bvh_joint_positions,
    summarize_values,
    transform_vectors,
)


# Blender/Willa world:
# - up/trunk +Z
# - forward -Y
# Current source BVH import convention:
# - trunk +Y
# - forward +Z
# So we bake a global +90 deg X rotation into the BVH itself.
WILLA_ALIGNMENT_BASIS = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def import_module_from_path(module_name: str, path: Path) -> ModuleType:
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


def load_pose_driver_module() -> ModuleType:
    return import_module_from_path(
        "export_dataset_pose_driver_sequences_module",
        REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "export_dataset_pose_driver_sequences.py",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_bvh",
        type=Path,
        help="Input source BVH.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output BVH path. Defaults to <input stem>_willa_aligned.bvh.",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate that the new BVH equals a global +90 deg X rotation of the input BVH.",
    )
    return parser.parse_args()


def axis_index(channel_name: str) -> int:
    axis = channel_name[0].upper()
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def default_output_path(input_bvh: Path) -> Path:
    return input_bvh.with_name(f"{input_bvh.stem}_willa_aligned.bvh")


def transform_offsets_in_hierarchy(lines: list[str], basis: np.ndarray) -> list[str]:
    output: list[str] = []
    in_hierarchy = True
    for line in lines:
        stripped = line.strip()
        if stripped == "MOTION":
            in_hierarchy = False
            output.append(line)
            continue
        if in_hierarchy and stripped.startswith("OFFSET "):
            prefix = line[: len(line) - len(line.lstrip())]
            _, x_text, y_text, z_text = stripped.split()
            values = np.asarray([float(x_text), float(y_text), float(z_text)], dtype=np.float64)
            rotated = transform_vectors(values, basis)
            output.append(
                f"{prefix}OFFSET {rotated[0]:.6f} {rotated[1]:.6f} {rotated[2]:.6f}"
            )
            continue
        output.append(line)
    return output


def transform_motion_channels(nodes: list[object], motion: np.ndarray, basis: np.ndarray) -> np.ndarray:
    transformed = np.asarray(motion, dtype=np.float64).copy()
    basis_inv = basis.T
    frame_count = int(transformed.shape[0])
    for node in nodes:
        channels = list(getattr(node, "channels", []))
        if not channels:
            continue
        start = int(getattr(node, "channel_start"))
        stop = start + len(channels)
        chunk = transformed[:, start:stop].copy()

        position_columns: list[tuple[int, int]] = []
        rotation_columns: list[int] = []
        rotation_order: list[str] = []
        for local_index, channel_name in enumerate(channels):
            if channel_name.endswith("position"):
                position_columns.append((local_index, axis_index(channel_name)))
            elif channel_name.endswith("rotation"):
                rotation_columns.append(local_index)
                rotation_order.append(channel_name[0].upper())

        if position_columns:
            positions = np.zeros((frame_count, 3), dtype=np.float64)
            for local_index, axis in position_columns:
                positions[:, axis] = chunk[:, local_index]
            positions = transform_vectors(positions, basis)
            for local_index, axis in position_columns:
                chunk[:, local_index] = positions[:, axis]

        if rotation_columns:
            eulers = np.stack([chunk[:, column] for column in rotation_columns], axis=1)
            matrices = Rotation.from_euler("".join(rotation_order), eulers, degrees=True).as_matrix()
            rotated_matrices = basis @ matrices @ basis_inv
            rotated_eulers = Rotation.from_matrix(rotated_matrices).as_euler("".join(rotation_order), degrees=True)
            for column_index, local_index in enumerate(rotation_columns):
                chunk[:, local_index] = rotated_eulers[:, column_index]

        transformed[:, start:stop] = chunk
    return transformed


def build_output_lines(hierarchy_lines: list[str], motion: np.ndarray, frame_time: float) -> list[str]:
    motion_lines = [
        "MOTION",
        f"Frames: {int(motion.shape[0])}",
        f"Frame Time: {float(frame_time):.8f}",
    ]
    motion_lines.extend(" ".join(f"{float(value):.6f}" for value in frame) for frame in motion)
    trimmed = []
    for line in hierarchy_lines:
        if line.strip() == "MOTION":
            break
        trimmed.append(line)
    return trimmed + motion_lines


def validate_alignment(input_bvh: Path, output_bvh: Path, parse_bvh) -> dict[str, float]:
    _, _, original_positions, _ = extract_bvh_joint_positions(input_bvh, parse_bvh, bone_order=BRIDGE_BONE_ORDER_22)
    _, _, aligned_positions, _ = extract_bvh_joint_positions(output_bvh, parse_bvh, bone_order=BRIDGE_BONE_ORDER_22)
    expected_positions = transform_vectors(original_positions, WILLA_ALIGNMENT_BASIS)
    error = np.linalg.norm(aligned_positions - expected_positions, axis=2)
    summary = summarize_values(error)
    return {
        "mean": float(summary["mean"]),
        "p95": float(summary["p95"]),
        "max": float(summary["max"]),
    }


def main() -> int:
    args = parse_args()
    input_bvh = args.input_bvh.expanduser().resolve()
    output_bvh = (args.output or default_output_path(input_bvh)).expanduser().resolve()

    pose_driver_module = load_pose_driver_module()
    nodes, motion, frame_time = pose_driver_module.parse_bvh(input_bvh)

    raw_lines = input_bvh.read_text(encoding="utf-8").splitlines()
    aligned_hierarchy = transform_offsets_in_hierarchy(raw_lines, WILLA_ALIGNMENT_BASIS)
    aligned_motion = transform_motion_channels(nodes, motion, WILLA_ALIGNMENT_BASIS)
    output_lines = build_output_lines(aligned_hierarchy, aligned_motion, frame_time)

    output_bvh.parent.mkdir(parents=True, exist_ok=True)
    output_bvh.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    print(f"wrote {output_bvh}")
    if args.validate:
        stats = validate_alignment(input_bvh, output_bvh, pose_driver_module.parse_bvh)
        print(
            "validation "
            f"mean={stats['mean']:.9f} "
            f"p95={stats['p95']:.9f} "
            f"max={stats['max']:.9f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
