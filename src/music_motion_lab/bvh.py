from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from scipy.spatial.transform import Rotation  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - Blender runtime fallback
    from math import degrees as deg_value, radians as rad_value

    from mathutils import Euler, Matrix, Vector  # type: ignore

    class Rotation:  # type: ignore[override]
        def __init__(self, matrix: Matrix) -> None:
            self._matrix = matrix.to_3x3()

        @classmethod
        def identity(cls) -> "Rotation":
            return cls(Matrix.Identity(3))

        @classmethod
        def from_euler(cls, order: str, angles: Iterable[float], degrees: bool = False) -> "Rotation":
            values = list(angles)
            if degrees:
                values = [rad_value(float(value)) for value in values]
            return cls(Euler(tuple(values), order).to_matrix())

        @classmethod
        def from_matrix(cls, matrix: np.ndarray) -> "Rotation":
            return cls(Matrix(np.asarray(matrix, dtype=np.float64).tolist()))

        def as_matrix(self) -> np.ndarray:
            return np.asarray(self._matrix, dtype=np.float64)

        def as_euler(self, order: str, degrees: bool = False) -> list[float]:
            values = list(self._matrix.to_euler(order))
            if degrees:
                values = [float(degrees_value) for degrees_value in map(deg_value, values)]
            return values

        def apply(self, vector: np.ndarray) -> np.ndarray:
            payload = self._matrix @ Vector(np.asarray(vector, dtype=np.float64).tolist())
            return np.asarray(payload, dtype=np.float64)

        def __mul__(self, other: "Rotation") -> "Rotation":
            return Rotation(self._matrix @ other._matrix)


@dataclass
class BvhNode:
    name: str
    parent: int | None
    offset: np.ndarray
    channels: list[str]
    channel_start: int
    end_site_offset: np.ndarray | None = None


@dataclass(frozen=True)
class FramePose:
    world_positions: dict[str, np.ndarray]
    world_rotations: dict[str, Rotation]
    local_translations: dict[str, np.ndarray]
    local_rotations: dict[str, Rotation]
    end_sites: dict[str, np.ndarray]


@dataclass(frozen=True)
class MotionSummary:
    nodes: list[BvhNode]
    motion: np.ndarray
    frame_time: float


def _axis_index(channel_name: str) -> int:
    axis = channel_name[0].upper()
    return {"X": 0, "Y": 1, "Z": 2}[axis]


def parse_bvh(input_path: Path) -> MotionSummary:
    lines = [line.rstrip() for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines or lines[0].strip() != "HIERARCHY":
        raise ValueError(f"invalid BVH header in {input_path}")

    nodes: list[BvhNode] = []
    channel_cursor = 0

    def parse_joint(index: int, parent: int | None) -> int:
        nonlocal channel_cursor
        parts = lines[index].split()
        if len(parts) < 2 or parts[0] not in {"ROOT", "JOINT"}:
            raise ValueError(f"expected ROOT/JOINT at line {index + 1}: {lines[index]}")
        node = BvhNode(
            name=parts[1],
            parent=parent,
            offset=np.zeros(3, dtype=np.float64),
            channels=[],
            channel_start=0,
            end_site_offset=None,
        )
        node_index = len(nodes)
        nodes.append(node)
        index += 1
        if lines[index].strip() != "{":
            raise ValueError(f"expected '{{' at line {index + 1}: {lines[index]}")
        index += 1
        while index < len(lines):
            line = lines[index].strip()
            if line == "}":
                return index + 1
            if line.startswith("OFFSET "):
                _, x, y, z = line.split()
                node.offset = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
                index += 1
                continue
            if line.startswith("CHANNELS "):
                parts = line.split()
                count = int(parts[1])
                node.channels = parts[2 : 2 + count]
                node.channel_start = channel_cursor
                channel_cursor += count
                index += 1
                continue
            if line.startswith("JOINT "):
                index = parse_joint(index, node_index)
                continue
            if line == "End Site":
                index += 1
                if lines[index].strip() != "{":
                    raise ValueError(f"expected '{{' after End Site at line {index + 1}")
                index += 1
                while index < len(lines):
                    site_line = lines[index].strip()
                    if site_line.startswith("OFFSET "):
                        _, x, y, z = site_line.split()
                        node.end_site_offset = np.asarray([float(x), float(y), float(z)], dtype=np.float64)
                    if site_line == "}":
                        break
                    index += 1
                if index >= len(lines):
                    raise ValueError("unterminated End Site block")
                index += 1
                continue
            if line == "MOTION":
                return index
            raise ValueError(f"unexpected BVH token at line {index + 1}: {line}")
        raise ValueError("unterminated BVH hierarchy")

    motion_start = parse_joint(1, None)
    if motion_start >= len(lines) or lines[motion_start].strip() != "MOTION":
        raise ValueError(f"missing MOTION section in {input_path}")
    if motion_start + 2 >= len(lines):
        raise ValueError(f"incomplete MOTION section in {input_path}")

    frame_count = int(lines[motion_start + 1].split(":", 1)[1].strip())
    frame_time = float(lines[motion_start + 2].split(":", 1)[1].strip())
    motion_values = []
    for line in lines[motion_start + 3 : motion_start + 3 + frame_count]:
        motion_values.append([float(value) for value in line.split()])
    motion = np.asarray(motion_values, dtype=np.float64)
    return MotionSummary(nodes=nodes, motion=motion, frame_time=frame_time)


def children_by_parent(nodes: list[BvhNode]) -> dict[int | None, list[int]]:
    mapping: dict[int | None, list[int]] = {None: []}
    for index, node in enumerate(nodes):
        mapping.setdefault(node.parent, []).append(index)
        mapping.setdefault(index, [])
    return mapping


def _local_transform(node: BvhNode, frame_values: np.ndarray) -> tuple[np.ndarray, Rotation]:
    channel_values = frame_values[node.channel_start : node.channel_start + len(node.channels)]
    local_translation = np.zeros(3, dtype=np.float64)
    rotation_order: list[str] = []
    rotation_values: list[float] = []
    for channel_name, value in zip(node.channels, channel_values.tolist()):
        if channel_name.endswith("position"):
            local_translation[_axis_index(channel_name)] = float(value)
        elif channel_name.endswith("rotation"):
            rotation_order.append(channel_name[0].upper())
            rotation_values.append(float(value))
    local_rotation = Rotation.from_euler("".join(rotation_order), rotation_values, degrees=True) if rotation_order else Rotation.identity()
    return local_translation, local_rotation


def frame_pose(nodes: list[BvhNode], frame_values: np.ndarray) -> FramePose:
    positions: dict[str, np.ndarray] = {}
    rotations: dict[str, Rotation] = {}
    local_translations: dict[str, np.ndarray] = {}
    local_rotations: dict[str, Rotation] = {}
    end_sites: dict[str, np.ndarray] = {}

    for index, node in enumerate(nodes):
        local_translation, local_rotation = _local_transform(node, frame_values)
        local_translations[node.name] = local_translation
        local_rotations[node.name] = local_rotation
        local_origin = np.asarray(node.offset, dtype=np.float64) + local_translation
        if node.parent is None:
            positions[node.name] = local_origin
            rotations[node.name] = local_rotation
        else:
            parent_name = nodes[node.parent].name
            positions[node.name] = positions[parent_name] + rotations[parent_name].apply(local_origin)
            rotations[node.name] = rotations[parent_name] * local_rotation
        if node.end_site_offset is not None:
            end_sites[node.name] = positions[node.name] + rotations[node.name].apply(node.end_site_offset)

    return FramePose(
        world_positions=positions,
        world_rotations=rotations,
        local_translations=local_translations,
        local_rotations=local_rotations,
        end_sites=end_sites,
    )


def rest_pose(nodes: list[BvhNode]) -> FramePose:
    channel_count = 0
    for node in nodes:
        channel_count = max(channel_count, node.channel_start + len(node.channels))
    return frame_pose(nodes, np.zeros((channel_count,), dtype=np.float64))


def channel_count(nodes: Iterable[BvhNode]) -> int:
    total = 0
    for node in nodes:
        total += len(node.channels)
    return total


def write_bvh(path: Path, nodes: list[BvhNode], motion: np.ndarray, frame_time: float) -> Path:
    child_map = children_by_parent(nodes)

    lines: list[str] = ["HIERARCHY"]

    def emit_joint(node_index: int, indent: int) -> None:
        node = nodes[node_index]
        prefix = "  " * indent
        keyword = "ROOT" if node.parent is None else "JOINT"
        lines.append(f"{prefix}{keyword} {node.name}")
        lines.append(f"{prefix}{{")
        lines.append(f"{prefix}  OFFSET {node.offset[0]:.6f} {node.offset[1]:.6f} {node.offset[2]:.6f}")
        if node.channels:
            channels = " ".join(node.channels)
            lines.append(f"{prefix}  CHANNELS {len(node.channels)} {channels}")
        for child_index in child_map.get(node_index, []):
            emit_joint(child_index, indent + 1)
        if node.end_site_offset is not None:
            lines.append(f"{prefix}  End Site")
            lines.append(f"{prefix}  {{")
            lines.append(
                f"{prefix}    OFFSET {node.end_site_offset[0]:.6f} {node.end_site_offset[1]:.6f} {node.end_site_offset[2]:.6f}"
            )
            lines.append(f"{prefix}  }}")
        lines.append(f"{prefix}}}")

    roots = child_map.get(None, [])
    if len(roots) != 1:
        raise ValueError("BVH writer expects exactly one root node")
    emit_joint(roots[0], 0)

    lines.append("MOTION")
    lines.append(f"Frames: {int(motion.shape[0])}")
    lines.append(f"Frame Time: {float(frame_time):.8f}")
    for row in motion:
        lines.append(" ".join(f"{float(value):.6f}" for value in row.tolist()))

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def rotation_channels(node: BvhNode) -> list[str]:
    return [channel for channel in node.channels if channel.endswith("rotation")]


def position_channels(node: BvhNode) -> list[str]:
    return [channel for channel in node.channels if channel.endswith("position")]


def rotation_matrix_to_channels(node: BvhNode, matrix: np.ndarray) -> list[float]:
    order = "".join(channel[0].upper() for channel in rotation_channels(node))
    if not order:
        return []
    return Rotation.from_matrix(np.asarray(matrix, dtype=np.float64)).as_euler(order, degrees=True).tolist()
