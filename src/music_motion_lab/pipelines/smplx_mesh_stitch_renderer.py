from __future__ import annotations

import html
import importlib.util
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

from ..utils import slugify, utc_now_iso, write_json


@dataclass(frozen=True)
class MeshCache:
    sequence_id: str
    path: Path
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    source_frame_indices: np.ndarray
    source_motion_path: str

    @property
    def frame_to_cache_index(self) -> dict[int, int]:
        return {int(frame): index for index, frame in enumerate(self.source_frame_indices.tolist())}


@dataclass(frozen=True)
class StitchedMeshSequence:
    vertices: np.ndarray
    joints: np.ndarray
    faces: np.ndarray
    scene_frame_start: int
    scene_frame_end: int
    fps: int
    step_reports: list[dict[str, Any]]
    transition_reports: list[dict[str, Any]]
    rhythm_lock_reports: list[dict[str, Any]]

    @property
    def frame_count(self) -> int:
        return int(self.vertices.shape[0])


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _project_path(project_root: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (project_root / candidate).resolve()


def _import_module_from_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_frame_ranges(frame_ranges: list[dict[str, Any]]) -> list[dict[str, int]]:
    ranges: list[tuple[int, int]] = []
    for frame_range in frame_ranges:
        start = _safe_int(frame_range.get("start"))
        end = _safe_int(frame_range.get("end_exclusive"), start + 1)
        if end > start:
            ranges.append((start, end))
    ranges.sort()
    merged: list[dict[str, int]] = []
    for start, end in ranges:
        if not merged or start > merged[-1]["end_exclusive"]:
            merged.append({"start": start, "end_exclusive": end})
        else:
            merged[-1]["end_exclusive"] = max(merged[-1]["end_exclusive"], end)
    return merged


def _smplx_model_path(motion_base_assets_root: Path) -> Path:
    candidates = [
        motion_base_assets_root / "models" / "smplx" / "SMPLX_NEUTRAL.npz",
        motion_base_assets_root / "models" / "SMPLX_NEUTRAL.npz",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("missing SMPL-X model: expected SMPLX_NEUTRAL.npz under motion-base-assets/models")


def load_finedance_smplx_mesh_frames(
    source_motion_path: Path,
    frame_ranges: list[dict[str, int]],
    motion_base_assets_root: Path,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load only the requested FineDance frames through the official SMPL-X path."""
    render_path = motion_base_assets_root / "tools" / "upstream" / "FineDance" / "render.py"
    if not render_path.exists():
        raise FileNotFoundError(f"FineDance render.py not found: {render_path}")
    if not source_motion_path.exists():
        raise FileNotFoundError(f"FineDance source motion not found: {source_motion_path}")

    render_module = _import_module_from_path("finedance_upstream_render_for_stitch", render_path)
    motion = np.asarray(render_module.motion_data_load_process(str(source_motion_path)), dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] < 159:
        raise ValueError(f"expected FineDance motion array [frames, >=159], got {motion.shape}")

    selected_frames: list[np.ndarray] = []
    selected_motion: list[np.ndarray] = []
    for frame_range in _normalize_frame_ranges(frame_ranges):
        start = max(0, int(frame_range["start"]))
        end = min(int(frame_range["end_exclusive"]), int(motion.shape[0]))
        if end <= start:
            continue
        selected_frames.append(np.arange(start, end, dtype=np.int32))
        selected_motion.append(motion[start:end])
    if not selected_motion:
        raise ValueError(f"no valid frame ranges for {source_motion_path}")

    motion_slice = np.concatenate(selected_motion, axis=0)
    source_frame_indices = np.concatenate(selected_frames, axis=0)

    import torch
    from smplx import SMPLX

    model_path = _smplx_model_path(motion_base_assets_root)
    model = SMPLX(str(model_path), use_pca=False, flat_hand_mean=True).eval()
    motion_tensor = torch.from_numpy(motion_slice)
    vertices_batches: list[np.ndarray] = []
    joints_batches: list[np.ndarray] = []
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, motion_tensor.shape[0], batch_size):
            batch = motion_tensor[start : start + batch_size]
            zeros_10 = torch.zeros((batch.shape[0], 10), dtype=torch.float32)
            zeros_3 = torch.zeros((batch.shape[0], 3), dtype=torch.float32)
            output = model(
                betas=zeros_10,
                transl=batch[:, :3],
                global_orient=batch[:, 3:6],
                body_pose=batch[:, 6:69],
                jaw_pose=zeros_3,
                leye_pose=zeros_3,
                reye_pose=zeros_3,
                left_hand_pose=batch[:, 69:114],
                right_hand_pose=batch[:, 114:159],
                expression=zeros_10,
            )
            vertices_batches.append(output.vertices.detach().cpu().numpy().astype(np.float32, copy=False))
            joints_batches.append(output.joints.detach().cpu().numpy().astype(np.float32, copy=False))

    vertices = np.concatenate(vertices_batches, axis=0)
    joints = np.concatenate(joints_batches, axis=0)
    faces = np.asarray(model.faces, dtype=np.int32)
    return vertices, joints, faces, source_frame_indices


def load_mesh_cache(cache_path: Path) -> MeshCache:
    with np.load(cache_path, allow_pickle=False) as payload:
        sequence_id = str(payload["sequence_id"].item()) if "sequence_id" in payload else cache_path.stem
        source_motion_path = str(payload["source_motion_path"].item()) if "source_motion_path" in payload else ""
        return MeshCache(
            sequence_id=sequence_id,
            path=cache_path,
            vertices=np.asarray(payload["vertices"], dtype=np.float32),
            joints=np.asarray(payload["joints"], dtype=np.float32),
            faces=np.asarray(payload["faces"], dtype=np.int32),
            source_frame_indices=np.asarray(payload["source_frame_indices"], dtype=np.int32),
            source_motion_path=source_motion_path,
        )


def materialize_mesh_caches(
    manifest: dict[str, Any],
    project_root: Path,
    motion_base_assets_root: Path,
    force: bool = False,
    batch_size: int = 128,
) -> tuple[dict[str, MeshCache], list[dict[str, Any]]]:
    caches: dict[str, MeshCache] = {}
    summaries: list[dict[str, Any]] = []
    for request in manifest.get("cache_requests", []) or []:
        sequence_id = str(request.get("sequence_id", "unknown") or "unknown")
        cache_path = _project_path(project_root, str(request.get("cache_path", "")))
        source_motion_path = _project_path(project_root, str(request.get("source_motion_path", "")))
        frame_ranges = _normalize_frame_ranges(list(request.get("frame_ranges", []) or []))
        status = "hit"
        if force or not cache_path.exists():
            status = "rebuilt" if force and cache_path.exists() else "miss_built"
            vertices, joints, faces, source_frame_indices = load_finedance_smplx_mesh_frames(
                source_motion_path=source_motion_path,
                frame_ranges=frame_ranges,
                motion_base_assets_root=motion_base_assets_root,
                batch_size=batch_size,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                cache_path,
                sequence_id=np.asarray(sequence_id),
                source_motion_path=np.asarray(str(source_motion_path)),
                vertices=vertices,
                joints=joints,
                faces=faces,
                source_frame_indices=source_frame_indices,
            )

        cache = load_mesh_cache(cache_path)
        caches[sequence_id] = cache
        summaries.append(
            {
                "sequence_id": sequence_id,
                "status": status,
                "cache_path": str(cache_path),
                "source_motion_path": str(source_motion_path),
                "frame_ranges": frame_ranges,
                "cached_frame_count": int(cache.vertices.shape[0]),
                "vertex_count": int(cache.vertices.shape[1]),
                "joint_count": int(cache.joints.shape[1]),
                "face_count": int(cache.faces.shape[0]),
            }
        )
    return caches, summaries


def _sample_step_from_cache(step: dict[str, Any], cache: MeshCache) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scene_start = _safe_int(step.get("scene_frame_start"), 1)
    scene_end = max(scene_start, _safe_int(step.get("scene_frame_end"), scene_start))
    target_count = scene_end - scene_start + 1
    source_start = _safe_int(step.get("source_frame_start"))
    source_end = max(source_start + 1, _safe_int(step.get("source_frame_end_exclusive"), source_start + 1))
    source_frames = np.rint(np.linspace(source_start, source_end - 1, num=target_count)).astype(np.int32)
    lookup = cache.frame_to_cache_index
    missing = [int(frame) for frame in source_frames.tolist() if int(frame) not in lookup]
    if missing:
        preview = ", ".join(str(frame) for frame in missing[:5])
        raise KeyError(f"cache {cache.path} missing source frames for step {step.get('index')}: {preview}")
    cache_indices = np.asarray([lookup[int(frame)] for frame in source_frames.tolist()], dtype=np.int32)
    return cache.vertices[cache_indices].copy(), cache.joints[cache_indices].copy(), source_frames


def _smoothstep(value: float) -> float:
    t = max(0.0, min(1.0, float(value)))
    return t * t * (3.0 - 2.0 * t)


def _mean_vertex_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right, axis=-1).mean())


def _root_xz_delta(left_root: np.ndarray, right_root: np.ndarray) -> float:
    return float(np.linalg.norm(left_root[[0, 2]] - right_root[[0, 2]]))


def _root_xyz_delta(left_root: np.ndarray, right_root: np.ndarray) -> float:
    return float(np.linalg.norm(left_root - right_root))


def _offset_mesh(vertices: np.ndarray, joints: np.ndarray, offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return vertices + offset.reshape(1, 1, 3), joints + offset.reshape(1, 1, 3)


def _fill_missing_frames(vertices: np.ndarray, joints: np.ndarray, filled: np.ndarray) -> None:
    if bool(filled.all()):
        return
    last_valid: int | None = None
    for index in range(len(filled)):
        if filled[index]:
            last_valid = index
        elif last_valid is not None:
            vertices[index] = vertices[last_valid]
            joints[index] = joints[last_valid]
            filled[index] = True
    next_valid: int | None = None
    for index in range(len(filled) - 1, -1, -1):
        if filled[index]:
            next_valid = index
        elif next_valid is not None:
            vertices[index] = vertices[next_valid]
            joints[index] = joints[next_valid]
            filled[index] = True


def _rhythm_lock_reports(manifest: dict[str, Any], scene_start: int, scene_end: int) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for step in manifest.get("steps", []) or []:
        for lock in step.get("rhythm_locks", []) or []:
            target_frame = _safe_int(lock.get("scene_frame"))
            if target_frame < scene_start:
                error = scene_start - target_frame
            elif target_frame > scene_end:
                error = target_frame - scene_end
            else:
                error = 0
            reports.append(
                {
                    "step_index": _safe_int(step.get("index")),
                    "kind": lock.get("kind", "rhythm_lock"),
                    "target_scene_frame": target_frame,
                    "rendered_scene_frame": min(max(target_frame, scene_start), scene_end),
                    "frame_error": int(error),
                    "time_sec": _safe_float(lock.get("time_sec")),
                }
            )
    return reports


def _max_temporal_delta(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    deltas = np.linalg.norm(values[1:] - values[:-1], axis=-1)
    return float(deltas.mean(axis=tuple(range(1, deltas.ndim))).max())


def _boundary_velocity_delta(previous_values: np.ndarray, incoming_values: np.ndarray, index: int | None = None) -> float:
    if previous_values.shape[0] < 2 or incoming_values.shape[0] < 2:
        return 0.0
    previous_delta = previous_values[-1] - previous_values[-2]
    incoming_delta = incoming_values[1] - incoming_values[0]
    if index is not None and previous_delta.ndim >= 2:
        previous_delta = previous_delta[index]
        incoming_delta = incoming_delta[index]
    return float(np.linalg.norm(incoming_delta - previous_delta, axis=-1).mean())


def _foot_slide_proxy(joints: np.ndarray) -> float:
    if joints.shape[0] < 2 or joints.shape[1] == 0:
        return 0.0
    candidate_indices = [index for index in (7, 8, 10, 11, 20, 21) if index < joints.shape[1]]
    if not candidate_indices:
        candidate_indices = list(range(min(4, joints.shape[1])))
    foot_deltas = np.linalg.norm(np.diff(joints[:, candidate_indices, :], axis=0), axis=-1)
    return float(np.percentile(foot_deltas, 90.0)) if foot_deltas.size else 0.0


def _smooth_array_window(values: np.ndarray, start_index: int, end_index: int, passes: int, strength: float = 0.72) -> None:
    if end_index <= start_index or passes <= 0:
        return
    start_index = max(0, start_index)
    end_index = min(values.shape[0] - 1, end_index)
    if end_index - start_index < 2:
        return
    kernel = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float32)
    kernel = kernel / kernel.sum()
    window = values[start_index : end_index + 1].copy()
    for _ in range(max(1, int(passes))):
        padded = np.pad(window, ((2, 2), (0, 0), (0, 0)), mode="edge")
        averaged = np.zeros_like(window)
        for kernel_index, weight in enumerate(kernel):
            averaged += padded[kernel_index : kernel_index + len(window)] * weight
        window = window * (1.0 - strength) + averaged * strength
    values[start_index : end_index + 1] = window


def _limit_temporal_delta_window(values: np.ndarray, start_index: int, end_index: int, target_delta: float) -> None:
    if end_index <= start_index or target_delta <= 0.0:
        return
    start_index = max(0, start_index)
    end_index = min(values.shape[0] - 1, end_index)
    for index in range(start_index + 1, end_index + 1):
        delta = values[index] - values[index - 1]
        mean_delta = float(np.linalg.norm(delta, axis=-1).mean())
        if mean_delta > target_delta:
            values[index] = values[index - 1] + delta * float(target_delta / max(mean_delta, 1e-8))


def _smooth_transition_windows(
    vertices: np.ndarray,
    joints: np.ndarray,
    transition_reports: list[dict[str, Any]],
    scene_start: int,
    smooth_frames: int,
    passes: int,
) -> None:
    if smooth_frames <= 0 or passes <= 0:
        for report in transition_reports:
            report["smoothing_frames"] = 0
            report["smoothing_passes"] = 0
        return
    for report in transition_reports:
        outgoing_end = _safe_int(report.get("outgoing_end_frame"))
        incoming_start = _safe_int(report.get("incoming_start_frame"), _safe_int(report.get("boundary_frame")))
        window_start = max(0, outgoing_end - scene_start - smooth_frames)
        window_end = min(vertices.shape[0] - 1, incoming_start - scene_start + smooth_frames)
        before_vertices = vertices[window_start : window_end + 1].copy()
        before_joints = joints[window_start : window_end + 1].copy()
        _smooth_array_window(vertices, window_start, window_end, passes=passes)
        _smooth_array_window(joints, window_start, window_end, passes=passes)
        after_vertices = vertices[window_start : window_end + 1]
        after_joints = joints[window_start : window_end + 1]
        report.update(
            {
                "smoothing_frames": int(smooth_frames),
                "smoothing_passes": int(passes),
                "smoothing_window": {
                    "start_frame": int(scene_start + window_start),
                    "end_frame": int(scene_start + window_end),
                },
                "max_temporal_vertex_delta_before_smoothing": round(_max_temporal_delta(before_vertices), 6),
                "max_temporal_vertex_delta_after_smoothing": round(_max_temporal_delta(after_vertices), 6),
                "max_temporal_joint_delta_before_smoothing": round(_max_temporal_delta(before_joints), 6),
                "max_temporal_joint_delta_after_smoothing": round(_max_temporal_delta(after_joints), 6),
            }
        )


def compose_stitched_mesh_sequence(
    manifest: dict[str, Any],
    caches_by_sequence: dict[str, MeshCache],
    transition_smooth_frames: int = 12,
    transition_smooth_passes: int = 2,
) -> StitchedMeshSequence:
    steps = list(manifest.get("steps", []) or [])
    if not steps:
        raise ValueError("stitch manifest has no steps")

    fps = _safe_int(manifest.get("fps"), 30)
    scene_start = min(_safe_int(step.get("scene_frame_start"), 1) for step in steps)
    scene_end = max(_safe_int(step.get("scene_frame_end"), scene_start) for step in steps)
    total_frames = scene_end - scene_start + 1

    first_cache = caches_by_sequence[str(steps[0].get("source_sequence"))]
    vertex_count = int(first_cache.vertices.shape[1])
    joint_count = int(first_cache.joints.shape[1])
    faces = first_cache.faces
    stitched_vertices = np.zeros((total_frames, vertex_count, 3), dtype=np.float32)
    stitched_joints = np.zeros((total_frames, joint_count, 3), dtype=np.float32)
    filled = np.zeros((total_frames,), dtype=bool)

    step_reports: list[dict[str, Any]] = []
    transition_reports: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None

    for step in steps:
        sequence_id = str(step.get("source_sequence", "unknown") or "unknown")
        if sequence_id not in caches_by_sequence:
            raise KeyError(f"missing mesh cache for source sequence {sequence_id}")
        cache = caches_by_sequence[sequence_id]
        if int(cache.vertices.shape[1]) != vertex_count:
            raise ValueError(f"cache vertex count mismatch for {sequence_id}")

        raw_vertices, raw_joints, source_frames = _sample_step_from_cache(step, cache)
        scene_frame_start = _safe_int(step.get("scene_frame_start"), scene_start)
        scene_frame_end = max(scene_frame_start, _safe_int(step.get("scene_frame_end"), scene_frame_start))
        start_index = scene_frame_start - scene_start
        end_index = start_index + raw_vertices.shape[0]
        raw_entry_root = raw_joints[0, 0].copy()
        raw_exit_root = raw_joints[-1, 0].copy()
        offset = np.zeros((3,), dtype=np.float32)
        if previous is None:
            offset[0] = -raw_entry_root[0]
            offset[2] = -raw_entry_root[2]
        else:
            previous_exit_root = previous["aligned_exit_root"]
            offset[:] = previous_exit_root - raw_entry_root

        aligned_vertices, aligned_joints = _offset_mesh(raw_vertices, raw_joints, offset)
        pre_blend_vertices_0 = aligned_vertices[0].copy()
        pre_blend_joints_0 = aligned_joints[0].copy()
        blend_in = max(0, _safe_int(step.get("blend_in_frames")))
        blend_out = max(0, _safe_int(step.get("blend_out_frames")))

        if previous is not None:
            previous_vertices = previous["aligned_vertices"]
            previous_joints = previous["aligned_joints"]
            requested_blend_count = min(blend_in, _safe_int(previous["blend_out"]), len(previous_vertices), len(aligned_vertices))
            blend_count = 0
            if requested_blend_count > 0:
                candidate_vertices = aligned_vertices.copy()
                candidate_joints = aligned_joints.copy()
                for local_index in range(requested_blend_count):
                    alpha = _smoothstep((local_index + 1) / float(requested_blend_count + 1))
                    previous_index = max(0, len(previous_vertices) - requested_blend_count + local_index)
                    candidate_vertices[local_index] = previous_vertices[previous_index] * (1.0 - alpha) + candidate_vertices[local_index] * alpha
                    candidate_joints[local_index] = previous_joints[previous_index] * (1.0 - alpha) + candidate_joints[local_index] * alpha
                before_delta = _mean_vertex_delta(previous_vertices[-1], pre_blend_vertices_0)
                candidate_delta = _mean_vertex_delta(previous_vertices[-1], candidate_vertices[0])
                if candidate_delta <= before_delta * 1.02 + 1e-6:
                    aligned_vertices = candidate_vertices
                    aligned_joints = candidate_joints
                    blend_count = requested_blend_count
            gap_start = int(previous["scene_frame_end"]) + 1
            gap_end = scene_frame_start - 1
            if gap_end >= gap_start:
                for scene_frame in range(gap_start, gap_end + 1):
                    alpha = _smoothstep((scene_frame - gap_start + 1) / float(gap_end - gap_start + 2))
                    buffer_index = scene_frame - scene_start
                    stitched_vertices[buffer_index] = previous_vertices[-1] * (1.0 - alpha) + aligned_vertices[0] * alpha
                    stitched_joints[buffer_index] = previous_joints[-1] * (1.0 - alpha) + aligned_joints[0] * alpha
                    filled[buffer_index] = True

            transition_reports.append(
                {
                    "index": len(transition_reports),
                    "outgoing_step": int(previous["step_index"]),
                    "incoming_step": _safe_int(step.get("index")),
                    "outgoing_end_frame": int(previous["scene_frame_end"]),
                    "incoming_start_frame": int(scene_frame_start),
                    "boundary_frame": scene_frame_start,
                    "gap_frames": max(0, scene_frame_start - int(previous["scene_frame_end"]) - 1),
                    "blend_frames": int(blend_count),
                    "blend_frames_requested": int(requested_blend_count),
                    "root_offset_xyz": [round(float(value), 6) for value in offset.tolist()],
                    "raw_root_xz_delta": round(_root_xz_delta(previous["raw_exit_root"], raw_entry_root), 6),
                    "raw_root_xyz_delta": round(_root_xyz_delta(previous["raw_exit_root"], raw_entry_root), 6),
                    "aligned_root_xz_delta_before_blend": round(
                        _root_xz_delta(previous["aligned_exit_root"], pre_blend_joints_0[0]),
                        6,
                    ),
                    "aligned_root_xyz_delta_before_blend": round(
                        _root_xyz_delta(previous["aligned_exit_root"], pre_blend_joints_0[0]),
                        6,
                    ),
                    "joint_delta_before_blend": round(_mean_vertex_delta(previous["aligned_joints"][-1], pre_blend_joints_0), 6),
                    "joint_delta_after_blend": round(_mean_vertex_delta(previous["aligned_joints"][-1], aligned_joints[0]), 6),
                    "vertex_delta_before_blend": round(_mean_vertex_delta(previous["aligned_vertices"][-1], pre_blend_vertices_0), 6),
                    "vertex_delta_after_blend": round(_mean_vertex_delta(previous["aligned_vertices"][-1], aligned_vertices[0]), 6),
                    "root_velocity_delta_proxy": round(
                        _boundary_velocity_delta(previous["aligned_joints"], aligned_joints, index=0),
                        6,
                    ),
                    "root_acceleration_discontinuity_proxy": round(
                        _boundary_velocity_delta(previous["aligned_joints"], aligned_joints, index=0) * float(fps),
                        6,
                    ),
                    "joint_jerk_proxy": round(_boundary_velocity_delta(previous["aligned_joints"], aligned_joints), 6),
                    "foot_slide_proxy": round(
                        max(_foot_slide_proxy(previous["aligned_joints"][-min(6, len(previous["aligned_joints"])) :]), _foot_slide_proxy(aligned_joints[: min(6, len(aligned_joints))])),
                        6,
                    ),
                }
            )

        for local_index in range(raw_vertices.shape[0]):
            buffer_index = start_index + local_index
            if buffer_index < 0 or buffer_index >= total_frames:
                continue
            if filled[buffer_index]:
                alpha = _smoothstep((local_index + 1) / float(max(2, raw_vertices.shape[0] + 1)))
                stitched_vertices[buffer_index] = stitched_vertices[buffer_index] * (1.0 - alpha) + aligned_vertices[local_index] * alpha
                stitched_joints[buffer_index] = stitched_joints[buffer_index] * (1.0 - alpha) + aligned_joints[local_index] * alpha
            else:
                stitched_vertices[buffer_index] = aligned_vertices[local_index]
                stitched_joints[buffer_index] = aligned_joints[local_index]
                filled[buffer_index] = True

        step_reports.append(
            {
                "index": _safe_int(step.get("index")),
                "unit_id": step.get("unit_id"),
                "source_sequence": sequence_id,
                "scene_frame_start": scene_frame_start,
                "scene_frame_end": scene_frame_end,
                "source_frame_start": int(source_frames[0]),
                "source_frame_end": int(source_frames[-1]),
                "sampled_frame_count": int(raw_vertices.shape[0]),
                "blend_in_frames": blend_in,
                "blend_out_frames": blend_out,
                "root_offset_xyz": [round(float(value), 6) for value in offset.tolist()],
            }
        )
        previous = {
            "step_index": _safe_int(step.get("index")),
            "scene_frame_end": scene_frame_end,
            "blend_out": blend_out,
            "raw_exit_root": raw_exit_root,
            "aligned_exit_root": aligned_joints[-1, 0].copy(),
            "aligned_vertices": aligned_vertices,
            "aligned_joints": aligned_joints,
        }

    _fill_missing_frames(stitched_vertices, stitched_joints, filled)
    if not bool(filled.all()):
        raise ValueError("unable to fill all stitched mesh frames")
    _smooth_transition_windows(
        vertices=stitched_vertices,
        joints=stitched_joints,
        transition_reports=transition_reports,
        scene_start=scene_start,
        smooth_frames=max(0, int(transition_smooth_frames)),
        passes=max(0, int(transition_smooth_passes)),
    )

    return StitchedMeshSequence(
        vertices=stitched_vertices,
        joints=stitched_joints,
        faces=faces,
        scene_frame_start=scene_start,
        scene_frame_end=scene_end,
        fps=fps,
        step_reports=step_reports,
        transition_reports=transition_reports,
        rhythm_lock_reports=_rhythm_lock_reports(manifest, scene_start, scene_end),
    )


def _display_vertices(vertices: np.ndarray) -> np.ndarray:
    displayed = vertices[..., [0, 2, 1]].copy()
    mins = displayed.min(axis=(0, 1))
    maxs = displayed.max(axis=(0, 1))
    center = (mins + maxs) * 0.5
    displayed[..., 0] -= center[0]
    displayed[..., 1] -= center[1]
    return displayed


def _bounds_for_preview(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mins = vertices.min(axis=(0, 1))
    maxs = vertices.max(axis=(0, 1))
    span = np.maximum(maxs - mins, 1e-3)
    pad = span * 0.1 + 0.03
    return mins - pad, maxs + pad


def _draw_mesh(ax: Any, vertices: np.ndarray, faces: np.ndarray, bounds_min: np.ndarray, bounds_max: np.ndarray, title: str) -> None:
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.view_init(elev=12, azim=-65)
    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    span = np.maximum(bounds_max - bounds_min, 1e-3)
    ax.set_box_aspect((float(span[0]), float(span[1]), float(span[2])))
    ax.set_title(title, fontsize=8)
    tris = vertices[faces]
    mesh = Poly3DCollection(tris, linewidths=0.015, alpha=1.0)
    mesh.set_facecolor((0.72, 0.78, 0.82, 1.0))
    mesh.set_edgecolor((0.10, 0.13, 0.15, 0.05))
    ax.add_collection3d(mesh)


def render_stitched_mesh_artifacts(
    stitched: StitchedMeshSequence,
    output_video: Path,
    output_strip: Path,
    face_stride: int = 12,
    render_frame_stride: int = 1,
    max_render_frames: int = 0,
) -> dict[str, Any]:
    mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/music_motion_lab_mplconfig"))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    import imageio.v2 as imageio
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_strip.parent.mkdir(parents=True, exist_ok=True)

    frame_stride = max(1, int(render_frame_stride))
    render_indices = np.arange(0, stitched.frame_count, frame_stride, dtype=np.int32)
    if max_render_frames > 0 and len(render_indices) > max_render_frames:
        render_indices = np.unique(np.linspace(0, stitched.frame_count - 1, num=max_render_frames, dtype=np.int32))
    if len(render_indices) == 0:
        render_indices = np.asarray([0], dtype=np.int32)

    sampled_vertices = _display_vertices(stitched.vertices[render_indices])
    sampled_faces = stitched.faces[:: max(1, int(face_stride))]
    bounds_min, bounds_max = _bounds_for_preview(sampled_vertices)

    keyframes = np.linspace(0, len(sampled_vertices) - 1, num=min(6, len(sampled_vertices)), dtype=np.int32)
    fig = plt.figure(figsize=(12, 7))
    for panel_index, sampled_index in enumerate(keyframes, start=1):
        ax = fig.add_subplot(2, 3, panel_index, projection="3d")
        scene_frame = stitched.scene_frame_start + int(render_indices[sampled_index])
        _draw_mesh(ax, sampled_vertices[sampled_index], sampled_faces, bounds_min, bounds_max, f"scene frame {scene_frame}")
    fig.suptitle("SMPL-X stitched mesh visual test")
    fig.tight_layout()
    fig.savefig(output_strip, dpi=150)
    plt.close(fig)

    video_fps = max(1, int(round(stitched.fps / float(frame_stride))))
    writer = imageio.get_writer(output_video, fps=video_fps)
    try:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for local_index, verts in enumerate(sampled_vertices):
            ax.cla()
            scene_frame = stitched.scene_frame_start + int(render_indices[local_index])
            _draw_mesh(ax, verts, sampled_faces, bounds_min, bounds_max, f"scene frame {scene_frame}")
            fig.tight_layout()
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            writer.append_data(frame)
        plt.close(fig)
    finally:
        writer.close()

    return {
        "video": str(output_video),
        "strip": str(output_strip),
        "rendered_frame_count": int(len(render_indices)),
        "render_frame_stride": int(frame_stride),
        "video_fps": int(video_fps),
        "face_stride": int(max(1, int(face_stride))),
        "sampled_face_count": int(len(sampled_faces)),
    }


def _max_metric(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return max(float(item.get(key, 0.0) or 0.0) for item in items)


def _timed_events(
    events: list[dict[str, Any]],
    start_time_sec: float,
    end_time_sec: float,
    fps: int,
    kind: str,
) -> list[dict[str, Any]]:
    mapped: list[dict[str, Any]] = []
    for item in events:
        time_sec = _safe_float(item.get("time_sec"), -1.0)
        if time_sec < start_time_sec - 1e-6 or time_sec > end_time_sec + 1e-6:
            continue
        mapped.append(
            {
                "kind": kind,
                "index": _safe_int(item.get("index"), len(mapped)),
                "source_beat_index": item.get("source_beat_index", item.get("beat_index", item.get("index"))),
                "time_sec": round(time_sec, 5),
                "local_time_sec": round(time_sec - start_time_sec, 5),
                "scene_frame": int(round(time_sec * fps)) + 1,
                "strength": round(_safe_float(item.get("strength")), 5),
                "level": item.get("level"),
                "band": item.get("band"),
                "is_downbeat": bool(item.get("is_downbeat", kind == "downbeat")),
            }
        )
    return mapped


def build_rhythm_mapping(manifest: dict[str, Any], song_event_map: dict[str, Any] | None) -> dict[str, Any]:
    steps = list(manifest.get("steps", []) or [])
    fps = _safe_int(manifest.get("fps"), 30)
    if steps:
        start_time_sec = min(_safe_float(step.get("start_time_sec")) for step in steps)
        end_time_sec = max(_safe_float(step.get("end_time_sec")) for step in steps)
    else:
        start_time_sec = 0.0
        end_time_sec = 0.0

    song = song_event_map or {}
    beats = _timed_events(list(song.get("beats", []) or []), start_time_sec, end_time_sec, fps, "beat")
    downbeats = _timed_events(list(song.get("downbeats", []) or []), start_time_sec, end_time_sec, fps, "downbeat")
    accents = _timed_events(list(song.get("accents", []) or []), start_time_sec, end_time_sec, fps, "accent")
    drum_source = list(song.get("drum_hits", []) or song.get("kick_hits", []) or song.get("accents", []) or song.get("downbeats", []) or [])
    drum_hits = _timed_events(drum_source, start_time_sec, end_time_sec, fps, "drum_hit")
    if not drum_hits:
        drum_hits = [dict(item, kind="drum_hit", strength=item.get("strength") or 1.0) for item in downbeats]
    step_maps: list[dict[str, Any]] = []
    for step in steps:
        step_start = _safe_float(step.get("start_time_sec"), start_time_sec)
        step_end = _safe_float(step.get("end_time_sec"), step_start)
        step_maps.append(
            {
                "index": _safe_int(step.get("index")),
                "unit_id": step.get("unit_id"),
                "source_sequence": step.get("source_sequence"),
                "section_label": step.get("section_label"),
                "target_time_sec": {"start": round(step_start, 5), "end": round(step_end, 5)},
                "local_time_sec": {"start": round(step_start - start_time_sec, 5), "end": round(step_end - start_time_sec, 5)},
                "scene_frames": {
                    "start": _safe_int(step.get("scene_frame_start")),
                    "end": _safe_int(step.get("scene_frame_end")),
                },
                "source_frames": {
                    "start": _safe_int(step.get("source_frame_start")),
                    "end_exclusive": _safe_int(step.get("source_frame_end_exclusive")),
                },
                "source_beat_range": dict(step.get("source_beat_range", {}) or {}),
                "speed_scale": _safe_float(step.get("speed_scale"), 1.0),
                "transition_score": _safe_float(step.get("transition_score")),
                "blend": {
                    "in_frames": _safe_int(step.get("blend_in_frames")),
                    "out_frames": _safe_int(step.get("blend_out_frames")),
                },
                "rhythm_locks": list(step.get("rhythm_locks", []) or []),
            }
        )

    return {
        "song_id": manifest.get("song_id") or song.get("song_id"),
        "source_audio_path": song.get("source_audio_path"),
        "beats_per_bar": song.get("beats_per_bar"),
        "duration_sec": song.get("duration_sec"),
        "preview_window_sec": {"start": round(start_time_sec, 5), "end": round(end_time_sec, 5)},
        "beats": beats,
        "downbeats": downbeats,
        "accents": accents,
        "drum_hits": drum_hits,
        "steps": step_maps,
        "summary": {
            "beat_count": len(beats),
            "downbeat_count": len(downbeats),
            "accent_count": len(accents),
            "drum_hit_count": len(drum_hits),
            "step_count": len(step_maps),
        },
    }


def build_mesh_stitch_report(
    manifest: dict[str, Any],
    stitched: StitchedMeshSequence,
    cache_summaries: list[dict[str, Any]],
    render_summary: dict[str, Any],
    output_report: Path,
    output_html: Path,
    song_event_map: dict[str, Any] | None = None,
) -> dict[str, Any]:
    transition_reports = stitched.transition_reports
    rhythm_lock_reports = stitched.rhythm_lock_reports
    rhythm_mapping = build_rhythm_mapping(manifest, song_event_map)
    report = {
        "schema_version": 1,
        "report_id": f"{slugify(str(manifest.get('manifest_id', 'smplx_stitch')))}_mesh_visual_report",
        "manifest_id": manifest.get("manifest_id"),
        "plan_id": manifest.get("plan_id"),
        "song_id": manifest.get("song_id"),
        "generated_at_utc": utc_now_iso(),
        "fps": int(stitched.fps),
        "scene": {
            "frame_start": int(stitched.scene_frame_start),
            "frame_end": int(stitched.scene_frame_end),
            "frame_count": int(stitched.frame_count),
        },
        "mesh": {
            "vertex_count": int(stitched.vertices.shape[1]),
            "joint_count": int(stitched.joints.shape[1]),
            "face_count": int(stitched.faces.shape[0]),
        },
        "render": render_summary,
        "artifacts": {
            "video": render_summary["video"],
            "strip": render_summary["strip"],
            "html": str(output_html),
            "report": str(output_report),
        },
        "cache": cache_summaries,
        "steps": stitched.step_reports,
        "rhythm_mapping": rhythm_mapping,
        "segment_mapping": rhythm_mapping["steps"],
        "transitions": transition_reports,
        "rhythm_locks": rhythm_lock_reports,
        "streaming": dict(manifest.get("streaming_review", {}) or {}),
        "metrics": {
            "transition_count": int(len(transition_reports)),
            "beat_count": int(rhythm_mapping["summary"]["beat_count"]),
            "downbeat_count": int(rhythm_mapping["summary"]["downbeat_count"]),
            "accent_count": int(rhythm_mapping["summary"]["accent_count"]),
            "drum_hit_count": int(rhythm_mapping["summary"]["drum_hit_count"]),
            "max_joint_delta_after_blend": round(_max_metric(transition_reports, "joint_delta_after_blend"), 6),
            "max_vertex_delta_after_blend": round(_max_metric(transition_reports, "vertex_delta_after_blend"), 6),
            "max_temporal_joint_delta_after_smoothing": round(_max_metric(transition_reports, "max_temporal_joint_delta_after_smoothing"), 6),
            "max_temporal_vertex_delta_after_smoothing": round(_max_metric(transition_reports, "max_temporal_vertex_delta_after_smoothing"), 6),
            "max_root_acceleration_discontinuity_proxy": round(_max_metric(transition_reports, "root_acceleration_discontinuity_proxy"), 6),
            "max_joint_jerk_proxy": round(_max_metric(transition_reports, "joint_jerk_proxy"), 6),
            "max_foot_slide_proxy": round(_max_metric(transition_reports, "foot_slide_proxy"), 6),
            "max_rhythm_lock_frame_error": int(_max_metric(rhythm_lock_reports, "frame_error")),
            "cache_miss_count": sum(1 for item in cache_summaries if str(item.get("status")) != "hit"),
        },
        "notes": [
            "This report renders real FineDance source SMPL-X vertices into a stitched visual preview.",
            "Boundary smoothing is a visual vertex/joint crossfade for review; it is not yet a pose-space reusable motion blend.",
        ],
    }
    write_json(output_report, report)
    return report


def build_mesh_stitch_review_html(report: dict[str, Any], video_href: str, strip_href: str, audio_href: str | None = None) -> str:
    title = html.escape(str(report.get("report_id", "SMPL-X mesh stitch review")))
    transitions = list(report.get("transitions", []) or [])
    locks = list(report.get("rhythm_locks", []) or [])
    segment_mapping = list(report.get("segment_mapping", []) or [])
    rhythm_mapping = dict(report.get("rhythm_mapping", {}) or {})
    streaming = dict(report.get("streaming", {}) or {})
    streaming_decisions = list(streaming.get("decisions", []) or [])
    rhythm_payload = json.dumps(
        {
            "previewWindow": rhythm_mapping.get("preview_window_sec", {}),
            "beats": rhythm_mapping.get("beats", []),
            "downbeats": rhythm_mapping.get("downbeats", []),
            "accents": rhythm_mapping.get("accents", []),
            "drumHits": rhythm_mapping.get("drum_hits", []),
            "segments": segment_mapping,
            "transitions": transitions,
        },
        ensure_ascii=False,
    ).replace("</", "<\\/")

    def transition_rows() -> str:
        if not transitions:
            return "<tr><td colspan=\"9\">No transitions.</td></tr>"
        rows = []
        for item in transitions:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('outgoing_step')))} -> {html.escape(str(item.get('incoming_step')))}</td>"
                f"<td>{html.escape(str(item.get('boundary_frame')))}</td>"
                f"<td>{html.escape(str(item.get('gap_frames')))}</td>"
                f"<td>{html.escape(str(item.get('blend_frames')))}</td>"
                f"<td>{html.escape(str(item.get('smoothing_frames', 0)))}</td>"
                f"<td>{html.escape(str(item.get('raw_root_xz_delta')))}</td>"
                f"<td>{html.escape(str(item.get('aligned_root_xz_delta_before_blend')))}</td>"
                f"<td>{html.escape(str(item.get('joint_delta_after_blend')))}</td>"
                f"<td>{html.escape(str(item.get('vertex_delta_after_blend')))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def streaming_rows() -> str:
        if not streaming_decisions:
            return "<tr><td colspan=\"13\">No streaming decision records.</td></tr>"
        rows = []
        for item in streaming_decisions:
            target = dict(item.get("target_time_sec", {}) or {})
            guard = dict(item.get("future_visibility_guard", {}) or {})
            score = dict(item.get("score_breakdown", {}) or {})
            source_frames = dict(item.get("source_frame_range", {}) or {})
            rejected = list(item.get("rejected_top_candidates", []) or [])
            reject_summary = " | ".join(
                f"{entry.get('source_sequence')}:{entry.get('unit_id')} => {','.join(list(entry.get('reasons', []) or []))}"
                for entry in rejected[:2]
            ) or "-"
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('index')))}</td>"
                f"<td>{html.escape(str(item.get('decision_time_sec')))}</td>"
                f"<td>{html.escape(str(item.get('playhead_sec')))}</td>"
                f"<td>{html.escape(str(item.get('available_audio_until_sec')))}</td>"
                f"<td>{html.escape(str(target.get('start')))}-{html.escape(str(target.get('end')))}s</td>"
                f"<td>{html.escape(str(item.get('selected_unit_id')))}</td>"
                f"<td>{html.escape(str(item.get('source_sequence')))}:{html.escape(str(source_frames.get('start')))}-{html.escape(str(source_frames.get('end_exclusive')))}</td>"
                f"<td>{html.escape(str(item.get('selected_from_tier')))}</td>"
                f"<td>{html.escape(','.join(list(item.get('cohort_source_sequences', []) or [])[:8]))}</td>"
                f"<td>{html.escape(str(item.get('target_energy')))} / {html.escape(str(item.get('target_bpm')))}</td>"
                f"<td>{html.escape(str(item.get('speed_scale')))} / {html.escape(str(item.get('score')))}</td>"
                f"<td>{html.escape(str(guard.get('passed')))} r={html.escape(str(score.get('rhythm_lock')))} t={html.escape(str(score.get('transition_smoothness')))}</td>"
                f"<td>{html.escape(reject_summary)}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def lock_rows() -> str:
        if not locks:
            return "<tr><td colspan=\"4\">No explicit step rhythm locks. Use beat/downbeat/accent map above for this smoke pass.</td></tr>"
        rows = []
        for item in locks:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('step_index')))}</td>"
                f"<td>{html.escape(str(item.get('kind')))}</td>"
                f"<td>{html.escape(str(item.get('target_scene_frame')))}</td>"
                f"<td>{html.escape(str(item.get('frame_error')))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def segment_rows() -> str:
        if not segment_mapping:
            return "<tr><td colspan=\"8\">No segments.</td></tr>"
        rows = []
        for item in segment_mapping:
            target = dict(item.get("target_time_sec", {}) or {})
            source_frames = dict(item.get("source_frames", {}) or {})
            source_beats = dict(item.get("source_beat_range", {}) or {})
            scene_frames = dict(item.get("scene_frames", {}) or {})
            blend = dict(item.get("blend", {}) or {})
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('index')))}</td>"
                f"<td>{html.escape(str(item.get('unit_id')))}</td>"
                f"<td>{html.escape(str(item.get('source_sequence')))}</td>"
                f"<td>{html.escape(str(source_beats.get('start')))}-{html.escape(str(source_beats.get('end_exclusive')))}</td>"
                f"<td>{html.escape(str(source_frames.get('start')))}-{html.escape(str(source_frames.get('end_exclusive')))}</td>"
                f"<td>{html.escape(str(target.get('start')))}-{html.escape(str(target.get('end')))}s</td>"
                f"<td>{html.escape(str(scene_frames.get('start')))}-{html.escape(str(scene_frames.get('end')))}</td>"
                f"<td>{html.escape(str(blend.get('in_frames')))} / {html.escape(str(blend.get('out_frames')))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    scene = dict(report.get("scene", {}))
    mesh = dict(report.get("mesh", {}))
    metrics = dict(report.get("metrics", {}))
    audio_html = (
        f'<audio id="audio" controls preload="metadata" src="{html.escape(audio_href)}"></audio>'
        if audio_href
        else '<span class="missing-audio">No audio linked</span>'
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #101214;
      --panel: #181d21;
      --panel2: #20272c;
      --text: #f3f1ea;
      --muted: #aeb8ba;
      --line: rgba(255,255,255,0.14);
      --blue: #74b9ff;
      --gold: #f4c95d;
      --green: #7bd88f;
      --coral: #ff7b63;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      letter-spacing: 0;
    }}
    .shell {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 380px;
      min-height: 100vh;
    }}
    main {{
      padding: 18px;
      min-width: 0;
    }}
    aside {{
      background: var(--panel);
      border-left: 1px solid var(--line);
      padding: 18px;
      min-width: 0;
    }}
    h1 {{ font-size: 22px; line-height: 1.15; margin: 0 0 14px; }}
    h2 {{ font-size: 15px; margin: 20px 0 8px; }}
    video, img {{
      display: block;
      width: 100%;
      max-height: 72vh;
      background: #050607;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    img {{ margin-top: 14px; max-height: none; }}
    .media-bar {{
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 10px;
      align-items: center;
      margin: 10px 0 12px;
    }}
    button {{
      border: 1px solid rgba(255,255,255,0.18);
      background: #f3f1ea;
      color: #101214;
      border-radius: 6px;
      height: 34px;
      padding: 0 13px;
      font-weight: 800;
      cursor: pointer;
    }}
    audio {{ width: 100%; height: 34px; }}
    .missing-audio {{ color: var(--coral); }}
    #rhythmCanvas {{
      display: block;
      width: 100%;
      height: 132px;
      margin: 12px 0 14px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #11161a;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--panel2);
      border-radius: 6px;
      padding: 10px;
      min-width: 0;
    }}
    .metric b {{ display: block; font-size: 18px; overflow-wrap: anywhere; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      border: 1px solid var(--line);
      background: #14181c;
      margin-top: 8px;
      table-layout: fixed;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px;
      text-align: left;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    th {{ color: var(--gold); font-weight: 700; }}
    .path {{
      color: var(--muted);
      overflow-wrap: anywhere;
      font-size: 12px;
      margin-top: 8px;
    }}
    @media (max-width: 980px) {{
      .shell {{ grid-template-columns: 1fr; }}
      aside {{ border-left: 0; border-top: 1px solid var(--line); }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <main>
      <h1>{title}</h1>
      <div class="media-bar">
        <button id="syncPlay" type="button">Play Sync</button>
        {audio_html}
      </div>
      <video controls src="{html.escape(video_href)}"></video>
      <canvas id="rhythmCanvas" width="1280" height="132"></canvas>
      <img src="{html.escape(strip_href)}" alt="SMPL-X stitched mesh strip">
    </main>
    <aside>
      <div class="grid">
        <div class="metric"><b>{html.escape(str(scene.get('frame_count')))}</b><span>stitched frames</span></div>
        <div class="metric"><b>{html.escape(str(report.get('fps')))}</b><span>source fps</span></div>
        <div class="metric"><b>{html.escape(str(mesh.get('vertex_count')))}</b><span>vertices</span></div>
        <div class="metric"><b>{html.escape(str(mesh.get('face_count')))}</b><span>faces</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('beat_count')))}</b><span>beats in preview</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('downbeat_count')))}</b><span>downbeats</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('drum_hit_count')))}</b><span>drum hits mapped</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('accent_count')))}</b><span>accents in preview</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('max_vertex_delta_after_blend')))}</b><span>max vertex delta after blend</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('max_temporal_vertex_delta_after_smoothing')))}</b><span>max temporal vertex delta after smoothing</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('max_rhythm_lock_frame_error')))}</b><span>max rhythm frame error</span></div>
      </div>
      <h2>Segments</h2>
      <table>
        <thead><tr><th>step</th><th>unit</th><th>seq</th><th>src beats</th><th>src frames</th><th>target time</th><th>scene frames</th><th>blend in/out</th></tr></thead>
        <tbody>{segment_rows()}</tbody>
      </table>
      <h2>Streaming Decisions</h2>
      <table>
        <thead><tr><th>step</th><th>decision</th><th>playhead</th><th>available</th><th>target</th><th>unit</th><th>source</th><th>tier</th><th>cohort songs</th><th>energy/bpm</th><th>speed/score</th><th>guard/scores</th><th>top rejects</th></tr></thead>
        <tbody>{streaming_rows()}</tbody>
      </table>
      <h2>Transitions</h2>
      <table>
        <thead><tr><th>steps</th><th>boundary</th><th>gap</th><th>blend</th><th>smooth</th><th>raw root</th><th>aligned root</th><th>joint after</th><th>vertex after</th></tr></thead>
        <tbody>{transition_rows()}</tbody>
      </table>
      <h2>Rhythm Locks</h2>
      <table>
        <thead><tr><th>step</th><th>kind</th><th>target frame</th><th>error</th></tr></thead>
        <tbody>{lock_rows()}</tbody>
      </table>
      <p class="path">{html.escape(str(report.get('artifacts', {}).get('report', '')))}</p>
    </aside>
  </div>
  <script id="rhythmData" type="application/json">{rhythm_payload}</script>
  <script>
    const rhythmData = JSON.parse(document.getElementById("rhythmData").textContent);
    const video = document.querySelector("video");
    const audio = document.getElementById("audio");
    const syncPlay = document.getElementById("syncPlay");
    const canvas = document.getElementById("rhythmCanvas");
    const ctx = canvas.getContext("2d");
    const preview = rhythmData.previewWindow || {{}};
    const start = Number(preview.start || 0);
    const end = Math.max(start + 0.001, Number(preview.end || start + 1));
    function xFor(time) {{
      return Math.max(0, Math.min(canvas.width, ((Number(time) - start) / (end - start)) * canvas.width));
    }}
    function draw() {{
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = "#11161a";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const segment of rhythmData.segments || []) {{
        const local = segment.target_time_sec || {{}};
        const x0 = xFor(local.start);
        const x1 = xFor(local.end);
        ctx.fillStyle = Number(segment.index || 0) % 2 ? "rgba(116,185,255,0.26)" : "rgba(123,216,143,0.22)";
        ctx.fillRect(x0, 14, Math.max(2, x1 - x0), 34);
        ctx.fillStyle = "#f3f1ea";
        ctx.font = "12px system-ui, sans-serif";
        ctx.fillText("step " + segment.index, x0 + 6, 35);
      }}
      for (const beat of rhythmData.beats || []) {{
        const x = xFor(beat.time_sec);
        ctx.strokeStyle = beat.is_downbeat ? "#f4c95d" : "rgba(255,255,255,0.35)";
        ctx.lineWidth = beat.is_downbeat ? 2 : 1;
        ctx.beginPath();
        ctx.moveTo(x, 58);
        ctx.lineTo(x, 100);
        ctx.stroke();
      }}
      for (const downbeat of rhythmData.downbeats || []) {{
        const x = xFor(downbeat.time_sec);
        ctx.fillStyle = "#f4c95d";
        ctx.fillRect(x - 2, 56, 4, 48);
      }}
      for (const accent of rhythmData.accents || []) {{
        const x = xFor(accent.time_sec);
        const height = 10 + Math.min(28, Number(accent.strength || 0) * 48);
        ctx.fillStyle = "#ff7b63";
        ctx.fillRect(x - 1.5, 108 - height, 3, height);
      }}
      for (const hit of rhythmData.drumHits || []) {{
        const x = xFor(hit.time_sec);
        ctx.fillStyle = "#7bd88f";
        ctx.beginPath();
        ctx.arc(x, 112, 4, 0, Math.PI * 2);
        ctx.fill();
      }}
      for (const transition of rhythmData.transitions || []) {{
        const segment = (rhythmData.segments || []).find((item) => Number(item.index) === Number(transition.incoming_step));
        if (!segment) continue;
        const x = xFor((segment.target_time_sec || {{}}).start);
        ctx.strokeStyle = "#bba7ff";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x, 8);
        ctx.lineTo(x, 122);
        ctx.stroke();
      }}
      const playhead = start + (video.currentTime || 0);
      const px = xFor(playhead);
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, canvas.height);
      ctx.stroke();
      ctx.fillStyle = "#aeb8ba";
      ctx.font = "12px system-ui, sans-serif";
      ctx.fillText("segments", 8, 12);
      ctx.fillText("beats/downbeats", 8, 72);
      ctx.fillText("drum hits / accents", 8, 120);
    }}
    syncPlay?.addEventListener("click", async () => {{
      if (audio) audio.currentTime = start + video.currentTime;
      await video.play();
      if (audio) await audio.play();
    }});
    video.addEventListener("play", () => {{
      if (audio && Math.abs(audio.currentTime - (start + video.currentTime)) > 0.2) audio.currentTime = start + video.currentTime;
      if (audio) audio.play();
    }});
    video.addEventListener("pause", () => audio?.pause());
    video.addEventListener("seeked", () => {{
      if (audio) audio.currentTime = start + video.currentTime;
      draw();
    }});
    function tick() {{
      if (audio && !video.paused && Math.abs(audio.currentTime - (start + video.currentTime)) > 0.35) {{
        audio.currentTime = start + video.currentTime;
      }}
      draw();
      requestAnimationFrame(tick);
    }}
    tick();
  </script>
</body>
</html>
"""


def build_smplx_mesh_stitch_visual_preview(
    manifest: dict[str, Any],
    project_root: Path,
    motion_base_assets_root: Path,
    output_video: Path,
    output_strip: Path,
    output_report: Path,
    output_html: Path,
    song_event_map: dict[str, Any] | None = None,
    audio_path: Path | None = None,
    force_cache: bool = False,
    batch_size: int = 128,
    face_stride: int = 12,
    render_frame_stride: int = 1,
    max_render_frames: int = 0,
    transition_smooth_frames: int = 12,
    transition_smooth_passes: int = 2,
) -> dict[str, Any]:
    caches, cache_summaries = materialize_mesh_caches(
        manifest=manifest,
        project_root=project_root,
        motion_base_assets_root=motion_base_assets_root,
        force=force_cache,
        batch_size=batch_size,
    )
    stitched = compose_stitched_mesh_sequence(
        manifest,
        caches,
        transition_smooth_frames=transition_smooth_frames,
        transition_smooth_passes=transition_smooth_passes,
    )
    render_summary = render_stitched_mesh_artifacts(
        stitched=stitched,
        output_video=output_video,
        output_strip=output_strip,
        face_stride=face_stride,
        render_frame_stride=render_frame_stride,
        max_render_frames=max_render_frames,
    )
    report = build_mesh_stitch_report(
        manifest=manifest,
        stitched=stitched,
        cache_summaries=cache_summaries,
        render_summary=render_summary,
        output_report=output_report,
        output_html=output_html,
        song_event_map=song_event_map,
    )
    video_href = os.path.relpath(output_video.resolve(), output_html.parent.resolve())
    strip_href = os.path.relpath(output_strip.resolve(), output_html.parent.resolve())
    audio_href = os.path.relpath(audio_path.resolve(), output_html.parent.resolve()) if audio_path is not None else None
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(
        build_mesh_stitch_review_html(report, video_href=video_href, strip_href=strip_href, audio_href=audio_href),
        encoding="utf-8",
    )
    return report
