from __future__ import annotations

import html
import importlib.util
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


def compose_stitched_mesh_sequence(manifest: dict[str, Any], caches_by_sequence: dict[str, MeshCache]) -> StitchedMeshSequence:
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
            offset[0] = previous_exit_root[0] - raw_entry_root[0]
            offset[2] = previous_exit_root[2] - raw_entry_root[2]

        aligned_vertices, aligned_joints = _offset_mesh(raw_vertices, raw_joints, offset)
        pre_blend_vertices_0 = aligned_vertices[0].copy()
        pre_blend_joints_0 = aligned_joints[0].copy()
        blend_in = max(0, _safe_int(step.get("blend_in_frames")))
        blend_out = max(0, _safe_int(step.get("blend_out_frames")))

        if previous is not None:
            previous_vertices = previous["aligned_vertices"]
            previous_joints = previous["aligned_joints"]
            blend_count = min(blend_in, _safe_int(previous["blend_out"]), len(previous_vertices), len(aligned_vertices))
            for local_index in range(blend_count):
                alpha = _smoothstep((local_index + 1) / float(blend_count + 1))
                previous_index = max(0, len(previous_vertices) - blend_count + local_index)
                aligned_vertices[local_index] = previous_vertices[previous_index] * (1.0 - alpha) + aligned_vertices[local_index] * alpha
                aligned_joints[local_index] = previous_joints[previous_index] * (1.0 - alpha) + aligned_joints[local_index] * alpha

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
                    "boundary_frame": scene_frame_start,
                    "gap_frames": max(0, scene_frame_start - int(previous["scene_frame_end"]) - 1),
                    "blend_frames": int(blend_count),
                    "root_offset_xyz": [round(float(value), 6) for value in offset.tolist()],
                    "raw_root_xz_delta": round(_root_xz_delta(previous["raw_exit_root"], raw_entry_root), 6),
                    "aligned_root_xz_delta_before_blend": round(
                        _root_xz_delta(previous["aligned_exit_root"], pre_blend_joints_0[0]),
                        6,
                    ),
                    "joint_delta_before_blend": round(_mean_vertex_delta(previous["aligned_joints"][-1], pre_blend_joints_0), 6),
                    "joint_delta_after_blend": round(_mean_vertex_delta(previous["aligned_joints"][-1], aligned_joints[0]), 6),
                    "vertex_delta_before_blend": round(_mean_vertex_delta(previous["aligned_vertices"][-1], pre_blend_vertices_0), 6),
                    "vertex_delta_after_blend": round(_mean_vertex_delta(previous["aligned_vertices"][-1], aligned_vertices[0]), 6),
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
    render_frame_stride: int = 2,
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


def build_mesh_stitch_report(
    manifest: dict[str, Any],
    stitched: StitchedMeshSequence,
    cache_summaries: list[dict[str, Any]],
    render_summary: dict[str, Any],
    output_report: Path,
    output_html: Path,
) -> dict[str, Any]:
    transition_reports = stitched.transition_reports
    rhythm_lock_reports = stitched.rhythm_lock_reports
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
        "transitions": transition_reports,
        "rhythm_locks": rhythm_lock_reports,
        "metrics": {
            "transition_count": int(len(transition_reports)),
            "max_joint_delta_after_blend": round(_max_metric(transition_reports, "joint_delta_after_blend"), 6),
            "max_vertex_delta_after_blend": round(_max_metric(transition_reports, "vertex_delta_after_blend"), 6),
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


def build_mesh_stitch_review_html(report: dict[str, Any], video_href: str, strip_href: str) -> str:
    title = html.escape(str(report.get("report_id", "SMPL-X mesh stitch review")))
    transitions = list(report.get("transitions", []) or [])
    locks = list(report.get("rhythm_locks", []) or [])

    def transition_rows() -> str:
        if not transitions:
            return "<tr><td colspan=\"8\">No transitions.</td></tr>"
        rows = []
        for item in transitions:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(item.get('outgoing_step')))} -> {html.escape(str(item.get('incoming_step')))}</td>"
                f"<td>{html.escape(str(item.get('boundary_frame')))}</td>"
                f"<td>{html.escape(str(item.get('gap_frames')))}</td>"
                f"<td>{html.escape(str(item.get('blend_frames')))}</td>"
                f"<td>{html.escape(str(item.get('raw_root_xz_delta')))}</td>"
                f"<td>{html.escape(str(item.get('aligned_root_xz_delta_before_blend')))}</td>"
                f"<td>{html.escape(str(item.get('joint_delta_after_blend')))}</td>"
                f"<td>{html.escape(str(item.get('vertex_delta_after_blend')))}</td>"
                "</tr>"
            )
        return "\n".join(rows)

    def lock_rows() -> str:
        if not locks:
            return "<tr><td colspan=\"4\">No rhythm locks.</td></tr>"
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

    scene = dict(report.get("scene", {}))
    mesh = dict(report.get("mesh", {}))
    metrics = dict(report.get("metrics", {}))
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
      <video controls src="{html.escape(video_href)}"></video>
      <img src="{html.escape(strip_href)}" alt="SMPL-X stitched mesh strip">
    </main>
    <aside>
      <div class="grid">
        <div class="metric"><b>{html.escape(str(scene.get('frame_count')))}</b><span>stitched frames</span></div>
        <div class="metric"><b>{html.escape(str(report.get('fps')))}</b><span>source fps</span></div>
        <div class="metric"><b>{html.escape(str(mesh.get('vertex_count')))}</b><span>vertices</span></div>
        <div class="metric"><b>{html.escape(str(mesh.get('face_count')))}</b><span>faces</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('max_vertex_delta_after_blend')))}</b><span>max vertex delta after blend</span></div>
        <div class="metric"><b>{html.escape(str(metrics.get('max_rhythm_lock_frame_error')))}</b><span>max rhythm frame error</span></div>
      </div>
      <h2>Transitions</h2>
      <table>
        <thead><tr><th>steps</th><th>boundary</th><th>gap</th><th>blend</th><th>raw root</th><th>aligned root</th><th>joint after</th><th>vertex after</th></tr></thead>
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
    force_cache: bool = False,
    batch_size: int = 128,
    face_stride: int = 12,
    render_frame_stride: int = 2,
    max_render_frames: int = 0,
) -> dict[str, Any]:
    caches, cache_summaries = materialize_mesh_caches(
        manifest=manifest,
        project_root=project_root,
        motion_base_assets_root=motion_base_assets_root,
        force=force_cache,
        batch_size=batch_size,
    )
    stitched = compose_stitched_mesh_sequence(manifest, caches)
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
    )
    video_href = os.path.relpath(output_video.resolve(), output_html.parent.resolve())
    strip_href = os.path.relpath(output_strip.resolve(), output_html.parent.resolve())
    output_html.parent.mkdir(parents=True, exist_ok=True)
    output_html.write_text(build_mesh_stitch_review_html(report, video_href=video_href, strip_href=strip_href), encoding="utf-8")
    return report
