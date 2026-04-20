from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..utils import slugify, utc_now_iso


def _safe_float(value: Any, fallback: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _safe_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _target_window(step: dict[str, Any], fps: int) -> dict[str, Any]:
    target_time = dict(step.get("target_time_sec", {}))
    start_time = _safe_float(target_time.get("start"))
    end_time = max(start_time + 1e-3, _safe_float(target_time.get("end"), start_time + 1.0))
    scene_frame_start = int(round(start_time * fps)) + 1
    scene_frame_end = max(scene_frame_start + 1, int(round(end_time * fps)))
    return {
        "start_time_sec": round(start_time, 5),
        "end_time_sec": round(end_time, 5),
        "scene_frame_start": scene_frame_start,
        "scene_frame_end": scene_frame_end,
    }


def _rhythm_locks(step: dict[str, Any], fps: int) -> list[dict[str, Any]]:
    start_beat = _safe_float(step.get("start_beat"))
    locks: list[dict[str, Any]] = []
    for index, time_sec in enumerate(step.get("expected_accent_hits", []) or []):
        absolute_time = _safe_float(time_sec)
        locks.append(
            {
                "index": index,
                "kind": "expected_accent",
                "time_sec": round(absolute_time, 5),
                "scene_frame": int(round(absolute_time * fps)) + 1,
                "beat_offset_estimate": round(max(0.0, absolute_time - _safe_float(step.get("target_time_sec", {}).get("start"))) * 2.0, 5),
                "source_start_beat": start_beat,
            }
        )
    return locks


def _unit_lookup(motion_library: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(unit["unit_id"]): dict(unit) for unit in motion_library.get("units", [])}


def _cache_request(plan_id: str, sequence_id: str, source_motion_path: str, frame_ranges: list[dict[str, int]]) -> dict[str, Any]:
    if frame_ranges:
        start_frame = min(item["start"] for item in frame_ranges)
        end_frame = max(item["end_exclusive"] for item in frame_ranges)
    else:
        start_frame = 0
        end_frame = 0
    cache_name = f"{slugify(sequence_id)}_{start_frame:05d}_{max(start_frame, end_frame - 1):05d}.npz"
    return {
        "sequence_id": sequence_id,
        "source_motion_path": source_motion_path,
        "frame_ranges": frame_ranges,
        "cache_path": f"outputs/smplx_mesh_previews/cache/{slugify(plan_id)}/{cache_name}",
        "cache_format": "npz: vertices float32 [frames, verts, 3], joints float32 [frames, joints, 3], faces int32 [faces, 3]",
    }


def build_smplx_stitch_preview_manifest(
    plan: dict[str, Any],
    motion_library: dict[str, Any],
    song_event_map: dict[str, Any] | None = None,
    fps: int = 30,
    blend_frames: int = 6,
    max_steps: int = 0,
) -> dict[str, Any]:
    units_by_id = _unit_lookup(motion_library)
    selected_steps = list(plan.get("steps", []))
    if max_steps > 0:
        selected_steps = selected_steps[:max_steps]

    manifest_steps: list[dict[str, Any]] = []
    ranges_by_sequence: dict[str, list[dict[str, int]]] = defaultdict(list)
    source_path_by_sequence: dict[str, str] = {}
    previous_step: dict[str, Any] | None = None
    transitions: list[dict[str, Any]] = []

    for plan_step in selected_steps:
        step_index = _safe_int(plan_step.get("index"), len(manifest_steps))
        unit_id = str(plan_step["selected_unit_id"])
        unit = units_by_id.get(unit_id)
        if unit is None:
            raise KeyError(f"plan step {step_index} references missing unit: {unit_id}")

        frame_range = dict(plan_step.get("source_frame_range") or unit.get("frame_range") or {})
        source_frame_start = _safe_int(frame_range.get("start"))
        source_frame_end = max(source_frame_start + 1, _safe_int(frame_range.get("end_exclusive"), source_frame_start + 1))
        target = _target_window(plan_step, fps=fps)
        source_sequence = str(plan_step.get("source_sequence") or unit.get("source_sequence") or "unknown")
        reference_artifacts = dict(unit.get("reference_artifacts", {}))
        reference_artifacts.update(dict(plan_step.get("reference_artifacts", {})))
        source_motion_path = str(reference_artifacts.get("source_motion_path", "") or "")
        if not source_motion_path:
            raise FileNotFoundError(f"source_motion_path missing for plan step {step_index}")

        ranges_by_sequence[source_sequence].append({"start": source_frame_start, "end_exclusive": source_frame_end})
        source_path_by_sequence[source_sequence] = source_motion_path
        blend_in = min(blend_frames if previous_step is not None else 0, max(0, (target["scene_frame_end"] - target["scene_frame_start"]) // 3))
        blend_out = min(blend_frames if step_index < len(selected_steps) - 1 else 0, max(0, (target["scene_frame_end"] - target["scene_frame_start"]) // 3))
        manifest_step = {
            "index": step_index,
            "unit_id": unit_id,
            "source_sequence": source_sequence,
            "source_motion_path": source_motion_path,
            "source_frame_start": source_frame_start,
            "source_frame_end_exclusive": source_frame_end,
            "source_beat_range": dict(plan_step.get("source_beat_range") or unit.get("beat_range") or {}),
            "scene_frame_start": target["scene_frame_start"],
            "scene_frame_end": target["scene_frame_end"],
            "start_time_sec": target["start_time_sec"],
            "end_time_sec": target["end_time_sec"],
            "speed_scale": _safe_float(plan_step.get("speed_scale"), 1.0),
            "blend_in_frames": blend_in,
            "blend_out_frames": blend_out,
            "rhythm_locks": _rhythm_locks(plan_step, fps=fps),
            "root_alignment": {
                "mode": "pelvis_yaw_translation_continuity",
                "preserve_beat_locked_frames": True,
                "entry_anchor": dict(unit.get("entry_anchor", {})),
                "exit_anchor": dict(unit.get("exit_anchor", {})),
            },
            "transition_score": _safe_float(plan_step.get("transition_score")),
            "section_label": plan_step.get("section_label"),
        }
        if previous_step is not None:
            transitions.append(
                {
                    "index": len(transitions),
                    "outgoing_step": previous_step["index"],
                    "incoming_step": step_index,
                    "boundary_frame": manifest_step["scene_frame_start"],
                    "blend_frames": min(blend_frames, manifest_step["blend_in_frames"], previous_step["blend_out_frames"]),
                    "incoming_transition_score": manifest_step["transition_score"],
                    "mode": "root_continuity_plus_slerp",
                }
            )
        manifest_steps.append(manifest_step)
        previous_step = manifest_step

    cache_requests = [
        _cache_request(
            plan_id=str(plan.get("plan_id", "smplx_stitch_plan")),
            sequence_id=sequence_id,
            source_motion_path=source_path_by_sequence[sequence_id],
            frame_ranges=frame_ranges,
        )
        for sequence_id, frame_ranges in sorted(ranges_by_sequence.items())
    ]

    scene_frame_end = max((step["scene_frame_end"] for step in manifest_steps), default=1)
    return {
        "schema_version": 1,
        "manifest_id": f"{plan.get('plan_id', 'smplx_stitch_plan')}_smplx_stitch_preview",
        "plan_id": plan.get("plan_id"),
        "song_id": plan.get("song_id") or (song_event_map or {}).get("song_id"),
        "library_id": plan.get("library_id") or motion_library.get("library_id"),
        "generated_at_utc": utc_now_iso(),
        "fps": int(fps),
        "mesh_backend": "smplx_neutral",
        "blend_policy": {
            "default_blend_frames": int(blend_frames),
            "rotation": "slerp",
            "root_translation": "continuous_offset",
            "root_yaw": "shortest_arc_offset",
            "beat_locks": "preserve_target_scene_frames",
        },
        "scene": {"frame_start": 1, "frame_end": scene_frame_end},
        "steps": manifest_steps,
        "transitions": transitions,
        "cache_requests": cache_requests,
        "notes": [
            "This is a lightweight SMPL-X source-mesh stitch manifest.",
            "It does not embed vertices; a renderer/cache builder should read cache_requests and generate mesh arrays from raw FineDance motion.",
        ],
    }
