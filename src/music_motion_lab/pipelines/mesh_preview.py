from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from ..utils import utc_now_iso


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _step_duration_sec(step: dict[str, Any], song_event_map: dict[str, Any]) -> tuple[float, float]:
    beats = list(song_event_map.get("beats", []))
    if not beats:
        return (0.0, 0.0)

    def beat_time(beat_index: float) -> float:
        if beat_index <= 0:
            return float(beats[0]["time_sec"])
        left_index = int(beat_index // 1)
        right_index = int(-(-beat_index // 1))
        if right_index >= len(beats):
            if len(beats) == 1:
                return float(beats[0]["time_sec"])
            interval = float(beats[-1]["time_sec"]) - float(beats[-2]["time_sec"])
            return float(beats[-1]["time_sec"]) + interval * float(beat_index - (len(beats) - 1))
        if left_index == right_index:
            return float(beats[left_index]["time_sec"])
        left_time = float(beats[left_index]["time_sec"])
        right_time = float(beats[right_index]["time_sec"])
        mix = beat_index - left_index
        return _lerp(left_time, right_time, mix)

    start_beat = float(step.get("start_beat", 0.0) or 0.0)
    target_beats = float(step.get("target_beats", 0.0) or 0.0)
    start_time = beat_time(start_beat)
    end_time = beat_time(start_beat + target_beats)
    return start_time, max(end_time, start_time + 1e-3)


def build_mesh_preview_manifest(
    preview_job: dict[str, Any],
    plan: dict[str, Any],
    song_event_map: dict[str, Any] | None,
    mesh_template_fbx: Path,
    fps: int = 24,
    basis_mode: str = "constraint_world_pelvis_local",
    max_steps: int = 0,
) -> dict[str, Any]:
    if not mesh_template_fbx.exists():
        raise FileNotFoundError(f"mesh template FBX not found: {mesh_template_fbx}")

    selected_steps = list(preview_job.get("steps", []))
    if max_steps > 0:
        selected_steps = selected_steps[:max_steps]

    plan_steps_by_index = {int(step["index"]): step for step in plan.get("steps", [])}
    manifest_steps: list[dict[str, Any]] = []
    sequences: dict[tuple[str, str, str], dict[str, Any]] = {}
    previous_scene_end = 0

    for preview_step in selected_steps:
        step_index = int(preview_step.get("index", 0))
        plan_step = plan_steps_by_index.get(step_index, {})
        reference_artifacts = dict(preview_step.get("reference_artifacts", {}))
        source_motion_fbx = str(reference_artifacts.get("sequence_motion_fbx", "") or "").strip()
        if not source_motion_fbx:
            raise FileNotFoundError(f"sequence_motion_fbx missing for step {step_index}")
        source_motion_path = Path(source_motion_fbx)
        if not source_motion_path.exists():
            raise FileNotFoundError(f"sequence motion FBX not found for step {step_index}: {source_motion_path}")

        frame_range = dict(preview_step.get("frame_range", {}))
        source_frame_start = int(frame_range.get("start", 0) or 0)
        source_frame_end = max(source_frame_start + 1, int(frame_range.get("end_exclusive", source_frame_start + 1) or (source_frame_start + 1))) - 1

        if song_event_map and plan_step:
            start_time_sec, end_time_sec = _step_duration_sec(plan_step, song_event_map)
            scene_frame_start = int(round(start_time_sec * fps)) + 1
            scene_frame_end = max(scene_frame_start + 1, int(round(end_time_sec * fps)))
        else:
            scene_frame_start = previous_scene_end + 1
            raw_length = max(2, source_frame_end - source_frame_start + 1)
            scene_frame_end = scene_frame_start + raw_length - 1
            start_time_sec = (scene_frame_start - 1) / float(max(fps, 1))
            end_time_sec = scene_frame_end / float(max(fps, 1))

        previous_scene_end = max(previous_scene_end, scene_frame_end)
        dataset = str(preview_step.get("dataset", "unknown") or "unknown")
        source_sequence = str(preview_step.get("source_sequence", "unknown") or "unknown")
        sequence_key = (dataset, source_sequence, str(source_motion_path))
        sequence_entry = sequences.setdefault(
            sequence_key,
            {
                "dataset": dataset,
                "source_sequence": source_sequence,
                "source_motion_fbx": str(source_motion_path),
                "step_indices": [],
            },
        )
        sequence_entry["step_indices"].append(step_index)

        manifest_steps.append(
            {
                "index": step_index,
                "dataset": dataset,
                "source_sequence": source_sequence,
                "source_motion_fbx": str(source_motion_path),
                "source_frame_start": source_frame_start,
                "source_frame_end": source_frame_end,
                "scene_frame_start": scene_frame_start,
                "scene_frame_end": scene_frame_end,
                "section_label": plan_step.get("section_label"),
                "target_energy": plan_step.get("switch_reason", {}).get("target_energy"),
                "transition_mode": plan_step.get("switch_reason", {}).get("transition_mode"),
                "speed_scale": float(plan_step.get("speed_scale", 1.0) or 1.0),
                "start_time_sec": round(float(start_time_sec), 5),
                "end_time_sec": round(float(end_time_sec), 5),
                "unit_id": preview_step.get("unit_id"),
            }
        )

    return {
        "schema_version": 1,
        "plan_id": preview_job.get("plan_id", plan.get("plan_id", "mesh_preview_plan")),
        "generated_at_utc": utc_now_iso(),
        "fps": int(fps),
        "basis_mode": basis_mode,
        "mesh_template_fbx": str(mesh_template_fbx),
        "scene": {
            "frame_start": 1,
            "frame_end": max((step["scene_frame_end"] for step in manifest_steps), default=1),
        },
        "sequences": sorted(sequences.values(), key=lambda item: (item["dataset"], item["source_sequence"])),
        "steps": manifest_steps,
        "notes": [
            "This manifest drives a true Blender skinned-mesh preview scene using a Mixamo-compatible mesh template.",
            "Each source sequence is retargeted once, then reused through NLA strips for the selected choreography windows.",
        ],
    }
