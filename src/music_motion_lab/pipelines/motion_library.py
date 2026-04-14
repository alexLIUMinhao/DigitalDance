from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..contracts import MotionUnitLibrary
from ..utils import load_json, stable_digest, utc_now_iso

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise RuntimeError("music-motion-lab motion library requires numpy.") from exc


COMMON_BONE_ORDER = [
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

LOWER_BODY_BONES = ["left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"]
UPPER_BODY_BONES = ["neck", "head", "left_collar", "right_collar", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
CORE_BONES = ["pelvis", "spine1", "spine2", "spine3"]
ANCHOR_BONES = ["left_foot", "right_foot", "left_wrist", "right_wrist", "head"]
DATASET_SOURCE_FPS = {
    "finedance": 30.0,
    "aistpp": 60.0,
}


def _style_tags_from_report(report: dict[str, Any]) -> list[str]:
    job_id = str(report.get("jobId", "") or "")
    tokens = [token for token in job_id.replace("__", "_").split("_") if token]
    blocked = {"dataset", "smpl", "datasets", "bvh", "job", "basic", "step"}
    deduped: list[str] = []
    for token in tokens:
        lowered = token.lower()
        if lowered in blocked or lowered in deduped:
            continue
        deduped.append(lowered)
    return deduped[:5] or [str(report.get("datasetName", "unknown")).lower()]


def _preferred_sections(energy: str, travel: str) -> list[str]:
    if energy == "high_energy":
        return ["chorus", "instrumental"]
    if energy == "mid_energy":
        return ["verse", "chorus"] if travel != "stationary" else ["verse", "intro"]
    return ["intro", "outro", "verse"]


def _scan_pose_driver_files(preview_root: Path) -> list[Path]:
    return sorted(preview_root.glob("dataset_pose_driver/*/*_bridge_pose_driver.json"))


def _scan_bridge_consistency(preview_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    results: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(preview_root.glob("bridge_consistency/*/*_bridge_consistency.json")):
        payload = load_json(path)
        key = (str(payload.get("datasetName", "")), str(payload.get("sequenceId", "")))
        results[key] = {"path": str(path), "payload": payload}
    return results


def _scan_retarget_reports(preview_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(preview_root.glob("retarget_pose_consistency/*/*/*_consistency_report.json")):
        payload = load_json(path)
        key = (str(payload.get("datasetName", "")), str(payload.get("sequenceId", "")))
        score = float(payload.get("summary", {}).get("jointPositionError", {}).get("mean", 999999.0) or 999999.0)
        existing = candidates.get(key)
        if existing is None or score < existing["score"]:
            candidates[key] = {"path": str(path), "payload": payload, "score": score}
    return candidates


def _retarget_health_bucket(mean_joint_error: float | None, p95_joint_error: float | None) -> str:
    mean_value = 999999.0 if mean_joint_error is None else float(mean_joint_error)
    p95_value = 999999.0 if p95_joint_error is None else float(p95_joint_error)
    if mean_value <= 1.0 and p95_value <= 1.6:
        return "preferred"
    if mean_value <= 2.5 and p95_value <= 5.0:
        return "usable"
    if mean_value <= 5.0 and p95_value <= 10.0:
        return "risky"
    return "avoid"


def _reshape_pose_driver_positions(payload: dict[str, Any]) -> np.ndarray:
    frame_count = int(payload.get("frameCount", 0) or 0)
    joint_count = int(payload.get("jointCount", 0) or 0)
    flat = np.asarray(payload.get("positions", []), dtype=np.float32)
    if flat.size != frame_count * joint_count * 3:
        raise ValueError(f"Unexpected pose-driver position payload shape: {flat.size} values for {frame_count}x{joint_count}x3")
    return flat.reshape(frame_count, joint_count, 3)


def _joint_index_map(bone_order: list[str]) -> dict[str, int]:
    return {name: index for index, name in enumerate(bone_order)}


def _pick_joint(bone_index: dict[str, int], *names: str) -> int | None:
    for name in names:
        if name in bone_index:
            return bone_index[name]
    return None


def _sequence_stats(sequence_positions: np.ndarray, bone_index: dict[str, int]) -> dict[str, Any]:
    foot_ids = [joint_id for joint_id in [_pick_joint(bone_index, "left_foot", "left_ankle"), _pick_joint(bone_index, "right_foot", "right_ankle")] if joint_id is not None]
    head_id = _pick_joint(bone_index, "head", "neck", "spine3", "pelvis")
    foot_heights = sequence_positions[:, foot_ids, 1].reshape(-1) if foot_ids else sequence_positions[:, :, 1].reshape(-1)
    floor_height = float(np.percentile(foot_heights, 5))
    head_heights = sequence_positions[:, head_id, 1] if head_id is not None else np.zeros(sequence_positions.shape[0], dtype=np.float32)
    body_scale = float(np.percentile(head_heights, 90) - floor_height)
    return {
        "floor_height": floor_height,
        "body_scale": max(0.25, body_scale),
    }


def _yaw_from_vector(vector_xz: np.ndarray) -> float:
    return math.degrees(math.atan2(float(vector_xz[0]), float(vector_xz[1])))


def _wrap_degrees(angle: float) -> float:
    wrapped = (angle + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _mean_motion(unit_positions: np.ndarray, bone_index: dict[str, int], bone_names: list[str]) -> float:
    joint_ids = [bone_index[name] for name in bone_names if name in bone_index]
    if not joint_ids or unit_positions.shape[0] < 2:
        return 0.0
    diffs = np.diff(unit_positions[:, joint_ids, :], axis=0)
    return float(np.linalg.norm(diffs, axis=2).mean())


def _resolve_foot_track(unit_positions: np.ndarray, bone_index: dict[str, int], side: str) -> np.ndarray:
    joint_id = _pick_joint(bone_index, f"{side}_foot", f"{side}_ankle")
    if joint_id is None:
        return np.zeros((unit_positions.shape[0], 3), dtype=np.float32)
    return unit_positions[:, joint_id, :]


def _support_ratio(foot_track: np.ndarray, floor_height: float, body_scale: float) -> float:
    if foot_track.shape[0] == 0:
        return 0.0
    heights = foot_track[:, 1]
    planar_vel = np.linalg.norm(np.diff(foot_track[:, [0, 2]], axis=0), axis=1)
    planar_vel = np.concatenate([planar_vel[:1], planar_vel], axis=0) if planar_vel.size else np.zeros(foot_track.shape[0], dtype=np.float32)
    near_floor = heights <= (floor_height + body_scale * 0.08)
    low_speed = planar_vel <= (body_scale * 0.03)
    return float(np.mean(near_floor & low_speed))


def _support_state(left_ratio: float, right_ratio: float) -> str:
    if left_ratio >= 0.45 and right_ratio >= 0.45:
        return "both_feet_planted"
    if left_ratio >= 0.45 and right_ratio < 0.30:
        return "left_support"
    if right_ratio >= 0.45 and left_ratio < 0.30:
        return "right_support"
    if left_ratio >= 0.18 and right_ratio >= 0.18:
        return "alternating_support"
    return "airborne_bias"


def _body_focus(lower_motion: float, upper_motion: float, core_motion: float) -> list[str]:
    if lower_motion > max(upper_motion * 1.35, core_motion * 1.1):
        return ["lower_body"]
    if upper_motion > max(lower_motion * 1.35, core_motion * 1.1):
        return ["upper_body"]
    if core_motion > max(lower_motion, upper_motion) * 1.2:
        return ["core_driven"]
    return ["full_body"]


def _travel_class(displacement: float, body_scale: float) -> str:
    if displacement < body_scale * 0.06:
        return "stationary"
    if displacement < body_scale * 0.18:
        return "localized"
    return "traveling"


def _facing_yaws(unit_positions: np.ndarray, bone_index: dict[str, int]) -> np.ndarray:
    left_shoulder = _pick_joint(bone_index, "left_shoulder", "left_collar", "left_hip")
    right_shoulder = _pick_joint(bone_index, "right_shoulder", "right_collar", "right_hip")
    left_hip = _pick_joint(bone_index, "left_hip")
    right_hip = _pick_joint(bone_index, "right_hip")

    yaws: list[float] = []
    previous_yaw = 0.0
    for frame in unit_positions:
        lateral = np.zeros(2, dtype=np.float32)
        if left_shoulder is not None and right_shoulder is not None:
            lateral += frame[right_shoulder, [0, 2]] - frame[left_shoulder, [0, 2]]
        if left_hip is not None and right_hip is not None:
            lateral += 0.65 * (frame[right_hip, [0, 2]] - frame[left_hip, [0, 2]])
        if float(np.linalg.norm(lateral)) <= 1e-6:
            yaws.append(previous_yaw)
            continue
        lateral = lateral / max(float(np.linalg.norm(lateral)), 1e-6)
        forward = np.asarray([lateral[1], -lateral[0]], dtype=np.float32)
        yaw = _yaw_from_vector(forward)
        previous_yaw = yaw
        yaws.append(yaw)
    return np.asarray(yaws, dtype=np.float32)


def _facing_change_class(start_yaw: float, end_yaw: float) -> str:
    delta = abs(_wrap_degrees(end_yaw - start_yaw))
    if delta < 20.0:
        return "stable"
    if delta < 60.0:
        return "slight_turn"
    if delta < 120.0:
        return "major_turn"
    return "sweeping_turn"


def _anchor_feature(unit_positions: np.ndarray, bone_index: dict[str, int], frame_index: int, body_scale: float) -> np.ndarray:
    pelvis_id = _pick_joint(bone_index, "pelvis")
    pelvis = unit_positions[frame_index, pelvis_id, :] if pelvis_id is not None else np.zeros(3, dtype=np.float32)
    components: list[np.ndarray] = []
    for bone_name in ANCHOR_BONES:
        joint_id = _pick_joint(bone_index, bone_name)
        if joint_id is None:
            components.append(np.zeros(3, dtype=np.float32))
            continue
        components.append((unit_positions[frame_index, joint_id, :] - pelvis) / max(body_scale, 1e-6))
    return np.concatenate(components, axis=0).astype(np.float32)


def _motion_intensity_metrics(unit_positions: np.ndarray, body_scale: float, fps: float) -> dict[str, float]:
    if unit_positions.shape[0] < 2:
        return {
            "mean_joint_speed_norm": 0.0,
            "p90_joint_speed_norm": 0.0,
            "mean_joint_accel_norm": 0.0,
            "intensity_score": 0.0,
        }

    normalized_scale = max(body_scale, 1e-6)
    velocities = np.diff(unit_positions, axis=0) * (fps / normalized_scale)
    speed = np.linalg.norm(velocities, axis=2)
    mean_joint_speed_norm = float(speed.mean())
    p90_joint_speed_norm = float(np.percentile(speed, 90))

    if velocities.shape[0] >= 2:
        accelerations = np.diff(velocities, axis=0) * fps
        accel_mag = np.linalg.norm(accelerations, axis=2)
        mean_joint_accel_norm = float(np.sqrt(max(float(accel_mag.mean()), 0.0)))
    else:
        mean_joint_accel_norm = 0.0

    intensity_score = (
        mean_joint_speed_norm * 0.48
        + p90_joint_speed_norm * 0.24
        + mean_joint_accel_norm * 0.28
    )
    return {
        "mean_joint_speed_norm": round(mean_joint_speed_norm, 5),
        "p90_joint_speed_norm": round(p90_joint_speed_norm, 5),
        "mean_joint_accel_norm": round(mean_joint_accel_norm, 5),
        "intensity_score": round(float(intensity_score), 5),
    }


def _assign_kinematic_energy(units: list[dict[str, Any]]) -> None:
    if not units:
        return

    scores = np.asarray([float(unit["_intensity_score"]) for unit in units], dtype=np.float32)
    min_score = float(scores.min())
    max_score = float(scores.max())
    score_span = max(max_score - min_score, 1e-6)
    low_threshold = float(np.quantile(scores, 0.35))
    high_threshold = float(np.quantile(scores, 0.72))

    all_equal = bool(np.allclose(scores, scores[0]))
    for unit in units:
        score = float(unit["_intensity_score"])
        percentile = (score - min_score) / score_span
        if all_equal:
            energy = "mid_energy"
        elif score <= low_threshold:
            energy = "low_energy"
        elif score >= high_threshold:
            energy = "high_energy"
        else:
            energy = "mid_energy"

        unit["energy"] = energy
        unit["energy_profile"] = {
            **unit.pop("_energy_metrics"),
            "energy_quantile": round(float(percentile), 5),
            "band_method": "kinematic_quantiles",
        }
        unit["preferred_section_labels"] = _preferred_sections(energy=energy, travel=unit["travel"])


def _derive_unit_semantics(
    unit_positions: np.ndarray,
    bone_index: dict[str, int],
    start_frame: int,
    end_frame_exclusive: int,
    body_scale: float,
    floor_height: float,
    fps: float,
) -> dict[str, Any]:
    pelvis_id = _pick_joint(bone_index, "pelvis")
    if pelvis_id is None:
        raise ValueError("pelvis joint is required for motion unit semantics")

    pelvis_track = unit_positions[:, pelvis_id, :]
    travel_vector = pelvis_track[-1, [0, 2]] - pelvis_track[0, [0, 2]]
    travel_distance = float(np.linalg.norm(travel_vector))
    travel_yaw = _yaw_from_vector(travel_vector) if travel_distance > 1e-6 else 0.0
    pelvis_planar_speed = float(np.linalg.norm(np.diff(pelvis_track[:, [0, 2]], axis=0), axis=1).mean() * fps) if unit_positions.shape[0] > 1 else 0.0

    lower_motion = _mean_motion(unit_positions, bone_index, LOWER_BODY_BONES)
    upper_motion = _mean_motion(unit_positions, bone_index, UPPER_BODY_BONES)
    core_motion = _mean_motion(unit_positions, bone_index, CORE_BONES)

    left_foot_track = _resolve_foot_track(unit_positions, bone_index, "left")
    right_foot_track = _resolve_foot_track(unit_positions, bone_index, "right")
    left_support_ratio = _support_ratio(left_foot_track, floor_height=floor_height, body_scale=body_scale)
    right_support_ratio = _support_ratio(right_foot_track, floor_height=floor_height, body_scale=body_scale)
    support_state = _support_state(left_support_ratio, right_support_ratio)

    yaws = _facing_yaws(unit_positions, bone_index)
    start_yaw = float(np.median(yaws[: max(1, min(4, len(yaws)))]))
    end_yaw = float(np.median(yaws[max(0, len(yaws) - 4):]))
    facing_delta = _wrap_degrees(end_yaw - start_yaw)

    entry_feature = _anchor_feature(unit_positions, bone_index, 0, body_scale)
    exit_feature = _anchor_feature(unit_positions, bone_index, unit_positions.shape[0] - 1, body_scale)
    intensity_metrics = _motion_intensity_metrics(unit_positions=unit_positions, body_scale=body_scale, fps=fps)

    anchor_label = "stable_support" if support_state in {"both_feet_planted", "left_support", "right_support"} else "dynamic_transition"
    return {
        "body_focus": _body_focus(lower_motion=lower_motion, upper_motion=upper_motion, core_motion=core_motion),
        "foot_contact": [support_state],
        "foot_contact_profile": {
            "left_support_ratio": round(left_support_ratio, 5),
            "right_support_ratio": round(right_support_ratio, 5),
        },
        "travel": _travel_class(travel_distance, body_scale=body_scale),
        "travel_profile": {
            "horizontal_displacement": round(travel_distance, 5),
            "travel_yaw_deg": round(travel_yaw, 5),
            "pelvis_planar_speed": round(pelvis_planar_speed, 5),
        },
        "facing_change": _facing_change_class(start_yaw, end_yaw),
        "facing_profile": {
            "start_yaw_deg": round(start_yaw, 5),
            "end_yaw_deg": round(end_yaw, 5),
            "delta_yaw_deg": round(facing_delta, 5),
        },
        "entry_anchor": {
            "label": anchor_label,
            "frame_index": start_frame,
            "contact_state": support_state,
            "facing_yaw_deg": round(start_yaw, 5),
            "travel_yaw_deg": round(travel_yaw, 5),
            "pelvis_planar_speed": round(pelvis_planar_speed, 5),
        },
        "exit_anchor": {
            "label": anchor_label,
            "frame_index": end_frame_exclusive - 1,
            "contact_state": support_state,
            "facing_yaw_deg": round(end_yaw, 5),
            "travel_yaw_deg": round(travel_yaw, 5),
            "pelvis_planar_speed": round(pelvis_planar_speed, 5),
        },
        "_entry_feature": entry_feature,
        "_exit_feature": exit_feature,
        "_energy_metrics": intensity_metrics,
        "_intensity_score": intensity_metrics["intensity_score"],
    }


def _compatibility_score(exit_unit: dict[str, Any], entry_unit: dict[str, Any]) -> float:
    exit_anchor = exit_unit["exit_anchor"]
    entry_anchor = entry_unit["entry_anchor"]

    facing_delta = abs(_wrap_degrees(float(exit_anchor["facing_yaw_deg"]) - float(entry_anchor["facing_yaw_deg"])))
    if facing_delta > 110.0:
        return -999.0

    pose_distance = float(np.linalg.norm(exit_unit["_exit_feature"] - entry_unit["_entry_feature"]))
    if pose_distance > 6.0:
        return -999.0

    score = 0.0
    if exit_unit["energy"] == entry_unit["energy"]:
        score += 2.0
    elif "mid_energy" in {exit_unit["energy"], entry_unit["energy"]}:
        score += 0.9

    if exit_unit["travel"] == entry_unit["travel"]:
        score += 1.1
    elif {"stationary", "localized"} == {exit_unit["travel"], entry_unit["travel"]}:
        score += 0.5

    exit_support = exit_anchor["contact_state"]
    entry_support = entry_anchor["contact_state"]
    if exit_support == entry_support:
        score += 1.3
    elif "both_feet_planted" in {exit_support, entry_support}:
        score += 0.9
    elif "alternating_support" in {exit_support, entry_support}:
        score += 0.35

    score += max(0.0, 1.0 - (facing_delta / 90.0))
    score += max(0.0, 2.0 - (pose_distance / 2.0))
    score += max(0.0, 0.8 - abs(float(exit_anchor["pelvis_planar_speed"]) - float(entry_anchor["pelvis_planar_speed"])) * 0.25)
    return score


def _trim_internal_fields(unit: dict[str, Any]) -> None:
    unit.pop("_entry_feature", None)
    unit.pop("_exit_feature", None)
    unit.pop("_intensity_score", None)
    unit.pop("_energy_metrics", None)


def build_motion_library_showcase(library: MotionUnitLibrary, sample_count: int = 8) -> dict[str, Any]:
    units = list(library.units)
    energy_counts = Counter(unit["energy"] for unit in units)
    travel_counts = Counter(unit["travel"] for unit in units)
    facing_counts = Counter(unit["facing_change"] for unit in units)
    support_counts = Counter(unit["foot_contact"][0] for unit in units if unit.get("foot_contact"))
    body_focus_counts = Counter((unit.get("body_focus") or ["unknown"])[0] for unit in units)
    sequence_counts = Counter(f"{unit['dataset']}:{unit['source_sequence']}" for unit in units)

    sampled_units = []
    for unit in units[:sample_count]:
        sampled_units.append(
            {
                "unit_id": unit["unit_id"],
                "dataset": unit["dataset"],
                "sequence": unit["source_sequence"],
                "energy": unit["energy"],
                "energy_profile": unit.get("energy_profile"),
                "travel": unit["travel"],
                "facing_change": unit["facing_change"],
                "body_focus": unit["body_focus"],
                "foot_contact": unit["foot_contact"],
                "entry_anchor": unit["entry_anchor"],
                "exit_anchor": unit["exit_anchor"],
                "compatible_next_units": unit["compatible_next_units"][:4],
            }
        )

    return {
        "schema_version": 1,
        "library_id": library.library_id,
        "generated_at_utc": library.generated_at_utc,
        "counts": {
            "unit_count": len(units),
            "sequence_count": len(sequence_counts),
            "energy": dict(energy_counts),
            "travel": dict(travel_counts),
            "facing_change": dict(facing_counts),
            "support_state": dict(support_counts),
            "body_focus": dict(body_focus_counts),
        },
        "top_sequences": [{"sequence": sequence, "unit_count": count} for sequence, count in sequence_counts.most_common(10)],
        "sampled_units": sampled_units,
        "notes": [
            "Showcase is a stage-summary artifact for reviewing automatically derived motion-unit semantics.",
            "Energy bands now come from kinematic intensity quantiles over the full base-action library, not dataset-name heuristics.",
        ],
    }


def build_motion_unit_library(
    preview_root: Path,
    fps: float = 30.0,
    unit_beats: float = 4.0,
    stride_beats: float = 1.0,
    assumed_bpm: float = 120.0,
) -> MotionUnitLibrary:
    pose_driver_paths = _scan_pose_driver_files(preview_root)
    bridge_reports = _scan_bridge_consistency(preview_root)
    retarget_reports = _scan_retarget_reports(preview_root)

    units: list[dict[str, Any]] = []
    units_by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for pose_driver_path in pose_driver_paths:
        payload = load_json(pose_driver_path)
        dataset = str(payload.get("datasetName", "unknown"))
        dataset_key = dataset.strip().lower()
        sequence_id = str(payload.get("sequenceId", pose_driver_path.stem))
        frame_count = int(payload.get("frameCount", 0) or 0)
        bone_order = list(payload.get("boneOrder", []))
        bone_index = _joint_index_map(bone_order)
        sequence_positions = _reshape_pose_driver_positions(payload)
        stats = _sequence_stats(sequence_positions, bone_index)
        key = (dataset, sequence_id)
        source_fps = DATASET_SOURCE_FPS.get(dataset_key, fps)
        frames_per_unit = max(1, int(round((unit_beats * 60.0 / max(assumed_bpm, 1.0)) * source_fps)))
        frames_per_stride = max(1, int(round((stride_beats * 60.0 / max(assumed_bpm, 1.0)) * source_fps)))

        bridge_report = bridge_reports.get(key, {})
        retarget_report = retarget_reports.get(key, {})
        retarget_payload = dict(retarget_report.get("payload", {}))
        style_tags = _style_tags_from_report(retarget_payload) if retarget_payload else [dataset]
        retarget_summary = dict(retarget_payload.get("summary", {}))
        retarget_joint_error = dict(retarget_summary.get("jointPositionError", {}))
        retarget_bone_angle = dict(retarget_summary.get("boneAngleDeg", {}))
        retarget_mean_joint_error = float(retarget_joint_error.get("mean", 999999.0) or 999999.0) if retarget_payload else None
        retarget_p95_joint_error = float(retarget_joint_error.get("p95", 999999.0) or 999999.0) if retarget_payload else None
        retarget_p95_bone_angle = float(retarget_bone_angle.get("p95", 999999.0) or 999999.0) if retarget_payload else None
        retarget_basis_mode = str(retarget_payload.get("retargetBasisMode", "unknown") or "unknown") if retarget_payload else "unknown"
        retarget_health = _retarget_health_bucket(retarget_mean_joint_error, retarget_p95_joint_error) if retarget_payload else "unknown"

        for start_frame in range(0, frame_count, frames_per_stride):
            end_frame = min(frame_count, start_frame + frames_per_unit)
            if end_frame - start_frame < max(10, frames_per_unit // 2):
                continue

            unit_positions = sequence_positions[start_frame:end_frame]
            duration_sec = (end_frame - start_frame) / max(source_fps, 1.0)
            duration_beats = duration_sec * max(assumed_bpm, 1.0) / 60.0
            semantics = _derive_unit_semantics(
                unit_positions=unit_positions,
                bone_index=bone_index,
                start_frame=start_frame,
                end_frame_exclusive=end_frame,
                body_scale=stats["body_scale"],
                floor_height=stats["floor_height"],
                fps=source_fps,
            )

            entry_pose_fingerprint = stable_digest({
                "dataset": dataset,
                "sequence": sequence_id,
                "frame": start_frame,
                "feature": semantics["_entry_feature"].tolist(),
            })
            exit_pose_fingerprint = stable_digest({
                "dataset": dataset,
                "sequence": sequence_id,
                "frame": end_frame - 1,
                "feature": semantics["_exit_feature"].tolist(),
            })

            unit = {
                "unit_id": f"{dataset}_{sequence_id}_{start_frame:04d}_{end_frame - 1:04d}",
                "dataset": dataset,
                "source_sequence": sequence_id,
                "skeleton_id": "bridge_truth_v1_common22",
                "frame_range": {"start": start_frame, "end_exclusive": end_frame},
                "duration_sec": round(duration_sec, 5),
                "duration_beats_estimate": round(duration_beats, 5),
                "energy": "mid_energy",
                "style_tags": style_tags,
                "body_focus": semantics["body_focus"],
                "foot_contact": semantics["foot_contact"],
                "foot_contact_profile": semantics["foot_contact_profile"],
                "facing_change": semantics["facing_change"],
                "facing_profile": semantics["facing_profile"],
                "travel": semantics["travel"],
                "travel_profile": semantics["travel_profile"],
                "preferred_section_labels": [],
                "entry_anchor": semantics["entry_anchor"],
                "exit_anchor": semantics["exit_anchor"],
                "entry_pose_fingerprint": entry_pose_fingerprint,
                "exit_pose_fingerprint": exit_pose_fingerprint,
                "reference_artifacts": {
                    "pose_driver_json": str(pose_driver_path),
                    "bridge_consistency_json": bridge_report.get("path"),
                    "reference_retarget_report_json": retarget_report.get("path"),
                },
                "bridge_quality": bridge_report.get("payload", {}).get("status", "unknown"),
                "retarget_quality": retarget_payload.get("status", "unknown") if retarget_payload else "unknown",
                "retarget_basis_mode": retarget_basis_mode,
                "retarget_mean_joint_error": round(retarget_mean_joint_error, 5) if retarget_mean_joint_error is not None else None,
                "retarget_p95_joint_error": round(retarget_p95_joint_error, 5) if retarget_p95_joint_error is not None else None,
                "retarget_p95_bone_angle_deg": round(retarget_p95_bone_angle, 5) if retarget_p95_bone_angle is not None else None,
                "retarget_health": retarget_health,
                "compatible_next_units": [],
                "_entry_feature": semantics["_entry_feature"],
                "_exit_feature": semantics["_exit_feature"],
                "_energy_metrics": semantics["_energy_metrics"],
                "_intensity_score": semantics["_intensity_score"],
            }
            units.append(unit)
            units_by_sequence[key].append(unit)

    _assign_kinematic_energy(units)

    units_by_energy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        units_by_energy[unit["energy"]].append(unit)

    for sequence_units in units_by_sequence.values():
        for index, unit in enumerate(sequence_units):
            compatibility_scores: list[tuple[float, str]] = []
            if index + 1 < len(sequence_units):
                compatibility_scores.append((999.0, sequence_units[index + 1]["unit_id"]))

            candidate_pool = units_by_energy[unit["energy"]] + units_by_energy.get("mid_energy", [])
            seen_ids = {unit["unit_id"]}
            for candidate in candidate_pool:
                candidate_id = candidate["unit_id"]
                if candidate_id in seen_ids:
                    continue
                seen_ids.add(candidate_id)
                score = _compatibility_score(unit, candidate)
                if score <= -100.0:
                    continue
                compatibility_scores.append((score, candidate_id))

            compatibility_scores.sort(key=lambda item: item[0], reverse=True)
            unit["compatible_next_units"] = [candidate_id for _, candidate_id in compatibility_scores[:6]]

    for unit in units:
        _trim_internal_fields(unit)

    notes = [
        "Units are sliced from shared pose-driver diagnostics so the new lab can stay isolated from Unity runtime code.",
        "Motion-unit semantics now derive travel, facing, support state, transition anchors, and kinematic energy from shared skeleton positions.",
        "The base-action library now uses overlapping windows so every available sequence contributes many reusable sampling candidates instead of only a few coarse clips.",
        "Retarget consistency metrics are attached per source sequence so the planner can avoid known-bad avatar mapping assets.",
    ]
    return MotionUnitLibrary(
        schema_version=1,
        library_id="motion_unit_library_v2",
        source_roots={"motion_base_assets_root": str(preview_root.parent)},
        units=units,
        notes=notes,
        generated_at_utc=utc_now_iso(),
    )
