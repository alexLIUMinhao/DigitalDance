from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..contracts import RetargetReport
from ..utils import load_json, utc_now_iso


def _dataset_and_sequence_from_report_path(report_path: str) -> tuple[str, str]:
    candidate = Path(report_path)
    parts = list(candidate.parts)
    if "retarget_pose_consistency" not in parts:
        return ("unknown", "unknown")
    base_index = parts.index("retarget_pose_consistency")
    if len(parts) <= base_index + 2:
        return ("unknown", "unknown")
    return (parts[base_index + 1], parts[base_index + 2])


def _resolve_sequence_motion_fbx(reference_artifacts: dict[str, Any]) -> str:
    report_path_raw = str(reference_artifacts.get("reference_retarget_report_json", "") or "").strip()
    if not report_path_raw:
        return ""

    report_path = Path(report_path_raw)
    if not report_path.exists():
        return ""

    dataset, _ = _dataset_and_sequence_from_report_path(report_path_raw)
    preview_cache_root = report_path.parents[3]
    stem = report_path.name.replace("_consistency_report.json", "")
    candidates = sorted((preview_cache_root / "dataset_sequences" / dataset).glob(f"**/{stem}.fbx"))
    return str(candidates[0]) if candidates else ""


def _build_preview_job(
    plan: dict[str, Any],
    motion_library: dict[str, Any],
    blender_path: str | None,
) -> dict[str, Any]:
    units_by_id = {unit["unit_id"]: unit for unit in motion_library.get("units", [])}
    steps = []
    for step in plan.get("steps", []):
        unit = units_by_id.get(step["selected_unit_id"], {})
        reference_artifacts = dict(unit.get("reference_artifacts", {}))
        dataset = str(unit.get("dataset", "") or "")
        if not dataset:
            dataset, _ = _dataset_and_sequence_from_report_path(str(reference_artifacts.get("reference_retarget_report_json", "")))
        sequence_motion_fbx = _resolve_sequence_motion_fbx(reference_artifacts)
        if sequence_motion_fbx:
            reference_artifacts["sequence_motion_fbx"] = sequence_motion_fbx
        steps.append(
            {
                "index": step["index"],
                "dataset": dataset,
                "unit_id": step["selected_unit_id"],
                "source_sequence": step.get("source_sequence"),
                "frame_range": unit.get("frame_range"),
                "reference_artifacts": reference_artifacts,
            }
        )

    return {
        "schema_version": 1,
        "plan_id": plan["plan_id"],
        "backend": {
            "name": "blender",
            "available": bool(blender_path),
            "path": blender_path or "",
        },
        "status": "ready" if blender_path else "backend_unavailable",
        "steps": steps,
        "command_template": (
            f"{blender_path} --background --python your_preview_driver.py -- --plan outputs/choreography_plans/{plan['plan_id']}.json"
            if blender_path
            else ""
        ),
        "generated_at_utc": utc_now_iso(),
    }


def _sequence_reports_for_plan(plan: dict[str, Any], motion_library: dict[str, Any]) -> list[dict[str, Any]]:
    units_by_id = {unit["unit_id"]: unit for unit in motion_library.get("units", [])}
    reports_by_sequence: dict[tuple[str, str], dict[str, Any]] = {}

    for step in plan.get("steps", []):
        unit = units_by_id.get(step["selected_unit_id"])
        if not unit:
            continue
        report_path = unit.get("reference_artifacts", {}).get("reference_retarget_report_json")
        if not report_path:
            continue
        payload = load_json(Path(report_path))
        key = (str(payload.get("datasetName", unit.get("dataset", "unknown"))), str(payload.get("sequenceId", unit.get("source_sequence", "unknown"))))
        reports_by_sequence[key] = payload

    sequence_reports: list[dict[str, Any]] = []
    for (dataset, sequence_id), payload in sorted(reports_by_sequence.items()):
        summary = payload.get("summary", {})
        bone_angle = summary.get("boneAngleDeg", {})
        joint_error = summary.get("jointPositionError", {})
        sequence_reports.append(
            {
                "dataset": dataset,
                "sequence_id": sequence_id,
                "status": payload.get("status", "unknown"),
                "issues": list(payload.get("issues", [])),
                "bone_angle_deg": {
                    "mean": bone_angle.get("mean"),
                    "p95": bone_angle.get("p95"),
                    "max": bone_angle.get("max"),
                },
                "joint_position_error": {
                    "mean": joint_error.get("mean"),
                    "p95": joint_error.get("p95"),
                    "max": joint_error.get("max"),
                },
                "source_report_json": str(payload.get("artifacts", {}).get("retargetPoseJson", "")) or "",
                "reference_report_json": "",
            }
        )
    return sequence_reports


def build_preview_and_retarget_report(
    plan: dict[str, Any],
    motion_library: dict[str, Any],
    avatar_id: str = "reference_avatar",
    blender_path: str | None = None,
) -> tuple[dict[str, Any], RetargetReport]:
    resolved_blender_path = blender_path or shutil.which("blender")
    preview_job = _build_preview_job(plan=plan, motion_library=motion_library, blender_path=resolved_blender_path)
    sequence_reports = _sequence_reports_for_plan(plan=plan, motion_library=motion_library)

    status_counts = defaultdict(int)
    for report in sequence_reports:
        status_counts[report["status"]] += 1

    summary = {
        "sequence_count": len(sequence_reports),
        "status_counts": dict(status_counts),
        "backend_available": bool(resolved_blender_path),
    }
    if sequence_reports:
        summary["worst_joint_error_mean"] = max(
            float(report.get("joint_position_error", {}).get("mean") or 0.0) for report in sequence_reports
        )
        summary["worst_bone_angle_max"] = max(
            float(report.get("bone_angle_deg", {}).get("max") or 0.0) for report in sequence_reports
        )

    report = RetargetReport(
        schema_version=1,
        plan_id=plan["plan_id"],
        avatar_id=avatar_id,
        status="backend_unavailable" if not resolved_blender_path else ("warning" if any(r["status"] != "pass" for r in sequence_reports) else "ok"),
        backend={"name": "blender", "available": bool(resolved_blender_path), "path": resolved_blender_path or ""},
        sequence_reports=sequence_reports,
        summary=summary,
        generated_at_utc=utc_now_iso(),
    )
    return preview_job, report
