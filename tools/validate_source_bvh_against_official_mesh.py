#!/usr/bin/env python3
"""Validate a source BVH against official mesh-preview joints."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.source_bvh_validation import (  # noqa: E402
    BRIDGE_BONE_ORDER_22,
    BVH_TO_TARGET_BASIS,
    build_pose_payload,
    compare_pose_payloads,
    compute_bone_vector_diagnostics,
    diagnose_validation,
    extract_bvh_joint_positions,
    remap_per_joint_error_names,
    write_json,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", default="168", help="Sequence id.")
    parser.add_argument(
        "--official-joints",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "upstream_baselines" / "finedance" / "168" / "finedance_168_official_mesh_joints.json",
        help="Official mesh joints json.",
    )
    parser.add_argument(
        "--source-bvh",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source.bvh",
        help="Source BVH to validate.",
    )
    parser.add_argument(
        "--fk-output",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source_fk_joints.json",
        help="Output path for FK direct joints json.",
    )
    parser.add_argument(
        "--import-output",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source_import_joints.json",
        help="Output path for Blender import joints json.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / "168" / "finedance_168_source_validation_report.json",
        help="Output path for validation report.",
    )
    parser.add_argument(
        "--offset-profile",
        default="raw_bridge",
        choices=["raw_bridge", "bridge_truth", "unknown"],
        help="Offset profile used to build the current source BVH.",
    )
    parser.add_argument(
        "--blender",
        help="Optional Blender executable override.",
    )
    parser.add_argument(
        "--blender-timeout-seconds",
        type=int,
        default=600,
        help="Timeout for Blender import roundtrip.",
    )
    return parser.parse_args()


def load_convert_module() -> ModuleType:
    return import_module_from_path(
        "convert_dataset_motion_module",
        REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "convert_dataset_motion.py",
    )


def load_pose_driver_module() -> ModuleType:
    return import_module_from_path(
        "export_dataset_pose_driver_sequences_module",
        REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "export_dataset_pose_driver_sequences.py",
    )


def blender_helper_path() -> Path:
    return REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "blender_dump_layer_poses.py"


def resolve_blender_executable(override: str | None) -> str:
    if override:
        return override
    app_path = Path("/Applications/Blender.app/Contents/MacOS/Blender")
    if app_path.exists():
        return str(app_path)
    blender_path = shutil.which("blender")
    if blender_path:
        return blender_path
    raise FileNotFoundError("Blender executable not found")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def aligned_positions_for_frame_indices(all_positions: Any, frame_indices: list[int]) -> tuple[list[int], Any]:
    payload = all_positions
    if not frame_indices:
        return list(range(int(payload.shape[0]))), payload
    max_requested = max(frame_indices)
    if max_requested < int(payload.shape[0]):
        return list(frame_indices), payload[np.asarray(frame_indices, dtype=np.int64)]
    if int(payload.shape[0]) == len(frame_indices):
        return list(frame_indices), payload
    raise ValueError(
        f"requested frameIndices exceed candidate frame count: max_requested={max_requested} frame_count={int(payload.shape[0])}"
    )


def run_blender_import_roundtrip(
    *,
    blender_exe: str,
    source_bvh_path: Path,
    frame_indices: list[int],
    output_path: Path,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    payload = {
        "inputPath": str(source_bvh_path),
        "outputPath": str(output_path),
        "frameIndices": [int(value) for value in frame_indices],
    }
    with tempfile.NamedTemporaryFile("w", suffix="_blender_dump_payload.json", delete=False, encoding="utf-8") as handle:
        handle.write(json.dumps(payload))
        payload_path = Path(handle.name)
    command = [
        blender_exe,
        "--background",
        "--factory-startup",
        "--python",
        str(blender_helper_path()),
        "--",
        str(payload_path),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=max(1, int(timeout_seconds)),
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        return None, {
            "status": "failed",
            "comparisonMode": "blender_import",
            "errorType": "timeout",
            "message": f"Blender import timed out after {timeout_seconds} seconds.",
            "stdoutTail": (exc.stdout or "")[-4000:],
            "stderrTail": (exc.stderr or "")[-4000:],
        }
    except subprocess.CalledProcessError as exc:
        return None, {
            "status": "failed",
            "comparisonMode": "blender_import",
            "errorType": "called_process_error",
            "returnCode": int(exc.returncode),
            "message": "Blender import helper exited with a non-zero status.",
            "stdoutTail": (exc.stdout or "")[-4000:],
            "stderrTail": (exc.stderr or "")[-4000:],
        }
    finally:
        payload_path.unlink(missing_ok=True)

    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    return output_payload, {
        "status": "ok",
        "comparisonMode": "blender_import",
        "stdoutTail": (result.stdout or "")[-4000:],
        "stderrTail": (result.stderr or "")[-4000:],
    }


def main() -> int:
    args = parse_args()
    convert_module = load_convert_module()
    pose_driver_module = load_pose_driver_module()

    official_payload = json.loads(args.official_joints.expanduser().resolve().read_text(encoding="utf-8"))
    official_positions = np.asarray(official_payload["poses"], dtype=np.float64)
    official_frame_indices = [int(value) for value in official_payload.get("frameIndices", [])]
    official_bone_order = list(official_payload.get("boneOrder", []) or BRIDGE_BONE_ORDER_22)

    all_bvh_names, _all_bvh_frame_indices, all_bvh_positions, bvh_frame_time = extract_bvh_joint_positions(
        args.source_bvh.expanduser().resolve(),
        pose_driver_module.parse_bvh,
        bone_order=official_bone_order,
        basis_matrix=BVH_TO_TARGET_BASIS,
    )
    aligned_frame_indices, fk_positions = aligned_positions_for_frame_indices(all_bvh_positions, official_frame_indices)
    fk_payload = build_pose_payload(
        input_path=str(args.source_bvh.expanduser().resolve()),
        bone_order=all_bvh_names,
        frame_indices=aligned_frame_indices,
        poses=fk_positions,
        include_preview_space=True,
        extra={
            "comparisonMode": "fk_direct",
            "bvhFrameTime": float(bvh_frame_time),
            "bvhFrameIndices": aligned_frame_indices,
        },
    )
    write_json(args.fk_output.expanduser().resolve(), fk_payload)
    fk_summary = compare_pose_payloads(official_payload, fk_payload, convert_module.compare_bridge_positions)
    fk_bone_diagnostics = compute_bone_vector_diagnostics(official_payload, fk_payload)

    blender_meta: dict[str, Any] | None = None
    import_payload: dict[str, Any] | None = None
    import_summary: dict[str, Any] | None = None
    import_bone_diagnostics: dict[str, Any] | None = None
    try:
        blender_exe = resolve_blender_executable(args.blender)
        import_payload, blender_meta = run_blender_import_roundtrip(
            blender_exe=blender_exe,
            source_bvh_path=args.source_bvh.expanduser().resolve(),
            frame_indices=official_frame_indices,
            output_path=args.import_output.expanduser().resolve(),
            timeout_seconds=int(args.blender_timeout_seconds),
        )
    except Exception as exc:  # noqa: BLE001
        blender_meta = {
            "status": "failed",
            "comparisonMode": "blender_import",
            "errorType": type(exc).__name__,
            "message": str(exc),
        }

    if import_payload is not None:
        if "previewSpacePoses" not in import_payload:
            import_payload["previewSpacePoses"] = build_pose_payload(
                input_path=str(args.source_bvh.expanduser().resolve()),
                bone_order=list((import_payload or {}).get("boneOrder", []) or official_bone_order),
                frame_indices=[int(value) for value in (import_payload or {}).get("frameIndices", [])],
                poses=np.asarray((import_payload or {}).get("poses", []), dtype=np.float64),
                include_preview_space=True,
            )["previewSpacePoses"]
            write_json(args.import_output.expanduser().resolve(), import_payload)
        import_summary = compare_pose_payloads(official_payload, import_payload, convert_module.compare_bridge_positions)
        import_bone_diagnostics = compute_bone_vector_diagnostics(official_payload, import_payload)

    fk_per_joint = remap_per_joint_error_names(fk_summary, fk_payload.get("boneOrder", []))
    import_per_joint = remap_per_joint_error_names(import_summary, (import_payload or {}).get("boneOrder", [])) if import_summary is not None else {}

    report_payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "sequenceId": str(args.sequence_id),
        "officialMeshSummaryRef": str(official_payload.get("sourceSummaryRef", "")),
        "officialMeshJointsPath": str(args.official_joints.expanduser().resolve()),
        "bvhPath": str(args.source_bvh.expanduser().resolve()),
        "offsetProfile": str(args.offset_profile),
        "sampleIndices": [int(value) for value in official_frame_indices],
        "comparisonResults": {
            "fk_direct": {
                "status": "ok",
                "comparisonMode": "fk_direct",
                "jointsPath": str(args.fk_output.expanduser().resolve()),
                "frameCountCompared": int(fk_summary.get("frameCountCompared", 0) or 0),
                "jointCountCompared": int(fk_summary.get("jointCountCompared", 0) or 0),
                "jointPositionError": dict(fk_summary.get("jointPositionError", {}) or {}),
                "perJointPositionError": fk_per_joint,
                "perJointPositionErrorByIndex": dict(fk_summary.get("perJointPositionError", {}) or {}),
                "boneVectorDiagnostics": fk_bone_diagnostics,
                "frameTime": float(bvh_frame_time),
                "boneOrder": list(fk_payload.get("boneOrder", [])),
            },
            "blender_import": (
                {
                    "status": "ok",
                    "comparisonMode": "blender_import",
                    "jointsPath": str(args.import_output.expanduser().resolve()),
                    "frameCountCompared": int(import_summary.get("frameCountCompared", 0) or 0),
                    "jointCountCompared": int(import_summary.get("jointCountCompared", 0) or 0),
                    "jointPositionError": dict(import_summary.get("jointPositionError", {}) or {}),
                    "perJointPositionError": import_per_joint,
                    "perJointPositionErrorByIndex": dict(import_summary.get("perJointPositionError", {}) or {}),
                    "boneVectorDiagnostics": import_bone_diagnostics,
                    "boneOrder": list((import_payload or {}).get("boneOrder", [])),
                    "sceneFrameIndices": list((import_payload or {}).get("sceneFrameIndices", [])),
                    "actionFrameRange": list((import_payload or {}).get("actionFrameRange", [])),
                    "stdoutTail": str((blender_meta or {}).get("stdoutTail", "")),
                    "stderrTail": str((blender_meta or {}).get("stderrTail", "")),
                }
                if import_payload is not None and import_summary is not None
                else {
                    **dict(blender_meta or {}),
                    "jointsPath": str(args.import_output.expanduser().resolve()),
                }
            ),
        },
        "diagnosis": diagnose_validation(fk_summary, import_summary),
    }
    write_json(args.report_output.expanduser().resolve(), report_payload)
    print(args.report_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
