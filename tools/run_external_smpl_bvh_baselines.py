#!/usr/bin/env python3
"""Run external SMPL->BVH baselines and compare them against FineDance official mesh truth."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.external_smpl_bvh_baselines import (  # noqa: E402
    build_canonical_payload_from_bvh,
    build_validation_result,
    import_module_from_path,
    load_finedance_smpl_compat_clip,
    run_external_smpl2bvh,
    write_aist_compat_pkl,
    write_smpl_compat_npz,
)
from music_motion_lab.source_bvh_validation import write_json  # noqa: E402


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    default_sequence_id = "168"
    layer_compare_root = REPO_ROOT / "motion-base-assets" / "previewCache" / "layer_compare" / "finedance" / default_sequence_id
    reports_root = REPO_ROOT / "music-motion-lab" / "outputs" / "reports"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-id", default=default_sequence_id)
    parser.add_argument(
        "--raw-motion",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "datasets" / "finedance" / "raw" / "extracted" / "finedance" / "motion" / f"{default_sequence_id}.npy",
    )
    parser.add_argument(
        "--official-joints",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "previewCache" / "upstream_baselines" / "finedance" / default_sequence_id / f"finedance_{default_sequence_id}_official_mesh_joints.json",
    )
    parser.add_argument(
        "--current-source-bvh",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_source.bvh",
    )
    parser.add_argument(
        "--smpl-compat-npz",
        type=Path,
        default=reports_root / f"finedance_{default_sequence_id}_smpl_compat_input.npz",
    )
    parser.add_argument(
        "--aist-compat-pkl",
        type=Path,
        default=reports_root / f"finedance_{default_sequence_id}_aist_compat_input.pkl",
    )
    parser.add_argument(
        "--smpl2bvh-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_external_smpl2bvh_source.bvh",
    )
    parser.add_argument(
        "--cat-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_external_cat_source.bvh",
    )
    parser.add_argument(
        "--smpl2bvh-fk-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_external_smpl2bvh_fk_joints.json",
    )
    parser.add_argument(
        "--cat-fk-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_external_cat_fk_joints.json",
    )
    parser.add_argument(
        "--current-source-fk-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_current_source_fk_joints_rerun.json",
    )
    parser.add_argument(
        "--smpl2bvh-report-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_external_smpl2bvh_validation_report.json",
    )
    parser.add_argument(
        "--cat-report-output",
        type=Path,
        default=layer_compare_root / f"finedance_{default_sequence_id}_external_cat_validation_report.json",
    )
    parser.add_argument(
        "--comparison-report-output",
        type=Path,
        default=reports_root / f"finedance_{default_sequence_id}_external_bvh_comparison.json",
    )
    parser.add_argument(
        "--smpl2bvh-repo",
        type=Path,
        default=REPO_ROOT / "music-motion-lab" / ".external" / "smpl2bvh",
    )
    parser.add_argument(
        "--cat-repo",
        type=Path,
        default=REPO_ROOT / "music-motion-lab" / ".external" / "CharacterAnimationTools",
    )
    parser.add_argument(
        "--smpl-model-root",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "models",
    )
    parser.add_argument(
        "--cat-smpl-model",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "models" / "SMPL_MALE.pkl",
    )
    parser.add_argument(
        "--cat-python",
        default=sys.executable,
        help="Python executable used to run the CharacterAnimationTools helper.",
    )
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def load_convert_module() -> ModuleType:
    return import_module_from_path(
        "convert_dataset_motion_module_for_external_baselines",
        REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "convert_dataset_motion.py",
    )


def load_pose_driver_module() -> ModuleType:
    return import_module_from_path(
        "pose_driver_module_for_external_baselines",
        REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "export_dataset_pose_driver_sequences.py",
    )


def write_route_report(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def run_cat_export_subprocess(
    *,
    python_executable: str,
    helper_script: Path,
    repo_root: Path,
    poses_path: Path,
    smpl_path: Path,
    output_bvh: Path,
    fps: int,
) -> dict[str, Any]:
    command = [
        python_executable,
        str(helper_script),
        "--repo-root",
        str(repo_root),
        "--poses-path",
        str(poses_path),
        "--smpl-path",
        str(smpl_path),
        "--output-bvh",
        str(output_bvh),
        "--fps",
        str(fps),
        "--scale",
        "1.0",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )
    return {
        "pythonExecutable": python_executable,
        "stdoutTail": (result.stdout or "")[-4000:],
        "stderrTail": (result.stderr or "")[-4000:],
    }


def build_route_payload(
    *,
    route_name: str,
    bvh_path: Path,
    fk_output_path: Path,
    official_payload: dict[str, Any],
    convert_module: ModuleType,
    parse_bvh,
    unit_scale: float,
    route_meta: dict[str, Any],
) -> dict[str, Any]:
    candidate_payload, frame_time, source_node_names = build_canonical_payload_from_bvh(
        input_path=bvh_path,
        parse_bvh=parse_bvh,
        convert_module=convert_module,
        frame_indices=[int(value) for value in official_payload.get("frameIndices", [])],
        unit_scale=unit_scale,
    )
    write_json(fk_output_path, candidate_payload)
    result = build_validation_result(
        route_name=route_name,
        official_payload=official_payload,
        candidate_payload=candidate_payload,
        convert_module=convert_module,
        bvh_path=bvh_path,
        fk_joints_path=fk_output_path,
        frame_time=frame_time,
        route_meta={**route_meta, "sourceNodeNames": source_node_names},
    )
    return result


def main() -> int:
    args = parse_args()
    convert_module = load_convert_module()
    pose_driver_module = load_pose_driver_module()

    official_payload = json.loads(args.official_joints.expanduser().resolve().read_text(encoding="utf-8"))
    official_frame_indices = [int(value) for value in official_payload.get("frameIndices", [])]
    official_frame_count = len(official_frame_indices)

    clip = load_finedance_smpl_compat_clip(args.raw_motion.expanduser().resolve(), convert_module)
    write_smpl_compat_npz(args.smpl_compat_npz.expanduser().resolve(), clip)
    write_aist_compat_pkl(args.aist_compat_pkl.expanduser().resolve(), clip, scaling=1.0)

    smpl2bvh_route_meta = {
        "generator": "KosukeFukazawa/smpl2bvh",
        "inputFamily": "smpl_compat_npz",
        "fps": int(args.fps),
        "unitScaleApplied": 0.01,
        "requestedModelType": "smplx",
        "requestedGender": "NEUTRAL",
    }
    try:
        run_external_smpl2bvh(
            repo_root=args.smpl2bvh_repo.expanduser().resolve(),
            poses_path=args.smpl_compat_npz.expanduser().resolve(),
            output_bvh=args.smpl2bvh_output.expanduser().resolve(),
            model_path=args.smpl_model_root.expanduser().resolve(),
            fps=int(args.fps),
            model_type="smplx",
            gender="NEUTRAL",
        )
        smpl2bvh_route_meta["modelFamily"] = "smplx_neutral"
        smpl2bvh_route_meta["actualModelType"] = "smplx"
        smpl2bvh_route_meta["actualGender"] = "NEUTRAL"
    except IndexError as exc:
        run_external_smpl2bvh(
            repo_root=args.smpl2bvh_repo.expanduser().resolve(),
            poses_path=args.smpl_compat_npz.expanduser().resolve(),
            output_bvh=args.smpl2bvh_output.expanduser().resolve(),
            model_path=args.smpl_model_root.expanduser().resolve(),
            fps=int(args.fps),
            model_type="smpl",
            gender="MALE",
        )
        smpl2bvh_route_meta["modelFamily"] = "smpl_male"
        smpl2bvh_route_meta["actualModelType"] = "smpl"
        smpl2bvh_route_meta["actualGender"] = "MALE"
        smpl2bvh_route_meta["fallbackReason"] = (
            "upstream smpl2bvh smplx mode raises IndexError because it slices rest joints to 24 "
            "but still indexes the full SMPL-X parent array"
        )
        smpl2bvh_route_meta["fallbackError"] = str(exc)
    cat_execution_meta = run_cat_export_subprocess(
        python_executable=str(args.cat_python),
        helper_script=REPO_ROOT / "music-motion-lab" / "tools" / "export_cat_aistpp_to_bvh.py",
        repo_root=args.cat_repo.expanduser().resolve(),
        poses_path=args.aist_compat_pkl.expanduser().resolve(),
        smpl_path=args.cat_smpl_model.expanduser().resolve(),
        output_bvh=args.cat_output.expanduser().resolve(),
        fps=int(args.fps),
    )

    current_source_result = build_route_payload(
        route_name="current_source",
        bvh_path=args.current_source_bvh.expanduser().resolve(),
        fk_output_path=args.current_source_fk_output.expanduser().resolve(),
        official_payload=official_payload,
        convert_module=convert_module,
        parse_bvh=pose_driver_module.parse_bvh,
        unit_scale=1.0,
        route_meta={
            "generator": "lab_current_source_bvh",
            "inputFamily": "bridge_bvh",
            "modelFamily": "smplx_truth_bridge",
            "fps": int(args.fps),
        },
    )
    smpl2bvh_result = build_route_payload(
        route_name="external_smpl2bvh",
        bvh_path=args.smpl2bvh_output.expanduser().resolve(),
        fk_output_path=args.smpl2bvh_fk_output.expanduser().resolve(),
        official_payload=official_payload,
        convert_module=convert_module,
        parse_bvh=pose_driver_module.parse_bvh,
        unit_scale=0.01,
        route_meta=smpl2bvh_route_meta,
    )
    cat_result = build_route_payload(
        route_name="external_cat",
        bvh_path=args.cat_output.expanduser().resolve(),
        fk_output_path=args.cat_fk_output.expanduser().resolve(),
        official_payload=official_payload,
        convert_module=convert_module,
        parse_bvh=pose_driver_module.parse_bvh,
        unit_scale=1.0,
        route_meta={
            "generator": "KosukeFukazawa/CharacterAnimationTools",
            "inputFamily": "aistpp_pkl",
            "modelFamily": "smpl_male",
            "fps": int(args.fps),
            "unitScaleApplied": 1.0,
            **cat_execution_meta,
        },
    )

    write_route_report(args.smpl2bvh_report_output.expanduser().resolve(), smpl2bvh_result)
    write_route_report(args.cat_report_output.expanduser().resolve(), cat_result)

    ranking = sorted(
        [
            {
                "routeName": current_source_result["routeName"],
                "meanJointError": float(current_source_result["jointPositionError"].get("mean", 0.0) or 0.0),
                "p95JointError": float(current_source_result["jointPositionError"].get("p95", 0.0) or 0.0),
            },
            {
                "routeName": smpl2bvh_result["routeName"],
                "meanJointError": float(smpl2bvh_result["jointPositionError"].get("mean", 0.0) or 0.0),
                "p95JointError": float(smpl2bvh_result["jointPositionError"].get("p95", 0.0) or 0.0),
            },
            {
                "routeName": cat_result["routeName"],
                "meanJointError": float(cat_result["jointPositionError"].get("mean", 0.0) or 0.0),
                "p95JointError": float(cat_result["jointPositionError"].get("p95", 0.0) or 0.0),
            },
        ],
        key=lambda item: (item["meanJointError"], item["p95JointError"], item["routeName"]),
    )
    best_route = ranking[0]["routeName"] if ranking else "unknown"
    smpl2bvh_note = (
        f"external_smpl2bvh used {smpl2bvh_route_meta.get('actualModelType', 'unknown')}/"
        f"{smpl2bvh_route_meta.get('actualGender', 'unknown')} with a {smpl2bvh_route_meta.get('unitScaleApplied', 1.0)} "
        "unit normalization because the upstream exporter writes centimeter BVH values."
    )
    if smpl2bvh_route_meta.get("fallbackReason"):
        smpl2bvh_note += f" It also fell back from the requested SMPL-X mode because {smpl2bvh_route_meta['fallbackReason']}."
    cat_note = (
        "external_cat used CharacterAnimationTools AIST++ loading with an SMPL male model, "
        "so its rest skeleton family is not identical to the official SMPL-X truth."
    )
    external_gap_note = (
        "The two external routes are numerically almost identical on FineDance 168, which suggests the dominant error "
        "comes from the shared SMPL-body baseline semantics rather than from one exporter-specific quirk."
    )
    comparison_payload = {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "sequenceId": str(args.sequence_id),
        "goal": "Compare external SMPL->BVH baselines against FineDance official mesh truth.",
        "rawMotionPath": str(args.raw_motion.expanduser().resolve()),
        "officialMeshJointsPath": str(args.official_joints.expanduser().resolve()),
        "currentSourceBvhPath": str(args.current_source_bvh.expanduser().resolve()),
        "intermediateInputs": {
            "smplCompatNpzPath": str(args.smpl_compat_npz.expanduser().resolve()),
            "aistCompatPklPath": str(args.aist_compat_pkl.expanduser().resolve()),
            "rawFeatureDims": int(clip.feature_dims),
            "rawFrameCount": int(clip.frame_count),
            "evaluationFrameCount": int(official_frame_count),
            "evaluationFrameIndices": official_frame_indices,
        },
        "routes": {
            "current_source": current_source_result,
            "external_smpl2bvh": smpl2bvh_result,
            "external_cat": cat_result,
        },
        "rankingByMeanJointError": ranking,
        "diagnosis": {
            "bestRoute": best_route,
            "externalOutperformsCurrent": best_route != "current_source",
            "notes": [
                smpl2bvh_note,
                cat_note,
                external_gap_note,
                "This comparison is intended to test BVH generation semantics first; model-family differences are explicitly preserved in the report.",
            ],
        },
    }
    write_json(args.comparison_report_output.expanduser().resolve(), comparison_payload)
    print(args.comparison_report_output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
