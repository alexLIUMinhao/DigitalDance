#!/usr/bin/env python3
"""Export official mesh-preview joint truth for a FineDance sequence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "music-motion-lab" / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.source_bvh_validation import (  # noqa: E402
    BRIDGE_BONE_ORDER_22,
    build_pose_payload,
    canonicalize_preview_positions,
    round_nested_array,
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
    parser.add_argument("--dataset", default="finedance", choices=["finedance"], help="Dataset name.")
    parser.add_argument("--sequence-id", default="168", help="Sequence id.")
    parser.add_argument("--catalog", type=Path, help="Optional dataset catalog override.")
    parser.add_argument("--summary", type=Path, help="Optional official mesh summary override.")
    parser.add_argument("--output", type=Path, help="Optional output json path.")
    return parser.parse_args()


def load_render_module() -> ModuleType:
    module_path = REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "render_official_mesh_preview.py"
    return import_module_from_path("render_official_mesh_preview_module", module_path)


def default_summary_path(render_module: ModuleType, entry: dict, dataset: str, sequence_id: str) -> Path:
    stem = f"{render_module.slugify(dataset)}_{render_module.slugify(sequence_id)}"
    return render_module.official_output_dir(entry) / f"{stem}_official_mesh_summary.json"


def default_output_path(render_module: ModuleType, entry: dict, dataset: str, sequence_id: str) -> Path:
    stem = f"{render_module.slugify(dataset)}_{render_module.slugify(sequence_id)}"
    return render_module.official_output_dir(entry) / f"{stem}_official_mesh_joints.json"


def main() -> int:
    args = parse_args()
    render_module = load_render_module()
    catalog_path = args.catalog if args.catalog is not None else render_module.DATASET_CATALOG_PATH
    entry = render_module.load_entry(catalog_path, args.dataset, args.sequence_id)
    summary_path = args.summary.expanduser().resolve() if args.summary is not None else default_summary_path(render_module, entry, args.dataset, args.sequence_id).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"official mesh summary not found: {summary_path}")

    output_path = args.output.expanduser().resolve() if args.output is not None else default_output_path(render_module, entry, args.dataset, args.sequence_id).resolve()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    vertices, joints, _faces = render_module.load_finedance_mesh(entry)
    sample_indices = render_module.choose_sample_indices(
        len(vertices),
        source_fps=int(summary.get("sourceFps", 30) or 30),
        preview_fps=int(summary.get("fps", 30) or 30),
        start_seconds=float(summary.get("startSeconds", 0.0) or 0.0),
        max_seconds=float(summary.get("maxSeconds", 0.0) or 0.0),
        max_frames=(int(summary["maxFrames"]) if summary.get("maxFrames") is not None else None),
    )
    sampled_joints = joints[sample_indices, : len(BRIDGE_BONE_ORDER_22), :]
    payload = build_pose_payload(
        input_path=str(render_module.raw_motion_path(entry)),
        bone_order=list(BRIDGE_BONE_ORDER_22),
        frame_indices=[int(value) for value in sample_indices.tolist()],
        poses=sampled_joints,
        extra={
            "datasetName": args.dataset,
            "sequenceId": str(args.sequence_id),
            "sourceSummaryRef": str(summary_path),
            "sourceVideoPath": str(((summary.get("artifacts") or {}).get("meshPreviewMp4", ""))),
            "sourceFps": int(summary.get("sourceFps", 30) or 30),
            "previewFps": int(summary.get("fps", 30) or 30),
            "startSeconds": float(summary.get("startSeconds", 0.0) or 0.0),
            "maxSeconds": float(summary.get("maxSeconds", 0.0) or 0.0),
            "maxFrames": (int(summary["maxFrames"]) if summary.get("maxFrames") is not None else None),
            "sampledFrameCount": int(len(sample_indices)),
            "previewSpacePoses": round_nested_array(canonicalize_preview_positions(sampled_joints)),
        },
    )
    write_json(output_path, payload)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
