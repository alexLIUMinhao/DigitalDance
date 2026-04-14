#!/usr/bin/env python3
"""Batch-export FineDance raw motions to an external smpl2bvh BVH dataset."""

from __future__ import annotations

import argparse
import json
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
    import_module_from_path,
    load_finedance_smpl_compat_clip,
    run_external_smpl2bvh,
    write_smpl_compat_npz,
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    default_input_dir = REPO_ROOT / "motion-base-assets" / "datasets" / "finedance" / "raw" / "extracted" / "finedance" / "motion"
    default_output_dir = REPO_ROOT / "motion-base-assets" / "datasets" / "finedance" / "external_smpl2bvh_dataset"
    default_work_dir = REPO_ROOT / "music-motion-lab" / "outputs" / "reports" / "finedance_external_smpl2bvh_inputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_input_dir)
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--work-dir", type=Path, default=default_work_dir)
    parser.add_argument(
        "--smpl2bvh-repo",
        type=Path,
        default=REPO_ROOT / "music-motion-lab" / ".external" / "smpl2bvh",
    )
    parser.add_argument(
        "--smpl-model-root",
        type=Path,
        default=REPO_ROOT / "motion-base-assets" / "models",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap for smoke testing.")
    return parser.parse_args()


def load_convert_module() -> ModuleType:
    return import_module_from_path(
        "convert_dataset_motion_module_for_external_dataset_batch",
        REPO_ROOT / "3d-digital-human" / "tools" / "motion_base" / "convert_dataset_motion.py",
    )


def enumerate_motion_files(input_dir: Path) -> list[Path]:
    return sorted(path for path in input_dir.glob("*.npy") if path.is_file())


def build_manifest_payload(
    *,
    input_dir: Path,
    output_dir: Path,
    bvh_dir: Path,
    sequence_records: list[dict[str, Any]],
    fps: int,
    force: bool,
) -> dict[str, Any]:
    completed = sum(1 for record in sequence_records if record.get("status") == "completed")
    skipped = sum(1 for record in sequence_records if record.get("status") == "skipped_existing")
    failed = sum(1 for record in sequence_records if record.get("status") == "failed")
    return {
        "schemaVersion": 1,
        "generatedAtUtc": utc_now_iso(),
        "datasetName": "FineDance external smpl2bvh BVH dataset",
        "inputDir": str(input_dir),
        "outputDir": str(output_dir),
        "bvhDir": str(bvh_dir),
        "generator": "KosukeFukazawa/smpl2bvh",
        "requestedModelType": "smplx",
        "requestedGender": "NEUTRAL",
        "actualModelType": "smpl",
        "actualGender": "MALE",
        "fps": int(fps),
        "forceRegenerate": bool(force),
        "sequenceCount": len(sequence_records),
        "statusSummary": {
            "completed": completed,
            "skippedExisting": skipped,
            "failed": failed,
        },
        "sequences": sequence_records,
        "notes": [
            "This dataset keeps the external smpl2bvh route as a reproducible baseline.",
            "The upstream smpl2bvh SMPL-X mode is currently unstable on this payload, so generation falls back to SMPL/MALE.",
            "Sequence ids are derived from the raw FineDance motion filenames that actually exist on disk.",
        ],
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    work_dir = args.work_dir.expanduser().resolve()
    bvh_dir = output_dir / "bvh"
    manifest_path = output_dir / "manifest.json"

    convert_module = load_convert_module()
    motion_files = enumerate_motion_files(input_dir)
    if args.limit > 0:
        motion_files = motion_files[: int(args.limit)]
    if not motion_files:
        raise FileNotFoundError(f"no motion .npy files found in {input_dir}")

    work_dir.mkdir(parents=True, exist_ok=True)
    bvh_dir.mkdir(parents=True, exist_ok=True)

    sequence_records: list[dict[str, Any]] = []
    for index, raw_motion_path in enumerate(motion_files, start=1):
        sequence_id = raw_motion_path.stem
        output_bvh = bvh_dir / f"finedance_{sequence_id}_external_smpl2bvh_source.bvh"
        work_npz = work_dir / f"finedance_{sequence_id}_smpl_compat_input.npz"
        record: dict[str, Any] = {
            "sequenceId": sequence_id,
            "index": index,
            "rawMotionPath": str(raw_motion_path),
            "outputBvhPath": str(output_bvh),
        }
        try:
            if output_bvh.exists() and not args.force:
                record["status"] = "skipped_existing"
                record["frameCount"] = None
                print(f"[{index:03d}/{len(motion_files):03d}] skip {sequence_id}")
                continue

            clip = load_finedance_smpl_compat_clip(raw_motion_path, convert_module)
            write_smpl_compat_npz(work_npz, clip)
            try:
                run_external_smpl2bvh(
                    repo_root=args.smpl2bvh_repo.expanduser().resolve(),
                    poses_path=work_npz,
                    output_bvh=output_bvh,
                    model_path=args.smpl_model_root.expanduser().resolve(),
                    fps=int(args.fps),
                    model_type="smplx",
                    gender="NEUTRAL",
                )
                record["modelType"] = "smplx"
                record["gender"] = "NEUTRAL"
                record["fallbackApplied"] = False
            except IndexError as exc:
                run_external_smpl2bvh(
                    repo_root=args.smpl2bvh_repo.expanduser().resolve(),
                    poses_path=work_npz,
                    output_bvh=output_bvh,
                    model_path=args.smpl_model_root.expanduser().resolve(),
                    fps=int(args.fps),
                    model_type="smpl",
                    gender="MALE",
                )
                record["modelType"] = "smpl"
                record["gender"] = "MALE"
                record["fallbackApplied"] = True
                record["fallbackReason"] = (
                    "upstream smpl2bvh smplx mode raises IndexError on this payload because it slices rest joints "
                    "to 24 while still indexing the full SMPL-X parent array"
                )
                record["fallbackError"] = str(exc)

            record["status"] = "completed"
            record["frameCount"] = int(clip.frame_count)
            record["featureDims"] = int(clip.feature_dims)
            print(f"[{index:03d}/{len(motion_files):03d}] done {sequence_id}")
        except Exception as exc:  # noqa: BLE001
            record["status"] = "failed"
            record["error"] = str(exc)
            print(f"[{index:03d}/{len(motion_files):03d}] fail {sequence_id}: {exc}", file=sys.stderr)
        finally:
            sequence_records.append(record)

    manifest = build_manifest_payload(
        input_dir=input_dir,
        output_dir=output_dir,
        bvh_dir=bvh_dir,
        sequence_records=sequence_records,
        fps=int(args.fps),
        force=bool(args.force),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0 if manifest["statusSummary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
