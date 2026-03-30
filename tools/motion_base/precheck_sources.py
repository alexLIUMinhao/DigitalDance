#!/usr/bin/env python3
"""Run technical prechecks over source video and raw motion inputs."""

from __future__ import annotations

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List

from common import (
    INTAKE_QUEUE_PATH,
    VIDEO_EXTENSIONS,
    command_exists,
    load_intake_queue,
    load_taxonomy,
    resolve_asset_path,
    write_json,
    utc_now_iso,
)

FPS_EPSILON = 0.25
DURATION_EPSILON_SEC = 0.25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INTAKE_QUEUE_PATH, help="Input intake queue.")
    parser.add_argument("--output", type=Path, default=INTAKE_QUEUE_PATH, help="Output intake queue.")
    parser.add_argument("--all", action="store_true", help="Recheck jobs even if they have already passed precheck.")
    return parser.parse_args()


def parse_fps(value: str) -> float:
    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except Exception:
        return 0.0


def ffprobe_metadata(path: Path) -> Dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    video_stream = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), {})
    return {
        "width": int(video_stream.get("width", 0) or 0),
        "height": int(video_stream.get("height", 0) or 0),
        "fps": parse_fps(str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0")),
        "durationSec": float(payload.get("format", {}).get("duration", 0.0) or 0.0),
        "formatName": str(payload.get("format", {}).get("format_name", "")),
    }


def imageio_ffmpeg_metadata(path: Path) -> Dict[str, Any]:
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        meta = next(reader)
    finally:
        reader.close()

    width, height = meta.get("source_size", meta.get("size", (0, 0)))
    return {
        "width": int(width or 0),
        "height": int(height or 0),
        "fps": float(meta.get("fps", 0.0) or 0.0),
        "durationSec": float(meta.get("duration", 0.0) or 0.0),
        "formatName": str(path.suffix.lower().lstrip(".")),
    }


def check_video(path: Path, gates: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"path": str(path), "checkedAtUtc": utc_now_iso(), "status": "hold", "issues": [], "warnings": []}
    try:
        if command_exists("ffprobe"):
            metadata = ffprobe_metadata(path)
        else:
            metadata = imageio_ffmpeg_metadata(path)
    except ImportError:
        summary["issues"].append("ffprobe_not_installed")
        return summary
    except (subprocess.CalledProcessError, OSError, RuntimeError, StopIteration, ValueError) as exc:
        summary["issues"].append(f"video_probe_failed:{type(exc).__name__}")
        return summary

    summary.update(metadata)
    min_width, min_height = gates["minimumResolution"]
    preferred_width, preferred_height = gates["preferredResolution"]

    if metadata["width"] < min_width or metadata["height"] < min_height:
        summary["issues"].append("resolution_below_minimum")
    elif metadata["width"] < preferred_width or metadata["height"] < preferred_height:
        summary["warnings"].append("resolution_below_preferred")

    minimum_fps = float(gates["minimumFps"])
    preferred_fps = float(gates["preferredFps"])
    if metadata["fps"] + FPS_EPSILON < minimum_fps:
        summary["issues"].append("fps_below_minimum")
    elif metadata["fps"] + FPS_EPSILON < preferred_fps:
        summary["warnings"].append("fps_below_preferred")

    duration = metadata["durationSec"]
    minimum_duration = float(gates["minimumDurationSec"])
    maximum_duration = float(gates["maximumDurationSec"])
    if duration + DURATION_EPSILON_SEC < minimum_duration or duration - DURATION_EPSILON_SEC > maximum_duration:
        summary["issues"].append("duration_outside_target_range")

    summary["manualChecks"] = [
        "single_person_visible",
        "full_body_visible",
        "fixed_camera",
        "no_hard_cuts",
        "clean_background",
        "low_occlusion",
    ]
    summary["status"] = "pass" if not summary["issues"] else "hold"
    return summary


def check_motion_file(path: Path) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"path": str(path), "checkedAtUtc": utc_now_iso(), "status": "pass", "issues": [], "warnings": []}
    if not path.exists():
        summary["issues"].append("missing_source_file")
        summary["status"] = "hold"
        return summary
    summary["sizeBytes"] = path.stat().st_size
    if path.suffix.lower() not in {".fbx", ".bvh"}:
        summary["issues"].append("unsupported_motion_extension")
        summary["status"] = "hold"
    return summary


def main() -> int:
    args = parse_args()
    taxonomy = load_taxonomy()
    gates = taxonomy["videoQualityGates"]
    queue = load_intake_queue() if not args.input.exists() else json.loads(args.input.read_text(encoding="utf-8"))
    updated = 0

    for job in queue.get("jobs", []):
        if not args.all and job.get("stage") not in {"discovered", "hold"}:
            continue

        source_kind = job.get("sourceAssetKind")
        source_path = resolve_asset_path("sourceVideo", job["sourceAssetRelPath"]) if source_kind == "video" else resolve_asset_path("rawFbx", job["sourceAssetRelPath"])
        if source_path is None:
            continue

        summary = check_video(source_path, gates) if source_kind == "video" else check_motion_file(source_path)
        job["precheckSummary"] = summary
        job["updatedAtUtc"] = utc_now_iso()

        if summary["status"] == "pass":
            job["stage"] = "precheck_passed"
        else:
            job["stage"] = "hold"
        updated += 1

    if args.output:
        write_json(args.output, queue)

    print(f"prechecked jobs={updated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
