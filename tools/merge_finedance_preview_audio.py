#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge FineDance preview mp4 with matching wav audio.")
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=Path(
            "/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/previewCache/upstream_baselines/finedance"
        ),
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=Path("/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/datasets/finedance/raw/extracted/finedance/music_wav"),
    )
    parser.add_argument("--ffmpeg-exe", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--sequence-id",
        action="append",
        default=[],
        help="Optional sequence id to merge. Repeat this flag to merge a subset only.",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        default=0.0,
        help="Optional fixed output duration. Use 0 to keep strict sync with source preview video length.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path(
            "/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/outputs/reports/finedance_preview_audio_merge_report.json"
        ),
    )
    return parser.parse_args()


def merge_one(
    ffmpeg_exe: Path,
    preview_mp4: Path,
    wav_path: Path,
    output_mp4: Path,
    target_seconds: float,
) -> tuple[str, str]:
    temp_path = output_mp4.with_suffix(".tmp.mp4")
    if float(target_seconds) > 0:
        freeze_seconds = max(float(target_seconds), 1.0)
        command = [
            str(ffmpeg_exe),
            "-y",
            "-i",
            str(preview_mp4),
            "-i",
            str(wav_path),
            "-filter:v",
            f"tpad=stop_mode=clone:stop_duration={freeze_seconds:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-t",
            f"{target_seconds:.3f}",
            str(temp_path),
        ]
    else:
        command = [
            str(ffmpeg_exe),
            "-y",
            "-i",
            str(preview_mp4),
            "-i",
            str(wav_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(temp_path),
        ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        return "failed", completed.stderr[-1200:]
    temp_path.replace(output_mp4)
    return "merged", ""


def main() -> int:
    args = parse_args()
    preview_root = args.preview_root.resolve()
    audio_root = args.audio_root.resolve()
    ffmpeg_exe = args.ffmpeg_exe.resolve()
    report_path = args.report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    selected_sequence_ids = {str(item).strip() for item in args.sequence_id if str(item).strip()}

    jobs: list[tuple[str, Path, Path, Path]] = []
    for seq_dir in sorted(path for path in preview_root.iterdir() if path.is_dir()):
        sequence_id = seq_dir.name
        if selected_sequence_ids and sequence_id not in selected_sequence_ids:
            continue
        preview_mp4 = seq_dir / f"finedance_{sequence_id}_official_mesh_preview.mp4"
        if not preview_mp4.exists():
            continue
        wav_path = audio_root / f"{sequence_id}.wav"
        output_mp4 = seq_dir / f"finedance_{sequence_id}_official_mesh_preview_with_audio.mp4"
        jobs.append((sequence_id, preview_mp4, wav_path, output_mp4))

    merged: list[str] = []
    skipped: list[str] = []
    missing_audio: list[str] = []
    failed: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {}
        for sequence_id, preview_mp4, wav_path, output_mp4 in jobs:
            if not wav_path.exists():
                missing_audio.append(sequence_id)
                continue
            if output_mp4.exists() and not args.force:
                skipped.append(sequence_id)
                continue
            futures[
                executor.submit(
                    merge_one,
                    ffmpeg_exe,
                    preview_mp4,
                    wav_path,
                    output_mp4,
                    float(args.target_seconds),
                )
            ] = sequence_id

        for future in as_completed(futures):
            sequence_id = futures[future]
            try:
                status, err = future.result()
            except Exception as exc:  # noqa: BLE001
                failed[sequence_id] = str(exc)
                continue
            if status == "merged":
                merged.append(sequence_id)
            else:
                failed[sequence_id] = err

    payload = {
        "previewRoot": str(preview_root),
        "audioRoot": str(audio_root),
        "selectedSequenceIds": sorted(selected_sequence_ids),
        "targetSeconds": float(args.target_seconds),
        "totalJobs": len(jobs),
        "mergedCount": len(merged),
        "skippedCount": len(skipped),
        "missingAudioCount": len(missing_audio),
        "failedCount": len(failed),
        "merged": sorted(merged),
        "skipped": sorted(skipped),
        "missingAudio": sorted(missing_audio),
        "failed": failed,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(report_path)
    print(
        f"merged={len(merged)} skipped={len(skipped)} missing_audio={len(missing_audio)} failed={len(failed)} total_jobs={len(jobs)}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
