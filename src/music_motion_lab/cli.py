from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

from .config import load_app_config
from .path_policy import ensure_output_path, resolve_input_path
from .utils import load_json, slugify, write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="music-motion-lab CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze-song", help="Generate song_event_map.json from a local audio file.")
    analyze.add_argument("--input", required=True, help="Audio file path. Shared music root is read-only.")
    analyze.add_argument("--song-id", required=True, help="Stable song identifier.")
    analyze.add_argument("--beats-per-bar", type=int, default=4)
    analyze.add_argument("--overrides", help="Optional manual override JSON inside this project or another allowed read-only root.")
    analyze.add_argument("--output", help="Optional output path inside outputs/.")

    library = subparsers.add_parser("build-motion-library", help="Build a base-action motion_unit_library.json from shared diagnostics.")
    library.add_argument("--output", help="Optional output path inside outputs/.")
    library.add_argument("--summary-output", help="Optional showcase summary path inside outputs/.")
    library.add_argument("--fps", type=float, default=30.0)
    library.add_argument("--unit-beats", type=float, default=4.0)
    library.add_argument("--stride-beats", type=float, default=1.0)
    library.add_argument("--assumed-bpm", type=float, default=120.0)

    plan = subparsers.add_parser("plan-choreography", help="Build choreography_plan.json from song and motion contracts.")
    plan.add_argument("--song-event-map", required=True)
    plan.add_argument("--motion-library", required=True)
    plan.add_argument("--output", help="Optional output path inside outputs/.")

    rhythmic_plan = subparsers.add_parser(
        "plan-rhythmic-choreography",
        help="Build a beat-locked choreography_plan.json from a song map and FineDance rhythmic SMPL-X library.",
    )
    rhythmic_plan.add_argument("--song-event-map", required=True)
    rhythmic_plan.add_argument("--motion-library", required=True)
    rhythmic_plan.add_argument("--output", help="Optional output path inside outputs/.")
    rhythmic_plan.add_argument("--beam-width", type=int, default=4)
    rhythmic_plan.add_argument("--candidate-limit", type=int, default=18)
    rhythmic_plan.add_argument("--max-speed-adjustment", type=float, default=0.10)

    preview = subparsers.add_parser("build-preview", help="Create preview_job.json and retarget_report.json.")
    preview.add_argument("--plan", required=True)
    preview.add_argument("--motion-library", required=True)
    preview.add_argument("--avatar-id", default="reference_avatar")
    preview.add_argument("--blender-path", help="Optional explicit Blender executable path. Overrides config and PATH lookup.")
    preview.add_argument("--preview-output", help="Optional preview job output path inside outputs/.")
    preview.add_argument("--report-output", help="Optional retarget report output path inside outputs/.")

    render = subparsers.add_parser("render-preview", help="Render a simple Blender preview video from preview_job.json.")
    render.add_argument("--preview-job", required=True)
    render.add_argument("--blender-path", help="Optional explicit Blender executable path. Overrides config and PATH lookup.")
    render.add_argument("--output-video", help="Optional preview video path inside outputs/.")
    render.add_argument("--output-blend", help="Optional .blend path inside outputs/.")
    render.add_argument("--max-steps", type=int, default=0, help="How many plan steps to render. Use 0 for all steps.")
    render.add_argument("--frame-stride", type=int, default=4)
    render.add_argument("--fps", type=int, default=12)

    launch = subparsers.add_parser("launch-preview", help="Launch Blender UI with a simple preview scene loaded from preview_job.json.")
    launch.add_argument("--preview-job", required=True)
    launch.add_argument("--blender-path", help="Optional explicit Blender executable path. Overrides config and PATH lookup.")
    launch.add_argument("--output-blend", help="Optional .blend path inside outputs/.")
    launch.add_argument("--max-steps", type=int, default=0, help="How many plan steps to load. Use 0 for all steps.")
    launch.add_argument("--frame-stride", type=int, default=4)
    launch.add_argument("--fps", type=int, default=12)

    mesh_preview = subparsers.add_parser("build-mesh-preview", help="Build a true skinned-mesh preview manifest for Blender.")
    mesh_preview.add_argument("--preview-job", required=True)
    mesh_preview.add_argument("--plan", required=True)
    mesh_preview.add_argument("--song-event-map")
    mesh_preview.add_argument("--mesh-template-fbx", help="Optional explicit mesh template FBX. Overrides config.")
    mesh_preview.add_argument("--output", help="Optional manifest output path inside outputs/.")
    mesh_preview.add_argument("--fps", type=int, default=24)
    mesh_preview.add_argument("--basis-mode", default="constraint_world_pelvis_local")
    mesh_preview.add_argument("--max-steps", type=int, default=0, help="How many plan steps to include. Use 0 for all steps.")

    smplx_stitch = subparsers.add_parser(
        "build-smplx-stitch-preview",
        help="Build a lightweight SMPL-X mesh stitch preview manifest from a rhythmic choreography plan.",
    )
    smplx_stitch.add_argument("--plan", required=True)
    smplx_stitch.add_argument("--motion-library", required=True)
    smplx_stitch.add_argument("--song-event-map")
    smplx_stitch.add_argument("--output", help="Optional manifest output path inside outputs/.")
    smplx_stitch.add_argument("--fps", type=int, default=30)
    smplx_stitch.add_argument("--blend-frames", type=int, default=6)
    smplx_stitch.add_argument("--max-steps", type=int, default=0, help="How many plan steps to include. Use 0 for all steps.")

    smplx_stitch_web = subparsers.add_parser(
        "build-smplx-stitch-web-preview",
        help="Build a self-contained HTML preview for a SMPL-X stitch manifest.",
    )
    smplx_stitch_web.add_argument("--manifest", required=True)
    smplx_stitch_web.add_argument("--song-event-map")
    smplx_stitch_web.add_argument("--audio", help="Optional audio path. Defaults to song_event_map.source_audio_path when available.")
    smplx_stitch_web.add_argument("--output", help="Optional HTML output path inside outputs/.")

    smplx_mesh_stitch = subparsers.add_parser(
        "build-smplx-stitch-mesh-preview",
        help="Build a true SMPL-X mesh video/HTML review from a stitch manifest.",
    )
    smplx_mesh_stitch.add_argument("--manifest", required=True)
    smplx_mesh_stitch.add_argument("--song-event-map", help="Optional song_event_map JSON for audio, beat, downbeat, and accent review.")
    smplx_mesh_stitch.add_argument("--audio", help="Optional audio path. Defaults to song_event_map.source_audio_path when available.")
    smplx_mesh_stitch.add_argument("--output-video", help="Optional MP4 output path inside outputs/.")
    smplx_mesh_stitch.add_argument("--output-strip", help="Optional strip PNG output path inside outputs/.")
    smplx_mesh_stitch.add_argument("--output-report", help="Optional report JSON output path inside outputs/.")
    smplx_mesh_stitch.add_argument("--output-html", help="Optional HTML review output path inside outputs/.")
    smplx_mesh_stitch.add_argument("--force-cache", action="store_true", help="Rebuild source mesh caches even if they already exist.")
    smplx_mesh_stitch.add_argument("--batch-size", type=int, default=128)
    smplx_mesh_stitch.add_argument("--face-stride", type=int, default=12)
    smplx_mesh_stitch.add_argument("--render-frame-stride", type=int, default=1)
    smplx_mesh_stitch.add_argument("--max-render-frames", type=int, default=0)
    smplx_mesh_stitch.add_argument("--transition-smooth-frames", type=int, default=12)
    smplx_mesh_stitch.add_argument("--transition-smooth-passes", type=int, default=2)

    mesh_launch = subparsers.add_parser("launch-mesh-preview", help="Launch Blender UI with a true skinned-mesh preview scene.")
    mesh_launch.add_argument("--manifest", required=True)
    mesh_launch.add_argument("--blender-path", help="Optional explicit Blender executable path. Overrides config and PATH lookup.")
    mesh_launch.add_argument("--output-blend", help="Optional .blend path inside outputs/.")

    willa_retarget = subparsers.add_parser("launch-willa-retarget", help="Launch Blender UI with Willa retargeted from a canonical Willa BVH.")
    willa_retarget.add_argument("--input-bvh", help="Optional input BVH path. Non-canonical BVHs are normalized before retarget.")
    willa_retarget.add_argument("--target-scene", help="Optional target .blend path. Defaults to the Willa retarget profile.")
    willa_retarget.add_argument("--profile", help="Optional Willa retarget profile path.")
    willa_retarget.add_argument(
        "--strategy",
        choices=["constraint-bake", "rest-pose-first"],
        help="Retarget execution strategy. Defaults to the profile strategy.",
    )
    willa_retarget.add_argument("--result-tag", help="Optional tag used when auto-versioning saved result artifacts.")
    willa_retarget.add_argument("--blender-path", help="Optional explicit Blender executable path. Overrides config and PATH lookup.")
    willa_retarget.add_argument("--output-blend", help="Optional output .blend path inside outputs/.")
    willa_retarget.add_argument("--report-output", help="Optional report JSON path inside outputs/.")

    web_preview = subparsers.add_parser("build-web-preview", help="Build a self-contained HTML mesh-like preview from preview_job.json.")
    web_preview.add_argument("--preview-job", required=True)
    web_preview.add_argument("--plan", help="Optional choreography plan for step timing and labels.")
    web_preview.add_argument("--song-event-map", help="Optional song event map for music timing.")
    web_preview.add_argument("--audio", help="Optional audio path. Defaults to song_event_map.source_audio_path when available.")
    web_preview.add_argument("--output", help="Optional HTML output path inside outputs/.")
    web_preview.add_argument("--max-steps", type=int, default=0, help="How many plan steps to include. Use 0 for all steps.")
    web_preview.add_argument("--frame-stride", type=int, default=0)
    web_preview.add_argument("--fps", type=int, default=24)
    web_preview.add_argument("--transition-frames", type=int, default=6)

    review = subparsers.add_parser("build-review-bundle", help="Create a stage-review bundle from current pipeline outputs.")
    review.add_argument("--song-event-map", required=True)
    review.add_argument("--motion-library-showcase", required=True)
    review.add_argument("--plan", required=True)
    review.add_argument("--retarget-report", required=True)
    review.add_argument("--output", help="Optional review bundle JSON path inside outputs/.")
    review.add_argument("--markdown-output", help="Optional markdown summary path inside outputs/.")

    truth = subparsers.add_parser(
        "build-dataset-truth-preview",
        help="Build an HTML sync preview for original dataset motion/music pairs.",
    )
    truth.add_argument("--dataset", default="finedance", choices=["finedance"])
    truth.add_argument("--sequence-ids", nargs="+", help="Dataset sequence IDs to include in the preview.")
    truth.add_argument("--all-sequences", action="store_true", help="Include every sequence available for the dataset.")
    truth.add_argument("--output", help="Optional HTML output path inside outputs/.")

    rhythmic_library = subparsers.add_parser(
        "build-finedance-rhythmic-library",
        help="Build a FineDance rhythmic-first SMPL-X motion-unit library from M2-2 Event Rail reports.",
    )
    rhythmic_library.add_argument(
        "--audio-feature-report",
        default="outputs/reports/dataset_truth_finedance_audio_features_all.json",
        help="M2-2 dataset truth audio feature report JSON.",
    )
    rhythmic_library.add_argument("--output", help="Optional library JSON path inside outputs/.")
    rhythmic_library.add_argument("--summary-output", help="Optional showcase summary path inside outputs/.")
    rhythmic_library.add_argument("--include-fallback", action="store_true", help="Include non-rhythmic-first FineDance entries too.")
    rhythmic_library.add_argument("--max-sequences", type=int, default=0, help="Optional limit for smoke builds. Use 0 for all selected sequences.")
    rhythmic_library.add_argument("--unit-beats", type=int, default=8)
    rhythmic_library.add_argument("--unit-beat-set", help="Comma-separated unit durations in beats, e.g. 2,4,8,16.")
    rhythmic_library.add_argument("--accent-unit-beats", type=int, default=4)
    rhythmic_library.add_argument("--max-unit-beats", type=int, default=16)
    rhythmic_library.add_argument("--min-source-sec", type=float, default=10.0, help="Drop units that start within the first N seconds of each FineDance source sequence.")
    rhythmic_library.add_argument("--coverage-report", action="store_true", help="Also write an M10 coverage report.")
    rhythmic_library.add_argument("--coverage-output", help="Optional coverage report path inside outputs/.")

    finedance_song = subparsers.add_parser(
        "build-finedance-song-event-map",
        help="Convert M2-2 FineDance audio features into a full-song song_event_map.json.",
    )
    finedance_song.add_argument("--sequence-id", required=True)
    finedance_song.add_argument(
        "--audio-feature-report",
        default="outputs/reports/dataset_truth_finedance_audio_features_all.json",
        help="M2-2 dataset truth audio feature report JSON.",
    )
    finedance_song.add_argument("--output", help="Optional song_event_map output path inside outputs/.")
    finedance_song.add_argument("--phrase-beats", type=int, default=8)

    stream_events = subparsers.add_parser(
        "simulate-streaming-song-events",
        help="Simulate endpoint-style streaming song events from a local audio file.",
    )
    stream_events.add_argument("--audio", required=True, help="Audio file path to simulate as a stream.")
    stream_events.add_argument("--song-id", help="Stable song id. Defaults to the audio filename stem.")
    stream_events.add_argument("--output", help="Optional JSONL output path inside outputs/.")
    stream_events.add_argument("--initial-buffer-sec", type=float, default=2.0)
    stream_events.add_argument("--lookahead-sec", type=float, default=2.0)
    stream_events.add_argument("--lookfront-sec", type=float, default=1.0)
    stream_events.add_argument("--chunk-ms", type=float, default=46.44)
    stream_events.add_argument("--beats-per-bar", type=int, default=4)
    stream_events.add_argument("--rolling-window-sec", type=float, default=8.0)

    annotate_units = subparsers.add_parser(
        "annotate-finedance-motion-units",
        help="Add M9 streaming retrieval annotations to a FineDance rhythmic SMPL-X motion library.",
    )
    annotate_units.add_argument("--input-library", required=True)
    annotate_units.add_argument("--output", help="Optional annotated library JSON path inside outputs/.")
    annotate_units.add_argument("--contact-mode", choices=["root", "joints"], default="root")

    stream_plan = subparsers.add_parser(
        "simulate-streaming-smplx-plan",
        help="Retrieve FineDance SMPL-X motion units from streaming song events with a 2-second lookahead guard.",
    )
    stream_plan.add_argument("--stream-events", required=True)
    stream_plan.add_argument("--library", required=True)
    stream_plan.add_argument("--output", help="Optional JSONL stream plan output path inside outputs/.")
    stream_plan.add_argument("--max-steps", type=int, default=0)
    stream_plan.add_argument("--planner-version", choices=["m9", "m12", "m15", "m17"], default="m17")
    stream_plan.add_argument("--tail-policy", choices=["none", "recover"], default="none")
    stream_plan.add_argument("--source-sequence-allowlist", help="Comma-separated FineDance source sequence ids for visually coherent planning.")
    stream_plan.add_argument("--initial-hold-sec", type=float, default=5.0, help="Hold an initial pose before starting retrieval.")
    stream_plan.add_argument("--initial-pose-mode", choices=["freeze_first", "neutral_rest", "neutral_idle"], default="neutral_idle")
    stream_plan.add_argument("--cohort-size", type=int, default=10, help="How many source songs to keep in the M15 style cohort before unit-level retrieval.")

    stream_render = subparsers.add_parser(
        "render-streaming-smplx-mesh-review",
        help="Render an M8-style SMPL-X mesh review from an M9 streaming plan JSONL.",
    )
    stream_render.add_argument("--stream-plan", required=True)
    stream_render.add_argument("--audio", required=True)
    stream_render.add_argument("--output-prefix", help="Output stem used when explicit output paths are omitted.")
    stream_render.add_argument("--output-video", help="Optional MP4 output path inside outputs/.")
    stream_render.add_argument("--output-strip", help="Optional strip PNG output path inside outputs/.")
    stream_render.add_argument("--output-report", help="Optional report JSON output path inside outputs/.")
    stream_render.add_argument("--output-html", help="Optional HTML review output path inside outputs/.")
    stream_render.add_argument("--manifest-output", help="Optional generated stitch manifest path inside outputs/.")
    stream_render.add_argument("--fps", type=int, default=30)
    stream_render.add_argument("--blend-frames", type=int, default=10)
    stream_render.add_argument("--force-cache", action="store_true")
    stream_render.add_argument("--batch-size", type=int, default=128)
    stream_render.add_argument("--face-stride", type=int, default=30)
    stream_render.add_argument("--render-frame-stride", type=int, default=2)
    stream_render.add_argument("--max-render-frames", type=int, default=0)
    stream_render.add_argument("--transition-smooth-frames", type=int, default=12)
    stream_render.add_argument("--transition-smooth-passes", type=int, default=2)

    coverage = subparsers.add_parser("evaluate-motion-library-coverage", help="Write an M10 coverage report for a FineDance motion library.")
    coverage.add_argument("--library", required=True)
    coverage.add_argument("--output", help="Optional coverage report path inside outputs/.")
    coverage.add_argument("--unit-beat-set", default="2,4,8,16")

    calibrate = subparsers.add_parser("calibrate-song-event-rail", help="Create a calibrated song event rail from streaming events and optional manual overrides.")
    calibrate.add_argument("--stream-events", required=True)
    calibrate.add_argument("--manual-overrides", help="Optional override JSON with beats/downbeats/accents/drum_hits.")
    calibrate.add_argument("--output", help="Optional calibrated event-map output path inside outputs/.")

    event_eval = subparsers.add_parser("evaluate-streaming-event-rail", help="Evaluate streaming beat/downbeat/drum events against a calibrated reference.")
    event_eval.add_argument("--stream-events", required=True)
    event_eval.add_argument("--reference-event-map", help="Optional calibrated event-map JSON. Defaults to self-consistency baseline.")
    event_eval.add_argument("--output", help="Optional event-rail evaluation report path inside outputs/.")
    event_eval.add_argument("--tolerance-sec", type=float, default=2.0 / 30.0)

    planner_eval = subparsers.add_parser("evaluate-streaming-smplx-plan", help="Evaluate an M12 streaming SMPL-X plan for speed, gaps, and future-visibility safety.")
    planner_eval.add_argument("--stream-plan", required=True)
    planner_eval.add_argument("--output", help="Optional planner evaluation report path inside outputs/.")

    transition_eval = subparsers.add_parser("build-motion-transition-report", help="Write an M13 transition-quality report from a mesh review report.")
    transition_eval.add_argument("--mesh-report", required=True)
    transition_eval.add_argument("--output", help="Optional transition report path inside outputs/.")

    runtime_bundle = subparsers.add_parser("export-unity-streaming-runtime-bundle", help="Export an M14 Unity/endpoint runtime bundle contract.")
    runtime_bundle.add_argument("--library", required=True)
    runtime_bundle.add_argument("--stream-plan")
    runtime_bundle.add_argument("--stream-events")
    runtime_bundle.add_argument("--output-dir", help="Optional output directory inside outputs/.")

    return parser


def _default_song_output(song_id: str) -> str:
    return f"song_event_maps/{slugify(song_id)}_song_event_map.json"


def _default_plan_output(song_id: str) -> str:
    return f"choreography_plans/{slugify(song_id)}_choreography_plan.json"


def _parse_int_set(raw_value: str | None) -> list[int] | None:
    if raw_value is None:
        return None
    values: list[int] = []
    for part in str(raw_value).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value > 0 and value not in values:
            values.append(value)
    return values or None


def _blender_app_bundle_path(blender_path: str) -> Path | None:
    candidate = Path(blender_path).expanduser().resolve()
    parts = candidate.parts
    if ".app" in candidate.name:
        return candidate
    if "Contents" in parts and "MacOS" in parts:
        contents_index = parts.index("Contents")
        return Path(*parts[:contents_index]).with_suffix(".app")
    for index, part in enumerate(parts):
        if part.endswith(".app"):
            return Path(*parts[: index + 1])
    return None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_app_config()

    if args.command == "analyze-song":
        from .pipelines.music_analysis import analyze_song

        input_path = resolve_input_path(config, args.input)
        override_payload = {}
        override_source = None
        if args.overrides:
            override_path = resolve_input_path(config, args.overrides)
            override_payload = load_json(override_path)
            override_source = str(override_path)
        payload = analyze_song(
            input_path=input_path,
            song_id=args.song_id,
            beats_per_bar=args.beats_per_bar,
            overrides=override_payload,
            override_source=override_source,
        ).to_dict()
        output_path = ensure_output_path(config, args.output or _default_song_output(args.song_id))
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "build-motion-library":
        from .pipelines.motion_library import build_motion_library_showcase, build_motion_unit_library

        preview_root = config.shared_roots.motion_base_assets_root / "previewCache"
        library = build_motion_unit_library(
            preview_root=preview_root,
            fps=args.fps,
            unit_beats=args.unit_beats,
            stride_beats=args.stride_beats,
            assumed_bpm=args.assumed_bpm,
        )
        payload = library.to_dict()
        showcase = build_motion_library_showcase(library)
        output_path = ensure_output_path(config, args.output or "motion_libraries/motion_unit_library.json")
        summary_output_path = ensure_output_path(config, args.summary_output or "motion_libraries/motion_unit_library_showcase.json")
        write_json(output_path, payload)
        write_json(summary_output_path, showcase)
        print(output_path)
        print(summary_output_path)
        return 0

    if args.command == "plan-choreography":
        from .pipelines.planner import build_choreography_plan

        song_event_map = load_json(resolve_input_path(config, args.song_event_map))
        motion_library = load_json(resolve_input_path(config, args.motion_library))
        payload = build_choreography_plan(song_event_map=song_event_map, motion_library=motion_library).to_dict()
        output_path = ensure_output_path(config, args.output or _default_plan_output(song_event_map["song_id"]))
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "plan-rhythmic-choreography":
        from .pipelines.rhythmic_planner import build_rhythmic_choreography_plan

        song_event_map = load_json(resolve_input_path(config, args.song_event_map))
        motion_library = load_json(resolve_input_path(config, args.motion_library))
        payload = build_rhythmic_choreography_plan(
            song_event_map=song_event_map,
            motion_library=motion_library,
            beam_width=args.beam_width,
            candidate_limit=args.candidate_limit,
            max_speed_adjustment=args.max_speed_adjustment,
        ).to_dict()
        output_path = ensure_output_path(config, args.output or f"choreography_plans/{slugify(song_event_map['song_id'])}_rhythmic_smplx_plan.json")
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "build-preview":
        from .pipelines.preview import build_preview_and_retarget_report

        plan = load_json(resolve_input_path(config, args.plan))
        motion_library = load_json(resolve_input_path(config, args.motion_library))
        preview_job, report = build_preview_and_retarget_report(
            plan=plan,
            motion_library=motion_library,
            avatar_id=args.avatar_id,
            blender_path=args.blender_path or (str(config.blender_path) if config.blender_path else None),
        )
        preview_output = ensure_output_path(
            config,
            args.preview_output or f"preview_jobs/{slugify(plan['plan_id'])}_preview_job.json",
        )
        report_output = ensure_output_path(
            config,
            args.report_output or f"retarget_reports/{slugify(plan['plan_id'])}_retarget_report.json",
        )
        write_json(preview_output, preview_job)
        write_json(report_output, report.to_dict())
        print(preview_output)
        print(report_output)
        return 0

    if args.command == "build-review-bundle":
        from .pipelines.review_bundle import build_review_bundle, build_review_bundle_markdown

        song_event_map = load_json(resolve_input_path(config, args.song_event_map))
        motion_library_showcase = load_json(resolve_input_path(config, args.motion_library_showcase))
        plan = load_json(resolve_input_path(config, args.plan))
        retarget_report = load_json(resolve_input_path(config, args.retarget_report))
        bundle = build_review_bundle(
            song_event_map=song_event_map,
            motion_library_showcase=motion_library_showcase,
            choreography_plan=plan,
            retarget_report=retarget_report,
        ).to_dict()
        output_path = ensure_output_path(config, args.output or f"review_bundles/{slugify(bundle['bundle_id'])}.json")
        markdown_output_path = ensure_output_path(
            config,
            args.markdown_output or f"review_bundles/{slugify(bundle['bundle_id'])}.md",
        )
        write_json(output_path, bundle)
        markdown_output_path.write_text(build_review_bundle_markdown(bundle), encoding="utf-8")
        print(output_path)
        print(markdown_output_path)
        return 0

    if args.command == "build-dataset-truth-preview":
        from .pipelines.dataset_truth_preview import (
            build_dataset_truth_audio_feature_report,
            build_dataset_truth_preview_document,
            list_dataset_sequence_ids,
            resolve_dataset_truth_entries,
        )

        if not args.all_sequences and not args.sequence_ids:
            parser.error("build-dataset-truth-preview requires --sequence-ids or --all-sequences.")
        sequence_ids = list_dataset_sequence_ids(config, args.dataset) if args.all_sequences else list(args.sequence_ids or [])
        suffix = "all" if args.all_sequences else "_".join(slugify(seq) for seq in sequence_ids)
        output_name = f"dataset_truth_{slugify(args.dataset)}_{suffix}.html"
        output_path = ensure_output_path(config, args.output or f"renders/{output_name}")
        entries = resolve_dataset_truth_entries(config=config, dataset_name=args.dataset, sequence_ids=sequence_ids)
        output_path.write_text(
            build_dataset_truth_preview_document(
                dataset_name=args.dataset,
                entries=entries,
                output_dir=output_path.parent,
            ),
            encoding="utf-8",
        )
        report_name = (
            f"dataset_truth_{slugify(args.dataset)}_audio_features_all.json"
            if args.all_sequences
            else f"dataset_truth_{slugify(args.dataset)}_audio_features_{suffix}.json"
        )
        report_path = ensure_output_path(config, f"reports/{report_name}")
        write_json(
            report_path,
            build_dataset_truth_audio_feature_report(dataset_name=args.dataset, entries=entries),
        )
        print(output_path)
        print(report_path)
        return 0

    if args.command == "build-finedance-rhythmic-library":
        from .pipelines.finedance_rhythmic_library import (
            build_motion_library_coverage_report,
            build_finedance_rhythmic_library_showcase,
            build_finedance_rhythmic_smplx_library,
        )

        report = load_json(resolve_input_path(config, args.audio_feature_report))
        raw_root = config.shared_roots.motion_base_assets_root / "datasets" / "finedance" / "raw" / "extracted" / "finedance"
        library = build_finedance_rhythmic_smplx_library(
            audio_feature_report=report,
            raw_root=raw_root,
            rhythmic_only=not args.include_fallback,
            unit_beats=args.unit_beats,
            unit_beat_set=_parse_int_set(args.unit_beat_set),
            accent_unit_beats=args.accent_unit_beats,
            max_unit_beats=args.max_unit_beats,
            max_sequences=args.max_sequences,
            min_source_sec=args.min_source_sec,
        )
        payload = library.to_dict()
        showcase = build_finedance_rhythmic_library_showcase(library)
        output_path = ensure_output_path(config, args.output or "motion_libraries/finedance_rhythmic_smplx_library.json")
        summary_output_path = ensure_output_path(
            config,
            args.summary_output or "motion_libraries/finedance_rhythmic_smplx_library_showcase.json",
        )
        write_json(output_path, payload)
        write_json(summary_output_path, showcase)
        print(output_path)
        print(summary_output_path)
        if args.coverage_report:
            coverage = build_motion_library_coverage_report(payload, required_unit_beats=_parse_int_set(args.unit_beat_set) or [2, 4, 8, 16])
            coverage_path = ensure_output_path(
                config,
                args.coverage_output or "reports/milestone_m10_motion_library_coverage_report.json",
            )
            write_json(coverage_path, coverage)
            print(coverage_path)
        return 0

    if args.command == "build-finedance-song-event-map":
        from .pipelines.finedance_song_event_map import build_finedance_song_event_map_from_report

        report = load_json(resolve_input_path(config, args.audio_feature_report))
        payload = build_finedance_song_event_map_from_report(
            audio_feature_report=report,
            sequence_id=args.sequence_id,
            motion_base_assets_root=config.shared_roots.motion_base_assets_root,
            phrase_beats=args.phrase_beats,
        )
        output_path = ensure_output_path(
            config,
            args.output or f"song_event_maps/{slugify(payload['song_id'])}_song_event_map.json",
        )
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "simulate-streaming-song-events":
        from .pipelines.streaming_smplx import build_streaming_song_event_records, write_jsonl

        audio_path = resolve_input_path(config, args.audio)
        song_id = args.song_id or slugify(audio_path.stem)
        records = build_streaming_song_event_records(
            audio_path=audio_path,
            song_id=song_id,
            initial_buffer_sec=args.initial_buffer_sec,
            lookahead_sec=args.lookahead_sec,
            lookfront_sec=args.lookfront_sec,
            chunk_ms=args.chunk_ms,
            beats_per_bar=args.beats_per_bar,
            rolling_window_sec=args.rolling_window_sec,
        )
        output_path = ensure_output_path(config, args.output or f"streaming/{slugify(song_id)}_stream_events.jsonl")
        write_jsonl(output_path, records)
        print(output_path)
        return 0

    if args.command == "annotate-finedance-motion-units":
        from .pipelines.streaming_smplx import annotate_finedance_motion_units

        input_path = resolve_input_path(config, args.input_library)
        motion_library = load_json(input_path)
        payload = annotate_finedance_motion_units(motion_library=motion_library, project_root=config.project_root, contact_mode=args.contact_mode)
        default_name = f"motion_libraries/{slugify(str(payload.get('library_id', input_path.stem)))}.json"
        output_path = ensure_output_path(config, args.output or default_name)
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "simulate-streaming-smplx-plan":
        from .pipelines.streaming_smplx import read_jsonl, simulate_streaming_smplx_plan_records, write_jsonl

        stream_events_path = resolve_input_path(config, args.stream_events)
        stream_events = read_jsonl(stream_events_path)
        library = load_json(resolve_input_path(config, args.library))
        records = simulate_streaming_smplx_plan_records(
            stream_event_records=stream_events,
            annotated_library=library,
            stream_events_path=str(stream_events_path),
            max_steps=args.max_steps,
            planner_version=args.planner_version,
            tail_policy=args.tail_policy,
            source_sequence_allowlist=[value.strip() for value in str(args.source_sequence_allowlist or "").split(",") if value.strip()] or None,
            initial_hold_sec=args.initial_hold_sec,
            initial_pose_mode=args.initial_pose_mode,
            cohort_size=args.cohort_size,
        )
        header = next((record for record in records if record.get("kind") == "stream_plan_header"), {})
        song_id = str(header.get("song_id", "stream_song") or "stream_song")
        output_path = ensure_output_path(config, args.output or f"streaming/{slugify(song_id)}_stream_plan.jsonl")
        write_jsonl(output_path, records)
        print(output_path)
        return 0

    if args.command == "evaluate-motion-library-coverage":
        from .pipelines.finedance_rhythmic_library import build_motion_library_coverage_report

        library = load_json(resolve_input_path(config, args.library))
        report = build_motion_library_coverage_report(library, required_unit_beats=_parse_int_set(args.unit_beat_set) or [2, 4, 8, 16])
        output_path = ensure_output_path(config, args.output or "reports/milestone_m10_motion_library_coverage_report.json")
        write_json(output_path, report)
        print(output_path)
        return 0

    if args.command == "calibrate-song-event-rail":
        from .pipelines.streaming_smplx import calibrate_song_event_rail, read_jsonl

        stream_events_path = resolve_input_path(config, args.stream_events)
        stream_events = read_jsonl(stream_events_path)
        overrides = load_json(resolve_input_path(config, args.manual_overrides)) if args.manual_overrides else None
        report = calibrate_song_event_rail(stream_event_records=stream_events, manual_overrides=overrides)
        output_path = ensure_output_path(config, args.output or f"song_event_maps/{slugify(str(report.get('song_id', 'stream_song')))}_calibrated_event_rail.json")
        write_json(output_path, report)
        print(output_path)
        return 0

    if args.command == "evaluate-streaming-event-rail":
        from .pipelines.streaming_smplx import evaluate_streaming_event_rail, read_jsonl

        stream_events = read_jsonl(resolve_input_path(config, args.stream_events))
        reference = load_json(resolve_input_path(config, args.reference_event_map)) if args.reference_event_map else None
        report = evaluate_streaming_event_rail(stream_event_records=stream_events, reference_event_map=reference, tolerance_sec=args.tolerance_sec)
        output_path = ensure_output_path(config, args.output or "reports/milestone_m11_streaming_event_rail_eval.json")
        write_json(output_path, report)
        print(output_path)
        return 0

    if args.command == "evaluate-streaming-smplx-plan":
        from .pipelines.streaming_smplx import evaluate_streaming_planner_records, read_jsonl

        stream_plan = read_jsonl(resolve_input_path(config, args.stream_plan))
        report = evaluate_streaming_planner_records(stream_plan)
        output_path = ensure_output_path(config, args.output or "reports/milestone_m12_streaming_planner_eval.json")
        write_json(output_path, report)
        print(output_path)
        return 0

    if args.command == "build-motion-transition-report":
        from .pipelines.streaming_smplx import build_motion_transition_report

        mesh_report = load_json(resolve_input_path(config, args.mesh_report))
        report = build_motion_transition_report(mesh_report)
        output_path = ensure_output_path(config, args.output or "reports/milestone_m13_motion_transition_report.json")
        write_json(output_path, report)
        print(output_path)
        return 0

    if args.command == "export-unity-streaming-runtime-bundle":
        from .pipelines.streaming_smplx import export_unity_streaming_runtime_bundle, read_jsonl

        library = load_json(resolve_input_path(config, args.library))
        stream_plan = read_jsonl(resolve_input_path(config, args.stream_plan)) if args.stream_plan else None
        stream_events = read_jsonl(resolve_input_path(config, args.stream_events)) if args.stream_events else None
        output_dir = ensure_output_path(config, args.output_dir or "runtime_bundles/unity_streaming_smplx_bundle")
        report = export_unity_streaming_runtime_bundle(
            output_dir=output_dir,
            annotated_library=library,
            stream_plan_records=stream_plan,
            stream_event_records=stream_events,
        )
        print(report["bundle_dir"])
        print(str(output_dir / "bundle_report.json"))
        return 0

    if args.command == "render-streaming-smplx-mesh-review":
        from .pipelines.streaming_smplx import read_jsonl, render_streaming_smplx_mesh_review

        stream_plan_path = resolve_input_path(config, args.stream_plan)
        stream_plan_records = read_jsonl(stream_plan_path)
        stream_header = next((record for record in stream_plan_records if record.get("kind") == "stream_plan_header"), {})
        stream_events_records = None
        stream_events_raw = stream_header.get("source_stream_events_path")
        if stream_events_raw:
            stream_events_path = resolve_input_path(config, str(stream_events_raw))
            stream_events_records = read_jsonl(stream_events_path)
        audio_path = resolve_input_path(config, args.audio)
        output_prefix = slugify(args.output_prefix or f"{stream_header.get('song_id', audio_path.stem)}_streaming")
        output_video = ensure_output_path(config, args.output_video or f"renders/{output_prefix}_mesh_preview.mp4")
        output_strip = ensure_output_path(config, args.output_strip or f"renders/{output_prefix}_mesh_strip.png")
        output_report = ensure_output_path(config, args.output_report or f"reports/{output_prefix}_mesh_report.json")
        output_html = ensure_output_path(config, args.output_html or f"renders/{output_prefix}_mesh_review.html")
        manifest_output = ensure_output_path(config, args.manifest_output or f"streaming/{output_prefix}_stitch_manifest.json")
        report = render_streaming_smplx_mesh_review(
            stream_plan_records=stream_plan_records,
            stream_event_records=stream_events_records,
            project_root=config.project_root,
            motion_base_assets_root=config.shared_roots.motion_base_assets_root,
            audio_path=audio_path,
            output_video=output_video,
            output_strip=output_strip,
            output_report=output_report,
            output_html=output_html,
            manifest_output=manifest_output,
            fps=args.fps,
            blend_frames=args.blend_frames,
            force_cache=args.force_cache,
            batch_size=args.batch_size,
            face_stride=args.face_stride,
            render_frame_stride=args.render_frame_stride,
            max_render_frames=args.max_render_frames,
            transition_smooth_frames=args.transition_smooth_frames,
            transition_smooth_passes=args.transition_smooth_passes,
        )
        print(report["artifacts"]["video"])
        print(report["artifacts"]["strip"])
        print(report["artifacts"]["html"])
        print(report["artifacts"]["report"])
        return 0

    if args.command == "build-mesh-preview":
        from .pipelines.mesh_preview import build_mesh_preview_manifest

        preview_job = load_json(resolve_input_path(config, args.preview_job))
        plan = load_json(resolve_input_path(config, args.plan))
        song_event_map = load_json(resolve_input_path(config, args.song_event_map)) if args.song_event_map else None
        mesh_template_fbx = Path(args.mesh_template_fbx).resolve() if args.mesh_template_fbx else config.mesh_template_fbx
        if mesh_template_fbx is None:
            parser.error("build-mesh-preview requires a mesh template FBX. Set config/paths.json tooling.meshTemplateFbx or pass --mesh-template-fbx.")
        payload = build_mesh_preview_manifest(
            preview_job=preview_job,
            plan=plan,
            song_event_map=song_event_map,
            mesh_template_fbx=mesh_template_fbx,
            fps=args.fps,
            basis_mode=args.basis_mode,
            max_steps=args.max_steps,
        )
        output_path = ensure_output_path(config, args.output or f"mesh_preview_jobs/{slugify(preview_job['plan_id'])}_mesh_preview.json")
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "build-smplx-stitch-preview":
        from .pipelines.smplx_stitch_preview import build_smplx_stitch_preview_manifest

        plan = load_json(resolve_input_path(config, args.plan))
        motion_library = load_json(resolve_input_path(config, args.motion_library))
        song_event_map = load_json(resolve_input_path(config, args.song_event_map)) if args.song_event_map else None
        payload = build_smplx_stitch_preview_manifest(
            plan=plan,
            motion_library=motion_library,
            song_event_map=song_event_map,
            fps=args.fps,
            blend_frames=args.blend_frames,
            max_steps=args.max_steps,
        )
        output_path = ensure_output_path(config, args.output or f"smplx_mesh_previews/{slugify(plan['plan_id'])}_stitch_manifest.json")
        write_json(output_path, payload)
        print(output_path)
        return 0

    if args.command == "build-smplx-stitch-web-preview":
        from .pipelines.smplx_stitch_web_preview import build_smplx_stitch_web_preview_document

        manifest = load_json(resolve_input_path(config, args.manifest))
        song_event_map = load_json(resolve_input_path(config, args.song_event_map)) if args.song_event_map else None
        output_path = ensure_output_path(
            config,
            args.output or f"renders/{slugify(str(manifest.get('manifest_id', 'smplx_stitch_preview')))}.html",
        )
        audio_href = None
        audio_path = None
        if args.audio:
            audio_path = resolve_input_path(config, args.audio)
        elif song_event_map and song_event_map.get("source_audio_path"):
            audio_path = resolve_input_path(config, str(song_event_map["source_audio_path"]))
        if audio_path is not None:
            audio_href = os.path.relpath(audio_path.resolve(), output_path.parent.resolve())
        output_path.write_text(
            build_smplx_stitch_web_preview_document(
                manifest=manifest,
                song_event_map=song_event_map,
                audio_href=audio_href,
            ),
            encoding="utf-8",
        )
        print(output_path)
        return 0

    if args.command == "build-smplx-stitch-mesh-preview":
        from .pipelines.smplx_mesh_stitch_renderer import build_smplx_mesh_stitch_visual_preview

        manifest = load_json(resolve_input_path(config, args.manifest))
        song_event_map = load_json(resolve_input_path(config, args.song_event_map)) if args.song_event_map else None
        audio_path = None
        if args.audio:
            audio_path = resolve_input_path(config, args.audio)
        elif song_event_map and song_event_map.get("source_audio_path"):
            audio_path = resolve_input_path(config, str(song_event_map["source_audio_path"]))
        stem = slugify(str(manifest.get("manifest_id", "smplx_stitch_mesh_preview"))).replace("_smplx_stitch_preview", "")
        output_video = ensure_output_path(config, args.output_video or f"renders/{stem}_mesh_preview.mp4")
        output_strip = ensure_output_path(config, args.output_strip or f"renders/{stem}_mesh_strip.png")
        output_report = ensure_output_path(config, args.output_report or f"reports/{stem}_mesh_report.json")
        output_html = ensure_output_path(config, args.output_html or f"renders/{stem}_mesh_review.html")
        report = build_smplx_mesh_stitch_visual_preview(
            manifest=manifest,
            project_root=config.project_root,
            motion_base_assets_root=config.shared_roots.motion_base_assets_root,
            output_video=output_video,
            output_strip=output_strip,
            output_report=output_report,
            output_html=output_html,
            song_event_map=song_event_map,
            audio_path=audio_path,
            force_cache=args.force_cache,
            batch_size=args.batch_size,
            face_stride=args.face_stride,
            render_frame_stride=args.render_frame_stride,
            max_render_frames=args.max_render_frames,
            transition_smooth_frames=args.transition_smooth_frames,
            transition_smooth_passes=args.transition_smooth_passes,
        )
        print(report["artifacts"]["video"])
        print(report["artifacts"]["strip"])
        print(report["artifacts"]["html"])
        print(report["artifacts"]["report"])
        return 0

    if args.command == "launch-mesh-preview":
        manifest_path = resolve_input_path(config, args.manifest)
        manifest = load_json(manifest_path)
        blender_path = args.blender_path or (str(config.blender_path) if config.blender_path else None)
        if not blender_path:
            parser.error("launch-mesh-preview requires Blender. Set config/paths.json tooling.blenderPath or pass --blender-path.")
        output_blend = ensure_output_path(
            config,
            args.output_blend or f"renders/{slugify(manifest['plan_id'])}_mesh_preview.blend",
        )
        script_path = config.project_root / "tools" / "blender_build_mesh_preview.py"
        command = [
            blender_path,
            "--python",
            str(script_path),
            "--",
            "--manifest",
            str(manifest_path),
            "--output-blend",
            str(output_blend),
        ]
        subprocess.run(command, check=True)
        print(output_blend)
        return 0

    if args.command == "launch-willa-retarget":
        from .willa_canonicalization import canonicalize_source_bvh_for_willa, default_canonical_bvh_path
        from .willa_retarget import (
            default_willa_retarget_profile_path,
            load_willa_retarget_profile,
            next_versioned_retarget_paths,
            normalize_retarget_strategy,
            validate_willa_retarget_profile,
        )

        profile_path = Path(args.profile).resolve() if args.profile else default_willa_retarget_profile_path(config.project_root)
        profile = load_willa_retarget_profile(config.project_root, profile_path)
        validate_willa_retarget_profile(profile)

        blender_path = args.blender_path or (str(config.blender_path) if config.blender_path else None)
        if not blender_path:
            parser.error("launch-willa-retarget requires Blender. Set config/paths.json tooling.blenderPath or pass --blender-path.")

        source_bvh = resolve_input_path(config, args.input_bvh) if args.input_bvh else Path(profile["defaultSourceBvh"])
        canonical_report = ensure_output_path(
            config,
            Path(profile["defaultCanonicalReport"]).relative_to(config.project_root).as_posix(),
        )
        if source_bvh.name.endswith("_canonical.bvh"):
            input_bvh = source_bvh
        else:
            default_canonical = Path(profile["defaultInputBvh"])
            if args.input_bvh:
                input_bvh = default_canonical_bvh_path(source_bvh)
            else:
                input_bvh = default_canonical
            canonicalize_source_bvh_for_willa(
                input_bvh=source_bvh,
                output_bvh=input_bvh,
                report_output=canonical_report,
            )
        target_scene = resolve_input_path(config, args.target_scene) if args.target_scene else Path(profile["targetSceneBlend"])
        default_strategy = normalize_retarget_strategy(str(profile.get("defaultRetargetStrategy", "constraint_bake") or "constraint_bake"))
        strategy = normalize_retarget_strategy(args.strategy or default_strategy)
        if args.output_blend:
            output_blend = ensure_output_path(config, args.output_blend)
        else:
            output_blend = Path(profile["defaultOutputBlend"]).resolve()
        if args.report_output:
            report_output = ensure_output_path(config, args.report_output)
        else:
            report_output = Path(profile["defaultReport"]).resolve()
        if not args.output_blend and not args.report_output:
            result_tag = args.result_tag or strategy
            output_blend, report_output, _ = next_versioned_retarget_paths(output_blend, report_output, result_tag)
        script_path = (
            config.project_root / "tools" / "blender_retarget_willa.py"
            if strategy == "constraint_bake"
            else config.project_root / "tools" / "blender_retarget_willa_rest_pose.py"
        )
        app_bundle = _blender_app_bundle_path(blender_path)
        if app_bundle is not None and app_bundle.suffix == ".app":
            bootstrap = tempfile.NamedTemporaryFile(
                prefix="music_motion_lab_willa_retarget_",
                suffix=".py",
                delete=False,
                mode="w",
                encoding="utf-8",
            )
            with bootstrap:
                bootstrap.write(
                    "\n".join(
                        [
                            "import runpy",
                            "import sys",
                            "sys.argv = [",
                            "    'Blender',",
                            "    '--',",
                            f"    '--input-bvh', {str(input_bvh)!r},",
                            f"    '--profile', {str(profile_path)!r},",
                            (
                                f"    '--basis-mode', {str(profile.get('defaultConstraintBakeBasisMode', 'constraint_world_pelvis_torso_local'))!r},"
                                if strategy == "constraint_bake"
                                else ""
                            ),
                            f"    '--output-blend', {str(output_blend)!r},",
                            f"    '--report-output', {str(report_output)!r},",
                            "]",
                            f"runpy.run_path({str(script_path)!r}, run_name='__main__')",
                            "",
                        ]
                    )
                )
            command = [
                "open",
                "-na",
                str(app_bundle),
                "--args",
                str(target_scene),
                "--python-expr",
                f"exec(compile(open({str(Path(bootstrap.name))!r}, 'r', encoding='utf-8').read(), {str(Path(bootstrap.name))!r}, 'exec'))",
            ]
        else:
            command = [
                blender_path,
                str(target_scene),
                "--python",
                str(script_path),
                "--",
                "--input-bvh",
                str(input_bvh),
                "--profile",
                str(profile_path),
            ]
            if strategy == "constraint_bake":
                command.extend(
                    [
                        "--basis-mode",
                        str(profile.get("defaultConstraintBakeBasisMode", "constraint_world_pelvis_torso_local")),
                    ]
                )
            command.extend(
                [
                    "--output-blend",
                    str(output_blend),
                    "--report-output",
                    str(report_output),
                ]
            )
        subprocess.Popen(command)
        print(output_blend)
        print(report_output)
        return 0

    if args.command == "render-preview":
        preview_job_path = resolve_input_path(config, args.preview_job)
        preview_job = load_json(preview_job_path)
        blender_path = args.blender_path or (str(config.blender_path) if config.blender_path else None)
        if not blender_path:
            parser.error("render-preview requires Blender. Set config/paths.json tooling.blenderPath or pass --blender-path.")
        output_video = ensure_output_path(
            config,
            args.output_video or f"renders/{slugify(preview_job['plan_id'])}_simple_preview.mp4",
        )
        output_blend = ensure_output_path(
            config,
            args.output_blend or f"renders/{slugify(preview_job['plan_id'])}_simple_preview.blend",
        )
        script_path = config.project_root / "tools" / "blender_render_preview.py"
        command = [
            blender_path,
            "--background",
            "--python",
            str(script_path),
            "--",
            "--preview-job",
            str(preview_job_path),
            "--output-video",
            str(output_video),
            "--output-blend",
            str(output_blend),
            "--max-steps",
            str(args.max_steps),
            "--frame-stride",
            str(args.frame_stride),
            "--fps",
            str(args.fps),
        ]
        subprocess.run(command, check=True)
        print(output_video)
        print(output_blend)
        return 0

    if args.command == "launch-preview":
        preview_job_path = resolve_input_path(config, args.preview_job)
        preview_job = load_json(preview_job_path)
        blender_path = args.blender_path or (str(config.blender_path) if config.blender_path else None)
        if not blender_path:
            parser.error("launch-preview requires Blender. Set config/paths.json tooling.blenderPath or pass --blender-path.")
        output_blend = ensure_output_path(
            config,
            args.output_blend or f"renders/{slugify(preview_job['plan_id'])}_simple_preview.blend",
        )
        script_path = config.project_root / "tools" / "blender_render_preview.py"
        command = [
            blender_path,
            "--python",
            str(script_path),
            "--",
            "--preview-job",
            str(preview_job_path),
            "--output-blend",
            str(output_blend),
            "--max-steps",
            str(args.max_steps),
            "--frame-stride",
            str(args.frame_stride),
            "--fps",
            str(args.fps),
        ]
        subprocess.Popen(command)
        print(output_blend)
        return 0

    if args.command == "build-web-preview":
        from .pipelines.web_preview import build_web_preview_document

        preview_job_path = resolve_input_path(config, args.preview_job)
        preview_job = load_json(preview_job_path)
        plan = load_json(resolve_input_path(config, args.plan)) if args.plan else None
        song_event_map = load_json(resolve_input_path(config, args.song_event_map)) if args.song_event_map else None
        output_path = ensure_output_path(
            config,
            args.output or f"renders/{slugify(preview_job['plan_id'])}_web_preview.html",
        )
        audio_href = None
        audio_path = None
        if args.audio:
            audio_path = resolve_input_path(config, args.audio)
        elif song_event_map and song_event_map.get("source_audio_path"):
            audio_path = resolve_input_path(config, str(song_event_map["source_audio_path"]))
        if audio_path is not None:
            audio_href = os.path.relpath(audio_path.resolve(), output_path.parent.resolve())
        output_path.write_text(
            build_web_preview_document(
                preview_job=preview_job,
                plan=plan,
                song_event_map=song_event_map,
                audio_href=audio_href,
                max_steps=args.max_steps,
                frame_stride=args.frame_stride,
                fps=args.fps,
                transition_frames=args.transition_frames,
            ),
            encoding="utf-8",
        )
        print(output_path)
        return 0

    parser.error(f"unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
