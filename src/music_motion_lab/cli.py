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
    rhythmic_library.add_argument("--accent-unit-beats", type=int, default=4)
    rhythmic_library.add_argument("--max-unit-beats", type=int, default=16)

    return parser


def _default_song_output(song_id: str) -> str:
    return f"song_event_maps/{slugify(song_id)}_song_event_map.json"


def _default_plan_output(song_id: str) -> str:
    return f"choreography_plans/{slugify(song_id)}_choreography_plan.json"


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
            accent_unit_beats=args.accent_unit_beats,
            max_unit_beats=args.max_unit_beats,
            max_sequences=args.max_sequences,
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
