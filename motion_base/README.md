# Motion Base

This directory holds an isolated motion-library database for future integration work.

Goals:
- keep database work off the current runtime path
- store only metadata, review records, and preview exports in git
- keep external FBX binaries outside the repo through a local asset-root config

Structure:
- `config/`: tracked config template plus ignored local asset-root override
- `intake/`: external source discovery queue and operator-facing job state
- `library/`: motion records and review records
- `review/`: candidate review feed plus in-progress review decisions
- `validation/`: candidate metrics and static validation output
- `preview/runtime_preview/`: generated runtime-compatible preview exports
- `taxonomy/`: controlled vocabulary and preview-export defaults

Expected workflow:
1. Copy [`asset_roots.example.json`](/Users/alex/Desktop/codex%20project/3d-digital/3d-digital-human/motion_base/config/asset_roots.example.json) to `motion_base/config/asset_roots.local.json`.
2. Point the configured roots at your external `rawFbx`, `sourceVideo`, `extractedMotion`, `approvedFbx`, and `previewCache` directories.
3. Run `tools/motion_base/sync_intake_queue.py` to discover new source assets.
4. Run `tools/motion_base/precheck_sources.py` to perform technical checks on source media.
5. For `video_rokoko` jobs, upload the video to Rokoko and register the exported result with `tools/motion_base/register_rokoko_export.py`.
6. Run `tools/motion_base/retarget_and_slice.py` to normalize and slice candidate phrases.
7. Run `tools/motion_base/validate_fbx_candidates.py` and `tools/motion_base/build_review_queue.py`.
8. Review candidates in Unity via `Tools > Motion Base > Review Queue`.
9. Run `tools/motion_base/ingest_motion.py --sync-approved` to write approved candidates into the formal library.
10. Run `tools/motion_base/validate_motion_base.py` and `tools/motion_base/export_runtime_preview.py` before any handoff.

Isolation contract:
- do not write to `Assets/Resources/Dance/Clips`
- do not write to `Assets/StreamingAssets/DanceData`
- do not modify runtime C# code from this workflow
- candidate FBX imports for review must stay under gitignored `Assets/MotionBaseReviewCache`
