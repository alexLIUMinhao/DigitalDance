# Motion Base

This directory holds an isolated motion-library database for future integration work.

Goals:
- keep database work off the current runtime path
- store only metadata, review records, and preview exports in git
- keep external FBX binaries outside the repo through a local asset-root config

Structure:
- `config/`: tracked config template plus ignored local asset-root override
- `intake/`: source catalog, dataset catalog, external discovery queue, and operator-facing job state
- `library/`: motion records and review records
- `review/`: candidate review feed plus in-progress review decisions
- `validation/`: candidate metrics and static validation output
- `preview/runtime_preview/`: generated runtime-compatible preview exports
- `taxonomy/`: controlled vocabulary and preview-export defaults

Expected workflow:
1. Copy [`asset_roots.example.json`](/Users/alex/Desktop/codex%20project/3d-digital/3d-digital-human/motion_base/config/asset_roots.example.json) to `motion_base/config/asset_roots.local.json`.
2. Point the configured roots at your external `rawFbx`, `sourceVideo`, `extractedMotion`, `approvedFbx`, and `previewCache` directories.
3. Prepare external dataset buckets under the same asset workspace, for example `motion-base-assets/datasets/<dataset>/{raw,manifest,converted_fbx}` and `motion-base-assets/models/{smpl,smplh,smplx}`.
4. Download first-wave datasets into the external dataset buckets. The current priority is `AIST++`, `FineDance`, and `PhantomDance`.
5. Run `tools/motion_base/register_dataset_catalog.py` to scan downloaded datasets into `motion_base/intake/dataset_catalog.json`.
6. Run `tools/motion_base/convert_dataset_motion.py` to convert dataset-native motion (`.pkl`, `.npy`, `.json`) into bridge assets under the external `rawFbx/datasets/...` tree.
7. Run `tools/motion_base/sync_intake_queue.py` to discover converted dataset motions first, then any curated FBX and source-video jobs.
8. Run `tools/motion_base/retarget_and_slice.py` to normalize and slice candidate phrases for dataset and curated-FBX jobs.
9. Run `tools/motion_base/validate_fbx_candidates.py` and `tools/motion_base/build_review_queue.py`.
10. Review candidate motions in Unity via `Tools > Motion Base > Studio`.
11. Run `tools/motion_base/ingest_motion.py --sync-approved` to write approved candidates into the formal library.
12. Run `tools/motion_base/validate_motion_base.py` and `tools/motion_base/export_runtime_preview.py` before any handoff.
13. Treat `sourceVideo -> Rokoko` as a secondary acquisition path. If you still want live provider intake, run `tools/motion_base/fetch_source_candidates.py`, `tools/motion_base/precheck_sources.py`, review in `Source Intake`, then register approved exports with `tools/motion_base/register_rokoko_export.py`.

Isolation contract:
- do not write to `Assets/Resources/Dance/Clips`
- do not write to `Assets/StreamingAssets/DanceData`
- do not modify runtime C# code from this workflow
- candidate FBX imports for review must stay under gitignored `Assets/MotionBaseReviewCache`
