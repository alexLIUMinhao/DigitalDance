# Milestone M5: FineDance Rhythmic-First SMPL-X Library

## Summary
- Branch: `codex/finedance-rhythmic-smplx-library`
- Mainline pivot: FineDance source SMPL-X action units and music-driven stitching are now the active task; Willa retargeting is out of scope for this branch.
- Source timing: M2-2 `dataset_truth_finedance_audio_features_all.json`, filtered to `analysis_priority.tier == rhythmic_first` by default.
- Source motion: raw FineDance `.npy` motion under `motion-base-assets/datasets/finedance/raw/extracted/finedance`.

## Implemented
- Added `build-finedance-rhythmic-library`.
  - Builds `schema_version=2` motion units with source frame ranges, beat ranges, keyframes, event-rail rhythm profiles, SMPL-X source references, entry/exit anchors, and compatible-next hints.
  - Keeps mesh data out of JSON; raw motion and future mesh caches remain referenced artifacts.
- Added `plan-rhythmic-choreography`.
  - Builds beat-locked choreography plans with beam search over the rhythmic SMPL-X library.
  - Records selected unit ids, source frames, target time windows, speed scale, transition score, and expected accent hits.
- Added `build-smplx-stitch-preview`.
  - Emits a lightweight stitch manifest with scene frames, source frames, blend windows, rhythm locks, transition records, and cache requests for future SMPL-X mesh generation.

## Verification
- Focused tests passed with the available local runner:
  - `env PYTHONPATH=src python -m unittest tests.test_finedance_rhythmic_library tests.test_rhythmic_planner tests.test_smplx_stitch_preview`
- Full available local test suite passed:
  - `env PYTHONPATH=src python -m unittest discover -s tests`
  - Result: `68` tests passed.
- `pytest` was requested in the plan, but the default local Python environment does not have it installed:
  - `python -m pytest tests` exits with `No module named pytest`.
- CLI smoke checks were run for:
  - rhythmic library build from the cached M2-2 report
  - rhythmic plan build from an existing song event map
  - SMPL-X stitch manifest build from the smoke plan/library

## Next
- Add the heavy renderer/cache worker that reads `cache_requests` and materializes SMPL-X mesh `.npz` files or review videos.
- Add transition-quality reports from actual generated vertices/joints once the mesh cache exists.
- Expand planner scoring with foot contact and full-joint pose features when all rhythmic-first source pose caches are available.
