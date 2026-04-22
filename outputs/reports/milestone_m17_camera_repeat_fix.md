# M17 Camera Centering And Repeat-Control Fix

## Summary

This patch keeps the M17 any-song streaming baseline, then fixes two visual review issues found on `unity_audio0_m17_fx2_preview_mesh_review.html`:

- The mesh preview camera now centers each rendered frame on the pelvis/root joint and uses a closer fixed actor scale.
- The M17 planner now discourages repeated retrieval of the same motion unit after three consecutive uses when an alternative exists.

## Code Changes

- `src/music_motion_lab/pipelines/smplx_mesh_stitch_renderer.py`
  - Changed preview display coordinates to use per-frame pelvis/root centering.
  - Changed review bounds to a closer actor-scale camera instead of using the full root trajectory.
  - Added render metadata: `camera_mode=pelvis_centered_close_review`.
- `src/music_motion_lab/pipelines/streaming_smplx.py`
  - Added `M17_MAX_CONSECUTIVE_SAME_UNIT = 3`.
  - Added a `same_unit_run_limit_3` rejection path when the same motion unit has already run three times.
  - Added planner/render metrics:
    - `max_consecutive_motion_unit_run`
    - `repeat_unit_hard_reject_count`
- `tests/test_streaming_smplx.py`
  - Added coverage for the M17 same-unit repeat cap.
- `tests/test_smplx_mesh_stitch_renderer.py`
  - Added coverage for pelvis-centered preview display bounds.

## Audio0 Fx2 Review Output

- HTML: `outputs/renders/unity_audio0_m17_fx2_center_repeatcap_mesh_review.html`
- MP4: `outputs/renders/unity_audio0_m17_fx2_center_repeatcap_mesh_preview.mp4`
- Strip: `outputs/renders/unity_audio0_m17_fx2_center_repeatcap_mesh_strip.png`
- Planner report: `outputs/reports/unity_audio0_m17_fx2_center_repeatcap_planner_eval.json`
- Mesh report: `outputs/reports/unity_audio0_m17_fx2_center_repeatcap_mesh_report.json`

## Metrics

- `max_consecutive_motion_unit_run = 3`
- `repeat_unit_hard_reject_count = 2`
- `gap_count = 0`
- `future_visibility_violations = 0`
- `max_streaming_lookahead_sec = 3.0`
- `camera_mode = pelvis_centered_close_review`
- `camera_horizontal_radius = 0.836108`

## Verification

- `env PYTHONPATH=src python -m unittest tests.test_streaming_smplx tests.test_smplx_mesh_stitch_renderer`
  - Passed: `18 tests`
- `env PYTHONPATH=src python -m unittest discover -s tests`
  - Passed: `91 tests`

## Remaining Work

- The repeat cap reduces obvious repeated unit runs, but it does not yet solve higher-level phrase repetition where different units look visually similar.
- Camera centering improves review readability, but Unity runtime still needs its own avatar-follow camera contract.
- Beat confidence and speed fallback remain the next highest-impact M17/M18 improvements.
