# Milestone M10-M14 · Streaming SMPL-X Optimization

Generated: 2026-04-21
Branch: `codex/finedance-rhythmic-smplx-library`

## Summary

This milestone upgrades the M9 local streaming smoke into a broader optimization pass for endpoint-oriented rhythmic SMPL-X choreography. It keeps Willa out of scope and continues to use FineDance source SMPL-X motion units.

## Implemented

- M10 action-library coverage:
  - Full rhythmic-first FineDance library rebuilt with `2/4/8/16` beat units.
  - Annotated with count grids, accent locks, movement quality, safe retime ranges, and joints-based contact proxy windows.
  - Coverage report: `outputs/reports/milestone_m10_motion_library_coverage_report.json`.
- M11 streaming event-rail reliability:
  - Streaming event generation now uses normalized online tempo hypotheses to reduce half/double-time mistakes.
  - Added calibration and event-rail evaluation CLIs.
  - Event report: `outputs/reports/milestone_m11_streaming_event_rail_eval.json`.
- M12 planner:
  - Added `--planner-version m12` and `--tail-policy recover`.
  - Planner now records phrase-aware mode, stronger diversity/speed penalties, and tail recovery decisions.
  - Planner reports:
    - `outputs/reports/milestone_m12_streaming_planner_eval.json`
    - `outputs/reports/milestone_m12_audio_pressure_planner_eval.json`
- M13 transition quality:
  - Added transition metrics for root acceleration discontinuity, joint jerk, and foot-slide proxies.
  - Added a temporal delta limiter inside transition smoothing windows.
  - Transition report: `outputs/reports/milestone_m13_motion_transition_report.json`.
- M14 runtime bundle:
  - Added Unity/endpoint runtime bundle export with compact motion-unit manifest, planner config, and stream samples.
  - Bundle report: `outputs/runtime_bundles/unity_streaming_smplx_bundle/bundle_report.json`.

## Key Metrics

- Motion library coverage:
  - Units: `10067`
  - Sequences: `100`
  - Duration coverage: `2b=3132`, `4b=2943`, `8b=2096`, `16b=1896`
  - Contact annotation ratio: `1.0`
  - Missing required beat units: none
- `audio4` M12 planner:
  - Decisions: `97`
  - Future visibility violations: `0`
  - Gaps: `0`
  - Speed scale outside `0.85-1.15`: `2/97` (`0.02062`)
  - Max non-tail speed scale: `1.15413`
- `audio.mp3` pressure planner:
  - Decisions: `139`
  - Future visibility violations: `0`
  - Gaps: `0`
  - Speed scale outside `0.85-1.15`: `0`
  - Max speed scale: `1.1328`
- M13 transition smoothing:
  - Max temporal vertex delta after smoothing: `0.048`
  - Max temporal joint delta after smoothing: `0.06`
  - Rhythm lock frame error: `0`
- Runtime bundle:
  - Compact unit count: `10067`
  - Motion-unit manifest size: `32348234` bytes
  - Python SMPL-X runtime required by endpoint bundle: no

## Verification

- `python -m py_compile src/music_motion_lab/cli.py src/music_motion_lab/pipelines/finedance_rhythmic_library.py src/music_motion_lab/pipelines/streaming_smplx.py src/music_motion_lab/pipelines/smplx_mesh_stitch_renderer.py`
- `env PYTHONPATH=src python -m unittest discover -s tests`
  - Result: `Ran 84 tests ... OK`
- `env PYTHONPATH=src python -m pytest tests`
  - Not run: local Python environment has no `pytest` module.

## Remaining Work

- Replace contact proxies with true SMPL-X foot-joint contact from cached joints for the production runtime bundle.
- Add manual beat/downbeat calibration files for all Unity music tracks instead of self-consistency event-rail evaluation.
- Validate the M12 full-library planner with true mesh render on selected non-`audio4` songs; current heavy visual render was refreshed on `audio4`.
- Port the runtime bundle consumer into Unity and measure device memory, load time, and playback frame rate.
