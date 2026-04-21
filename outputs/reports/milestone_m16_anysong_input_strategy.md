# Milestone M16 · Any-Song Streaming Input Strategy `2+2+1`

Generated: 2026-04-21
Branch: `codex/finedance-rhythmic-smplx-library`

## Summary

This milestone locks the streaming simulator onto the new any-song-first default input strategy:

- `2s` initial buffer before playback
- `2s` planner lookahead
- `1s` transport/lookfront reserve

The implementation goal is not per-song tuning. It standardizes a single future-visibility contract that later planner and transition work must respect across different songs.

## Implemented

- Streaming event rail defaults:
  - Added explicit `lookahead_sec=2.0` and `lookfront_sec=1.0` defaults next to the existing `initial_buffer_sec=2.0`.
  - Event stream headers and per-tick records now carry `lookahead_sec`, `lookfront_sec`, `planning_audio_until_sec`, and `total_future_sec`.
- Planner guard alignment:
  - Streaming planner now reads the new header contract and computes decision timing from `total_future_sec` instead of the old single-buffer assumption.
  - Future-visibility guards now report `lookahead_sec`, `lookfront_sec`, `total_future_sec`, and `used_audio_until_sec`.
  - Initial hold metadata was aligned to the same future-visibility contract.
- Any-song stabilization kept active:
  - M15 cohort retrieval remains the main planner.
  - Low-confidence fallback, sequence dwell, and stricter cross-sequence transition rejection remain enabled so the broader input horizon improves rhythm lock without increasing visual jitter.

## Key Outputs

- Planner eval:
  - `outputs/reports/unity_audio4_m16_anysong_planner_eval.json`
    - `decision_count=98`
    - `future_visibility_violations=0`
    - `gap_count=0`
    - `speed_scale_outside_default_range_ratio=0.0102`
    - `max_non_tail_speed_scale=1.0831`
    - `transition_hard_reject_count=13`
    - `cross_sequence_transition_count=1`
  - `outputs/reports/unity_audio_m16_anysong_planner_eval.json`
    - `decision_count=154`
    - `future_visibility_violations=0`
    - `gap_count=0`
    - `speed_scale_outside_default_range_ratio=0.01299`
    - `max_non_tail_speed_scale=1.15234`
    - `transition_hard_reject_count=66`
    - `cross_sequence_transition_count=5`
- Review artifacts:
  - `outputs/renders/unity_audio4_m16_anysong_mesh_review.html`
  - `outputs/renders/unity_audio4_m16_anysong_mesh_preview.mp4`
  - `outputs/reports/unity_audio4_m16_anysong_mesh_report.json`

## Interpretation

- The new `2+2+1` contract held cleanly on both songs: no future-visibility violations and no playback gaps.
- Compared with the previous any-song baseline, `audio4` became more conservative and stable:
  - cross-sequence transitions dropped to `1`
  - non-tail speed stayed within `1.0831`
- The longer `audio` run also stayed inside the current speed acceptance band while keeping cross-sequence switching relatively low for a multi-song pool.

This is not the final quality target yet, but it is a better generic operating point for later rhythm and transition optimization than the previous tighter future window.

## Verification

- `python -m py_compile src/music_motion_lab/cli.py src/music_motion_lab/pipelines/streaming_smplx.py tests/test_streaming_smplx.py`
- `env PYTHONPATH=src python -m unittest discover -s tests`
  - Result: `Ran 88 tests ... OK`

## Remaining Work

- Keep optimizing against the unified `2+2+1` input contract instead of song-specific tuning.
- Improve transition quality further with stronger pose/contact-aware smoothing so the visual result catches up with the improved planner metrics.
- Expand any-song evaluation beyond the current Unity sample songs into a larger batch report.
