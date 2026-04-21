# Milestone M15 · Rhythmic-First Multi-Song Streaming Retrieval

Generated: 2026-04-21
Branch: `codex/finedance-rhythmic-smplx-library`

## Summary

This milestone replaces the single-sequence streaming review path with a rhythmic-first multi-song candidate pool. Every selected FineDance source song now contributes only middle-segment motion units by applying a shared `10s` source cutoff, and the planner uses an `m15` two-stage retrieval strategy: style cohort first, unit retrieval second.

## Implemented

- Rhythmic-first multi-song middle-segment library:
  - Built from the existing FineDance `rhythmic_first` subset.
  - Uniform `--min-source-sec 10` cutoff across all source songs.
  - Unit metadata now keeps `priority_tier`, `source_song_bpm`, `source_song_energy`, `source_song_style_tags`, and `source_song_quality_weight`.
- M15 planner:
  - Added style-cohort retrieval with a default `cohort_size=10`.
  - Cohort ranking uses BPM, energy, movement-quality proxy, style tags, and source-quality weight.
  - Unit scoring now records `selected_from_sequence`, `selected_from_tier`, `cohort_source_sequences`, `cohort_rankings`, and `rejected_top_candidates`.
  - Hard gates reject low-confidence candidates when `rhythm_lock < 0.40` or `transition_smoothness < 0.45`.
- Review/report visibility:
  - HTML streaming table now shows source tier, current cohort songs, and top rejected candidates.
  - Planner/render reports now expose `rhythm_hard_reject_count`, `transition_hard_reject_count`, `cross_sequence_transition_count`, and cohort coverage.

## Key Outputs

- Coverage report:
  - `outputs/reports/milestone_m15_rhythmic_first_midcut10_coverage_report.json`
  - `unit_count=8200`, `sequence_count=100`, `priority_tier.rhythmic_first=8200`
- Planner eval:
  - `outputs/reports/unity_audio4_m15_rhythmic_first_midcut10_planner_eval.json`
  - `decision_count=97`
  - `future_visibility_violations=0`
  - `speed_scale_outside_default_range_ratio=0.08247`
  - `max_non_tail_speed_scale=1.15413`
  - `rhythm_hard_reject_count=114`
  - `cross_sequence_transition_count=33`
- Review artifacts:
  - `outputs/renders/unity_audio4_m15_rhythmic_first_midcut10_mesh_review.html`
  - `outputs/renders/unity_audio4_m15_rhythmic_first_midcut10_mesh_preview.mp4`
  - `outputs/reports/unity_audio4_m15_rhythmic_first_midcut10_mesh_report.json`

## Verification

- `python -m py_compile src/music_motion_lab/cli.py src/music_motion_lab/pipelines/finedance_rhythmic_library.py src/music_motion_lab/pipelines/streaming_smplx.py src/music_motion_lab/pipelines/smplx_mesh_stitch_renderer.py tests/test_finedance_rhythmic_library.py tests/test_streaming_smplx.py`
- `env PYTHONPATH=src python -m unittest discover -s tests`
  - Result: `Ran 88 tests ... OK`

## Remaining Work

- Reduce cross-sequence transition count without collapsing back to a single-sequence planner.
- Strengthen transition hard rejects by adding more explicit contact/pose mismatch penalties.
- Extend the same `m15` interface from rhythmic-first to the full `203`-song FineDance pool once the multi-song rhythmic-first behavior is visually acceptable.
