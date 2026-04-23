# M21 Dual Window Quality And Accent-Retime Milestone

## Goal

M21 keeps the any-song streaming contract from M19/M20, but adds a second low-frequency context layer so the planner is not only chasing immediate drum hits. The short window remains real-time safe for beat locks; the phrase context only changes music intent, source-song cohort choice, candidate quality filtering, and next-action preparation.

## Streaming Inputs

- Hard real-time window: `history=2s`, `main=2s`, `future=1s`.
- Phrase/style context: `history=16s`, `future=4s`.
- `available_audio_until_sec` still follows the hard window and is checked by the future guard.
- The phrase future is marked `style_state_and_prepare_only`; it is not used to place current-window rhythm locks.
- If a runtime cannot see the 4s phrase future, the contract falls back to prediction from the 16s history trend.

## Music Intent

Each tick now emits `music_intent`:

- `smooth_flow`: lower accent density, smoother energy motion, prefer longer `8/16 beat` smooth units and lower transition risk.
- `drum_lock`: strong beat/drum confidence, prefer `2/4 beat` units with clear body accents.
- `build_up` / `drop_prepare`: future phrase energy or accent density is rising, prepare larger or sharper movement while the main window still decides the actual action boundary.

## Motion Quality Annotation

`annotate-finedance-motion-units` now writes:

- `motion_quality_score`
- `accent_clarity_score`
- `danceability_score`
- `transition_risk_score`
- `retime_stress_score`
- `quality_bucket`
- `quality_thresholds`
- `quality_threshold_source`
- `quality_gate_passed`
- `quality_gate_reasons`

Thresholds are adaptive by bucket: `duration_beats / energy / movement_quality`. Small buckets fall back to global quantiles and mark `quality_threshold_source=global_fallback_small_bucket`.

## Candidate Retrieval

M21 retrieval order:

1. Select a source-song cohort from phrase/style context.
2. Score unit quality and apply adaptive quality gates.
3. Score visible main-window drum/body-accent lock.
4. Check local accent retime feasibility.
5. Apply transition, speed, repetition, and diversity constraints.
6. If the cohort has no valid unit, expand to the wider same-duration library before using recovery ranking.

The selected plan still preserves the M17/M18 constraints: first 5s neutral idle, final 5s recover, consecutive same unit ideally `<=2`, hard total same unit use `<=5`, and non-tail speed within `0.85-1.15`.

## Accent-Retime Feasibility

M21 does not only check whole-segment `speed_scale`. For visible drum-anchor locks it computes local speed between source body-accent anchors and target drum anchors:

- preferred local speed: `0.90-1.10`
- hard segment speed: `0.85-1.15`
- local warp hard bound: `0.80-1.20`

Candidates that need excessive local stretch/compression are rejected or heavily downranked. Review pages expose `retime_stress_score` and `anchor_speed_segments`.

## Generated Reviews

- `outputs/renders/unity_audio0_m21_dual_quality_mesh_review.html`
- `outputs/renders/unity_audio1_m21_dual_quality_mesh_review.html`
- `outputs/renders/unity_audio2_m21_dual_quality_mesh_review.html`
- `outputs/renders/unity_audio3_m21_dual_quality_mesh_review.html`
- `outputs/renders/unity_audio4_m21_dual_quality_mesh_review.html`

## Planner Metrics

| Song | Decisions | Gap | Future Violations | Max Non-tail Speed | Max Same Run | Max Total Uses | Lock Error Frames | Post-warp Body Accent P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| audio0 | 56 | 0 | 0 | 1.10000 | 1 | 4 | 1.5000 | 0.0 |
| audio1 | 65 | 0 | 0 | 1.10000 | 2 | 3 | 1.4532 | 0.0 |
| audio2 | 50 | 0 | 0 | 1.14493 | 2 | 3 | 1.7205 | 0.0 |
| audio3 | 84 | 0 | 0 | 1.12354 | 2 | 3 | 1.5555 | 0.0 |
| audio4 | 58 | 0 | 0 | 1.09420 | 2 | 3 | 1.5627 | 0.0 |

## Current Limitations

- The phrase future is simulated from local files; runtime integration still needs a real transport-buffer contract.
- Some selected recovery windows still have high retime stress when no fully valid motion exists, though full-song gaps remain zero.
- Mesh transition smoothness is still not solved at production level: `max_temporal_vertex_delta_after_smoothing` is around `0.074-0.113` across this batch.
- M22 should focus on pose-space/contact-aware transition smoothing and a better stable fallback state for no-valid-candidate windows.

## Verification

- `env PYTHONPATH=music-motion-lab/src python3 -m unittest discover -s music-motion-lab/tests`
- Rendered real SMPL-X mesh review HTML/MP4/strip/report for Unity `audio0-audio4`.
