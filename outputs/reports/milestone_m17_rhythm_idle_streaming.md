# M17 Rhythm Correction And Neutral Idle Streaming Batch

## Summary

- Goal: keep the project on `FineDance rhythmic-first + Any-song first` streaming SMPL-X choreography.
- This milestone introduced `m17`, a stricter rhythm-first planner with:
  - fixed `2s initial buffer + 2s lookahead + 1s lookfront`
  - first `5s` rendered as independent `smplx_neutral_idle`
  - stronger tempo hysteresis on the streaming event rail
  - harder rhythm / transition / non-tail speed rejection
  - low-confidence continuation that prefers calm continuity over short chaotic cuts
- Batch outputs were regenerated for `audio0-4`, with three effect variants per song:
  - `fx1`: steady (`cohort_size=6`)
  - `fx2`: balanced (`cohort_size=10`)
  - `fx3`: expressive (`cohort_size=14`)

## Code Changes

- `src/music_motion_lab/pipelines/streaming_smplx.py`
  - added `m17`
  - added `initial_pose_mode`
  - added synthetic `smplx_neutral_idle` / `smplx_neutral_rest` decision support
  - added tempo hysteresis, stronger rhythm scoring, and bounded initial idle extension
  - added planner/report metrics: `low_confidence_continuation_count`, `initial_upright_hold_sec`, `first_dance_start_sec`
- `src/music_motion_lab/pipelines/smplx_mesh_stitch_renderer.py`
  - added synthetic SMPL-X pose rendering for neutral idle / rest
  - extended stitch compose path to handle synthetic steps
  - isolated cache paths by plan digest so `fx` variants do not collide
- `src/music_motion_lab/cli.py`
  - exposed `m17`
  - exposed `--initial-pose-mode`
  - moved CLI defaults to the new M17 batch baseline
- `tests/test_streaming_smplx.py`
  - added M17 tests for neutral idle insertion, delayed first dance, and hard reject accounting

## Batch Outputs

- Render pages:
  - `outputs/renders/unity_audio0_m17_fx1_preview_mesh_review.html`
  - `outputs/renders/unity_audio0_m17_fx2_preview_mesh_review.html`
  - `outputs/renders/unity_audio0_m17_fx3_preview_mesh_review.html`
  - `outputs/renders/unity_audio1_m17_fx1_preview_mesh_review.html`
  - `outputs/renders/unity_audio1_m17_fx2_preview_mesh_review.html`
  - `outputs/renders/unity_audio1_m17_fx3_preview_mesh_review.html`
  - `outputs/renders/unity_audio2_m17_fx1_preview_mesh_review.html`
  - `outputs/renders/unity_audio2_m17_fx2_preview_mesh_review.html`
  - `outputs/renders/unity_audio2_m17_fx3_preview_mesh_review.html`
  - `outputs/renders/unity_audio3_m17_fx1_preview_mesh_review.html`
  - `outputs/renders/unity_audio3_m17_fx2_preview_mesh_review.html`
  - `outputs/renders/unity_audio3_m17_fx3_preview_mesh_review.html`
  - `outputs/renders/unity_audio4_m17_fx1_preview_mesh_review.html`
  - `outputs/renders/unity_audio4_m17_fx2_preview_mesh_review.html`
  - `outputs/renders/unity_audio4_m17_fx3_preview_mesh_review.html`

## Planner Snapshot

- `audio0`
  - `fx1`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=30`, `cross_sequence_transition_count=8`, `max_non_tail_speed_scale=2.03124`
  - `fx2`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=30`, `cross_sequence_transition_count=6`, `max_non_tail_speed_scale=2.03124`
  - `fx3`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=28`, `cross_sequence_transition_count=7`, `max_non_tail_speed_scale=2.08983`
- `audio1`
  - `fx1`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=79`, `cross_sequence_transition_count=7`, `max_non_tail_speed_scale=1.63576`
  - `fx2`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=80`, `cross_sequence_transition_count=5`, `max_non_tail_speed_scale=1.66017`
  - `fx3`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=78`, `cross_sequence_transition_count=8`, `max_non_tail_speed_scale=1.76811`
- `audio2`
  - `fx1`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=28`, `cross_sequence_transition_count=8`, `max_non_tail_speed_scale=1.4186`
  - `fx2`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=29`, `cross_sequence_transition_count=6`, `max_non_tail_speed_scale=2.09706`
  - `fx3`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=28`, `cross_sequence_transition_count=6`, `max_non_tail_speed_scale=1.90032`
- `audio3`
  - `fx1`: `first_dance_start_sec=5.02403`, `initial_upright_hold_sec=5.02403`, `gap_count=0`, `low_confidence_continuation_count=2`, `cross_sequence_transition_count=6`, `max_non_tail_speed_scale=1.1379`
  - `fx2`: `first_dance_start_sec=5.02403`, `initial_upright_hold_sec=5.02403`, `gap_count=0`, `low_confidence_continuation_count=2`, `cross_sequence_transition_count=7`, `max_non_tail_speed_scale=1.1379`
  - `fx3`: `first_dance_start_sec=5.02403`, `initial_upright_hold_sec=5.02403`, `gap_count=0`, `low_confidence_continuation_count=2`, `cross_sequence_transition_count=10`, `max_non_tail_speed_scale=1.1379`
- `audio4`
  - `fx1`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=13`, `cross_sequence_transition_count=7`, `max_non_tail_speed_scale=1.18964`
  - `fx2`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=13`, `cross_sequence_transition_count=7`, `max_non_tail_speed_scale=1.18964`
  - `fx3`: `first_dance_start_sec=6.0`, `low_confidence_continuation_count=13`, `cross_sequence_transition_count=7`, `max_non_tail_speed_scale=1.18964`

## What Improved

- First dance no longer starts at `33s+`; the bounded initial idle fix brought the first dance start back to roughly `5-6s`.
- The M17 batch now renders a true synthetic neutral idle instead of freezing the first dance source frame.
- Variant-specific cache collisions were removed.
- The tiny `audio3` handoff gap (`5.0s -> 5.02403s`) was fixed by extending the synthetic initial hold to the first credible dance boundary.

## What Is Still Not Good Enough

- The non-tail speed target from the M17 plan (`<= 1.08`) is still not being met on several songs.
- Low-confidence continuation remains too frequent on `audio0-2`, especially `audio1`.
- Cross-sequence transitions are still higher than desired for a fully calm runtime baseline.

## Next Focus

1. Tighten M17 recovery so fallback cannot escape the non-tail speed target so easily.
2. Improve event-rail confidence calibration for low-confidence songs before retrieval, especially `audio0-2`.
3. Reduce cross-sequence switching further by making same-sequence continuity cheaper than risky expression changes.
