# M22 Stable Fallback And Contact-Aware Transition Smoothing

## Goal

M22 continues the FineDance rhythmic-first any-song streaming SMPL-X choreography line. It builds on M21 and targets two remaining failure modes:

- No-valid-candidate windows should not retrieve marginal motions with high retime stress.
- Mesh transitions should reduce visible vertex/joint jumps at segment boundaries.

## Method

### Planner

- Added `planner-version m22`.
- M22 keeps the M21 dual-window strategy:
  - hard real-time window: `2s history + 2s main + 1s future`
  - phrase context: `16s history + 4s future`, used only for style/state preparation
- M22 keeps M21 quality gates, drum-anchor/body-accent matching, local accent retime feasibility, repeat limits, intro neutral idle, and outro neutral recover.
- M22 tightens recovery behavior:
  - Preferred candidate retime stress remains conservative.
  - Selected non-tail motions are kept below the M21 hard limit and M22 avoids recovery candidates above `0.68` retime stress.
  - If no valid motion exists and the best recovery candidate is still too risky, M22 inserts a short synthetic `m22_stable_groove_fallback` neutral bridge instead of forcing a visibly bad motion.
  - The fallback is intentionally sparse: it is only used for high retime stress, hard speed violation, risky transition, or severe quality-gate failure.

### Mesh Transition

- Boundary smoothing now applies adaptive temporal-delta limits after the existing visual smoothing pass.
- Each transition report records:
  - vertex/joint delta limits
  - temporal vertex/joint delta before and after smoothing
  - foot-slide proxy before and after smoothing
  - root velocity discontinuity proxy before and after smoothing
- This is still a visual-review blend, not full runtime IK. It gives a cleaner target for the next Unity/runtime motion-blend implementation.

## Generated Artifacts

- `outputs/streaming/unity_audio[0-4]_m22_transition_fallback_stream_events.jsonl`
- `outputs/streaming/unity_audio[0-4]_m22_transition_fallback_stream_plan.jsonl`
- `outputs/reports/unity_audio[0-4]_m22_transition_fallback_planner_eval.json`
- `outputs/renders/unity_audio[0-4]_m22_transition_fallback_mesh_review.html`
- `outputs/renders/unity_audio[0-4]_m22_transition_fallback_mesh_preview.mp4`
- `outputs/renders/unity_audio[0-4]_m22_transition_fallback_mesh_strip.png`
- `outputs/reports/unity_audio[0-4]_m22_transition_fallback_mesh_report.json`

## Batch Metrics

| song | fallback | gap | future violation | max lock error frames | max retime stress | max speed | max motion run | max total uses | max temporal vertex delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| audio0 | 2 | 0 | 0 | 1.4763 | 0.67514 | 1.13043 | 1 | 4 | 0.052 |
| audio1 | 2 | 0 | 0 | 1.2312 | 0.63111 | 1.11594 | 1 | 2 | 0.052 |
| audio2 | 3 | 0 | 0 | 1.6149 | 0.67611 | 1.12319 | 1 | 2 | 0.052 |
| audio3 | 0 | 0 | 0 | 1.5555 | 0.6539 | 1.11594 | 2 | 2 | 0.052 |
| audio4 | 2 | 0 | 0 | 1.5513 | 0.55 | 1.12319 | 1 | 2 | 0.047577 |

## Verification

- `env PYTHONPATH=music-motion-lab/src python3 -m unittest music-motion-lab/tests/test_streaming_smplx.py`
- Generated and opened `outputs/renders/unity_audio4_m22_transition_fallback_mesh_review.html`.

## Remaining Work

- Replace the synthetic fallback bridge with a learned or curated low-risk groove hold so recovery windows still feel dance-like.
- Move from visual vertex/joint smoothing toward pose-space blend plus contact-aware IK for the Unity/runtime bundle.
- Continue subjective review: M22 is safer and smoother than M21, but rhythm feel still depends on the quality of online drum-anchor detection.
