# M19 Windowed State-Machine Streaming Choreography

## Summary

M19 upgrades the M18 streaming choreography baseline from a loose rolling-window event view to an explicit three-part endpoint window:

- `history_sec = 2.0`: audio and motion context behind the current playhead.
- `main_window_sec = 2.0`: the current planning window used for rhythm scoring and concrete motion-unit selection.
- `future_sec = 1.0`: short transport/lookfront buffer used only to prepare upcoming accents or transitions.

The planner remains any-song first. It does not tune per song and keeps the M18 behavior for neutral intro, outro recover, conservative retiming, repeat limits, centered review camera, and the hybrid choreography state machine.

## Method

`simulate-streaming-song-events` now supports `--window-contract history_main_future --history-sec 2 --main-window-sec 2 --future-sec 1`. Each tick records:

- `history_window_sec = [playhead - 2s, playhead]`
- `main_window_sec = [playhead, playhead + 2s]`
- `future_window_sec = [playhead + 2s, playhead + 3s]`
- `visible_window_sec = [playhead - 2s, playhead + 3s]`

Beat, downbeat, drum, and accent events are annotated with `window_role` and also summarized as `history_events`, `main_events`, and `future_events`.

`simulate-streaming-smplx-plan --planner-version m19` uses these windows as follows:

- History context records previous unit, previous source sequence, previous choreography state, and past event density for continuity.
- Main window events drive rhythm lock scoring, target energy, segment hypotheses, and speed retime.
- Future events can trigger `accent_prepare` or transition preparation, but they do not replace the main-window rhythm score.

## Generated Reviews

Generated one complete M19 mesh review for each Unity music file:

- `outputs/renders/unity_audio0_m19_windowed_mesh_review.html`
- `outputs/renders/unity_audio1_m19_windowed_mesh_review.html`
- `outputs/renders/unity_audio2_m19_windowed_mesh_review.html`
- `outputs/renders/unity_audio3_m19_windowed_mesh_review.html`
- `outputs/renders/unity_audio4_m19_windowed_mesh_review.html`

Each review page includes a streaming decision table with the history/main/future window used for each step, and the timeline canvas draws thin H/M/F bands above the segment rhythm map.

## Metrics

| song | gap | future violations | max consecutive unit run | max total unit uses | max non-tail speed | lock error frames | final pose | outro recover sec |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| audio0 | 0 | 0 | 2 | 4 | 1.13333 | 0.87960 | `smplx_neutral_idle` | 11.81997 |
| audio1 | 0 | 0 | 2 | 5 | 1.10000 | 0.72000 | `smplx_neutral_idle` | 6.73997 |
| audio2 | 0 | 0 | 2 | 3 | 1.05861 | 0.48060 | `smplx_neutral_idle` | 7.33997 |
| audio3 | 0 | 0 | 2 | 5 | 1.14652 | 0.72030 | `smplx_neutral_idle` | 8.47668 |
| audio4 | 0 | 0 | 2 | 5 | 1.14257 | 0.60750 | `smplx_neutral_idle` | 5.06932 |

All five planner reports pass the M19 acceptance checks for no gaps, no future visibility violations, repeat limits, non-tail speed <= `1.15`, final neutral pose, and high-confidence lock error <= `2` frames.

## Notes And Next Steps

- M19 improves the streaming contract and state visibility rather than changing the FineDance motion library itself.
- The mesh reports still show transition vertex deltas above the stricter M18 visual target on some songs, so the next motion-quality milestone should focus on pose-space/root/contact-aware blending rather than only vertex smoothing.
- Subjective review still reports that actions do not feel precisely snapped to drum hits. M20 should focus on kick/snare/downbeat separation, true motion-accent annotation, and intra-unit pose-time warping so multiple action accents land on musical drum hits instead of only aligning segment starts.
- Runtime integration should use the same `history/main/future` contract so Unity can keep a small past context and a one-second transport lookfront without reading the whole song.
