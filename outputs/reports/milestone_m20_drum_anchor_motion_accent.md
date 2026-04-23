# M20 Drum-Anchor Motion-Accent Lock

## Goal

M20 continues the any-song streaming SMPL-X choreography path from M19. The specific problem addressed here is subjective rhythm perception: previous plans could report low frame error while the visible body action still did not feel precisely snapped to kick/snare/downbeat accents.

## Method

- The streaming event rail now emits `drum_anchor_events`.
  - Sources: downbeat, low-band kick, high-attack/snare-like hits, and generic onset accents.
  - Each anchor carries `anchor_role`, `priority`, `confidence`, `strength`, and `window_role`.
- FineDance motion units now include `drum_lock_frames`.
  - These combine count-grid locks, root-speed motion accents, source frames, local frames, beat offsets, and lock roles.
  - The planner can distinguish a theoretical count from a body/action accent frame.
- `planner-version m20` changes rhythm scoring from motion-lock-centric to drum-anchor-centric.
  - For each visible drum anchor in the 2s main window, it finds the best matching motion/body accent lock.
  - Future 1s remains prepare-only, following the M19 window contract.
  - The score weights are more rhythm-heavy: rhythm `0.67`, transition `0.18`, style/BPM `0.08`, quality `0.04`, diversity `0.03`.
- M20 emits a `source_time_warp` per selected motion step.
  - Mode: `piecewise_linear_motion_accent_to_drum_anchor`.
  - This aligns selected source motion accent frames to the drum-anchor scene frames while preserving segment start/end.
  - Mesh review HTML shows the source time-warp mode and anchor count per decision.
- The visual timeline now draws drum anchors in addition to beats, downbeats, drum hits, accents, H/M/F windows, states, and transitions.

## Generated Reviews

- `outputs/renders/unity_audio0_m20_drum_anchor_mesh_review.html`
- `outputs/renders/unity_audio1_m20_drum_anchor_mesh_review.html`
- `outputs/renders/unity_audio2_m20_drum_anchor_mesh_review.html`
- `outputs/renders/unity_audio3_m20_drum_anchor_mesh_review.html`
- `outputs/renders/unity_audio4_m20_drum_anchor_mesh_review.html`

## Planner Metrics

| song | gaps | future violations | max same run | max total use | max non-tail speed | high-conf lock err | post-warp body err p95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| audio0 | 0 | 0 | 2 | 4 | 1.08333 | 1.5165 frames | 0.0 frames |
| audio1 | 0 | 0 | 2 | 4 | 1.10000 | 1.3005 frames | 0.0 frames |
| audio2 | 0 | 0 | 2 | 4 | 1.13333 | 1.5105 frames | 0.0 frames |
| audio3 | 0 | 0 | 2 | 4 | 1.14652 | 1.7781 frames | 0.0 frames |
| audio4 | 0 | 0 | 2 | 5 | 1.15000 | 1.3170 frames | 0.0 frames |

## Mesh Metrics

| song | max temporal vertex delta after smoothing |
|---|---:|
| audio0 | 0.069146 |
| audio1 | 0.111450 |
| audio2 | 0.069558 |
| audio3 | 0.078128 |
| audio4 | 0.088820 |

## Verification

- `env PYTHONPATH=music-motion-lab/src python3 -m unittest discover -s music-motion-lab/tests`
  - Passed: `95` tests.
- Rendered real SMPL-X mesh review HTML/MP4/strip/report for `audio0-audio4`.

## Notes And Next Step

M20 specifically improves the rhythm perception path by aligning body accent frames to drum anchors. It does not fully solve motion smoothness: several mesh reports still exceed the strict M13/M18 visual transition target. The next milestone should focus on pose-space/root/contact-aware blending so the new drum-locked timing does not introduce sharper visual motion changes at boundaries.
