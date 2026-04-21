# M9 Streaming SMPL-X Mesh Choreography Simulator

Status: saved milestone for the endpoint-style local-song streaming simulation.

## Goal

Upgrade the M8 offline full-song review into a streaming simulation: a local song is treated as if it were arriving during playback, with an initial 2-second buffer and a continuous 2-second lookahead guard. The runtime target remains FineDance source SMPL-X mesh, not Willa.

## Implemented

- Added `simulate-streaming-song-events` to emit JSONL tick records with rolling onset, low-band/kick, high-attack, tempo hypotheses, beat phase, downbeat confidence, drum hits, accents, and `available_audio_until_sec`.
- Added `annotate-finedance-motion-units` to enrich FineDance units with count-grid locks, accent lock frames, root/motion accents, foot-contact proxies, entry/exit anchors, movement quality, and safe retime ranges.
- Added `simulate-streaming-smplx-plan` to retrieve motion units under the 2-second visibility guard with fixed scoring weights: rhythm lock 45%, transition 25%, style/energy/BPM 20%, diversity 10%.
- Added `render-streaming-smplx-mesh-review` to convert the streaming plan into the existing true SMPL-X mesh stitch renderer and produce M8-style MP4/HTML review artifacts with a streaming decision table.

## Generated Smoke Artifacts

- `outputs/streaming/unity_audio4_stream_events.jsonl`
- `outputs/streaming/unity_audio4_stream_plan.jsonl`
- `outputs/streaming/unity_audio4_streaming_stitch_manifest.json`
- `outputs/renders/unity_audio4_streaming_mesh_preview.mp4`
- `outputs/renders/unity_audio4_streaming_mesh_strip.png`
- `outputs/renders/unity_audio4_streaming_mesh_review.html`
- `outputs/reports/milestone_m9_streaming_smplx_mesh_report.json`

## Acceptance Snapshot

- Song: `3d-digital-human/Assets/Resources/music/audio4.mp3`.
- Initial buffer: `2.0s`.
- Streaming decisions: `97`.
- Future visibility violations: `0`.
- Max streaming lookahead: `2.0s`.
- Scene frames: `5447`; rendered review frames: `2724` at `15fps`.
- Transitions: `96`.
- Rhythm lock max frame error: `0`.
- Max temporal vertex delta after smoothing: `0.059308`, meeting the M9 smoke target of `<= 0.06`.
- Speed scale outside default `0.85-1.15`: `1` decision, the final short song-tail segment.

## Pressure Test

- Also ran streaming events and planning for `audio.mp3`.
- Result: `102` decisions, `0` future visibility violations, max lookahead `2.0s`.
- Limitation: `94` decisions fell outside the default speed scale range because the current smoke action library is still narrow and based on FineDance sequence `001`; it lacks enough 2-beat, low-speed, and style-diverse alternatives for this track.

## Verification

- `env PYTHONPATH=src python -m unittest discover -s tests` passed: `81` tests.
- `python -m pytest tests` could not run because `pytest` is not installed in the default Python environment.

## Next Step

Expand the FineDance action library beyond the smoke sequence and explicitly generate 2/4/8-beat units, then rerun the M9 streaming planner on all Unity music tracks. The next quality upgrade should replace review-only vertex smoothing with pose/root-channel transition blending.
