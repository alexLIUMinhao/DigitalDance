# M8 Unity Music Mesh Review Set

Status: saved milestone for the Unity music full-song SMPL-X mesh review pass.

## Goal

Use the music files from `3d-digital-human/Assets/Resources/music` as target songs for the FineDance rhythmic-first SMPL-X stitching review pipeline.

## Source Music

- `audio.mp3` - 239.00s review, 71 stitched segments, 561 beats, 553 drum-hit markers.
- `audio1.mp3` - 238.97s review, 37 stitched segments, 292 beats, 519 drum-hit markers.
- `audio2.mp3` - 213.60s review, 34 stitched segments, 268 beats, 488 drum-hit markers.
- `audio3.mp3` - 208.93s review, 27 stitched segments, 214 beats, 494 drum-hit markers.
- `audio4.mp3` - 181.17s review, 49 stitched segments, 387 beats, 387 drum-hit markers.

## Generated Local Reviews

- `outputs/renders/unity_music_mesh_review_index.html`
- `outputs/renders/unity_audio_mesh_review.html`
- `outputs/renders/unity_audio1_mesh_review.html`
- `outputs/renders/unity_audio2_mesh_review.html`
- `outputs/renders/unity_audio3_mesh_review.html`
- `outputs/renders/unity_audio4_mesh_review.html`

The in-app review checkpoint was confirmed on `outputs/renders/unity_audio4_mesh_review.html`, with the full review set available from the index page above.

## Render Settings

- SMPL-X source mesh stitching from the current FineDance rhythmic smoke library.
- Transition smoothing: 12 frames, 2 passes.
- Render frame stride: 2, producing 15fps review videos for full-song inspection.
- Face stride: 30 for faster complete-song preview rendering.

## Milestone Acceptance

- Five Unity music tracks were processed into song event maps, rhythmic plans, stitch manifests, mesh review videos, HTML review pages, and JSON reports.
- Each review page includes linked music playback, synchronized review controls, beat/downbeat/drum timeline markers, source segment tables, transition metrics, and rhythm-lock frame error reporting.
- Rhythm-lock frame error is `0` in the generated reports.
- The highest reported smoothed temporal vertex delta across the five long-song previews is `0.055929`.
- Local unit verification for the branch previously passed with `76` tests; `pytest` remains unavailable in the default Python environment.

## Notes

- The song event maps are generated with automatic onset/beat analysis and currently carry manual-review flags; beat/downbeat correction should be added before final choreography decisions.
- Movement diversity is still limited by the current source library, which is based on FineDance sequence `001`.
- The generated render, JSON, and HTML review artifacts are local ignored outputs; this milestone document is the durable Git-tracked checkpoint for the M8 review state.

## Next Step

Select the best-looking Unity music track, add manual beat/downbeat overrides if needed, then re-render that selected track at full 30fps before expanding the FineDance source action library beyond sequence `001`.
