# M8 Unity Music Mesh Review Set

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

## Render Settings

- SMPL-X source mesh stitching from the current FineDance rhythmic smoke library.
- Transition smoothing: 12 frames, 2 passes.
- Render frame stride: 2, producing 15fps review videos for full-song inspection.
- Face stride: 30 for faster complete-song preview rendering.

## Notes

- The song event maps are generated with automatic onset/beat analysis and currently carry manual-review flags; beat/downbeat correction should be added before final choreography decisions.
- Movement diversity is still limited by the current source library, which is based on FineDance sequence `001`.
