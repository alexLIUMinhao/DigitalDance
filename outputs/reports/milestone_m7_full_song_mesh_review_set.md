# M7 Full Song Mesh Review Set

## Goal

Generate more complete-song review examples for the FineDance rhythmic-first SMPL-X mesh stitching workflow.

## Implemented

- Added `build-finedance-song-event-map` to convert M2-2 FineDance audio feature reports into full-song `song_event_map` JSON files.
- Generated complete-song plans, stitch manifests, mesh previews, reports, and HTML review pages for one local song and three FineDance dataset songs.
- Added a local index page for quick browser navigation across the complete-song reviews.

## Generated Complete-Song Reviews

- `outputs/renders/audio_full_mesh_review.html` - 237.23s, 70 stitched segments, 557 beats.
- `outputs/renders/finedance_001_full_mesh_review.html` - 59.50s, 15 stitched segments, 115 beats.
- `outputs/renders/finedance_060_full_mesh_review.html` - 58.77s, 8 stitched segments, 65 beats.
- `outputs/renders/finedance_088_full_mesh_review.html` - 39.83s, 10 stitched segments, 78 beats.
- `outputs/renders/full_song_mesh_review_index.html` links all four local review pages.

## Notes

- The current motion library used for these reviews is still the smoke FineDance rhythmic library from source sequence `001`, so the target music varies but source motion material is limited.
- The next useful step is rebuilding the motion library from more FineDance sequences, then re-running the same complete-song review set to increase movement diversity.
