# M6 SMPL-X Mesh Visual Stitch Test

## Goal

Validate the final FineDance source-SMPL-X mesh stitching path from an M2-2 rhythmic stitch manifest, without mapping to Willa.

## Implemented

- Added a true SMPL-X mesh stitch renderer that consumes a stitch manifest, materializes FineDance source mesh caches, aligns root X/Z continuity, bridges timeline gaps, and applies a short vertex/joint crossfade at transitions.
- Added MP4, strip PNG, JSON report, and HTML review outputs for visual inspection.
- Kept heavy dependencies behind the smoke render command; unit tests use small fake mesh caches and do not require `torch`, `smplx`, `imageio`, or `matplotlib`.

## Expected Smoke Outputs

- `outputs/renders/_smoke_audio_stitch_mesh_preview.mp4`
- `outputs/renders/_smoke_audio_stitch_mesh_strip.png`
- `outputs/renders/_smoke_audio_stitch_mesh_review.html`
- `outputs/reports/_smoke_audio_stitch_mesh_report.json`

## Notes

- The current transition smoothing is a visual vertex/joint crossfade for review. If the visual result is acceptable, the next upgrade should move blending earlier into pose/root channels so generated units can become a reusable motion asset.
- Rhythm locks are evaluated in scene-frame space; the renderer keeps the planned target frames and reports lock frame error.
