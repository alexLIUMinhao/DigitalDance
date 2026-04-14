# Milestone M3: Source BVH Alignment

Timestamp: 2026-04-09 13:36 CST

Status:
- `FineDance 168` source BVH content is now numerically aligned with the upstream official mesh-preview joint truth.

Validated Artifacts:
- Official truth joints:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/previewCache/upstream_baselines/finedance/168/finedance_168_official_mesh_joints.json`
- Source BVH:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/previewCache/layer_compare/finedance/168/finedance_168_source.bvh`
- Validation report:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/previewCache/layer_compare/finedance/168/finedance_168_source_validation_report.json`
- Validation viewer:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/previewCache/layer_compare/finedance/168/finedance_168_source_validation_viewer.html`

Direct FK Metrics:
- Mean joint error: `7.572461446940387e-07`
- P95 joint error: `1.5431477569805975e-06`
- Max joint error: `2.0779316639794985e-06`

What Was Fixed:
- Added BVH-basis conversion for root positions, offsets, and local rotations before writing the source BVH.
- Added BVH-basis-aware extraction in the validation path.
- Fixed a BVH motion-channel ordering bug: channels now follow hierarchy traversal order instead of raw bridge joint index order.

Current Interpretation:
- `official joints` and `source FK` now overlap numerically.
- If any visible mismatch remains in DCC tools or viewers, the remaining issue is likely importer/runtime behavior rather than source BVH content.

Next Suggested Focus:
- Blender/import runtime diagnosis.
