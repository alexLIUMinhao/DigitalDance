# Milestone M4: FineDance External smpl2bvh Dataset (203 Sequences)

## Summary
- Built a full FineDance external `smpl2bvh` BVH dataset covering all `203` raw motion files that exist in the upstream `motion/` directory.
- Output dataset root:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/datasets/finedance/external_smpl2bvh_dataset`
- Output BVH directory:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/datasets/finedance/external_smpl2bvh_dataset/bvh`
- Output manifest:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/datasets/finedance/external_smpl2bvh_dataset/manifest.json`

## Generation Contract
- Source motions: FineDance raw `.npy` files under:
  - `/Users/alex/Desktop/codex project/3d-digital/motion-base-assets/datasets/finedance/raw/extracted/finedance/motion`
- Route: external `smpl2bvh`
- Upstream reference:
  - `KosukeFukazawa/smpl2bvh`
- Frame rate:
  - `30 fps`
- Actual model family used for the batch:
  - `SMPL / MALE`

## Important Note
- The upstream `smpl2bvh` `SMPL-X` mode currently raises an `IndexError` on this FineDance payload.
- Because of that, the batch exporter automatically fell back to:
  - `modelType = smpl`
  - `gender = MALE`
- This fallback is recorded per sequence in the dataset manifest.

## Final Batch Result
- Sequence count:
  - `203`
- Completed:
  - `203`
- Failed:
  - `0`
- Dataset size:
  - `588M`
- Total frames:
  - `826983`
- Per-sequence frame count range:
  - `min = 521`
  - `max = 11781`

## Supporting Code
- Batch builder:
  - `/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/tools/build_finedance_external_smpl2bvh_dataset.py`
- Batch builder tests:
  - `/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/tests/test_build_finedance_external_smpl2bvh_dataset.py`

## Why This Milestone Matters
- This gives the project a reproducible external-BVH dataset baseline for all FineDance sequences, instead of relying on one-off exports.
- It also cleanly separates:
  - source-truth BVH experiments
  - Willa retarget experiments
  - reusable external dataset artifacts

## Next Suggested Step
- Spot-check a few representative sequences in Blender from this dataset before resuming the `Willa` retarget line.
