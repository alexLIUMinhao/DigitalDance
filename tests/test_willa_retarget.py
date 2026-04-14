from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from music_motion_lab.cli import _blender_app_bundle_path
from music_motion_lab.willa_retarget import (
    BRIDGE_BONE_ORDER_22,
    angle_deg,
    default_willa_retarget_profile_path,
    load_willa_retarget_profile,
    next_versioned_retarget_paths,
    normalize_retarget_strategy,
    resolve_reference_vector,
    validate_willa_retarget_profile,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class WillaRetargetProfileTests(unittest.TestCase):
    def test_profile_path_exists(self) -> None:
        self.assertTrue(default_willa_retarget_profile_path(PROJECT_ROOT).exists())

    def test_profile_loads_and_has_all_22_mappings(self) -> None:
        profile = load_willa_retarget_profile(PROJECT_ROOT)
        validate_willa_retarget_profile(profile)
        self.assertEqual(profile["sourceBoneOrder"], BRIDGE_BONE_ORDER_22)
        self.assertEqual(len(profile["boneMap"]), 22)
        self.assertEqual(profile["boneMap"]["pelvis"], "J_Bip_C_Hips")
        self.assertEqual(profile["boneMap"]["left_wrist"], "J_Bip_L_Hand")
        self.assertEqual(profile["boneMap"]["right_wrist"], "J_Bip_R_Hand")
        self.assertEqual(profile["referenceDatasetProfile"], "finedance")
        self.assertEqual(profile["basisMode"], "local_rest_delta")
        self.assertFalse(profile["useBasisMapOverrides"])
        self.assertEqual(profile["rotationDisabledSourceJoints"], [])
        self.assertEqual(
            profile["referenceSourceToTargetBasisJoints"],
            [
                "spine1",
                "spine2",
                "spine3",
                "neck",
                "head",
                "left_collar",
                "right_collar",
                "left_shoulder",
                "right_shoulder",
                "left_elbow",
                "right_elbow",
                "left_wrist",
                "right_wrist",
                "left_ankle",
                "right_ankle",
                "left_foot",
                "right_foot",
            ],
        )
        for joint_name in ["left_collar", "right_collar", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_foot", "right_foot"]:
            self.assertIn(joint_name, profile["basisMapOverrides"])
        for joint_name in ["left_collar", "right_collar", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_foot", "right_foot"]:
            self.assertIn(joint_name, profile["basisCorrectionTransforms"])
        self.assertTrue(profile["defaultSourceBvh"].endswith("finedance_168_source_willa_aligned.bvh"))
        self.assertTrue(profile["defaultInputBvh"].endswith("finedance_168_source_willa_canonical_posy_r001.bvh"))
        self.assertTrue(profile["defaultCanonicalReport"].endswith("finedance_168_source_willa_canonicalization_posy_r001.json"))
        self.assertEqual(profile["defaultRetargetStrategy"], "constraint_bake")
        self.assertEqual(profile["defaultConstraintBakeBasisMode"], "constraint_world_pelvis_torso_local")
        self.assertEqual(profile["humanToTarget"]["hips"], "J_Bip_C_Hips")

    def test_root_and_motion_root_remain_distinct(self) -> None:
        profile = load_willa_retarget_profile(PROJECT_ROOT)
        self.assertEqual(profile["targetRootBone"], "Root")
        self.assertEqual(profile["targetMotionRootBone"], "J_Bip_C_Hips")
        self.assertNotEqual(profile["targetRootBone"], profile["targetMotionRootBone"])

    def test_target_parent_chain_uses_willa_bone_names(self) -> None:
        profile = load_willa_retarget_profile(PROJECT_ROOT)
        target_parents = dict(profile["targetParents"])
        self.assertIsNone(target_parents["J_Bip_C_Hips"])
        self.assertEqual(target_parents["J_Bip_C_Spine"], "J_Bip_C_Hips")
        self.assertEqual(target_parents["J_Bip_L_Foot"], "J_Bip_L_LowerLeg")
        self.assertEqual(target_parents["J_Bip_L_Hand"], "J_Bip_L_LowerArm")


class WillaRetargetMathTests(unittest.TestCase):
    def test_resolve_reference_vector_pair_and_cross(self) -> None:
        positions = {
            "root": [0.0, 0.0, 0.0],
            "left": [1.0, 0.0, 0.0],
            "right": [0.0, 1.0, 0.0],
            "up": [0.0, 0.0, 1.0],
        }
        positions = {key: __import__("numpy").asarray(value, dtype=float) for key, value in positions.items()}
        parents = {"root": None, "left": "root", "right": "root", "up": "root"}
        pair = resolve_reference_vector(positions, parents, "root", {"mode": "pair", "from": "left", "to": "right"})
        cross = resolve_reference_vector(
            positions,
            parents,
            "root",
            {
                "mode": "cross",
                "a": {"mode": "child", "joint": "left"},
                "b": {"mode": "child", "joint": "right"},
            },
        )
        self.assertAlmostEqual(float(pair[0]), 1.0)
        self.assertAlmostEqual(float(pair[1]), -1.0)
        self.assertAlmostEqual(float(cross[2]), 1.0)

    def test_angle_deg_returns_zero_for_matching_vectors(self) -> None:
        import numpy as np

        self.assertAlmostEqual(angle_deg(np.asarray([1.0, 0.0, 0.0]), np.asarray([2.0, 0.0, 0.0])), 0.0)


class WillaRetargetCliTests(unittest.TestCase):
    def test_strategy_normalization(self) -> None:
        self.assertEqual(normalize_retarget_strategy("constraint-bake"), "constraint_bake")
        self.assertEqual(normalize_retarget_strategy("rest-pose-first"), "rest_pose_first")

    def test_next_versioned_retarget_paths_increments(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            blend_base = root / "renders" / "finedance_168_willa_retarget.blend"
            report_base = root / "reports" / "finedance_168_willa_retarget_report.json"
            blend_base.parent.mkdir(parents=True, exist_ok=True)
            report_base.parent.mkdir(parents=True, exist_ok=True)
            first_blend, first_report, first_index = next_versioned_retarget_paths(
                blend_base,
                report_base,
                "constraint-bake",
            )
            self.assertEqual(first_index, 1)
            first_blend.touch()
            first_report.touch()
            second_blend, second_report, second_index = next_versioned_retarget_paths(
                blend_base,
                report_base,
                "constraint-bake",
            )
            self.assertEqual(second_index, 2)
            self.assertTrue(first_blend.name.endswith("constraint_bake_r001.blend"))
            self.assertTrue(second_blend.name.endswith("constraint_bake_r002.blend"))
            self.assertTrue(first_report.name.endswith("constraint_bake_r001.json"))
            self.assertTrue(second_report.name.endswith("constraint_bake_r002.json"))

    def test_blender_app_bundle_path_from_binary(self) -> None:
        path = _blender_app_bundle_path("/Applications/Blender.app/Contents/MacOS/Blender")
        self.assertEqual(path, Path("/Applications/Blender.app"))

    def test_blender_app_bundle_path_from_app_bundle(self) -> None:
        path = _blender_app_bundle_path("/Applications/Blender.app")
        self.assertEqual(path, Path("/Applications/Blender.app"))


if __name__ == "__main__":
    unittest.main()
