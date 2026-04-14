from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.source_bvh_validation import (
    BVH_TO_TARGET_BASIS,
    build_pose_payload,
    compute_bone_vector_diagnostics,
    ValidationThresholds,
    choose_sample_indices_from_summary,
    diagnose_validation,
    extract_bvh_joint_positions,
    remap_per_joint_error_names,
)


def load_pose_driver_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "3d-digital-human" / "tools" / "motion_base" / "export_dataset_pose_driver_sequences.py"
    if str(module_path.parent) not in sys.path:
        sys.path.insert(0, str(module_path.parent))
    spec = importlib.util.spec_from_file_location("pose_driver_module_for_tests", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


class SourceBvhValidationTests(unittest.TestCase):
    def test_choose_sample_indices_from_summary(self) -> None:
        summary = {
            "sourceFps": 30,
            "fps": 15,
            "startSeconds": 1.0,
            "maxSeconds": 2.0,
            "maxFrames": None,
        }
        indices = choose_sample_indices_from_summary(120, summary)
        self.assertEqual(indices.tolist()[:4], [30, 32, 34, 36])
        self.assertEqual(indices.tolist()[-1], 88)
        self.assertEqual(len(indices), 30)

    def test_extract_bvh_joint_positions_applies_root_rotation(self) -> None:
        pose_driver_module = load_pose_driver_module()
        bvh_text = """HIERARCHY
ROOT pelvis
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT left_hip
  {
    OFFSET 1.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.000000 1.000000 0.000000
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.03333333
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
1.000000 2.000000 3.000000 90.000000 0.000000 0.000000 0.000000 0.000000 0.000000
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            bvh_path = Path(temp_dir) / "simple.bvh"
            bvh_path.write_text(bvh_text, encoding="utf-8")
            bone_order, frame_indices, positions, frame_time = extract_bvh_joint_positions(
                bvh_path,
                pose_driver_module.parse_bvh,
                bone_order=["pelvis", "left_hip"],
            )

        self.assertEqual(bone_order, ["pelvis", "left_hip"])
        self.assertEqual(frame_indices, [0, 1])
        self.assertAlmostEqual(frame_time, 0.03333333, places=6)
        np.testing.assert_allclose(positions[0, 0], [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(positions[0, 1], [1.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(positions[1, 0], [1.0, 2.0, 3.0], atol=1e-6)
        np.testing.assert_allclose(positions[1, 1], [1.0, 3.0, 3.0], atol=1e-6)

    def test_extract_bvh_joint_positions_can_apply_basis_conversion(self) -> None:
        pose_driver_module = load_pose_driver_module()
        bvh_text = """HIERARCHY
ROOT pelvis
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT left_hip
  {
    OFFSET 1.000000 2.000000 3.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.000000 1.000000 0.000000
    }
  }
}
MOTION
Frames: 1
Frame Time: 0.03333333
4.000000 5.000000 6.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            bvh_path = Path(temp_dir) / "basis_test.bvh"
            bvh_path.write_text(bvh_text, encoding="utf-8")
            _bone_order, _frame_indices, positions, _frame_time = extract_bvh_joint_positions(
                bvh_path,
                pose_driver_module.parse_bvh,
                bone_order=["pelvis", "left_hip"],
                basis_matrix=BVH_TO_TARGET_BASIS,
            )
        np.testing.assert_allclose(positions[0, 0], [4.0, 6.0, 5.0], atol=1e-6)
        np.testing.assert_allclose(positions[0, 1], [5.0, 9.0, 7.0], atol=1e-6)

    def test_diagnose_validation_prefers_importer_when_fk_matches(self) -> None:
        fk_summary = {
            "jointPositionError": {
                "mean": 1e-7,
            }
        }
        import_summary = {
            "jointPositionError": {
                "mean": 5e-2,
            }
        }
        diagnosis = diagnose_validation(fk_summary, import_summary, thresholds=ValidationThresholds())
        self.assertEqual(diagnosis["likely_root_cause"], "bvh_import_or_viewer_convention")

    def test_build_pose_payload_can_include_preview_space(self) -> None:
        poses = np.asarray([[[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]], dtype=np.float64)
        payload = build_pose_payload(
            input_path="demo.npy",
            bone_order=["root", "child"],
            frame_indices=[0],
            poses=poses,
            include_preview_space=True,
        )
        self.assertIn("previewSpacePoses", payload)
        self.assertEqual(payload["previewSpacePoses"][0][0], [0.0, 0.0, 2.0])

    def test_remap_per_joint_error_names_uses_bone_order(self) -> None:
        summary = {
            "perJointPositionError": {
                "0": {"mean": 0.1},
                "1": {"mean": 0.2},
            }
        }
        remapped = remap_per_joint_error_names(summary, ["pelvis", "left_hip"])
        self.assertEqual(set(remapped.keys()), {"pelvis", "left_hip"})
        self.assertEqual(remapped["left_hip"]["mean"], 0.2)

    def test_compute_bone_vector_diagnostics_reports_direction_error(self) -> None:
        truth_payload = {
            "boneOrder": ["pelvis", "left_hip", "left_knee"],
            "poses": [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            ],
        }
        candidate_payload = {
            "boneOrder": ["pelvis", "left_hip", "left_knee"],
            "poses": [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]],
            ],
        }
        diagnostics = compute_bone_vector_diagnostics(truth_payload, candidate_payload)
        left_hip_stats = diagnostics["perBone"]["left_hip"]
        self.assertAlmostEqual(left_hip_stats["directionAngleDeg"]["mean"], 90.0, places=5)
        self.assertAlmostEqual(left_hip_stats["lengthError"]["mean"], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
