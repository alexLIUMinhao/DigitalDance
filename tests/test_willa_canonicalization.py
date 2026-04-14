from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from music_motion_lab.bvh import frame_pose, parse_bvh
from music_motion_lab.willa_canonicalization import (
    _orientation_basis,
    target_world_basis,
    canonicalize_source_bvh_for_willa,
)


SYNTHETIC_SOURCE_BVH = """HIERARCHY
ROOT pelvis
{
  OFFSET 0.000000 0.000000 0.000000
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT left_hip
  {
    OFFSET -1.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT left_knee
    {
      OFFSET 0.000000 -2.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT left_ankle
      {
        OFFSET 0.000000 -2.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT left_foot
        {
          OFFSET 0.000000 -0.500000 1.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET 0.000000 0.000000 1.000000
          }
        }
      }
    }
  }
  JOINT right_hip
  {
    OFFSET 1.000000 0.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT right_knee
    {
      OFFSET 0.000000 -2.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT right_ankle
      {
        OFFSET 0.000000 -2.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT right_foot
        {
          OFFSET 0.000000 -0.500000 1.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
            OFFSET 0.000000 0.000000 1.000000
          }
        }
      }
    }
  }
  JOINT spine1
  {
    OFFSET 0.000000 1.000000 0.000000
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT spine2
    {
      OFFSET 0.000000 1.000000 0.000000
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT spine3
      {
        OFFSET 0.000000 1.000000 0.000000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT neck
        {
          OFFSET 0.000000 1.000000 0.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT head
          {
            OFFSET 0.000000 1.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            End Site
            {
              OFFSET 0.000000 0.500000 0.000000
            }
          }
        }
        JOINT left_collar
        {
          OFFSET -1.000000 0.000000 0.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT left_shoulder
          {
            OFFSET -1.000000 0.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT left_elbow
            {
              OFFSET -1.000000 0.000000 0.000000
              CHANNELS 3 Zrotation Xrotation Yrotation
              JOINT left_wrist
              {
                OFFSET -1.000000 0.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                  OFFSET -0.500000 0.000000 0.000000
                }
              }
            }
          }
        }
        JOINT right_collar
        {
          OFFSET 1.000000 0.000000 0.000000
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT right_shoulder
          {
            OFFSET 1.000000 0.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT right_elbow
            {
              OFFSET 1.000000 0.000000 0.000000
              CHANNELS 3 Zrotation Xrotation Yrotation
              JOINT right_wrist
              {
                OFFSET 1.000000 0.000000 0.000000
                CHANNELS 3 Zrotation Xrotation Yrotation
                End Site
                {
                  OFFSET 0.500000 0.000000 0.000000
                }
              }
            }
          }
        }
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.03333333
0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
2.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000 0.000000
"""


class WillaCanonicalizationTests(unittest.TestCase):
    def test_target_world_basis_respects_requested_forward(self) -> None:
        neg_y_basis = target_world_basis(target_up_label="+Z", target_forward_label="-Y")
        pos_y_basis = target_world_basis(target_up_label="+Z", target_forward_label="+Y")
        self.assertTrue(np.allclose(neg_y_basis[:, 1], np.asarray([0.0, 0.0, 1.0]), atol=1e-6))
        self.assertTrue(np.allclose(neg_y_basis[:, 2], np.asarray([0.0, -1.0, 0.0]), atol=1e-6))
        self.assertTrue(np.allclose(neg_y_basis[:, 0], np.asarray([1.0, 0.0, 0.0]), atol=1e-6))
        self.assertTrue(np.allclose(pos_y_basis[:, 1], np.asarray([0.0, 0.0, 1.0]), atol=1e-6))
        self.assertTrue(np.allclose(pos_y_basis[:, 2], np.asarray([0.0, 1.0, 0.0]), atol=1e-6))
        self.assertTrue(np.allclose(pos_y_basis[:, 0], np.asarray([-1.0, 0.0, 0.0]), atol=1e-6))

    def test_canonicalization_inserts_static_root_and_aligns_positive_y_orientation_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_bvh = tmp_path / "source.bvh"
            output_bvh = tmp_path / "canonical.bvh"
            report_path = tmp_path / "canonical_report.json"
            input_bvh.write_text(SYNTHETIC_SOURCE_BVH, encoding="utf-8")

            report = canonicalize_source_bvh_for_willa(
                input_bvh=input_bvh,
                output_bvh=output_bvh,
                report_output=report_path,
            )

            self.assertEqual(report["frameCount"], 2)
            self.assertAlmostEqual(report["fps"], 30.0, places=4)
            self.assertTrue(report["canonicalRoot"]["isStatic"])
            self.assertTrue(report_path.exists())

            parsed = parse_bvh(output_bvh)
            self.assertEqual(parsed.nodes[0].name, "Root")
            self.assertEqual(parsed.nodes[1].name, "pelvis")
            self.assertEqual(parsed.nodes[1].parent, 0)
            self.assertEqual(parsed.motion.shape[0], 2)
            self.assertAlmostEqual(parsed.frame_time, 0.03333333, places=6)
            self.assertTrue(np.allclose(parsed.motion[:, :6], 0.0, atol=1e-6))

            pose0 = frame_pose(parsed.nodes, parsed.motion[0])
            root_world = pose0.world_positions["Root"]
            self.assertTrue(np.allclose(root_world, np.zeros((3,), dtype=np.float64), atol=1e-6))

            basis = _orientation_basis({name: np.asarray(value, dtype=np.float64) for name, value in pose0.world_positions.items()})
            self.assertTrue(np.allclose(basis[:, 1], np.asarray([0.0, 0.0, 1.0]), atol=1e-6))
            self.assertTrue(np.allclose(basis[:, 2], np.asarray([0.0, 1.0, 0.0]), atol=1e-6))

    def test_canonicalization_supports_positive_y_forward_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_bvh = tmp_path / "source.bvh"
            output_bvh = tmp_path / "canonical_pos_y.bvh"
            input_bvh.write_text(SYNTHETIC_SOURCE_BVH, encoding="utf-8")

            canonicalize_source_bvh_for_willa(
                input_bvh=input_bvh,
                output_bvh=output_bvh,
                target_forward_label="+Y",
            )

            parsed = parse_bvh(output_bvh)
            pose0 = frame_pose(parsed.nodes, parsed.motion[0])
            basis = _orientation_basis({name: np.asarray(value, dtype=np.float64) for name, value in pose0.world_positions.items()})
            self.assertTrue(np.allclose(basis[:, 1], np.asarray([0.0, 0.0, 1.0]), atol=1e-6))
            self.assertTrue(np.allclose(basis[:, 2], np.asarray([0.0, 1.0, 0.0]), atol=1e-6))


if __name__ == "__main__":
    unittest.main()
