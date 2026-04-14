import unittest
import tempfile
from pathlib import Path

from music_motion_lab.pipelines.web_preview import build_web_preview_document


class WebPreviewTests(unittest.TestCase):
    def test_build_web_preview_document_contains_plan_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pose_driver_path = Path(tmpdir) / "pose_driver.json"
            pose_driver_path.write_text(
                """
{
  "boneOrder": ["pelvis", "left_hip", "right_hip", "spine1", "spine3", "neck", "head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_foot", "right_foot"],
  "jointCount": 19,
  "frameCount": 2,
  "positions": [
    0,0,1, -0.2,0,0.8, 0.2,0,0.8, 0,0,1.4, 0,0,1.8, 0,0,2.0, 0,0,2.25,
    -0.35,0,1.75, 0.35,0,1.75, -0.55,0,1.45, 0.55,0,1.45, -0.75,0,1.25, 0.75,0,1.25,
    -0.18,0,0.35, 0.18,0,0.35, -0.18,0,0.05, 0.18,0,0.05, -0.22,0,-0.05, 0.22,0,-0.05,
    0.1,0,1, -0.1,0,0.82, 0.3,0,0.82, 0.1,0,1.42, 0.1,0,1.82, 0.1,0,2.02, 0.1,0,2.27,
    -0.25,0,1.77, 0.45,0,1.77, -0.45,0,1.5, 0.65,0,1.5, -0.62,0,1.35, 0.82,0,1.35,
    -0.08,0,0.32, 0.32,0,0.32, -0.02,0,0.02, 0.36,0,0.02, -0.08,0,-0.08, 0.44,0,-0.08
  ]
}
""".strip(),
                encoding="utf-8",
            )
            document = build_web_preview_document(
                preview_job={
                    "plan_id": "demo_plan",
                    "steps": [
                        {
                            "index": 0,
                            "unit_id": "demo_unit_a",
                            "source_sequence": "161",
                            "frame_range": {"start": 0, "end_exclusive": 2},
                            "reference_artifacts": {
                                "pose_driver_json": str(pose_driver_path),
                            },
                        },
                        {
                            "index": 1,
                            "unit_id": "demo_unit_b",
                            "source_sequence": "164",
                            "frame_range": {"start": 0, "end_exclusive": 2},
                            "reference_artifacts": {
                                "pose_driver_json": str(pose_driver_path),
                            },
                        }
                    ],
                },
                max_steps=0,
                frame_stride=1,
                fps=12,
            )
            self.assertIn("demo_plan", document)
            self.assertIn("demo_unit_a", document)
            self.assertIn("demo_unit_b", document)
            self.assertIn("mesh-like preview", document)


if __name__ == "__main__":
    unittest.main()
