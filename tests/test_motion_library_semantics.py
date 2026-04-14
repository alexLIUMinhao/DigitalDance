import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from music_motion_lab.pipelines.motion_library import COMMON_BONE_ORDER, _derive_unit_semantics, _joint_index_map, build_motion_unit_library


class MotionLibrarySemanticsTests(unittest.TestCase):
    def test_semantics_capture_travel_and_support(self) -> None:
        bone_index = _joint_index_map(COMMON_BONE_ORDER)
        frame_count = 10
        positions = np.zeros((frame_count, len(COMMON_BONE_ORDER), 3), dtype=np.float32)

        for frame in range(frame_count):
            t = frame / float(frame_count - 1)
            positions[frame, bone_index["pelvis"], :] = np.asarray([0.45 * t, 0.0, 0.0], dtype=np.float32)
            positions[frame, bone_index["head"], :] = np.asarray([0.45 * t, 1.2, 0.0], dtype=np.float32)
            positions[frame, bone_index["left_shoulder"], :] = np.asarray([0.45 * t - 0.25, 0.85, 0.0], dtype=np.float32)
            positions[frame, bone_index["right_shoulder"], :] = np.asarray([0.45 * t + 0.25, 0.85, 0.0], dtype=np.float32)
            positions[frame, bone_index["left_hip"], :] = np.asarray([0.45 * t - 0.15, -0.2, 0.0], dtype=np.float32)
            positions[frame, bone_index["right_hip"], :] = np.asarray([0.45 * t + 0.15, -0.2, 0.0], dtype=np.float32)
            positions[frame, bone_index["left_foot"], :] = np.asarray([0.45 * t - 0.18, -1.0, 0.0], dtype=np.float32)
            positions[frame, bone_index["right_foot"], :] = np.asarray([0.45 * t + 0.12 + 0.06 * t, -0.72 + 0.06 * t, 0.08 * t], dtype=np.float32)
            positions[frame, bone_index["left_wrist"], :] = np.asarray([0.45 * t - 0.42, 0.55, 0.02], dtype=np.float32)
            positions[frame, bone_index["right_wrist"], :] = np.asarray([0.45 * t + 0.42, 0.55, -0.02], dtype=np.float32)

        semantics = _derive_unit_semantics(
            unit_positions=positions,
            bone_index=bone_index,
            start_frame=0,
            end_frame_exclusive=frame_count,
            body_scale=2.2,
            floor_height=-1.0,
            fps=30.0,
        )

        self.assertEqual(semantics["travel"], "traveling")
        self.assertEqual(semantics["foot_contact"], ["left_support"])
        self.assertEqual(semantics["facing_change"], "stable")
        self.assertIn(semantics["body_focus"][0], {"full_body", "lower_body"})

    def test_motion_library_builds_overlapping_base_actions_and_kinematic_energy(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_root = Path(tmpdir) / "previewCache"
            pose_dir = preview_root / "dataset_pose_driver" / "finedance"
            pose_dir.mkdir(parents=True, exist_ok=True)
            pose_driver_path = pose_dir / "demo_bridge_pose_driver.json"

            frame_count = 90
            positions = np.zeros((frame_count, len(COMMON_BONE_ORDER), 3), dtype=np.float32)
            bone_index = _joint_index_map(COMMON_BONE_ORDER)
            for frame in range(frame_count):
                if frame < 30:
                    phase = frame / 30.0
                    amp = 0.02
                elif frame < 60:
                    phase = (frame - 30) / 30.0
                    amp = 0.18
                else:
                    phase = (frame - 60) / 30.0
                    amp = 0.42

                pelvis_x = amp * phase
                positions[frame, bone_index["pelvis"], :] = np.asarray([pelvis_x, 0.0, 0.0], dtype=np.float32)
                positions[frame, bone_index["spine1"], :] = np.asarray([pelvis_x, 0.28, 0.0], dtype=np.float32)
                positions[frame, bone_index["spine2"], :] = np.asarray([pelvis_x, 0.56, 0.0], dtype=np.float32)
                positions[frame, bone_index["spine3"], :] = np.asarray([pelvis_x, 0.86, 0.0], dtype=np.float32)
                positions[frame, bone_index["neck"], :] = np.asarray([pelvis_x, 1.08, 0.0], dtype=np.float32)
                positions[frame, bone_index["head"], :] = np.asarray([pelvis_x, 1.28, 0.0], dtype=np.float32)
                positions[frame, bone_index["left_hip"], :] = np.asarray([pelvis_x - 0.18, -0.08, 0.0], dtype=np.float32)
                positions[frame, bone_index["right_hip"], :] = np.asarray([pelvis_x + 0.18, -0.08, 0.0], dtype=np.float32)
                positions[frame, bone_index["left_knee"], :] = np.asarray([pelvis_x - 0.18, -0.56, 0.02 * phase], dtype=np.float32)
                positions[frame, bone_index["right_knee"], :] = np.asarray([pelvis_x + 0.18, -0.56, -0.02 * phase], dtype=np.float32)
                positions[frame, bone_index["left_ankle"], :] = np.asarray([pelvis_x - 0.18, -0.98, 0.03 * phase], dtype=np.float32)
                positions[frame, bone_index["right_ankle"], :] = np.asarray([pelvis_x + 0.18, -0.98, -0.03 * phase], dtype=np.float32)
                positions[frame, bone_index["left_foot"], :] = np.asarray([pelvis_x - 0.22, -1.03, 0.04 * phase], dtype=np.float32)
                positions[frame, bone_index["right_foot"], :] = np.asarray([pelvis_x + 0.22, -1.03, -0.04 * phase], dtype=np.float32)

                arm_swing = amp * 1.8
                positions[frame, bone_index["left_collar"], :] = np.asarray([pelvis_x - 0.14, 0.94, 0.0], dtype=np.float32)
                positions[frame, bone_index["right_collar"], :] = np.asarray([pelvis_x + 0.14, 0.94, 0.0], dtype=np.float32)
                positions[frame, bone_index["left_shoulder"], :] = np.asarray([pelvis_x - 0.32, 0.92, arm_swing], dtype=np.float32)
                positions[frame, bone_index["right_shoulder"], :] = np.asarray([pelvis_x + 0.32, 0.92, -arm_swing], dtype=np.float32)
                positions[frame, bone_index["left_elbow"], :] = np.asarray([pelvis_x - 0.55, 0.66, arm_swing * 1.4], dtype=np.float32)
                positions[frame, bone_index["right_elbow"], :] = np.asarray([pelvis_x + 0.55, 0.66, -arm_swing * 1.4], dtype=np.float32)
                positions[frame, bone_index["left_wrist"], :] = np.asarray([pelvis_x - 0.78, 0.48, arm_swing * 1.8], dtype=np.float32)
                positions[frame, bone_index["right_wrist"], :] = np.asarray([pelvis_x + 0.78, 0.48, -arm_swing * 1.8], dtype=np.float32)

            payload = {
                "datasetName": "finedance",
                "sequenceId": "demo",
                "boneOrder": COMMON_BONE_ORDER,
                "jointCount": len(COMMON_BONE_ORDER),
                "frameCount": frame_count,
                "positions": positions.reshape(-1).tolist(),
            }
            pose_driver_path.write_text(json.dumps(payload), encoding="utf-8")

            library = build_motion_unit_library(
                preview_root=preview_root,
                unit_beats=2.0,
                stride_beats=1.0,
                assumed_bpm=120.0,
            )

            self.assertGreaterEqual(len(library.units), 5)
            energy_labels = {unit["energy"] for unit in library.units}
            self.assertIn("low_energy", energy_labels)
            self.assertIn("high_energy", energy_labels)
            self.assertTrue(all(unit["energy_profile"]["band_method"] == "kinematic_quantiles" for unit in library.units))


if __name__ == "__main__":
    unittest.main()
