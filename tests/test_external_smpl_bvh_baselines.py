from __future__ import annotations

import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.external_smpl_bvh_baselines import (
    JOINT_ERROR_GROUPS,
    SMPL_COMPAT_BONE_ORDER_24,
    build_smpl_compat_quats,
    compute_group_error_summary,
    load_finedance_smpl_compat_clip,
    stabilize_zero_rotvecs,
    write_aist_compat_pkl,
    write_smpl_compat_npz,
)


def identity_quats(frame_count: int, joint_count: int) -> np.ndarray:
    payload = np.zeros((frame_count, joint_count, 4), dtype=np.float64)
    payload[..., 3] = 1.0
    return payload


class ExternalSmplBvhBaselineTests(unittest.TestCase):
    def test_build_smpl_compat_quats_maps_bridge_hands(self) -> None:
        source_joint_names = list(SMPL_COMPAT_BONE_ORDER_24)
        convert_module = SimpleNamespace(
            FINEDANCE_BRIDGE_JOINT_MAP={name: name for name in SMPL_COMPAT_BONE_ORDER_24},
            FINEDANCE_JOINT_INDEX={name: index for index, name in enumerate(source_joint_names)},
            safe_quaternions=lambda value: value,
        )
        full_quats = identity_quats(frame_count=2, joint_count=len(source_joint_names))
        left_hand_index = convert_module.FINEDANCE_JOINT_INDEX["left_hand"]
        right_hand_index = convert_module.FINEDANCE_JOINT_INDEX["right_hand"]
        full_quats[:, left_hand_index, :] = Rotation.from_euler("z", 45, degrees=True).as_quat()
        full_quats[:, right_hand_index, :] = Rotation.from_euler("z", -30, degrees=True).as_quat()

        compat_quats = build_smpl_compat_quats(full_quats, convert_module)

        np.testing.assert_allclose(compat_quats[:, 22, :], full_quats[:, left_hand_index, :], atol=1e-8)
        np.testing.assert_allclose(compat_quats[:, 23, :], full_quats[:, right_hand_index, :], atol=1e-8)

    def test_load_finedance_smpl_compat_clip_outputs_axis_angle(self) -> None:
        frame_count = 3
        full_quats = identity_quats(frame_count=frame_count, joint_count=len(SMPL_COMPAT_BONE_ORDER_24))
        full_quats[:, 0, :] = Rotation.from_euler("x", 15, degrees=True).as_quat()
        convert_module = SimpleNamespace(
            FINEDANCE_BRIDGE_JOINT_MAP={name: name for name in SMPL_COMPAT_BONE_ORDER_24},
            FINEDANCE_JOINT_INDEX={name: index for index, name in enumerate(SMPL_COMPAT_BONE_ORDER_24)},
            safe_quaternions=lambda value: value,
            load_finedance_motion_components=lambda _path: (np.ones((frame_count, 3), dtype=np.float64), full_quats, 315),
        )

        clip = load_finedance_smpl_compat_clip(Path("demo.npy"), convert_module)

        self.assertEqual(clip.feature_dims, 315)
        self.assertEqual(clip.frame_count, frame_count)
        self.assertEqual(clip.axis_angle.shape, (frame_count, 24, 3))
        self.assertEqual(clip.trans.shape, (frame_count, 3))
        self.assertGreater(abs(float(clip.axis_angle[0, 0, 0])), 0.0)

    def test_write_intermediate_formats_round_trip(self) -> None:
        clip = SimpleNamespace(
            frame_count=2,
            axis_angle=np.arange(2 * 24 * 3, dtype=np.float64).reshape((2, 24, 3)),
            trans=np.arange(6, dtype=np.float64).reshape((2, 3)),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            npz_path = root / "input.npz"
            pkl_path = root / "input.pkl"
            write_smpl_compat_npz(npz_path, clip)  # type: ignore[arg-type]
            write_aist_compat_pkl(pkl_path, clip)  # type: ignore[arg-type]
            npz_payload = np.load(npz_path)
            self.assertEqual(npz_payload["poses"].shape, (1, 2, 24, 3))
            self.assertEqual(npz_payload["trans"].shape, (1, 2, 3))
            with pkl_path.open("rb") as handle:
                pkl_payload = pickle.load(handle)
            self.assertEqual(np.asarray(pkl_payload["smpl_poses"]).shape, (2, 72))
            self.assertEqual(np.asarray(pkl_payload["smpl_trans"]).shape, (2, 3))
            self.assertTrue(np.all(np.linalg.norm(npz_payload["poses"].reshape(-1, 3), axis=1) > 0.0))

    def test_stabilize_zero_rotvecs_replaces_exact_zeros(self) -> None:
        payload = np.zeros((2, 24, 3), dtype=np.float64)
        stabilized = stabilize_zero_rotvecs(payload, epsilon=1e-8)
        self.assertTrue(np.all(np.linalg.norm(stabilized.reshape(-1, 3), axis=1) > 0.0))

    def test_compute_group_error_summary_uses_expected_groups(self) -> None:
        bone_order = ["pelvis"] + [bone for group in JOINT_ERROR_GROUPS.values() for bone in group if bone != "pelvis"]
        bone_order = list(dict.fromkeys(bone_order))
        truth = np.zeros((2, len(bone_order), 3), dtype=np.float64)
        candidate = np.zeros_like(truth)
        candidate[:, bone_order.index("left_wrist"), 0] = 2.0
        candidate[:, bone_order.index("right_knee"), 1] = 1.0

        summary = compute_group_error_summary(truth, candidate, bone_order)

        self.assertIn("arm", summary)
        self.assertIn("leg", summary)
        self.assertGreater(float(summary["arm"]["mean"]), 0.0)
        self.assertGreater(float(summary["leg"]["mean"]), 0.0)


if __name__ == "__main__":
    unittest.main()
