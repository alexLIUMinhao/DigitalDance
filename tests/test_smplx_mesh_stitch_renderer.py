import tempfile
import unittest
from pathlib import Path

import numpy as np

from music_motion_lab.pipelines.smplx_mesh_stitch_renderer import (
    MeshCache,
    build_mesh_stitch_review_html,
    compose_stitched_mesh_sequence,
    load_mesh_cache,
)


def _fake_cache() -> MeshCache:
    vertices = np.zeros((8, 4, 3), dtype=np.float32)
    joints = np.zeros((8, 1, 3), dtype=np.float32)
    local_shape = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.2, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    for frame in range(8):
        root_x = float(frame if frame < 4 else 10 + frame - 4)
        local_jump = 0.0 if frame < 4 else 3.0
        vertices[frame] = local_shape + np.asarray([root_x, local_jump, 0.0], dtype=np.float32)
        joints[frame, 0] = np.asarray([root_x, local_jump, 0.0], dtype=np.float32)
    return MeshCache(
        sequence_id="001",
        path=Path("fake.npz"),
        vertices=vertices,
        joints=joints,
        faces=np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int32),
        source_frame_indices=np.arange(8, dtype=np.int32),
        source_motion_path="/tmp/fake.npy",
    )


def _manifest() -> dict:
    return {
        "manifest_id": "demo_stitch",
        "fps": 30,
        "steps": [
            {
                "index": 0,
                "unit_id": "a",
                "source_sequence": "001",
                "source_frame_start": 0,
                "source_frame_end_exclusive": 4,
                "scene_frame_start": 1,
                "scene_frame_end": 4,
                "blend_in_frames": 0,
                "blend_out_frames": 2,
                "rhythm_locks": [{"kind": "expected_accent", "scene_frame": 3, "time_sec": 0.1}],
            },
            {
                "index": 1,
                "unit_id": "b",
                "source_sequence": "001",
                "source_frame_start": 4,
                "source_frame_end_exclusive": 8,
                "scene_frame_start": 7,
                "scene_frame_end": 10,
                "blend_in_frames": 2,
                "blend_out_frames": 0,
                "rhythm_locks": [{"kind": "expected_accent", "scene_frame": 8, "time_sec": 0.25}],
            },
        ],
    }


class SmplxMeshStitchRendererTests(unittest.TestCase):
    def test_composes_root_aligned_blended_sequence_from_fake_cache(self) -> None:
        stitched = compose_stitched_mesh_sequence(_manifest(), {"001": _fake_cache()})

        self.assertEqual(stitched.frame_count, 10)
        self.assertEqual(stitched.scene_frame_start, 1)
        self.assertEqual(stitched.scene_frame_end, 10)
        self.assertFalse(np.isnan(stitched.vertices).any())
        self.assertEqual(len(stitched.transition_reports), 1)
        transition = stitched.transition_reports[0]
        self.assertEqual(transition["gap_frames"], 2)
        self.assertEqual(transition["blend_frames"], 2)
        self.assertGreater(transition["raw_root_xz_delta"], 0.0)
        self.assertEqual(transition["aligned_root_xz_delta_before_blend"], 0.0)
        self.assertLess(transition["vertex_delta_after_blend"], transition["vertex_delta_before_blend"])
        self.assertEqual(max(lock["frame_error"] for lock in stitched.rhythm_lock_reports), 0)

    def test_load_mesh_cache_roundtrip(self) -> None:
        cache = _fake_cache()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.npz"
            np.savez_compressed(
                path,
                sequence_id=np.asarray(cache.sequence_id),
                source_motion_path=np.asarray(cache.source_motion_path),
                vertices=cache.vertices,
                joints=cache.joints,
                faces=cache.faces,
                source_frame_indices=cache.source_frame_indices,
            )

            loaded = load_mesh_cache(path)

        self.assertEqual(loaded.sequence_id, "001")
        self.assertEqual(loaded.frame_to_cache_index[7], 7)
        self.assertEqual(loaded.faces.shape, (2, 3))

    def test_review_html_links_video_strip_and_metrics(self) -> None:
        report = {
            "report_id": "demo_report",
            "fps": 30,
            "scene": {"frame_count": 10},
            "mesh": {"vertex_count": 4, "face_count": 2},
            "metrics": {"max_vertex_delta_after_blend": 1.25, "max_rhythm_lock_frame_error": 0},
            "artifacts": {"report": "/tmp/report.json"},
            "transitions": [{"outgoing_step": 0, "incoming_step": 1, "boundary_frame": 7}],
            "rhythm_locks": [{"step_index": 0, "kind": "expected_accent", "target_scene_frame": 3, "frame_error": 0}],
        }

        document = build_mesh_stitch_review_html(report, video_href="preview.mp4", strip_href="strip.png")

        self.assertIn("<video controls src=\"preview.mp4\"", document)
        self.assertIn("strip.png", document)
        self.assertIn("max vertex delta after blend", document)


if __name__ == "__main__":
    unittest.main()
