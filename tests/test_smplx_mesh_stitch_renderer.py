import tempfile
import unittest
from pathlib import Path

import numpy as np

from music_motion_lab.pipelines.smplx_mesh_stitch_renderer import (
    MeshCache,
    build_mesh_stitch_review_html,
    build_rhythm_mapping,
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
                "start_time_sec": 0.0,
                "end_time_sec": 0.12,
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
                "start_time_sec": 0.2,
                "end_time_sec": 0.32,
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
            "metrics": {
                "beat_count": 2,
                "downbeat_count": 1,
                "drum_hit_count": 1,
                "accent_count": 1,
                "max_vertex_delta_after_blend": 1.25,
                "max_rhythm_lock_frame_error": 0,
            },
            "artifacts": {"report": "/tmp/report.json"},
            "segment_mapping": [
                {
                    "index": 0,
                    "unit_id": "unit_a",
                    "source_sequence": "001",
                    "source_beat_range": {"start": 10, "end_exclusive": 18},
                    "source_frames": {"start": 100, "end_exclusive": 180},
                    "target_time_sec": {"start": 1.0, "end": 3.0},
                    "scene_frames": {"start": 31, "end": 90},
                    "blend": {"in_frames": 0, "out_frames": 6},
                }
            ],
            "rhythm_mapping": {
                "preview_window_sec": {"start": 1.0, "end": 3.0},
                "beats": [{"time_sec": 1.0, "is_downbeat": True}],
                "downbeats": [{"time_sec": 1.0}],
                "accents": [{"time_sec": 2.0, "strength": 0.9}],
                "drum_hits": [{"time_sec": 1.0}],
            },
            "transitions": [{"outgoing_step": 0, "incoming_step": 1, "boundary_frame": 7}],
            "rhythm_locks": [{"step_index": 0, "kind": "expected_accent", "target_scene_frame": 3, "frame_error": 0}],
        }

        document = build_mesh_stitch_review_html(report, video_href="preview.mp4", strip_href="strip.png", audio_href="../music/audio.mp3")

        self.assertIn("<audio id=\"audio\"", document)
        self.assertIn("<video controls src=\"preview.mp4\"", document)
        self.assertIn("strip.png", document)
        self.assertIn("unit_a", document)
        self.assertIn("beats in preview", document)
        self.assertIn("drum hits mapped", document)
        self.assertIn("max vertex delta after blend", document)

    def test_rhythm_mapping_extracts_song_events_for_preview_window(self) -> None:
        mapping = build_rhythm_mapping(
            _manifest(),
            {
                "song_id": "demo",
                "source_audio_path": "/tmp/audio.mp3",
                "beats_per_bar": 4,
                "duration_sec": 20.0,
                "beats": [
                    {"index": 0, "time_sec": 0.05, "strength": 0.1, "is_downbeat": True},
                    {"index": 1, "time_sec": 0.2, "strength": 0.2, "is_downbeat": False},
                ],
                "downbeats": [{"index": 0, "time_sec": 0.05, "source_beat_index": 0}],
                "accents": [{"index": 0, "time_sec": 0.25, "strength": 0.8, "kind": "accent_peak"}],
            },
        )

        self.assertEqual(mapping["summary"]["step_count"], 2)
        self.assertEqual(mapping["summary"]["beat_count"], 2)
        self.assertEqual(mapping["summary"]["downbeat_count"], 1)
        self.assertEqual(mapping["summary"]["accent_count"], 1)
        self.assertEqual(mapping["summary"]["drum_hit_count"], 1)
        self.assertEqual(mapping["steps"][0]["unit_id"], "a")
        self.assertEqual(mapping["beats"][0]["scene_frame"], 3)

    def test_rhythm_mapping_uses_downbeats_as_drum_hit_fallback_inside_window(self) -> None:
        mapping = build_rhythm_mapping(
            _manifest(),
            {
                "song_id": "demo",
                "beats": [{"index": 0, "time_sec": 0.05, "strength": 0.2, "is_downbeat": True}],
                "downbeats": [{"index": 0, "time_sec": 0.05, "source_beat_index": 0}],
                "accents": [{"index": 0, "time_sec": 5.0, "strength": 0.8}],
            },
        )

        self.assertEqual(mapping["summary"]["accent_count"], 0)
        self.assertEqual(mapping["summary"]["drum_hit_count"], 1)
        self.assertEqual(mapping["drum_hits"][0]["kind"], "drum_hit")


if __name__ == "__main__":
    unittest.main()
