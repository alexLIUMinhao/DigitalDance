import unittest

from music_motion_lab.pipelines.smplx_stitch_preview import build_smplx_stitch_preview_manifest


def _library() -> dict:
    return {
        "schema_version": 2,
        "library_id": "finedance_rhythmic_smplx_library_v1",
        "units": [
            {
                "unit_id": "unit_a",
                "source_sequence": "001",
                "frame_range": {"start": 10, "end_exclusive": 70},
                "beat_range": {"start": 0, "end_exclusive": 8},
                "entry_anchor": {"root_yaw_deg": 0.0},
                "exit_anchor": {"root_yaw_deg": 8.0},
                "reference_artifacts": {"source_motion_path": "/tmp/finedance/motion/001.npy"},
            },
            {
                "unit_id": "unit_b",
                "source_sequence": "001",
                "frame_range": {"start": 70, "end_exclusive": 130},
                "beat_range": {"start": 8, "end_exclusive": 16},
                "entry_anchor": {"root_yaw_deg": 8.0},
                "exit_anchor": {"root_yaw_deg": 10.0},
                "reference_artifacts": {"source_motion_path": "/tmp/finedance/motion/001.npy"},
            },
        ],
    }


def _plan() -> dict:
    return {
        "schema_version": 2,
        "plan_id": "demo_rhythmic_smplx_plan",
        "song_id": "demo",
        "library_id": "finedance_rhythmic_smplx_library_v1",
        "steps": [
            {
                "index": 0,
                "selected_unit_id": "unit_a",
                "source_sequence": "001",
                "source_frame_range": {"start": 10, "end_exclusive": 70},
                "source_beat_range": {"start": 0, "end_exclusive": 8},
                "target_time_sec": {"start": 0.0, "end": 2.0},
                "speed_scale": 1.0,
                "expected_accent_hits": [0.5, 1.5],
                "transition_score": 0.0,
            },
            {
                "index": 1,
                "selected_unit_id": "unit_b",
                "source_sequence": "001",
                "source_frame_range": {"start": 70, "end_exclusive": 130},
                "source_beat_range": {"start": 8, "end_exclusive": 16},
                "target_time_sec": {"start": 2.0, "end": 4.0},
                "speed_scale": 1.0,
                "expected_accent_hits": [2.5],
                "transition_score": 6.0,
            },
        ],
    }


class SmplxStitchPreviewTests(unittest.TestCase):
    def test_builds_manifest_with_cache_requests_and_transitions(self) -> None:
        manifest = build_smplx_stitch_preview_manifest(
            plan=_plan(),
            motion_library=_library(),
            fps=30,
            blend_frames=6,
        )

        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["mesh_backend"], "smplx_neutral")
        self.assertEqual(manifest["scene"], {"frame_start": 1, "frame_end": 120})
        self.assertEqual(len(manifest["steps"]), 2)
        self.assertEqual(manifest["steps"][0]["blend_in_frames"], 0)
        self.assertEqual(manifest["steps"][0]["blend_out_frames"], 6)
        self.assertEqual(manifest["steps"][1]["blend_in_frames"], 6)
        self.assertEqual(manifest["steps"][0]["rhythm_locks"][0]["scene_frame"], 16)
        self.assertEqual(len(manifest["transitions"]), 1)
        self.assertEqual(manifest["transitions"][0]["boundary_frame"], 61)
        self.assertEqual(len(manifest["cache_requests"]), 1)
        self.assertEqual(manifest["cache_requests"][0]["frame_ranges"][0], {"start": 10, "end_exclusive": 70})

    def test_missing_unit_fails_fast(self) -> None:
        plan = _plan()
        plan["steps"][0]["selected_unit_id"] = "missing"
        with self.assertRaises(KeyError):
            build_smplx_stitch_preview_manifest(plan=plan, motion_library=_library())


if __name__ == "__main__":
    unittest.main()
