import unittest

from music_motion_lab.pipelines.smplx_stitch_web_preview import build_smplx_stitch_web_preview_document


class SmplxStitchWebPreviewTests(unittest.TestCase):
    def test_document_embeds_manifest_and_audio(self) -> None:
        manifest = {
            "manifest_id": "demo_manifest",
            "plan_id": "demo_plan",
            "song_id": "demo_song",
            "library_id": "demo_library",
            "fps": 30,
            "mesh_backend": "smplx_neutral",
            "scene": {"frame_start": 1, "frame_end": 60},
            "steps": [
                {
                    "index": 0,
                    "unit_id": "unit_a",
                    "source_sequence": "001",
                    "start_time_sec": 0.0,
                    "end_time_sec": 2.0,
                    "scene_frame_start": 1,
                    "scene_frame_end": 60,
                    "blend_in_frames": 0,
                    "blend_out_frames": 6,
                    "rhythm_locks": [{"time_sec": 0.5, "scene_frame": 16}],
                    "root_alignment": {
                        "entry_anchor": {"root_translation": [0, 0, 0], "root_yaw_deg": 0},
                        "exit_anchor": {"root_translation": [1, 0, 0], "root_yaw_deg": 12},
                    },
                }
            ],
            "transitions": [],
            "cache_requests": [],
        }

        document = build_smplx_stitch_web_preview_document(
            manifest=manifest,
            song_event_map={"beats": [], "accents": [], "downbeats": [], "sections": []},
            audio_href="../music/demo.wav",
        )

        self.assertIn("<canvas id=\"stageCanvas\"", document)
        self.assertIn("demo_manifest", document)
        self.assertIn("unit_a", document)
        self.assertIn("../music/demo.wav", document)


if __name__ == "__main__":
    unittest.main()
