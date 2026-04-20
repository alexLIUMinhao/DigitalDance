import unittest

from music_motion_lab.pipelines.rhythmic_planner import build_rhythmic_choreography_plan


def _song_event_map() -> dict:
    beats = [{"index": index, "time_sec": index * 0.5, "strength": 0.7 if index % 4 == 0 else 0.2, "is_downbeat": index % 4 == 0} for index in range(17)]
    return {
        "schema_version": 1,
        "song_id": "demo_song",
        "source_audio_path": "demo.wav",
        "duration_sec": 8.0,
        "beats_per_bar": 4,
        "tempo_hypotheses": [{"label": "primary", "bpm": 120.0, "confidence": 0.9}],
        "beats": beats,
        "downbeats": [{"index": index // 4, "time_sec": beats[index]["time_sec"], "source_beat_index": index} for index in range(0, 17, 4)],
        "accents": [
            {"index": 0, "time_sec": 1.0, "strength": 0.8},
            {"index": 1, "time_sec": 5.0, "strength": 0.9},
        ],
        "phrases": [
            {"index": 0, "label": "phrase", "start_beat": 0, "end_beat_exclusive": 8, "start_time_sec": 0.0, "end_time_sec": 4.0},
            {"index": 1, "label": "phrase", "start_beat": 8, "end_beat_exclusive": 16, "start_time_sec": 4.0, "end_time_sec": 8.0},
        ],
        "sections": [
            {"index": 0, "label": "chorus", "start_beat": 0, "end_beat_exclusive": 16},
        ],
        "confidence": {"overall": 0.9},
        "manual_review_required": False,
        "analysis_mode": "automatic",
        "applied_override_keys": [],
        "generated_at_utc": "2026-04-20T00:00:00+00:00",
    }


def _unit(unit_id: str, energy: str, compatible_next_units: list[str] | None = None) -> dict:
    return {
        "unit_id": unit_id,
        "dataset": "finedance",
        "source_sequence": unit_id.split("_")[1],
        "skeleton_id": "finedance_smplx_source_v1",
        "mesh_backend": "smplx_neutral",
        "frame_range": {"start": 0, "end_exclusive": 120},
        "beat_range": {"start": 0, "end_exclusive": 8},
        "duration_beats": 8,
        "duration_beats_estimate": 8.0,
        "segment_source": "downbeat_phrase",
        "energy": energy,
        "style_tags": ["street", "hiphop"],
        "rhythm_profile": {"accent_count": 2},
        "entry_anchor": {"planar_speed": 0.2, "root_yaw_deg": 0.0},
        "exit_anchor": {"planar_speed": 0.22, "root_yaw_deg": 4.0},
        "reference_artifacts": {"source_motion_path": f"motion/{unit_id}.npy"},
        "compatible_next_units": compatible_next_units or [],
    }


class RhythmicPlannerTests(unittest.TestCase):
    def test_builds_beat_locked_plan_from_rhythmic_library(self) -> None:
        library = {
            "schema_version": 2,
            "library_id": "finedance_rhythmic_smplx_library_v1",
            "units": [
                _unit("finedance_001_a", "high_energy", ["finedance_002_b"]),
                _unit("finedance_002_b", "high_energy"),
                _unit("finedance_003_c", "low_energy"),
            ],
        }

        plan = build_rhythmic_choreography_plan(_song_event_map(), library, beam_width=2).to_dict()

        self.assertEqual(plan["schema_version"], 2)
        self.assertEqual(plan["plan_id"], "demo_song_rhythmic_smplx_plan")
        self.assertEqual(len(plan["steps"]), 2)
        self.assertEqual(plan["steps"][0]["start_beat"], 0)
        self.assertEqual(plan["steps"][0]["target_beats"], 8.0)
        self.assertEqual(plan["steps"][0]["speed_scale"], 1.0)
        self.assertEqual(plan["steps"][1]["selected_unit_id"], "finedance_002_b")
        self.assertTrue(plan["steps"][1]["switch_reason"]["compatible_from_previous"])
        self.assertEqual(plan["steps"][1]["target_time_sec"], {"start": 4.0, "end": 8.0})

    def test_rejects_empty_library(self) -> None:
        with self.assertRaises(ValueError):
            build_rhythmic_choreography_plan(_song_event_map(), {"library_id": "empty", "units": []})


if __name__ == "__main__":
    unittest.main()
