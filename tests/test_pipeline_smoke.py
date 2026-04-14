import unittest

from music_motion_lab.pipelines.planner import build_choreography_plan


class PlannerSmokeTests(unittest.TestCase):
    def test_planner_builds_steps_from_phrase_ranges(self) -> None:
        song_event_map = {
            "song_id": "demo_song",
            "manual_review_required": False,
            "beats": [{"index": index, "time_sec": float(index)} for index in range(16)],
            "phrases": [
                {"index": 0, "start_beat": 0, "end_beat_exclusive": 8, "start_time_sec": 0.0, "end_time_sec": 7.0},
                {"index": 1, "start_beat": 8, "end_beat_exclusive": 16, "start_time_sec": 8.0, "end_time_sec": 15.0},
            ],
            "sections": [
                {"index": 0, "label": "intro", "start_beat": 0, "end_beat_exclusive": 8},
                {"index": 1, "label": "chorus", "start_beat": 8, "end_beat_exclusive": 16},
            ],
            "accents": [],
        }
        motion_library = {
            "library_id": "motion_unit_library_v1",
            "units": [
                {
                    "unit_id": "low_a",
                    "duration_beats_estimate": 8.0,
                    "energy": "low_energy",
                    "preferred_section_labels": ["intro"],
                    "compatible_next_units": ["high_b"],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "001",
                },
                {
                    "unit_id": "high_b",
                    "duration_beats_estimate": 8.0,
                    "energy": "high_energy",
                    "preferred_section_labels": ["chorus"],
                    "compatible_next_units": [],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "002",
                },
            ],
        }

        plan = build_choreography_plan(song_event_map=song_event_map, motion_library=motion_library).to_dict()
        self.assertEqual(plan["steps"][0]["selected_unit_id"], "low_a")
        self.assertEqual(plan["steps"][1]["selected_unit_id"], "high_b")

    def test_planner_prefers_compatible_chain_inside_same_section(self) -> None:
        song_event_map = {
            "song_id": "demo_song",
            "manual_review_required": False,
            "beats": [{"index": index, "time_sec": float(index)} for index in range(24)],
            "phrases": [
                {"index": 0, "start_beat": 0, "end_beat_exclusive": 8, "start_time_sec": 0.0, "end_time_sec": 7.0},
                {"index": 1, "start_beat": 8, "end_beat_exclusive": 16, "start_time_sec": 8.0, "end_time_sec": 15.0},
            ],
            "sections": [
                {"index": 0, "label": "verse", "start_beat": 0, "end_beat_exclusive": 16},
            ],
            "accents": [],
            "downbeats": [],
        }
        motion_library = {
            "library_id": "motion_unit_library_v1",
            "units": [
                {
                    "unit_id": "verse_a",
                    "duration_beats_estimate": 8.0,
                    "energy": "mid_energy",
                    "travel": "localized",
                    "foot_contact": ["both_feet_planted"],
                    "body_focus": ["upper_body"],
                    "entry_anchor": {"label": "stable_support"},
                    "preferred_section_labels": ["verse"],
                    "compatible_next_units": ["verse_b"],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "001",
                },
                {
                    "unit_id": "verse_b",
                    "duration_beats_estimate": 8.0,
                    "energy": "mid_energy",
                    "travel": "localized",
                    "foot_contact": ["both_feet_planted"],
                    "body_focus": ["upper_body"],
                    "entry_anchor": {"label": "stable_support"},
                    "preferred_section_labels": ["verse"],
                    "compatible_next_units": [],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "001",
                },
                {
                    "unit_id": "verse_c",
                    "duration_beats_estimate": 8.0,
                    "energy": "mid_energy",
                    "travel": "localized",
                    "foot_contact": ["both_feet_planted"],
                    "body_focus": ["upper_body"],
                    "entry_anchor": {"label": "stable_support"},
                    "preferred_section_labels": ["verse"],
                    "compatible_next_units": [],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "999",
                },
            ],
        }

        plan = build_choreography_plan(song_event_map=song_event_map, motion_library=motion_library).to_dict()
        self.assertEqual(plan["steps"][0]["selected_unit_id"], "verse_a")
        self.assertEqual(plan["steps"][1]["selected_unit_id"], "verse_b")
        self.assertTrue(plan["steps"][1]["switch_reason"]["compatible_from_previous"])

    def test_planner_raises_energy_for_accent_dense_verse(self) -> None:
        song_event_map = {
            "song_id": "demo_song",
            "manual_review_required": False,
            "beats": [{"index": index, "time_sec": float(index)} for index in range(24)],
            "phrases": [
                {"index": 0, "start_beat": 0, "end_beat_exclusive": 8, "start_time_sec": 0.0, "end_time_sec": 7.0},
            ],
            "sections": [
                {"index": 0, "label": "verse", "start_beat": 0, "end_beat_exclusive": 8},
            ],
            "accents": [
                {"index": index, "time_sec": 0.5 + index * 0.8, "strength": 0.45, "kind": "accent_peak"}
                for index in range(6)
            ],
            "downbeats": [{"index": 0, "time_sec": 0.0}],
        }
        motion_library = {
            "library_id": "motion_unit_library_v1",
            "units": [
                {
                    "unit_id": "mid_a",
                    "duration_beats_estimate": 8.0,
                    "energy": "mid_energy",
                    "travel": "localized",
                    "foot_contact": ["both_feet_planted"],
                    "body_focus": ["upper_body"],
                    "entry_anchor": {"label": "dynamic_transition"},
                    "preferred_section_labels": ["verse"],
                    "compatible_next_units": [],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "001",
                },
                {
                    "unit_id": "high_b",
                    "duration_beats_estimate": 8.0,
                    "energy": "high_energy",
                    "travel": "traveling",
                    "foot_contact": ["alternating_support"],
                    "body_focus": ["full_body"],
                    "entry_anchor": {"label": "dynamic_transition"},
                    "preferred_section_labels": ["verse", "chorus"],
                    "compatible_next_units": [],
                    "bridge_quality": "pass",
                    "retarget_quality": "unknown",
                    "reference_artifacts": {},
                    "source_sequence": "002",
                },
            ],
        }

        plan = build_choreography_plan(song_event_map=song_event_map, motion_library=motion_library).to_dict()
        self.assertEqual(plan["steps"][0]["selected_unit_id"], "high_b")
        self.assertEqual(plan["steps"][0]["switch_reason"]["target_energy"], "high_energy")


if __name__ == "__main__":
    unittest.main()
