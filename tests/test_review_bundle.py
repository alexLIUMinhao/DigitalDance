import unittest

from music_motion_lab.pipelines.review_bundle import build_review_bundle, build_review_bundle_markdown


class ReviewBundleTests(unittest.TestCase):
    def test_review_bundle_collects_highlights(self) -> None:
        bundle = build_review_bundle(
            song_event_map={
                "song_id": "demo_song",
                "analysis_mode": "automatic",
                "manual_review_required": False,
                "override_source": None,
                "applied_override_keys": [],
                "duration_sec": 120.0,
                "beats_per_bar": 4,
                "beats": [{}] * 32,
                "phrases": [{}] * 4,
                "sections": [{"label": "intro"}, {"label": "chorus"}],
                "confidence": {"overall": 0.8},
                "tempo_hypotheses": [{"bpm": 96.0}],
            },
            motion_library_showcase={
                "library_id": "motion_unit_library_v1",
                "counts": {"unit_count": 10, "travel": {"traveling": 4, "localized": 3, "stationary": 3}},
                "sampled_units": [{"unit_id": "u1", "energy": "mid_energy", "travel": "localized", "facing_change": "stable", "foot_contact": ["both_feet_planted"]}],
            },
            choreography_plan={
                "plan_id": "demo_plan",
                "steps": [
                    {"index": 0, "section_label": "intro", "selected_unit_id": "u1", "source_sequence": "001"},
                    {"index": 1, "section_label": "chorus", "selected_unit_id": "u2", "source_sequence": "002"},
                ],
            },
            retarget_report={
                "status": "backend_unavailable",
                "backend": {"available": False},
                "summary": {"sequence_count": 2},
                "sequence_reports": [
                    {
                        "dataset": "finedance",
                        "sequence_id": "168",
                        "status": "fail",
                        "joint_position_error": {"mean": 4.2},
                        "bone_angle_deg": {"max": 160.0},
                        "issues": ["joint_error_mean_too_high"],
                    }
                ],
            },
        ).to_dict()

        self.assertEqual(bundle["song_id"], "demo_song")
        self.assertEqual(bundle["plan_id"], "demo_plan")
        self.assertTrue(bundle["highlights"])
        markdown = build_review_bundle_markdown(bundle)
        self.assertIn("# Review Bundle: demo_song", markdown)
        self.assertIn("worst retarget hotspot", " ".join(bundle["highlights"]).lower())


if __name__ == "__main__":
    unittest.main()
