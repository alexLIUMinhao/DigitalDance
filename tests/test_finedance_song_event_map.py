import unittest
from pathlib import Path

from music_motion_lab.pipelines.finedance_song_event_map import build_finedance_song_event_map_from_report


class FineDanceSongEventMapTests(unittest.TestCase):
    def test_converts_m2_2_audio_features_to_song_event_map(self) -> None:
        report = {
            "entries": [
                {
                    "sequence_id": "001",
                    "style_tags": {"song_name": "demo", "coarse_style": "Street"},
                    "analysis_priority": {"tier": "rhythmic_first"},
                    "audio_analysis_panel": {"clip_end_sec": 10.0},
                    "audio_features": {
                        "bpm": 120,
                        "beats_per_bar": 4,
                        "energy_label": "high",
                        "beats": [
                            {"index": index, "time_sec": index * 0.5, "strength": 0.2 + index * 0.01, "is_downbeat": index % 4 == 0}
                            for index in range(17)
                        ],
                        "strong_beats": [
                            {"index": 0, "beat_index": 0, "time_sec": 0.0, "is_downbeat": True, "level": 4},
                            {"index": 1, "beat_index": 4, "time_sec": 2.0, "is_downbeat": True, "level": 3},
                        ],
                        "accent_candidates": [{"index": 0, "source_beat_index": 4, "time_sec": 2.0, "strength": 0.8, "band": "low"}],
                        "drum_hits": [{"index": 0, "time_sec": 2.0, "strength": 0.7}],
                    },
                }
            ]
        }

        event_map = build_finedance_song_event_map_from_report(
            audio_feature_report=report,
            sequence_id="001",
            motion_base_assets_root=Path("/assets"),
            phrase_beats=8,
        )

        self.assertEqual(event_map["song_id"], "finedance_001_full_song")
        self.assertEqual(event_map["source_audio_path"], "/assets/datasets/finedance/raw/extracted/finedance/music_wav/001.wav")
        self.assertEqual(len(event_map["beats"]), 17)
        self.assertEqual(len(event_map["downbeats"]), 2)
        self.assertEqual(len(event_map["accents"]), 1)
        self.assertEqual(len(event_map["drum_hits"]), 1)
        self.assertEqual(len(event_map["phrases"]), 2)
        self.assertEqual(event_map["sections"][0]["label"], "intro")

    def test_missing_sequence_fails_fast(self) -> None:
        with self.assertRaises(KeyError):
            build_finedance_song_event_map_from_report(
                audio_feature_report={"entries": []},
                sequence_id="missing",
                motion_base_assets_root=Path("/assets"),
            )


if __name__ == "__main__":
    unittest.main()
