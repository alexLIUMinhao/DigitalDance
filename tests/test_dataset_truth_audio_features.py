import unittest

import numpy as np

from music_motion_lab.pipelines.dataset_truth_preview import (
    _build_elastic_beats,
    _build_mel_spectrogram_panel,
    _build_motion_accents,
    _build_multi_band_accents,
    _build_waveform_panel,
    _clip_section_entries,
    _assign_strong_beat_levels,
    _classify_analysis_priority,
    _classify_energy_labels,
    _drum_hits_from_beats,
)


class DatasetTruthAudioFeatureTests(unittest.TestCase):
    def test_assign_strong_beat_levels_applies_downbeat_bonus(self) -> None:
        beats = [
            {"index": 0, "time_sec": 0.0, "strength": 0.10, "is_downbeat": True},
            {"index": 1, "time_sec": 0.5, "strength": 0.20, "is_downbeat": False},
            {"index": 2, "time_sec": 1.0, "strength": 0.30, "is_downbeat": False},
            {"index": 3, "time_sec": 1.5, "strength": 0.40, "is_downbeat": False},
        ]
        strong_beats = _assign_strong_beat_levels(beats, downbeat_bonus=0.25)
        beat_index_to_level = {item["beat_index"]: item["level"] for item in strong_beats}
        self.assertIn(0, beat_index_to_level)
        self.assertGreaterEqual(beat_index_to_level[0], 3)
        self.assertEqual(max(beat_index_to_level.values()), 4)

    def test_classify_energy_labels_quantile_bands(self) -> None:
        labels = _classify_energy_labels([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        self.assertEqual(labels.count("low_energy"), 2)
        self.assertEqual(labels.count("mid_energy"), 2)
        self.assertEqual(labels.count("high_energy"), 2)

    def test_drum_hits_are_sorted_and_clipped(self) -> None:
        low_band_envelope = np.asarray([0.1, 0.8, 0.2, 0.95, 0.3, 0.9], dtype=np.float32)
        beats = [
            {"index": 0, "time_sec": 0.0, "strength": 0.15, "is_downbeat": True},
            {"index": 1, "time_sec": 0.8, "strength": 0.50, "is_downbeat": False},
            {"index": 2, "time_sec": 1.6, "strength": 0.42, "is_downbeat": False},
            {"index": 3, "time_sec": 2.4, "strength": 0.55, "is_downbeat": True},
            {"index": 4, "time_sec": 3.2, "strength": 0.43, "is_downbeat": False},
        ]
        hits = _drum_hits_from_beats(
            beats=beats,
            low_band_envelope=low_band_envelope,
            sample_rate=640,
            hop_size=512,
            clip_start_seconds=0.0,
            clip_duration_seconds=3.0,
        )
        self.assertGreaterEqual(len(hits), 1)
        self.assertTrue(all(hits[index]["time_sec"] < hits[index + 1]["time_sec"] for index in range(len(hits) - 1)))
        self.assertTrue(all(item["time_sec"] <= 3.0 for item in hits))

    def test_waveform_panel_uses_fixed_bin_count(self) -> None:
        audio = np.linspace(-1.0, 1.0, 1024, dtype=np.float32)
        panel = _build_waveform_panel(
            audio=audio,
            sample_rate=256,
            clip_start_seconds=0.0,
            clip_duration_seconds=4.0,
            bin_count=32,
        )
        self.assertEqual(panel["bin_count"], 32)
        self.assertGreater(len(panel["min_b64"]), 0)
        self.assertGreater(len(panel["max_b64"]), 0)

    def test_mel_panel_quantizes_to_expected_shape(self) -> None:
        frame_times_sec = np.linspace(0.0, 1.0, 10, dtype=np.float32)
        mel = np.linspace(0.0, 1.0, 48 * 10, dtype=np.float32).reshape(48, 10)
        panel = _build_mel_spectrogram_panel(
            frame_times_sec=frame_times_sec,
            mel_spectrogram=mel,
            clip_start_seconds=0.0,
            clip_duration_seconds=1.0,
            mel_bin_count=48,
            time_bin_count=24,
        )
        self.assertEqual(panel["mel_bin_count"], 48)
        self.assertEqual(panel["time_bin_count"], 24)
        self.assertGreater(len(panel["values_b64"]), 0)

    def test_clip_section_entries_truncates_to_clip_window(self) -> None:
        clipped = _clip_section_entries(
            sections=[
                {"label": "intro", "start_time_sec": 0.0, "end_time_sec": 4.0, "confidence": 0.5},
                {"label": "chorus", "start_time_sec": 4.0, "end_time_sec": 8.0, "confidence": 0.8},
            ],
            clip_start_seconds=2.0,
            clip_duration_seconds=3.0,
        )
        self.assertEqual(len(clipped), 2)
        self.assertEqual(clipped[0]["start_time_sec"], 0.0)
        self.assertEqual(clipped[0]["end_time_sec"], 2.0)
        self.assertEqual(clipped[1]["start_time_sec"], 2.0)
        self.assertEqual(clipped[1]["end_time_sec"], 3.0)

    def test_build_elastic_beats_emits_confidence_and_local_spacing(self) -> None:
        onset = np.zeros((80,), dtype=np.float32)
        for index, peak_index in enumerate([3, 11, 19, 27, 36, 44, 53, 61, 70]):
            onset[peak_index] = 0.6 + 0.04 * index
        result = _build_elastic_beats(
            onset_envelope=onset,
            sample_rate=800,
            hop_size=100,
            clip_start_seconds=0.0,
            clip_duration_seconds=10.0,
            beats_per_bar=4,
            profile="rhythmic_first",
        )
        self.assertGreater(len(result["beats"]), 5)
        self.assertIn("confidence", result["beats"][0])
        self.assertIn("local_spacing_sec", result["beats"][0])
        self.assertIn("periodic_prior", result["beats"][0])
        self.assertEqual(result["profile"], "rhythmic_first")
        self.assertEqual(result["beats"][0]["profile"], "rhythmic_first")
        self.assertGreaterEqual(len(result["local_tempo_segments"]), 1)
        self.assertGreater(result["beat_confidence_mean"], 0.0)

    def test_build_motion_accents_uses_motion_score_fields(self) -> None:
        beats = [
            {"index": 0, "time_sec": 0.0, "strength": 0.20, "confidence": 0.60, "periodic_prior": 1.0, "local_spacing_sec": 0.5, "is_downbeat": True, "profile": "rhythmic_first"},
            {"index": 1, "time_sec": 0.5, "strength": 0.45, "confidence": 0.80, "periodic_prior": 0.4, "local_spacing_sec": 0.5, "is_downbeat": False, "profile": "rhythmic_first"},
            {"index": 2, "time_sec": 1.0, "strength": 0.25, "confidence": 0.55, "periodic_prior": 0.6, "local_spacing_sec": 0.5, "is_downbeat": False, "profile": "rhythmic_first"},
            {"index": 3, "time_sec": 1.5, "strength": 0.60, "confidence": 0.90, "periodic_prior": 0.35, "local_spacing_sec": 0.5, "is_downbeat": False, "profile": "rhythmic_first"},
        ]
        low_band = np.asarray([0.2, 0.6, 0.25, 0.7, 0.3, 0.75], dtype=np.float32)
        accents = _build_motion_accents(
            beats=beats,
            low_band_envelope=low_band,
            sample_rate=640,
            hop_size=320,
            clip_start_seconds=0.0,
            sections=[{"label": "verse", "start_time_sec": 1.45, "end_time_sec": 3.0, "confidence": 0.8}],
            profile="rhythmic_first",
        )
        self.assertGreaterEqual(len(accents), 1)
        self.assertIn("motion_score", accents[0])
        self.assertIn("section_importance", accents[0])
        self.assertIn("spacing_consistency", accents[0])
        self.assertEqual(accents[0]["profile"], "rhythmic_first")
        self.assertTrue(all(1 <= item["level"] <= 4 for item in accents))

    def test_build_multi_band_accents_emits_band_labels(self) -> None:
        beats = [
            {"index": 0, "time_sec": 0.0, "strength": 0.3, "confidence": 0.7, "local_spacing_sec": 0.5},
            {"index": 1, "time_sec": 0.5, "strength": 0.4, "confidence": 0.8, "local_spacing_sec": 0.5},
            {"index": 2, "time_sec": 1.0, "strength": 0.35, "confidence": 0.75, "local_spacing_sec": 0.5},
        ]
        band_envelopes = {
            "low": np.asarray([0.82, 0.1, 0.88, 0.1, 0.84], dtype=np.float32),
            "low_mid": np.asarray([0.2, 0.1, 0.91, 0.1, 0.86], dtype=np.float32),
            "high_attack": np.asarray([0.79, 0.2, 0.1, 0.82, 0.83], dtype=np.float32),
        }
        accents = _build_multi_band_accents(
            beats=beats,
            band_envelopes=band_envelopes,
            sample_rate=400,
            hop_size=100,
            clip_start_seconds=0.0,
            profile="rhythmic_first",
        )
        self.assertGreaterEqual(len(accents), 1)
        self.assertTrue(all(item["band"] in {"low", "low_mid", "high_attack"} for item in accents))
        self.assertTrue(all("confidence" in item for item in accents))
        self.assertTrue(all(item["profile"] == "rhythmic_first" for item in accents))

    def test_classify_analysis_priority_uses_style_seed_and_energy_or_confidence(self) -> None:
        rhythmic = _classify_analysis_priority("Street", "Popping", "mid_energy", 0.52)
        self.assertEqual(rhythmic["tier"], "rhythmic_first")
        self.assertTrue(rhythmic["style_seed"])
        self.assertIn("street_seed", rhythmic["reason_tags"])

        fallback = _classify_analysis_priority("Classic", "ShenYun", "low_energy", 0.42)
        self.assertEqual(fallback["tier"], "fallback")
        self.assertFalse(fallback["style_seed"])
        self.assertIn("fallback_classic", fallback["reason_tags"])


if __name__ == "__main__":
    unittest.main()
