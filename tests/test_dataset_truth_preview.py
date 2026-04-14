import tempfile
import unittest
from pathlib import Path

from music_motion_lab.config import load_app_config
from music_motion_lab.pipelines.dataset_truth_preview import (
    build_dataset_truth_audio_feature_report,
    build_dataset_truth_preview_document,
    list_dataset_sequence_ids,
    resolve_dataset_truth_entries,
)


class DatasetTruthPreviewTests(unittest.TestCase):
    def test_build_dataset_truth_preview_document_contains_assets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            document = build_dataset_truth_preview_document(
                dataset_name="finedance",
                output_dir=output_dir,
                entries=[
                    {
                        "dataset_name": "finedance",
                        "sequence_id": "161",
                        "motion_path": "/tmp/demo/161.npy",
                        "music_path": "/tmp/demo/161.wav",
                        "label_path": "/tmp/demo/161.json",
                        "preview_video_path": "/tmp/demo/161.mp4",
                        "motion_frame_count": 4181,
                        "fps": 30.0,
                        "motion_duration_seconds": 139.37,
                        "audio_duration_seconds": 139.12,
                        "preview_duration_seconds": 12.0,
                        "clip_duration_seconds": 12.0,
                        "duration_delta_seconds": 0.25,
                        "audio_features": {
                            "bpm": 126.4,
                            "beats_per_bar": 4,
                            "energy_score": 0.53,
                            "energy_label": "mid_energy",
                            "beat_tracking": {
                                "mode": "elastic_grid",
                                "profile": "rhythmic_first",
                                "average_confidence": 0.72,
                                "local_tempo_segments": [{"start_time_sec": 0.0, "end_time_sec": 8.0, "bpm": 126.0}],
                            },
                            "energy_curve": [{"time_sec": 0.0, "value": 0.22}],
                            "beats": [{"index": 0, "time_sec": 0.0, "strength": 0.3, "is_downbeat": True, "confidence": 0.8, "periodic_prior": 1.0, "profile": "rhythmic_first"}],
                            "strong_beats": [{"index": 0, "beat_index": 0, "time_sec": 0.0, "strength": 0.3, "motion_score": 0.5, "confidence": 0.76, "level": 3, "is_downbeat": True, "profile": "rhythmic_first"}],
                            "accent_candidates": [{"index": 0, "time_sec": 0.1, "band": "low", "strength": 0.7, "confidence": 0.74, "level": 3, "profile": "rhythmic_first"}],
                            "drum_hits": [{"index": 0, "time_sec": 0.1, "strength": 0.7, "confidence": 0.74, "band": "low"}],
                        },
                        "analysis_priority": {"tier": "rhythmic_first", "style_seed": True, "reason_tags": ["street_seed", "mid_energy"]},
                        "audio_analysis_panel": {
                            "clip_start_sec": 0.0,
                            "clip_end_sec": 12.0,
                            "waveform": {"bin_count": 2, "encoding": "uint8/base64", "min_b64": "AA==", "max_b64": "/w=="},
                            "mel_spectrogram": {"mel_bin_count": 48, "time_bin_count": 2, "encoding": "uint8/base64", "values_b64": "AA=="},
                            "onset_curve": {"point_count": 2, "encoding": "uint8/base64", "values_b64": "AA=="},
                            "low_band_curve": {"point_count": 2, "encoding": "uint8/base64", "values_b64": "AA=="},
                            "sections": [{"index": 0, "label": "intro", "start_time_sec": 0.0, "end_time_sec": 2.0, "confidence": 0.5}],
                        },
                        "label_summary": {
                            "songName": "demo song",
                            "coarseStyle": "pop",
                            "fineStyle": "jazz",
                        },
                        "preview_summary": {},
                    }
                ],
            )
            self.assertIn("Milestone M2-2 · finedance dataset truth viewer", document)
            self.assertIn("demo song", document)
            self.assertIn("milestone: M2-2", document)
            self.assertIn("sequences in page: 1", document)
            self.assertIn("music wav", document)
            self.assertIn("official mesh preview mp4", document)
            self.assertIn("Play", document)
            self.assertIn("Search sequence, song, or style", document)
            self.assertIn("Motion Frames", document)
            self.assertIn("Preview Frames", document)
            self.assertIn("MP4 Size", document)
            self.assertIn("Previous", document)
            self.assertIn("Next", document)
            self.assertIn("scrollIntoView", document)
            self.assertIn('grid-template-columns: repeat(4, minmax(0, 1fr))', document)
            self.assertIn('"dataset_name": "finedance"', document)
            self.assertNotIn("&quot;dataset_name&quot;", document)
            self.assertIn("Rhythmic-First", document)
            self.assertIn("priorityFilters", document)
            self.assertIn('"analysis_priority"', document)
            self.assertIn("musicTimelineCanvas", document)
            self.assertIn("audioAnalysisCanvas", document)
            self.assertIn("Audio Analysis Panel", document)
            self.assertIn("Mel spectrogram", document)
            self.assertIn("Onset / Flux", document)
            self.assertIn("Low-band Envelope", document)
            self.assertIn("Event Rail", document)
            self.assertIn("motion accents (L1-L4)", document)
            self.assertIn("beats (confidence)", document)
            self.assertLess(document.index('id="truthVideo"'), document.index('class="analysis-panel"'))
            self.assertLess(document.index('class="analysis-panel"'), document.index('class="controls"'))
            self.assertIn("rhythmic-first", document)
            self.assertIn('"energy_label": "mid_energy"', document)
            self.assertIn('"audio_analysis_panel"', document)

    def test_resolve_dataset_truth_entries_finds_real_finedance_assets(self) -> None:
        config = load_app_config(project_root=Path("/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"))
        entries = resolve_dataset_truth_entries(config=config, dataset_name="finedance", sequence_ids=["161"])
        self.assertEqual(len(entries), 1)
        entry = entries[0]
        self.assertTrue(Path(entry["motion_path"]).exists())
        self.assertTrue(Path(entry["music_path"]).exists())
        self.assertTrue(Path(entry["preview_video_path"]).exists())
        self.assertGreater(entry["motion_frame_count"], 0)
        self.assertGreater(entry["audio_duration_seconds"], 0.0)
        self.assertGreater(entry["clip_duration_seconds"], 0.0)
        self.assertIn("audio_features", entry)
        self.assertIn("beat_tracking", entry["audio_features"])
        self.assertIn("energy_label", entry["audio_features"])
        self.assertIn("strong_beats", entry["audio_features"])
        self.assertIn("accent_candidates", entry["audio_features"])
        self.assertIn("drum_hits", entry["audio_features"])
        self.assertIn("analysis_priority", entry)
        self.assertIn(entry["analysis_priority"]["tier"], {"rhythmic_first", "fallback"})
        self.assertIn("profile", entry["audio_features"]["beat_tracking"])
        self.assertIn("local_tempo_segments", entry["audio_features"]["beat_tracking"])

    def test_build_document_uses_finedance_style_fallback_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            document = build_dataset_truth_preview_document(
                dataset_name="finedance",
                output_dir=output_dir,
                entries=[
                    {
                        "dataset_name": "finedance",
                        "sequence_id": "001",
                        "motion_path": "/tmp/demo/001.npy",
                        "music_path": "/tmp/demo/001.wav",
                        "label_path": "/tmp/demo/001.json",
                        "preview_video_path": "/tmp/demo/001.mp4",
                        "motion_frame_count": 100,
                        "fps": 30.0,
                        "motion_duration_seconds": 3.33,
                        "audio_duration_seconds": 3.30,
                        "preview_duration_seconds": 3.30,
                        "clip_duration_seconds": 3.30,
                        "duration_delta_seconds": 0.03,
                        "audio_features": {
                            "bpm": 92.0,
                            "beats_per_bar": 4,
                            "energy_score": 0.2,
                            "energy_label": "low_energy",
                            "beat_tracking": {"mode": "elastic_grid", "profile": "fallback", "average_confidence": 0.61, "local_tempo_segments": []},
                            "energy_curve": [],
                            "beats": [],
                            "strong_beats": [],
                            "accent_candidates": [],
                            "drum_hits": [],
                        },
                        "analysis_priority": {"tier": "fallback", "style_seed": False, "reason_tags": ["fallback_folk"]},
                        "audio_analysis_panel": {
                            "clip_start_sec": 0.0,
                            "clip_end_sec": 3.3,
                            "waveform": {"bin_count": 2, "encoding": "uint8/base64", "min_b64": "AA==", "max_b64": "/w=="},
                            "mel_spectrogram": {"mel_bin_count": 48, "time_bin_count": 2, "encoding": "uint8/base64", "values_b64": "AA=="},
                            "onset_curve": {"point_count": 2, "encoding": "uint8/base64", "values_b64": "AA=="},
                            "low_band_curve": {"point_count": 2, "encoding": "uint8/base64", "values_b64": "AA=="},
                            "sections": [],
                        },
                        "label_summary": {
                            "name": "folk demo",
                            "style1": "Folk",
                            "style2": "Dai",
                        },
                        "preview_summary": {},
                    }
                ],
            )
            self.assertIn("folk demo", document)
            self.assertIn("Folk / Dai", document)

    def test_build_audio_feature_report_contains_style_tags_and_features(self) -> None:
        report = build_dataset_truth_audio_feature_report(
            dataset_name="finedance",
            entries=[
                {
                    "sequence_id": "010",
                    "label_summary": {"name": "demo", "style1": "Street", "style2": "Korean"},
                    "analysis_priority": {"tier": "rhythmic_first", "style_seed": True, "reason_tags": ["street_seed", "high_energy"]},
                    "audio_features": {"energy_label": "high_energy", "bpm": 140.0},
                    "audio_analysis_panel": {"waveform": {"bin_count": 512}},
                }
            ],
        )
        self.assertEqual(report["dataset_name"], "finedance")
        self.assertEqual(report["entry_count"], 1)
        self.assertEqual(report["entries"][0]["style_tags"]["coarse_style"], "Street")
        self.assertEqual(report["entries"][0]["analysis_priority"]["tier"], "rhythmic_first")
        self.assertEqual(report["entries"][0]["audio_features"]["energy_label"], "high_energy")
        self.assertIn("audio_analysis_panel", report["entries"][0])

    def test_list_dataset_sequence_ids_counts_finedance_entries(self) -> None:
        config = load_app_config(project_root=Path("/Users/alex/Desktop/codex project/3d-digital/music-motion-lab"))
        sequence_ids = list_dataset_sequence_ids(config=config, dataset_name="finedance")
        self.assertEqual(len(sequence_ids), 203)
        self.assertIn("001", sequence_ids)
        self.assertIn("211", sequence_ids)


if __name__ == "__main__":
    unittest.main()
