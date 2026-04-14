import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from music_motion_lab.pipelines.music_analysis import analyze_song


class MusicAnalysisTests(unittest.TestCase):
    def test_manual_overrides_replace_detected_structure(self) -> None:
        sample_rate = 22050
        duration_sec = 2.0
        time_axis = np.linspace(0.0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
        signal = 0.15 * np.sin(2.0 * np.pi * 220.0 * time_axis)

        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = Path(tmp_dir) / "test.wav"
            sf.write(audio_path, signal, sample_rate)

            payload = analyze_song(
                input_path=audio_path,
                song_id="test_song",
                overrides={
                    "bpm": 80.0,
                    "beats_per_bar": 4,
                    "beats": [0.0, 0.75, 1.5],
                    "downbeats": [0.0],
                    "accents": [0.0, 1.5],
                    "phrases": [
                        {"label": "phrase_a", "startBeat": 0, "endBeatExclusive": 2},
                        {"label": "phrase_b", "startBeat": 2, "endBeatExclusive": 3}
                    ],
                    "sections": [
                        {"label": "intro", "startBeat": 0, "endBeatExclusive": 2},
                        {"label": "outro", "startBeat": 2, "endBeatExclusive": 3}
                    ]
                },
                override_source="config/test_override.json",
            ).to_dict()

        self.assertEqual(payload["analysis_mode"], "manual_override")
        self.assertEqual(payload["override_source"], "config/test_override.json")
        self.assertEqual(payload["applied_override_keys"], ["accents", "beats", "beats_per_bar", "bpm", "downbeats", "phrases", "sections"])
        self.assertFalse(payload["manual_review_required"])
        self.assertEqual([entry["time_sec"] for entry in payload["beats"]], [0.0, 0.75, 1.5])
        self.assertEqual([entry["time_sec"] for entry in payload["downbeats"]], [0.0])
        self.assertEqual([entry["label"] for entry in payload["sections"]], ["intro", "outro"])


if __name__ == "__main__":
    unittest.main()
