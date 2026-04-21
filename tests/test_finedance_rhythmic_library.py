import tempfile
import unittest
from pathlib import Path

import numpy as np

from music_motion_lab.pipelines.finedance_rhythmic_library import (
    build_motion_library_coverage_report,
    build_finedance_rhythmic_library_showcase,
    build_finedance_rhythmic_smplx_library,
)


def _write_motion(raw_root: Path, sequence_id: str, frame_count: int = 360) -> None:
    motion_dir = raw_root / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)
    payload = np.zeros((frame_count, 315), dtype=np.float32)
    payload[:, 0] = np.linspace(0.0, 1.0, frame_count, dtype=np.float32)
    payload[:, 2] = np.linspace(0.0, 0.25, frame_count, dtype=np.float32)
    payload[:, 3:9] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    np.save(motion_dir / f"{sequence_id}.npy", payload)


def _report() -> dict:
    beats = [
        {
            "index": index,
            "time_sec": round(index * 0.5, 5),
            "strength": 0.6 if index % 4 == 0 else 0.25,
            "is_downbeat": index % 4 == 0,
        }
        for index in range(17)
    ]
    return {
        "dataset_name": "finedance",
        "entry_count": 2,
        "entries": [
            {
                "sequence_id": "001",
                "style_tags": {"song_name": "demo", "coarse_style": "Street", "fine_style": "Jazz"},
                "analysis_priority": {"tier": "rhythmic_first"},
                "audio_features": {
                    "bpm": 120.0,
                    "energy_label": "high_energy",
                    "beat_tracking": {"profile": "rhythmic_first", "global_bpm": 120.0},
                    "beats": beats,
                    "accent_candidates": [
                        {
                            "source_beat_index": 8,
                            "time_sec": 4.0,
                            "band": "low",
                            "strength": 0.9,
                            "confidence": 0.88,
                            "level": 4,
                        }
                    ],
                },
            },
            {
                "sequence_id": "002",
                "style_tags": {"song_name": "fallback", "coarse_style": "Latin", "fine_style": "Salsa"},
                "analysis_priority": {"tier": "fallback"},
                "audio_features": {"beats": beats},
            },
        ],
    }


class FineDanceRhythmicLibraryTests(unittest.TestCase):
    def test_builds_rhythmic_first_smplx_units_from_event_rail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "finedance"
            _write_motion(raw_root, "001")
            _write_motion(raw_root, "002")

            library = build_finedance_rhythmic_smplx_library(
                audio_feature_report=_report(),
                raw_root=raw_root,
            )

            self.assertEqual(library.schema_version, 2)
            self.assertEqual(library.library_id, "finedance_rhythmic_smplx_library_v1")
            self.assertTrue(library.units)
            self.assertTrue(all(unit["source_sequence"] == "001" for unit in library.units))
            first = library.units[0]
            self.assertEqual(first["skeleton_id"], "finedance_smplx_source_v1")
            self.assertEqual(first["mesh_backend"], "smplx_neutral")
            self.assertEqual(first["duration_beats"], 8)
            self.assertEqual(first["beat_range"], {"start": 0, "end_exclusive": 8})
            self.assertEqual(first["frame_range"]["start"], 0)
            self.assertGreater(first["frame_range"]["end_exclusive"], first["frame_range"]["start"])
            self.assertTrue(first["keyframes"])
            self.assertEqual(first["rhythm_profile"]["source_profile"], "rhythmic_first")
            self.assertEqual(first["priority_tier"], "rhythmic_first")
            self.assertEqual(first["source_song_bpm"], 120.0)
            self.assertEqual(first["source_song_energy"], "high_energy")
            self.assertGreaterEqual(first["source_song_quality_weight"], 0.99)
            self.assertIn("source_motion_path", first["reference_artifacts"])

    def test_can_drop_early_source_seconds_from_all_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "finedance"
            _write_motion(raw_root, "001", frame_count=900)

            library = build_finedance_rhythmic_smplx_library(
                audio_feature_report=_report(),
                raw_root=raw_root,
                min_source_sec=2.0,
            )

            self.assertTrue(library.units)
            self.assertTrue(all(int(unit["frame_range"]["start"]) >= 60 for unit in library.units))

    def test_include_fallback_and_showcase_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "finedance"
            _write_motion(raw_root, "001")
            _write_motion(raw_root, "002")

            library = build_finedance_rhythmic_smplx_library(
                audio_feature_report=_report(),
                raw_root=raw_root,
                rhythmic_only=False,
            )
            sequences = {unit["source_sequence"] for unit in library.units}
            self.assertEqual(sequences, {"001", "002"})

            showcase = build_finedance_rhythmic_library_showcase(library)
            self.assertEqual(showcase["library_id"], library.library_id)
            self.assertEqual(showcase["counts"]["sequence_count"], 2)
            self.assertGreaterEqual(showcase["counts"]["unit_count"], len(library.units))

    def test_attaches_compatible_next_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "finedance"
            _write_motion(raw_root, "001")

            library = build_finedance_rhythmic_smplx_library(
                audio_feature_report=_report(),
                raw_root=raw_root,
            )

            chainable = [unit for unit in library.units if unit["compatible_next_units"]]
            self.assertTrue(chainable)
            self.assertTrue(chainable[0]["compatible_next_units"][0].startswith("finedance_001_"))

    def test_builds_multi_beat_set_and_coverage_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "finedance"
            _write_motion(raw_root, "001")

            library = build_finedance_rhythmic_smplx_library(
                audio_feature_report=_report(),
                raw_root=raw_root,
                unit_beat_set=[2, 4, 8, 16],
            )
            durations = {unit["duration_beats"] for unit in library.units}

            self.assertTrue({2, 4, 8, 16}.issubset(durations))
            report = build_motion_library_coverage_report(library, required_unit_beats=[2, 4, 8, 16])
            self.assertTrue(report["acceptance"]["has_required_2_4_8_16_units"])
            self.assertFalse(report["coverage"]["missing_unit_beats"])
            self.assertIn("rhythmic_first", report["counts"]["priority_tier"])


if __name__ == "__main__":
    unittest.main()
