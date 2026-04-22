import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from music_motion_lab.pipelines.streaming_smplx import (
    annotate_finedance_motion_units,
    build_motion_transition_report,
    build_streaming_song_event_records,
    calibrate_song_event_rail,
    evaluate_streaming_event_rail,
    evaluate_streaming_planner_records,
    export_unity_streaming_runtime_bundle,
    simulate_streaming_smplx_plan_records,
    stream_events_to_song_event_map,
    stream_plan_to_stitch_manifest,
)


def _write_pulse_audio(path: Path, duration_sec: float = 3.2, sample_rate: int = 22050) -> None:
    sample_count = int(duration_sec * sample_rate)
    audio = np.zeros((sample_count,), dtype=np.float32)
    for time_sec in np.arange(0.0, duration_sec, 0.5):
        start = int(time_sec * sample_rate)
        end = min(sample_count, start + int(0.035 * sample_rate))
        if end > start:
            audio[start:end] += np.hanning(end - start).astype(np.float32) * 0.9
    sf.write(str(path), audio, sample_rate)


def _write_motion(raw_root: Path, sequence_id: str = "001", frame_count: int = 180) -> Path:
    motion_dir = raw_root / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)
    motion = np.zeros((frame_count, 315), dtype=np.float32)
    motion[:, 0] = np.linspace(0.0, 0.8, frame_count, dtype=np.float32)
    motion[:, 2] = np.sin(np.linspace(0.0, np.pi, frame_count, dtype=np.float32)) * 0.2
    motion[:, 3:9] = np.asarray([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    path = motion_dir / f"{sequence_id}.npy"
    np.save(path, motion)
    return path


def _library(source_motion_path: Path) -> dict:
    unit = {
        "unit_id": "finedance_001_good",
        "source_sequence": "001",
        "frame_range": {"start": 0, "end_exclusive": 120},
        "beat_range": {"start": 0, "end_exclusive": 4},
        "duration_sec": 2.0,
        "duration_beats": 4,
        "duration_beats_estimate": 4.0,
        "energy": "mid_energy",
        "style_tags": ["street"],
        "priority_tier": "rhythmic_first",
        "source_song_bpm": 120.0,
        "source_song_energy": "mid_energy",
        "source_song_style_tags": ["street"],
        "source_song_quality_weight": 1.0,
        "keyframes": [
            {"kind": "downbeat", "beat_offset": 0, "frame_index": 0, "strength": 0.95, "is_downbeat": True},
            {"kind": "beat", "beat_offset": 1, "frame_index": 30, "strength": 0.25, "is_downbeat": False},
            {"kind": "accent", "beat_offset": 2, "frame_index": 60, "strength": 0.8, "is_downbeat": False},
            {"kind": "beat", "beat_offset": 3, "frame_index": 90, "strength": 0.25, "is_downbeat": False},
        ],
        "rhythm_profile": {"source_bpm": 120.0, "accent_count": 2},
        "entry_anchor": {"planar_speed": 0.2, "root_yaw_deg": 0.0},
        "exit_anchor": {"planar_speed": 0.22, "root_yaw_deg": 4.0},
        "transition_profile": {"yaw_delta_deg": 4.0},
        "reference_artifacts": {"source_motion_path": str(source_motion_path)},
        "compatible_next_units": ["finedance_001_good"],
    }
    weaker = dict(unit)
    weaker["unit_id"] = "finedance_001_weak"
    weaker["keyframes"] = [{"kind": "beat", "beat_offset": 1, "frame_index": 30, "strength": 0.1, "is_downbeat": False}]
    weaker["compatible_next_units"] = []
    other_sequence = dict(unit)
    other_sequence["unit_id"] = "finedance_002_other"
    other_sequence["source_sequence"] = "002"
    other_sequence["style_tags"] = ["street", "jazz"]
    other_sequence["source_song_style_tags"] = ["street", "jazz"]
    other_sequence["source_song_quality_weight"] = 0.92
    other_sequence["compatible_next_units"] = []
    other_sequence["entry_anchor"] = {"planar_speed": 0.21, "root_yaw_deg": 3.0}
    other_sequence["exit_anchor"] = {"planar_speed": 0.24, "root_yaw_deg": 8.0}
    bad_sequence = dict(unit)
    bad_sequence["unit_id"] = "finedance_003_bad"
    bad_sequence["source_sequence"] = "003"
    bad_sequence["style_tags"] = ["jazz"]
    bad_sequence["source_song_style_tags"] = ["jazz"]
    bad_sequence["energy"] = "high_energy"
    bad_sequence["source_song_energy"] = "high_energy"
    bad_sequence["source_song_quality_weight"] = 0.72
    bad_sequence["priority_tier"] = "fallback"
    bad_sequence["keyframes"] = [{"kind": "beat", "beat_offset": 1, "frame_index": 30, "strength": 0.1, "is_downbeat": False}]
    bad_sequence["compatible_next_units"] = []
    bad_sequence["entry_anchor"] = {"planar_speed": 2.0, "root_yaw_deg": 170.0}
    bad_sequence["exit_anchor"] = {"planar_speed": 2.25, "root_yaw_deg": 178.0}
    return {
        "schema_version": 2,
        "library_id": "demo_library",
        "source_roots": {},
        "units": [weaker, unit, other_sequence, bad_sequence],
        "notes": [],
        "generated_at_utc": "2026-04-21T00:00:00+00:00",
    }


class StreamingSmplxTests(unittest.TestCase):
    def test_streaming_events_are_deterministic_and_do_not_emit_future_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "pulse.wav"
            _write_pulse_audio(audio_path)

            first = build_streaming_song_event_records(audio_path, "pulse", initial_buffer_sec=2.0, chunk_ms=100.0)
            second = build_streaming_song_event_records(audio_path, "pulse", initial_buffer_sec=2.0, chunk_ms=100.0)

        first_without_time = [dict(record) for record in first]
        second_without_time = [dict(record) for record in second]
        first_without_time[0].pop("generated_at_utc", None)
        second_without_time[0].pop("generated_at_utc", None)
        self.assertEqual(first_without_time, second_without_time)
        header = first[0]
        self.assertEqual(header["kind"], "stream_header")
        self.assertEqual(header["lookahead_sec"], 2.0)
        self.assertEqual(header["lookfront_sec"], 1.0)
        self.assertEqual(header["total_future_sec"], 3.0)
        ticks = [record for record in first if record["kind"] == "tick"]
        self.assertTrue(ticks)
        self.assertLessEqual(ticks[0]["available_audio_until_sec"], 3.0)
        for tick in ticks:
            available = float(tick["available_audio_until_sec"])
            self.assertLessEqual(float(tick["lookahead_sec"]), 2.00001)
            self.assertLessEqual(float(tick["lookfront_sec"]), 1.00001)
            self.assertLessEqual(float(tick["total_future_sec"]), 3.00001)
            for key in ("beats", "downbeats", "drum_hits", "accents"):
                for event in tick[key]:
                    self.assertLessEqual(float(event["time_sec"]), available + 1e-8)

    def test_annotation_adds_streaming_retrieval_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            raw_root = Path(tmpdir) / "finedance"
            motion_path = _write_motion(raw_root)
            annotated = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir))
            joint_annotated = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")

        unit = annotated["units"][0]
        self.assertTrue(annotated["library_id"].endswith("_m9_annotated"))
        for key in (
            "count_grid",
            "accent_lock_frames",
            "motion_accent_frames",
            "foot_contact_windows",
            "entry_pose_anchor",
            "exit_pose_anchor",
            "root_velocity",
            "yaw_delta",
            "energy_curve",
            "movement_quality",
            "safe_retime_range",
        ):
            self.assertIn(key, unit)
        self.assertTrue(unit["count_grid"])
        self.assertGreater(unit["safe_retime_range"]["max"], unit["safe_retime_range"]["min"])
        self.assertTrue(joint_annotated["library_id"].endswith("_m10_annotated"))
        self.assertEqual(joint_annotated["units"][0]["annotation_profile"]["contact_mode"], "joints")

    def test_streaming_planner_prefers_downbeat_locked_unit_and_keeps_future_guard(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir))
        stream_events = [
            {
                "kind": "stream_header",
                "song_id": "demo",
                "duration_sec": 4.2,
                "initial_buffer_sec": 2.0,
                "lookahead_sec": 2.0,
                "lookfront_sec": 1.0,
                "total_future_sec": 3.0,
                "beats_per_bar": 4,
                "source_audio_path": "demo.wav",
            },
            {
                "kind": "tick",
                "tick_index": 0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 3.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.9},
                "beats": [
                    {"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.9, "is_downbeat": index % 4 == 0}
                    for index in range(5)
                ],
                "downbeats": [{"index": 0, "time_sec": 0.0, "confidence": 0.9, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 0.0, "strength": 0.95, "kind": "kick"}],
                "accents": [{"index": 0, "time_sec": 1.0, "strength": 0.8, "kind": "accent_peak"}],
            },
            {
                "kind": "tick",
                "tick_index": 20,
                "playhead_sec": 2.0,
                "available_audio_until_sec": 4.2,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.9},
                "beats": [
                    {"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.9, "is_downbeat": index % 4 == 0}
                    for index in range(9)
                ],
                "downbeats": [{"index": 1, "time_sec": 2.0, "confidence": 0.9, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 2.0, "strength": 0.95, "kind": "kick"}],
                "accents": [{"index": 0, "time_sec": 3.0, "strength": 0.8, "kind": "accent_peak"}],
            },
        ]

        plan = simulate_streaming_smplx_plan_records(stream_events, library, stream_events_path="/tmp/events.jsonl", max_steps=2)
        decisions = [record for record in plan if record["kind"] == "decision"]

        self.assertEqual(len(decisions), 2)
        self.assertEqual(decisions[0]["selected_unit_id"], "finedance_001_good")
        for decision in decisions:
            self.assertTrue(decision["future_visibility_guard"]["passed"])
            self.assertLessEqual(
                float(decision["available_audio_until_sec"]),
                float(decision["playhead_sec"]) + 3.00001,
            )
        self.assertGreater(decisions[0]["score_breakdown"]["rhythm_lock"], 0.5)

    def test_streaming_planner_can_constrain_source_sequences(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
        stream_events = [
            {"kind": "stream_header", "song_id": "demo", "duration_sec": 2.1, "initial_buffer_sec": 2.0, "beats_per_bar": 4},
            {
                "kind": "tick",
                "tick_index": 0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 2.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.9},
                "beats": [{"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.9, "is_downbeat": index % 4 == 0} for index in range(5)],
                "downbeats": [{"index": 0, "time_sec": 0.0, "confidence": 0.9, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 0.0, "strength": 0.95, "kind": "kick"}],
                "accents": [],
            },
        ]

        plan = simulate_streaming_smplx_plan_records(
            stream_events,
            library,
            planner_version="m12",
            source_sequence_allowlist=["001"],
            max_steps=1,
        )
        header = plan[0]
        decisions = [record for record in plan if record["kind"] == "decision"]

        self.assertEqual(header["source_sequence_allowlist"], ["001"])
        self.assertEqual(decisions[0]["source_sequence"], "001")

    def test_m17_can_insert_neutral_idle_and_delay_until_confident_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
        stream_events = [
            {
                "kind": "stream_header",
                "song_id": "hold_demo",
                "duration_sec": 7.2,
                "initial_buffer_sec": 2.0,
                "lookahead_sec": 2.0,
                "lookfront_sec": 1.0,
                "total_future_sec": 3.0,
                "beats_per_bar": 4,
            },
            {
                "kind": "tick",
                "tick_index": 20,
                "playhead_sec": 2.0,
                "available_audio_until_sec": 5.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.42},
                "beats": [{"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.42, "is_downbeat": index % 4 == 0} for index in range(11)],
                "downbeats": [{"index": 0, "time_sec": 4.0, "confidence": 0.42, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 0.0, "strength": 0.95, "kind": "kick"}],
                "accents": [],
            },
            {
                "kind": "tick",
                "tick_index": 26,
                "playhead_sec": 2.6,
                "available_audio_until_sec": 5.6,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.92},
                "beats": [{"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.92, "is_downbeat": index % 4 == 0} for index in range(12)],
                "downbeats": [{"index": 0, "time_sec": 4.0, "confidence": 0.92, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 5.5, "strength": 0.95, "kind": "kick"}],
                "accents": [],
            },
            {
                "kind": "tick",
                "tick_index": 30,
                "playhead_sec": 3.0,
                "available_audio_until_sec": 6.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.95},
                "beats": [{"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.95, "is_downbeat": index % 4 == 0} for index in range(13)],
                "downbeats": [{"index": 0, "time_sec": 4.0, "confidence": 0.95, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 5.5, "strength": 0.95, "kind": "kick"}],
                "accents": [],
            },
        ]

        plan = simulate_streaming_smplx_plan_records(
            stream_events,
            library,
            planner_version="m17",
            initial_hold_sec=5.0,
            initial_pose_mode="neutral_idle",
            max_steps=3,
        )
        evaluation = evaluate_streaming_planner_records(plan)
        header = plan[0]
        decisions = [record for record in plan if record["kind"] == "decision"]
        manifest = stream_plan_to_stitch_manifest(plan, fps=30, blend_frames=10)

        self.assertEqual(header["planner_version"], "m17")
        self.assertEqual(header["initial_hold_sec"], 5.0)
        self.assertEqual(header["initial_pose_mode"], "neutral_idle")
        self.assertEqual(decisions[0]["switch_reason"]["mode"], "initial_upright_hold")
        self.assertEqual(decisions[0]["target_time_sec"], {"start": 0.0, "end": 5.0})
        self.assertEqual(decisions[0]["pose_source"], "smplx_neutral_idle")
        self.assertGreaterEqual(decisions[1]["target_time_sec"]["start"], 5.0)
        self.assertEqual(manifest["steps"][0]["pose_source"], "smplx_neutral_idle")
        self.assertIsNone(manifest["steps"][0]["source_motion_path"])
        self.assertEqual(len(manifest["cache_requests"]), 1)
        self.assertGreaterEqual(evaluation["metrics"]["low_confidence_continuation_count"], 1)
        self.assertEqual(evaluation["metrics"]["initial_upright_hold_sec"], 5.0)
        self.assertGreaterEqual(evaluation["metrics"]["first_dance_start_sec"], 5.0)

    def test_m12_tail_policy_extends_final_segment_and_evaluator_reports_acceptance(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
        stream_events = [
            {
                "kind": "stream_header",
                "song_id": "tail_demo",
                "duration_sec": 4.2,
                "initial_buffer_sec": 2.0,
                "beats_per_bar": 4,
                "source_audio_path": "demo.wav",
            },
            {
                "kind": "tick",
                "tick_index": 0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 2.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.9},
                "beats": [
                    {"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.9, "is_downbeat": index % 4 == 0}
                    for index in range(5)
                ],
                "downbeats": [{"index": 0, "time_sec": 0.0, "confidence": 0.9, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 0.0, "strength": 0.95, "kind": "kick"}],
                "accents": [{"index": 0, "time_sec": 1.0, "strength": 0.8, "kind": "accent_peak"}],
                "segment_hypotheses": [{"start_time_sec": 0.0, "end_time_sec": 2.0, "duration_beats": 4, "confidence": 0.9}],
            },
            {
                "kind": "tick",
                "tick_index": 20,
                "playhead_sec": 2.0,
                "available_audio_until_sec": 4.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.9},
                "beats": [
                    {"index": index, "time_sec": index * 0.5, "strength": 0.7, "confidence": 0.9, "is_downbeat": index % 4 == 0}
                    for index in range(9)
                ],
                "downbeats": [{"index": 1, "time_sec": 2.0, "confidence": 0.9, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 2.0, "strength": 0.95, "kind": "kick"}],
                "accents": [{"index": 0, "time_sec": 3.0, "strength": 0.8, "kind": "accent_peak"}],
                "segment_hypotheses": [{"start_time_sec": 2.0, "end_time_sec": 4.0, "duration_beats": 4, "confidence": 0.9}],
            },
        ]

        plan = simulate_streaming_smplx_plan_records(
            stream_events,
            library,
            stream_events_path="/tmp/events.jsonl",
            planner_version="m12",
            tail_policy="recover",
        )
        decisions = [record for record in plan if record["kind"] == "decision"]
        evaluation = evaluate_streaming_planner_records(plan)

        self.assertEqual(decisions[-1]["target_time_sec"]["end"], 4.2)
        self.assertTrue(decisions[-1]["switch_reason"]["tail_extended_to_song_end"])
        self.assertTrue(evaluation["acceptance"]["no_gaps"])
        self.assertTrue(evaluation["acceptance"]["non_tail_max_speed_le_1_25"])

    def test_m15_builds_multi_song_cohort_and_tracks_hard_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            _write_motion(Path(tmpdir) / "finedance", sequence_id="002")
            _write_motion(Path(tmpdir) / "finedance", sequence_id="003")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
        stream_events = [
            {
                "kind": "stream_header",
                "song_id": "m15_demo",
                "duration_sec": 4.2,
                "initial_buffer_sec": 2.0,
                "beats_per_bar": 4,
            },
            {
                "kind": "tick",
                "tick_index": 0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 2.0,
                "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.95},
                "beats": [{"index": index, "time_sec": index * 0.5, "strength": 0.75, "confidence": 0.9, "is_downbeat": index % 4 == 0} for index in range(5)],
                "downbeats": [{"index": 0, "time_sec": 0.0, "confidence": 0.9, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 0.0, "strength": 0.92, "kind": "kick"}, {"index": 1, "time_sec": 1.0, "strength": 0.85, "kind": "snare"}],
                "accents": [{"index": 0, "time_sec": 1.0, "strength": 0.82, "kind": "accent_peak"}],
            },
        ]

        plan = simulate_streaming_smplx_plan_records(
            stream_events,
            library,
            planner_version="m15",
            cohort_size=3,
            max_steps=2,
        )
        decisions = [record for record in plan if record["kind"] == "decision"]
        evaluation = evaluate_streaming_planner_records(plan)

        self.assertEqual(plan[0]["planner_version"], "m15")
        self.assertEqual(plan[0]["cohort_size"], 3)
        self.assertGreaterEqual(len(decisions[0]["cohort_source_sequences"]), 3)
        self.assertEqual(decisions[0]["selected_from_tier"], "rhythmic_first")
        self.assertIn("001", evaluation["cohort_source_sequences"])
        self.assertIn("transition_hard_reject_count", evaluation["metrics"])
        self.assertIn("rhythm_hard_reject_count", evaluation["metrics"])

    def test_m17_tracks_speed_and_transition_hard_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
        stream_events = [
            {
                "kind": "stream_header",
                "song_id": "m17_demo",
                "duration_sec": 3.5,
                "initial_buffer_sec": 2.0,
                "lookahead_sec": 2.0,
                "lookfront_sec": 1.0,
                "total_future_sec": 3.0,
                "beats_per_bar": 4,
            },
            {
                "kind": "tick",
                "tick_index": 0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 3.0,
                "beat_phase": {"bpm": 360.0, "spacing_sec": 1.0 / 6.0, "offset_sec": 0.0, "confidence": 0.95},
                "beats": [{"index": index, "time_sec": index / 6.0, "strength": 0.75, "confidence": 0.95, "is_downbeat": index % 4 == 0} for index in range(19)],
                "downbeats": [{"index": 0, "time_sec": 0.0, "confidence": 0.95, "is_downbeat": True}],
                "drum_hits": [{"index": 0, "time_sec": 0.0, "strength": 0.95, "kind": "kick"}],
                "accents": [],
            },
        ]

        plan = simulate_streaming_smplx_plan_records(
            stream_events,
            library,
            planner_version="m17",
            initial_hold_sec=0.0,
            max_steps=1,
        )
        decisions = [record for record in plan if record["kind"] == "decision"]
        rejected = decisions[0]["rejected_top_candidates"]
        evaluation = evaluate_streaming_planner_records(plan)

        self.assertTrue(
            any("non_tail_speed_outside_0_92_1_08" in list(item.get("reasons", []) or []) for item in rejected)
        )
        self.assertGreaterEqual(evaluation["metrics"]["non_tail_speed_hard_reject_count"], 1)

    def test_m17_prefers_changing_after_two_consecutive_same_motion_units(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
        good = next(unit for unit in library["units"] if unit["unit_id"] == "finedance_001_good")
        alternative = dict(good)
        alternative["unit_id"] = "finedance_001_alt"
        for unit in library["units"]:
            unit["compatible_next_units"] = ["finedance_001_good", "finedance_001_alt"]
        alternative["compatible_next_units"] = ["finedance_001_good", "finedance_001_alt"]
        library["units"].append(alternative)

        beats = [
            {"index": index, "time_sec": index * 0.5, "strength": 0.8, "confidence": 0.95, "is_downbeat": index % 4 == 0}
            for index in range(27)
        ]
        stream_events = [
            {
                "kind": "stream_header",
                "song_id": "m17_repeat_demo",
                "duration_sec": 12.2,
                "initial_buffer_sec": 2.0,
                "lookahead_sec": 2.0,
                "lookfront_sec": 1.0,
                "total_future_sec": 3.0,
                "beats_per_bar": 4,
            }
        ]
        for tick_index, playhead_sec in enumerate([0.0, 1.0, 3.0, 5.0, 7.0, 9.0]):
            available_until = playhead_sec + 3.0
            visible_beats = [beat for beat in beats if beat["time_sec"] <= available_until + 1e-8]
            stream_events.append(
                {
                    "kind": "tick",
                    "tick_index": tick_index,
                    "playhead_sec": playhead_sec,
                    "available_audio_until_sec": available_until,
                    "beat_phase": {"bpm": 120.0, "spacing_sec": 0.5, "offset_sec": 0.0, "confidence": 0.95},
                    "beats": visible_beats,
                    "downbeats": [beat for beat in visible_beats if beat["is_downbeat"]],
                    "drum_hits": [{"index": index, "time_sec": beat["time_sec"], "strength": 0.95, "kind": "kick"} for index, beat in enumerate(visible_beats[::2])],
                    "accents": [{"index": index, "time_sec": beat["time_sec"], "strength": 0.85, "kind": "accent_peak"} for index, beat in enumerate(visible_beats[1::2])],
                }
            )

        plan = simulate_streaming_smplx_plan_records(
            stream_events,
            library,
            planner_version="m17",
            initial_hold_sec=0.0,
            max_steps=6,
        )
        decisions = [record for record in plan if record["kind"] == "decision"]
        selected_ids = [decision["selected_unit_id"] for decision in decisions]
        max_run = 0
        current_id = None
        current_run = 0
        for unit_id in selected_ids:
            if unit_id == current_id:
                current_run += 1
            else:
                current_id = unit_id
                current_run = 1
            max_run = max(max_run, current_run)
        evaluation = evaluate_streaming_planner_records(plan)

        self.assertLessEqual(max_run, 2)
        self.assertLessEqual(evaluation["metrics"]["max_consecutive_motion_unit_run"], 2)
        self.assertIn("repeat_unit_hard_reject_count", evaluation["metrics"])
        self.assertIn("repeat_unit_preferred_reject_count", evaluation["metrics"])
        self.assertTrue(evaluation["acceptance"]["max_consecutive_motion_unit_run_le_2"])

    def test_stream_plan_to_manifest_is_continuous_and_carries_decisions(self) -> None:
        decisions = [
            {
                "kind": "stream_plan_header",
                "song_id": "demo",
                "library_id": "library",
                "initial_buffer_sec": 2.0,
            },
            {
                "kind": "decision",
                "index": 0,
                "decision_time_sec": 0.0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 2.0,
                "target_time_sec": {"start": 0.0, "end": 2.0},
                "selected_unit_id": "unit_a",
                "source_sequence": "001",
                "source_frame_range": {"start": 0, "end_exclusive": 60},
                "source_beat_range": {"start": 0, "end_exclusive": 4},
                "source_motion_path": "/tmp/001.npy",
                "speed_scale": 1.0,
                "score": 0.9,
                "score_breakdown": {"transition_smoothness": 1.0, "rhythm_lock": 1.0},
                "future_visibility_guard": {"passed": True},
                "rhythm_locks": [{"kind": "stream_rhythm_lock", "time_sec": 0.0, "role": "count_1_downbeat", "score": 1.0}],
            },
            {
                "kind": "decision",
                "index": 1,
                "decision_time_sec": 0.0,
                "playhead_sec": 0.0,
                "available_audio_until_sec": 2.0,
                "target_time_sec": {"start": 2.0, "end": 4.0},
                "selected_unit_id": "unit_b",
                "source_sequence": "001",
                "source_frame_range": {"start": 60, "end_exclusive": 120},
                "source_beat_range": {"start": 4, "end_exclusive": 8},
                "source_motion_path": "/tmp/001.npy",
                "speed_scale": 1.0,
                "score": 0.8,
                "score_breakdown": {"transition_smoothness": 0.7, "rhythm_lock": 0.8},
                "future_visibility_guard": {"passed": True},
                "rhythm_locks": [{"kind": "stream_rhythm_lock", "time_sec": 2.0, "role": "count_1_downbeat", "score": 1.0}],
            },
        ]

        manifest = stream_plan_to_stitch_manifest(decisions, fps=30, blend_frames=10)

        self.assertEqual(manifest["steps"][0]["scene_frame_end"] + 1, manifest["steps"][1]["scene_frame_start"])
        self.assertEqual(len(manifest["streaming_review"]["decisions"]), 2)
        self.assertEqual(manifest["steps"][1]["rhythm_locks"][0]["scene_frame"], 61)

    def test_stream_events_aggregate_to_song_event_map(self) -> None:
        records = [
            {"kind": "stream_header", "song_id": "demo", "duration_sec": 1.0, "source_audio_path": "demo.wav", "beats_per_bar": 4},
            {
                "kind": "tick",
                "available_audio_until_sec": 1.0,
                "beats": [{"time_sec": 0.0, "confidence": 0.9, "is_downbeat": True, "strength": 0.8}],
                "downbeats": [{"time_sec": 0.0, "confidence": 0.9, "is_downbeat": True}],
                "accents": [{"time_sec": 0.5, "strength": 0.7}],
                "drum_hits": [{"time_sec": 0.0, "strength": 0.9, "kind": "kick"}],
            },
        ]

        event_map = stream_events_to_song_event_map(records)

        self.assertEqual(event_map["song_id"], "demo")
        self.assertEqual(len(event_map["beats"]), 1)
        self.assertEqual(len(event_map["downbeats"]), 1)
        self.assertEqual(event_map["beats"][0]["count"], 1)
        self.assertEqual(len(event_map["drum_hits"]), 1)
        self.assertTrue(event_map["manual_review_required"])

    def test_calibration_event_eval_transition_report_and_runtime_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "bundle"
            motion_path = _write_motion(Path(tmpdir) / "finedance")
            library = annotate_finedance_motion_units(_library(motion_path), project_root=Path(tmpdir), contact_mode="joints")
            records = [
                {"kind": "stream_header", "song_id": "demo", "duration_sec": 1.0, "source_audio_path": "demo.wav", "beats_per_bar": 4},
                {
                    "kind": "tick",
                    "playhead_sec": 0.0,
                    "available_audio_until_sec": 1.0,
                    "beats": [{"time_sec": 0.0, "confidence": 0.9, "is_downbeat": True, "strength": 0.8}],
                    "downbeats": [{"time_sec": 0.0, "confidence": 0.9, "is_downbeat": True}],
                    "accents": [{"time_sec": 0.5, "strength": 0.7}],
                    "drum_hits": [{"time_sec": 0.0, "strength": 0.9, "kind": "kick"}],
                },
            ]
            calibrated = calibrate_song_event_rail(records, manual_overrides={"beats": [{"time_sec": 0.0, "confidence": 1.0}]})
            event_eval = evaluate_streaming_event_rail(records, reference_event_map=calibrated)
            transition_report = build_motion_transition_report(
                {
                    "report_id": "mesh_report",
                    "song_id": "demo",
                    "metrics": {
                        "max_temporal_vertex_delta_after_smoothing": 0.04,
                        "max_root_acceleration_discontinuity_proxy": 0.1,
                        "max_joint_jerk_proxy": 0.2,
                        "max_foot_slide_proxy": 0.01,
                    },
                    "transitions": [{"index": 0, "outgoing_step": 0, "incoming_step": 1, "blend_frames": 10}],
                }
            )
            bundle_report = export_unity_streaming_runtime_bundle(
                output_dir=output_dir,
                annotated_library=library,
                stream_event_records=records,
            )
            bundle_manifest_exists = (output_dir / "bundle_manifest.json").exists()

        self.assertFalse(calibrated["manual_review_required"])
        self.assertTrue(event_eval["acceptance"]["future_safe"])
        self.assertTrue(transition_report["acceptance"]["vertex_delta_le_0_05"])
        self.assertTrue(bundle_report["acceptance"]["does_not_require_python_smplx_runtime"])
        self.assertTrue(bundle_manifest_exists)


if __name__ == "__main__":
    unittest.main()
