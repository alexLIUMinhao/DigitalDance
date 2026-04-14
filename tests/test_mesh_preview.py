import json
import tempfile
import unittest
from pathlib import Path

from music_motion_lab.pipelines.mesh_preview import build_mesh_preview_manifest
from music_motion_lab.pipelines.preview import build_preview_and_retarget_report


class MeshPreviewTests(unittest.TestCase):
    def test_build_preview_adds_sequence_motion_fbx_when_report_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preview_cache = Path(tmpdir) / "previewCache"
            report_dir = preview_cache / "retarget_pose_consistency" / "finedance" / "161"
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "contemporary_finedance_finedance_161_consistency_report.json"
            report_path.write_text(
                json.dumps(
                    {
                        "datasetName": "finedance",
                        "sequenceId": "161",
                        "status": "pass",
                        "summary": {},
                        "artifacts": {},
                    }
                ),
                encoding="utf-8",
            )
            sequence_fbx = preview_cache / "dataset_sequences" / "finedance" / "contemporary" / "finedance" / "contemporary_finedance_finedance_161.fbx"
            sequence_fbx.parent.mkdir(parents=True, exist_ok=True)
            sequence_fbx.write_text("dummy", encoding="utf-8")

            plan = {
                "plan_id": "demo_plan",
                "steps": [{"index": 0, "selected_unit_id": "unit_a", "source_sequence": "161"}],
            }
            motion_library = {
                "units": [
                    {
                        "unit_id": "unit_a",
                        "dataset": "finedance",
                        "source_sequence": "161",
                        "frame_range": {"start": 0, "end_exclusive": 60},
                        "reference_artifacts": {
                            "reference_retarget_report_json": str(report_path),
                        },
                    }
                ]
            }

            preview_job, _ = build_preview_and_retarget_report(plan=plan, motion_library=motion_library, blender_path="/Applications/Blender")
            self.assertEqual(preview_job["steps"][0]["dataset"], "finedance")
            self.assertEqual(preview_job["steps"][0]["reference_artifacts"]["sequence_motion_fbx"], str(sequence_fbx))

    def test_build_mesh_preview_manifest_preserves_song_timing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            mesh_template = Path(tmpdir) / "template.fbx"
            mesh_template.write_text("dummy", encoding="utf-8")
            sequence_fbx = Path(tmpdir) / "sequence.fbx"
            sequence_fbx.write_text("dummy", encoding="utf-8")

            preview_job = {
                "plan_id": "demo_plan",
                "steps": [
                    {
                        "index": 0,
                        "dataset": "finedance",
                        "unit_id": "unit_a",
                        "source_sequence": "161",
                        "frame_range": {"start": 0, "end_exclusive": 60},
                        "reference_artifacts": {"sequence_motion_fbx": str(sequence_fbx)},
                    },
                    {
                        "index": 1,
                        "dataset": "finedance",
                        "unit_id": "unit_b",
                        "source_sequence": "161",
                        "frame_range": {"start": 60, "end_exclusive": 120},
                        "reference_artifacts": {"sequence_motion_fbx": str(sequence_fbx)},
                    },
                ],
            }
            plan = {
                "plan_id": "demo_plan",
                "steps": [
                    {"index": 0, "start_beat": 0, "target_beats": 4, "section_label": "intro", "speed_scale": 1.0, "switch_reason": {"target_energy": "low_energy", "transition_mode": "hold"}},
                    {"index": 1, "start_beat": 4, "target_beats": 4, "section_label": "verse", "speed_scale": 1.0, "switch_reason": {"target_energy": "mid_energy", "transition_mode": "flow"}},
                ],
            }
            song_event_map = {
                "beats": [{"index": i, "time_sec": float(i)} for i in range(9)],
            }

            manifest = build_mesh_preview_manifest(
                preview_job=preview_job,
                plan=plan,
                song_event_map=song_event_map,
                mesh_template_fbx=mesh_template,
                fps=24,
                max_steps=0,
            )
            self.assertEqual(manifest["scene"]["frame_start"], 1)
            self.assertEqual(manifest["scene"]["frame_end"], 192)
            self.assertEqual(len(manifest["sequences"]), 1)
            self.assertEqual(manifest["steps"][0]["scene_frame_start"], 1)
            self.assertEqual(manifest["steps"][1]["scene_frame_start"], 97)


if __name__ == "__main__":
    unittest.main()
