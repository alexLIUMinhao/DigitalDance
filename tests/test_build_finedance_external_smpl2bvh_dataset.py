from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "build_finedance_external_smpl2bvh_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_finedance_external_smpl2bvh_dataset", TOOL_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class BuildFineDanceExternalSmpl2BvhDatasetTests(unittest.TestCase):
    def test_enumerate_motion_files_sorts_existing_npy_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "010.npy").write_bytes(b"0")
            (root / "002.npy").write_bytes(b"0")
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            result = MODULE.enumerate_motion_files(root)

            self.assertEqual([path.name for path in result], ["002.npy", "010.npy"])

    def test_build_manifest_payload_summarizes_status_counts(self) -> None:
        payload = MODULE.build_manifest_payload(
            input_dir=Path("/tmp/in"),
            output_dir=Path("/tmp/out"),
            bvh_dir=Path("/tmp/out/bvh"),
            sequence_records=[
                {"sequenceId": "001", "status": "completed"},
                {"sequenceId": "002", "status": "skipped_existing"},
                {"sequenceId": "003", "status": "failed"},
            ],
            fps=30,
            force=False,
        )

        self.assertEqual(payload["sequenceCount"], 3)
        self.assertEqual(payload["statusSummary"]["completed"], 1)
        self.assertEqual(payload["statusSummary"]["skippedExisting"], 1)
        self.assertEqual(payload["statusSummary"]["failed"], 1)
        self.assertEqual(payload["actualModelType"], "smpl")


if __name__ == "__main__":
    unittest.main()
