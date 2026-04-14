from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from music_motion_lab.willa_contract_diagnostic import (  # noqa: E402
    build_space_contract_report,
    parse_space_descriptor,
    vector_to_axis_label,
)


class WillaContractDiagnosticTests(unittest.TestCase):
    def test_parse_space_descriptor_extracts_forward_and_up(self) -> None:
        payload = parse_space_descriptor("Z-up; ground=XY; forward=-Y")
        self.assertEqual(payload["forwardLabel"], "-Y")
        self.assertEqual(payload["upLabel"], "+Z")

    def test_vector_to_axis_label_prefers_nearest_axis(self) -> None:
        summary = vector_to_axis_label([0.0, -0.98, -0.05])
        self.assertEqual(summary["label"], "-Y")
        self.assertLess(summary["angleDeg"], 5.0)

    def test_build_space_contract_report_flags_forward_sign_mismatch(self) -> None:
        canonical_report = {
            "firstFrameOrientation": {
                "after": {
                    "forward": [0.0, -1.0, 0.0],
                    "up": [0.0, 0.0, 1.0],
                }
            }
        }
        reference_alignment_profile = {
            "defaults": {
                "canonicalSourceSpace": "Z-up; ground=XY; forward=+Y",
                "unityTargetSpace": "Y-up; ground=XZ; forward=+Z",
                "globalAxisTransform": {
                    "mode": "swap_yz",
                    "matrix": [
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0],
                        [0.0, 1.0, 0.0],
                    ],
                },
            }
        }
        report = build_space_contract_report(
            canonical_report=canonical_report,
            reference_alignment_profile=reference_alignment_profile,
        )
        self.assertIn("canonical_forward_mismatch_with_reference_profile", report["issues"])
        self.assertIn("global_axis_transform_forward_mismatch", report["issues"])


if __name__ == "__main__":
    unittest.main()
