import unittest
from pathlib import Path

from music_motion_lab.config import load_app_config
from music_motion_lab.path_policy import ensure_output_path, resolve_input_path


class PathPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_app_config()

    def test_output_path_stays_inside_outputs(self) -> None:
        path = ensure_output_path(self.config, "song_event_maps/test.json")
        self.assertTrue(str(path).startswith(str(self.config.outputs_root)))

    def test_output_path_rejects_escape(self) -> None:
        with self.assertRaises(ValueError):
            ensure_output_path(self.config, "../music/illegal.json")

    def test_output_path_accepts_outputs_prefixed_argument(self) -> None:
        path = ensure_output_path(self.config, "outputs/song_event_maps/test_prefixed.json")
        self.assertEqual(
            path,
            Path("/Users/alex/Desktop/codex project/3d-digital/music-motion-lab/outputs/song_event_maps/test_prefixed.json"),
        )

    def test_shared_input_is_allowed(self) -> None:
        path = resolve_input_path(self.config, "../music/audio.mp3")
        self.assertEqual(path, Path("/Users/alex/Desktop/codex project/3d-digital/music/audio.mp3"))

    def test_unknown_input_root_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_input_path(self.config, "/tmp/not-allowed/outside.json")


if __name__ == "__main__":
    unittest.main()
