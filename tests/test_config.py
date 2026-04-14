import tempfile
import unittest
from pathlib import Path

from music_motion_lab.config import load_app_config


class ConfigTests(unittest.TestCase):
    def test_loads_optional_blender_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True)
            (root / "config" / "paths.json").write_text(
                """
{
  "schemaVersion": 1,
  "sharedRoots": {
    "musicRoot": "../music",
    "motionBaseAssetsRoot": "../motion-base-assets",
    "unityReferenceRoot": "../3d-digital-human"
  },
  "tooling": {
    "blenderPath": "/Applications/Blender.app/Contents/MacOS/Blender"
  }
}
""".strip(),
                encoding="utf-8",
            )
            config = load_app_config(root)
            self.assertEqual(str(config.blender_path), "/Applications/Blender.app/Contents/MacOS/Blender")


if __name__ == "__main__":
    unittest.main()
