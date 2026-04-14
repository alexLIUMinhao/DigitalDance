from pathlib import Path
import sys
Path('/tmp/blender_path_probe.txt').write_text('\n'.join(sys.argv), encoding='utf-8')
