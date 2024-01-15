from pathlib import Path
import sys
import os
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
def get_gif(name_gif):
    gif_path= ROOT / 'gif'
    return str(Path('.') / gif_path / name_gif)


