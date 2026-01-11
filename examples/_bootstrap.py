import os
import sys


def setup_path() -> None:
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    tinyfin_py = os.path.join(root, "tinyfin", "python")
    if os.path.isdir(tinyfin_py) and tinyfin_py not in sys.path:
        sys.path.insert(0, tinyfin_py)
