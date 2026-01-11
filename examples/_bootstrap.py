import os
import sys


def setup_path() -> None:
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
