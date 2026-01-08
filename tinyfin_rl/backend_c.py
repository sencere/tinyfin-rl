import ctypes
import os
from typing import Optional


class CBackend:
    def __init__(self, lib_path: Optional[str] = None):
        self.lib_path = lib_path or os.environ.get("TINYFIN_LIB")
        self.lib: Optional[ctypes.CDLL] = None

    def load(self, lib_path: Optional[str] = None) -> ctypes.CDLL:
        path = lib_path or self.lib_path
        if not path:
            raise ValueError("lib_path is required (or set TINYFIN_LIB)")
        self.lib = ctypes.CDLL(path)
        return self.lib

    def is_loaded(self) -> bool:
        return self.lib is not None

    def get_symbol(self, name: str):
        if self.lib is None:
            raise RuntimeError("C backend not loaded")
        return getattr(self.lib, name)
