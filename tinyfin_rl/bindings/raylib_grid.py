import ctypes
import os
from ctypes import c_int, c_double, c_char_p
from pathlib import Path

from .env_plugin import TfrlValue, TFRL_VALUE_I64


def default_grid_render_lib() -> str:
    override = os.environ.get("TFRL_GRID_RENDER_LIB")
    if override:
        return override
    root = Path(__file__).resolve().parents[2]
    render_lib = root / "environments" / "first-env" / "libfirst_env_render.so"
    return str(render_lib)


class GridRaylibRenderer:
    def __init__(self, lib_path: str | None = None, grid_size: int = 10, cell_size: int = 48, title: str = "tinyfin-rl grid"):
        self._lib_path = lib_path or default_grid_render_lib()
        self._lib = ctypes.CDLL(self._lib_path)
        self._lib.tfrl_renderer_init.argtypes = [c_int, c_int, c_char_p]
        self._lib.tfrl_renderer_init.restype = c_int
        self._lib.tfrl_renderer_draw.argtypes = [ctypes.POINTER(TfrlValue), c_double, c_int]
        self._lib.tfrl_renderer_draw.restype = None
        self._lib.tfrl_renderer_should_close.argtypes = []
        self._lib.tfrl_renderer_should_close.restype = c_int
        self._lib.tfrl_renderer_close.argtypes = []
        self._lib.tfrl_renderer_close.restype = None
        ok = self._lib.tfrl_renderer_init(int(grid_size), int(cell_size), title.encode("utf-8"))
        if ok != 1:
            raise RuntimeError("tfrl_grid_render_init failed")

    def draw(self, obs: int, reward: float, done: bool) -> None:
        val = TfrlValue()
        val.type = TFRL_VALUE_I64
        val.as_.i64 = int(obs)
        self._lib.tfrl_renderer_draw(ctypes.byref(val), float(reward), 1 if done else 0)

    def should_close(self) -> bool:
        return bool(self._lib.tfrl_renderer_should_close())

    def close(self) -> None:
        self._lib.tfrl_renderer_close()
