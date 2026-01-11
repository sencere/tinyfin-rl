import ctypes
import os
import sys


class Color(ctypes.Structure):
    _fields_ = [
        ("r", ctypes.c_ubyte),
        ("g", ctypes.c_ubyte),
        ("b", ctypes.c_ubyte),
        ("a", ctypes.c_ubyte),
    ]


def _find_raylib_library() -> str:
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    candidates = []
    if sys.platform.startswith("linux"):
        names = ["libraylib.so"]
    elif sys.platform == "darwin":
        names = ["libraylib.dylib"]
    else:
        names = ["raylib.dll"]
    for name in names:
        candidates.extend([
            os.path.join(root, "raylib-src", "src", name),
            os.path.join(root, "raylib-src", "build", "raylib", name),
            name,
        ])
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("raylib shared library not found; build raylib-src/src")


class RaylibAPI:
    def __init__(self) -> None:
        lib_path = _find_raylib_library()
        self._lib = ctypes.CDLL(lib_path)

        self._init_window = self._lib.InitWindow
        self._init_window.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
        self._init_window.restype = None

        self.set_target_fps = self._lib.SetTargetFPS
        self.set_target_fps.argtypes = [ctypes.c_int]
        self.set_target_fps.restype = None

        self.begin_drawing = self._lib.BeginDrawing
        self.begin_drawing.argtypes = []
        self.begin_drawing.restype = None

        self.clear_background = self._lib.ClearBackground
        self.clear_background.argtypes = [Color]
        self.clear_background.restype = None

        self.draw_rectangle = self._lib.DrawRectangle
        self.draw_rectangle.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, Color]
        self.draw_rectangle.restype = None

        self.end_drawing = self._lib.EndDrawing
        self.end_drawing.argtypes = []
        self.end_drawing.restype = None

        self.close_window = self._lib.CloseWindow
        self.close_window.argtypes = []
        self.close_window.restype = None

        self._set_window_title = self._lib.SetWindowTitle
        self._set_window_title.argtypes = [ctypes.c_char_p]
        self._set_window_title.restype = None

        self.RAYWHITE = Color(245, 245, 245, 255)
        self.LIGHTGRAY = Color(200, 200, 200, 255)
        self.GREEN = Color(0, 228, 48, 255)
        self.BLUE = Color(0, 121, 241, 255)

    def _encode(self, value: str) -> bytes:
        return value.encode("utf-8")

    def init_window(self, width: int, height: int, title: str) -> None:
        self._init_window(width, height, self._encode(title))

    def set_window_title(self, title: str) -> None:
        self._set_window_title(self._encode(title))


rl = RaylibAPI()
