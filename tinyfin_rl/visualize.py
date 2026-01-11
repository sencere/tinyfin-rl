from typing import Optional


class Visualizer:
    def on_reset(self, env, observation):
        pass

    def on_step(self, env, observation, action, reward, done):
        pass

    def close(self):
        pass


class RaylibGridworldVisualizer(Visualizer):
    def __init__(self, cell_size: int = 64, fps: int = 30, title: str = "Tinyfin-RL Gridworld"):
        rl = None
        try:
            import pyray as rl  # type: ignore
        except ImportError:
            rl = None
        if rl is None or not hasattr(rl, "init_window"):
            try:
                from .raylib_ctypes import rl as ctypes_rl
            except Exception as exc:
                raise ImportError("raylib bindings not available; build raylib-src/src or install pyray") from exc
            rl = ctypes_rl
        self.rl = rl
        self.cell_size = cell_size
        self.fps = fps
        self.title = title
        self._initialized = False

    def _ensure_window(self, width: int, height: int) -> None:
        if self._initialized:
            return
        self.rl.init_window(width, height, self.title)
        self.rl.set_target_fps(self.fps)
        self._initialized = True

    def on_reset(self, env, observation):
        width = env.width * self.cell_size
        height = env.height * self.cell_size
        self._ensure_window(width, height)
        self._draw(env)

    def on_step(self, env, observation, action, reward, done):
        self._draw(env)
        if done:
            self.rl.set_window_title(f"{self.title} - done")

    def _draw(self, env) -> None:
        rl = self.rl
        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)
        for y in range(env.height):
            for x in range(env.width):
                rect_x = x * self.cell_size
                rect_y = y * self.cell_size
                color = rl.LIGHTGRAY
                if [x, y] == env.goal_pos:
                    color = rl.GREEN
                if [x, y] == env.agent_pos:
                    color = rl.BLUE
                rl.draw_rectangle(rect_x, rect_y, self.cell_size - 2, self.cell_size - 2, color)
        rl.end_drawing()

    def close(self):
        if self._initialized:
            self.rl.close_window()
            self._initialized = False
