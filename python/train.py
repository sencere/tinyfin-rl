import os
import subprocess
from dataclasses import dataclass


@dataclass
class TrainConfig:
    algo: str = "dqn"
    env: str = "maze_rooms"
    steps: int = 1000
    seed: int = 0
    gamma: float = 0.99
    lr: float = 0.05
    epsilon: float = 0.1
    clip_eps: float = 0.2
    steps_per_batch: int = 64
    epochs: int = 2
    render: str = "off"
    render_fps: int = 60
    trace_out: str | None = None
    save_path: str | None = None
    load_path: str | None = None


def _binary_path() -> str:
    return os.environ.get("TFRL_BIN", os.path.join("build", "tinyfin-rl"))


def train(cfg: TrainConfig) -> None:
    cmd = [
        _binary_path(),
        "train",
        "--algo",
        cfg.algo,
        "--env",
        cfg.env,
        "--steps",
        str(cfg.steps),
        "--seed",
        str(cfg.seed),
        "--gamma",
        str(cfg.gamma),
        "--lr",
        str(cfg.lr),
        "--epsilon",
        str(cfg.epsilon),
        "--clip-eps",
        str(cfg.clip_eps),
        "--steps-per-batch",
        str(cfg.steps_per_batch),
        "--epochs",
        str(cfg.epochs),
        "--render",
        cfg.render,
        "--render-fps",
        str(cfg.render_fps),
    ]
    if cfg.trace_out:
        cmd += ["--trace-out", cfg.trace_out]
    if cfg.save_path:
        cmd += ["--save", cfg.save_path]
    if cfg.load_path:
        cmd += ["--load", cfg.load_path]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    train(TrainConfig())
