import ctypes
import os
from ctypes import c_int32, c_float, c_char_p
from pathlib import Path

from .env_plugin import EnvPlugin, TfrlEnv


class DqnConfig(ctypes.Structure):
    _fields_ = [
        ("obs_n", c_int32),
        ("action_n", c_int32),
        ("steps", c_int32),
        ("eval_steps", c_int32),
        ("gamma", c_float),
        ("lr", c_float),
        ("epsilon", c_float),
        ("save_path", c_char_p),
        ("load_path", c_char_p),
    ]


class PpoConfig(ctypes.Structure):
    _fields_ = [
        ("obs_n", c_int32),
        ("action_n", c_int32),
        ("steps_per_batch", c_int32),
        ("updates", c_int32),
        ("epochs", c_int32),
        ("eval_steps", c_int32),
        ("gamma", c_float),
        ("lr", c_float),
        ("clip_eps", c_float),
        ("save_path", c_char_p),
        ("load_path", c_char_p),
    ]


class TrainMetrics(ctypes.Structure):
    _fields_ = [
        ("return_mean", c_float),
        ("loss", c_float),
    ]


def _default_train_lib() -> str:
    override = os.environ.get("TFRL_TRAIN_LIB")
    if override:
        return override
    root = Path(__file__).resolve().parents[2]
    return str(root / "build" / "libtfrl_train.so")


class TrainLib:
    def __init__(self, lib_path: str | None = None):
        path = lib_path or _default_train_lib()
        self.lib = ctypes.CDLL(path)
        self.lib.tfrl_train_dqn.argtypes = [ctypes.POINTER(TfrlEnv), ctypes.POINTER(DqnConfig), ctypes.POINTER(TrainMetrics)]
        self.lib.tfrl_train_dqn.restype = c_int32
        self.lib.tfrl_train_ppo.argtypes = [ctypes.POINTER(TfrlEnv), ctypes.POINTER(PpoConfig), ctypes.POINTER(TrainMetrics)]
        self.lib.tfrl_train_ppo.restype = c_int32

    def train_dqn(self, plugin: EnvPlugin, config: DqnConfig) -> tuple[int, TrainMetrics]:
        metrics = TrainMetrics()
        status = self.lib.tfrl_train_dqn(ctypes.byref(plugin._env), ctypes.byref(config), ctypes.byref(metrics))
        return int(status), metrics

    def train_ppo(self, plugin: EnvPlugin, config: PpoConfig) -> tuple[int, TrainMetrics]:
        metrics = TrainMetrics()
        status = self.lib.tfrl_train_ppo(ctypes.byref(plugin._env), ctypes.byref(config), ctypes.byref(metrics))
        return int(status), metrics
