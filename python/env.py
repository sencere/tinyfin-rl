import ctypes
import os
from ctypes import c_int32, c_uint64, c_double, c_char_p, c_float
from dataclasses import dataclass

TFRL_MAX_BOX_DIMS = 32


class TfrlObs(ctypes.Structure):
    _fields_ = [
        ("index", c_int32),
        ("data_len", c_int32),
        ("data", c_float * TFRL_MAX_BOX_DIMS),
    ]


class TfrlAction(ctypes.Structure):
    _fields_ = [
        ("index", c_int32),
        ("data_len", c_int32),
        ("data", c_float * TFRL_MAX_BOX_DIMS),
    ]


class TfrlStepResult(ctypes.Structure):
    _fields_ = [("observation", TfrlObs), ("reward", c_double), ("done", c_int32)]


class TfrlEnvConfig(ctypes.Structure):
    _fields_ = [("name", c_char_p), ("seed", c_uint64)]


class TfrlEnvSpec(ctypes.Structure):
    _fields_ = [
        ("name", c_char_p),
        ("obs_n", c_int32),
        ("action_n", c_int32),
        ("max_steps", c_int32),
        ("width", c_int32),
        ("height", c_int32),
        ("obs_type", c_int32),
        ("action_type", c_int32),
        ("obs_dims", c_int32),
        ("action_dims", c_int32),
        ("obs_shape", c_int32 * 2),
        ("action_shape", c_int32 * 2),
        ("obs_dtype", c_int32),
        ("action_dtype", c_int32),
        ("obs_low", c_double),
        ("obs_high", c_double),
        ("action_low", c_double),
        ("action_high", c_double),
        ("agent_count", c_int32),
    ]


def _default_env_lib() -> str:
    override = os.environ.get("TFRL_ENV_LIB")
    if override:
        return override
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "build", "libtfrl_env.so")


@dataclass
class EnvSpec:
    name: str
    obs_n: int
    action_n: int
    max_steps: int
    width: int
    height: int
    obs_type: int
    action_type: int
    obs_dims: int
    action_dims: int
    obs_shape: tuple[int, int]
    action_shape: tuple[int, int]
    obs_dtype: int
    action_dtype: int
    obs_low: float
    obs_high: float
    action_low: float
    action_high: float
    agent_count: int


class EnvLib:
    def __init__(self, lib_path: str | None = None):
        path = lib_path or _default_env_lib()
        self.lib = ctypes.CDLL(path)
        self.lib.tfrl_env_create.argtypes = [ctypes.POINTER(TfrlEnvConfig)]
        self.lib.tfrl_env_create.restype = ctypes.c_void_p
        self.lib.tfrl_env_destroy.argtypes = [ctypes.c_void_p]
        self.lib.tfrl_env_destroy.restype = None
        self.lib.tfrl_env_reset.argtypes = [ctypes.c_void_p, c_uint64]
        self.lib.tfrl_env_reset.restype = TfrlObs
        self.lib.tfrl_env_step.argtypes = [ctypes.c_void_p, TfrlAction]
        self.lib.tfrl_env_step.restype = TfrlStepResult
        self.lib.tfrl_env_get_spec.argtypes = [ctypes.c_void_p]
        self.lib.tfrl_env_get_spec.restype = ctypes.POINTER(TfrlEnvSpec)

    def create(self, name: str = "maze_rooms", seed: int = 0) -> "EnvHandle":
        cfg = TfrlEnvConfig(name=name.encode(), seed=seed)
        handle = self.lib.tfrl_env_create(ctypes.byref(cfg))
        return EnvHandle(self, handle)


class EnvHandle:
    def __init__(self, lib: EnvLib, handle: int):
        self._lib = lib
        self._handle = handle

    def reset(self, seed: int = 0) -> int | list[float]:
        obs = self._lib.lib.tfrl_env_reset(self._handle, seed)
        if obs.data_len > 0:
            return [float(obs.data[i]) for i in range(obs.data_len)]
        return int(obs.index)

    def step(self, action: int | list[float] | tuple[float, ...]) -> tuple[int | list[float], float, bool]:
        act = TfrlAction()
        if isinstance(action, (list, tuple)):
            act.data_len = min(len(action), TFRL_MAX_BOX_DIMS)
            for i in range(act.data_len):
                act.data[i] = float(action[i])
        else:
            act.index = int(action)
        res = self._lib.lib.tfrl_env_step(self._handle, act)
        if res.observation.data_len > 0:
            obs_val = [float(res.observation.data[i]) for i in range(res.observation.data_len)]
        else:
            obs_val = int(res.observation.index)
        return obs_val, float(res.reward), bool(res.done)

    def spec(self) -> EnvSpec:
        ptr = self._lib.lib.tfrl_env_get_spec(self._handle)
        spec = ptr.contents
        return EnvSpec(
            name=spec.name.decode(),
            obs_n=spec.obs_n,
            action_n=spec.action_n,
            max_steps=spec.max_steps,
            width=spec.width,
            height=spec.height,
            obs_type=spec.obs_type,
            action_type=spec.action_type,
            obs_dims=spec.obs_dims,
            action_dims=spec.action_dims,
            obs_shape=(spec.obs_shape[0], spec.obs_shape[1]),
            action_shape=(spec.action_shape[0], spec.action_shape[1]),
            obs_dtype=spec.obs_dtype,
            action_dtype=spec.action_dtype,
            obs_low=spec.obs_low,
            obs_high=spec.obs_high,
            action_low=spec.action_low,
            action_high=spec.action_high,
            agent_count=spec.agent_count,
        )

    def close(self) -> None:
        if self._handle:
            self._lib.lib.tfrl_env_destroy(self._handle)
            self._handle = None
