import ctypes
import os
from ctypes import c_int, c_uint32, c_uint64, c_int64, c_double, c_size_t, c_void_p, c_char_p
from pathlib import Path


TFRL_VALUE_EMPTY = 0
TFRL_VALUE_I64 = 1
TFRL_VALUE_F64 = 2
TFRL_VALUE_BUFFER = 3

TFRL_SPACE_UNKNOWN = 0
TFRL_SPACE_DISCRETE = 1
TFRL_SPACE_BOX = 2


class TfrlBuffer(ctypes.Structure):
    _fields_ = [
        ("data", c_void_p),
        ("len", c_size_t),
        ("elem_size", c_size_t),
    ]


class TfrlValueUnion(ctypes.Union):
    _fields_ = [
        ("i64", c_int64),
        ("f64", c_double),
        ("buffer", TfrlBuffer),
    ]


class TfrlValue(ctypes.Structure):
    _fields_ = [
        ("type", c_int),
        ("as_", TfrlValueUnion),
    ]


class TfrlStep(ctypes.Structure):
    _fields_ = [
        ("observation", TfrlValue),
        ("reward", c_double),
        ("terminated", c_int),
        ("truncated", c_int),
    ]


RESET_FN = ctypes.CFUNCTYPE(c_int, c_void_p, ctypes.POINTER(TfrlValue))
STEP_FN = ctypes.CFUNCTYPE(c_int, c_void_p, ctypes.POINTER(TfrlValue), ctypes.POINTER(TfrlStep))
SEED_FN = ctypes.CFUNCTYPE(c_int, c_void_p, c_uint64)
DESTROY_FN = ctypes.CFUNCTYPE(None, c_void_p)


class TfrlEnvVTable(ctypes.Structure):
    _fields_ = [
        ("reset", RESET_FN),
        ("step", STEP_FN),
        ("seed", SEED_FN),
        ("destroy", DESTROY_FN),
    ]


class TfrlEnv(ctypes.Structure):
    _fields_ = [
        ("ctx", c_void_p),
        ("vtable", ctypes.POINTER(TfrlEnvVTable)),
    ]


CREATE_FN = ctypes.CFUNCTYPE(TfrlEnv)
PLUGIN_DESTROY_FN = ctypes.CFUNCTYPE(None, ctypes.POINTER(TfrlEnv))


class TfrlEnvPlugin(ctypes.Structure):
    _fields_ = [
        ("api_version", c_uint32),
        ("name", c_char_p),
        ("create", CREATE_FN),
        ("destroy", PLUGIN_DESTROY_FN),
    ]


class TfrlEnvMetadata(ctypes.Structure):
    _fields_ = [
        ("api_version", c_uint32),
        ("name", c_char_p),
        ("observation_type", c_int),
        ("action_type", c_int),
        ("observation_n", c_int64),
        ("action_n", c_int64),
        ("observation_dims", c_int64),
        ("action_dims", c_int64),
        ("observation_low", c_double),
        ("observation_high", c_double),
        ("action_low", c_double),
        ("action_high", c_double),
    ]


def _load_plugin(lib_path: str) -> TfrlEnvPlugin:
    lib = ctypes.CDLL(lib_path)
    getter = lib.tfrl_env_plugin_get
    getter.restype = ctypes.POINTER(TfrlEnvPlugin)
    plugin_ptr = getter()
    if not plugin_ptr:
        raise RuntimeError("tfrl_env_plugin_get returned NULL")
    plugin = plugin_ptr.contents
    plugin._lib = lib  # Keep a reference to avoid GC.
    return plugin


def _value_from_py(value) -> TfrlValue:
    out = TfrlValue()
    if isinstance(value, float):
        out.type = TFRL_VALUE_F64
        out.as_.f64 = float(value)
    else:
        out.type = TFRL_VALUE_I64
        out.as_.i64 = int(value)
    return out


def _value_to_py(value: TfrlValue):
    if value.type == TFRL_VALUE_I64:
        return int(value.as_.i64)
    if value.type == TFRL_VALUE_F64:
        return float(value.as_.f64)
    if value.type == TFRL_VALUE_BUFFER:
        buf = value.as_.buffer
        if not buf.data or buf.len == 0:
            return b""
        return ctypes.string_at(buf.data, buf.len * buf.elem_size)
    return None


def default_grid_plugin_lib() -> str:
    override = os.environ.get("TFRL_ENV_PLUGIN_LIB")
    if override:
        return override
    root = Path(__file__).resolve().parents[2]
    env_lib = root / "environments" / "first-env" / "libfirst_env.so"
    return str(env_lib)


class EnvPlugin:
    def __init__(self, lib_path: str | None = None):
        self._lib_path = lib_path or default_grid_plugin_lib()
        self._plugin = _load_plugin(self._lib_path)
        self._env = self._plugin.create()
        self._metadata = self._load_metadata()
        self._closed = False

    @property
    def name(self) -> str:
        return self._plugin.name.decode("utf-8") if self._plugin.name else "unknown"

    def metadata(self) -> dict:
        if self._metadata is None:
            return {}
        meta = self._metadata
        name = meta.name.decode("utf-8") if meta.name else self.name
        return {
            "name": name,
            "observation_type": int(meta.observation_type),
            "action_type": int(meta.action_type),
            "observation_n": int(meta.observation_n),
            "action_n": int(meta.action_n),
            "observation_dims": int(meta.observation_dims),
            "action_dims": int(meta.action_dims),
            "observation_low": float(meta.observation_low),
            "observation_high": float(meta.observation_high),
            "action_low": float(meta.action_low),
            "action_high": float(meta.action_high),
        }

    def reset(self):
        obs = TfrlValue()
        status = self._env.vtable.contents.reset(self._env.ctx, ctypes.byref(obs))
        if status != 0:
            raise RuntimeError(f"env.reset failed with status {status}")
        return _value_to_py(obs)

    def step(self, action):
        act = _value_from_py(action)
        step = TfrlStep()
        status = self._env.vtable.contents.step(self._env.ctx, ctypes.byref(act), ctypes.byref(step))
        if status != 0:
            raise RuntimeError(f"env.step failed with status {status}")
        done = bool(step.terminated or step.truncated)
        return _value_to_py(step.observation), float(step.reward), done, {}

    def seed(self, seed: int) -> int:
        seed_fn = self._env.vtable.contents.seed
        if not seed_fn:
            return -1
        status = seed_fn(self._env.ctx, c_uint64(seed))
        return int(status)

    def close(self) -> None:
        if self._closed:
            return
        if self._plugin.destroy:
            self._plugin.destroy(ctypes.byref(self._env))
        elif self._env.vtable and self._env.vtable.contents.destroy:
            self._env.vtable.contents.destroy(self._env.ctx)
        self._closed = True

    def _load_metadata(self) -> TfrlEnvMetadata | None:
        try:
            getter = self._plugin._lib.tfrl_env_metadata_get
        except AttributeError:
            return None
        getter.restype = ctypes.POINTER(TfrlEnvMetadata)
        meta_ptr = getter()
        if not meta_ptr:
            return None
        return meta_ptr.contents

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
