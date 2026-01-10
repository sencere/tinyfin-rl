import os
import sys

here = os.path.dirname(__file__)
root = os.path.normpath(os.path.join(here, ".."))
sys.path.insert(0, root)

from tinyfin_rl.emulation import EmulationSpec, flatten, infer_spec, unflatten


def test_flatten_unflatten_list():
    value = [1.0, 2.0, 3.0]
    flat, spec = flatten(value)
    restored, rest = unflatten(flat, spec)
    assert restored == value
    assert rest == []


def test_flatten_unflatten_nested():
    value = (1.0, [2.0, 3.0])
    spec = infer_spec(value)
    flat, _ = flatten(value, spec)
    restored, rest = unflatten(flat, spec)
    assert restored == value
    assert rest == []


def test_emulation_spec_roundtrip():
    obs = [0.0, 1.0]
    action = (2.0,)
    spec = EmulationSpec(obs_spec=infer_spec(obs), action_spec=infer_spec(action))
    obs_flat = spec.flatten_obs(obs)
    act_flat = spec.flatten_action(action)
    assert spec.unflatten_obs(obs_flat) == obs
    assert spec.unflatten_action(act_flat) == action
