# Python Env Bridge

The Python env bridge lets the C runner drive Gymnasium, PettingZoo, or Gym-Retro
environments while keeping the training loop in C.

## Start the bridge

```bash
python3 python/bridge_server.py --socket /tmp/tfrl_py_bridge.sock
```

Set the socket path for the C runner:

```bash
export TFRL_PY_BRIDGE=/tmp/tfrl_py_bridge.sock
```

## Gymnasium

```bash
./build/tinyfin-rl train --algo dqn --env py:gymnasium:CartPole-v1 --steps 2000
```

## PettingZoo (parallel API)

Use the module path under `pettingzoo.*`:

```bash
./build/tinyfin-rl train --algo dqn --env py:pettingzoo:butterfly.pistonball_v6 --steps 2000 --share-policy
```

## Gym-Retro

```bash
./build/tinyfin-rl train --algo dqn --env py:retro:Airstriker-Genesis --steps 2000
```

## Notes

- Only Discrete and 1D Box spaces (dims <= 4) are supported.
- PettingZoo assumes the parallel API and homogeneous spaces.
- Rendering uses the Python env's own viewer, not Tinyfin-RL snapshots.
