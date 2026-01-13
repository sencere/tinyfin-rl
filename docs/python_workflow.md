# Python Workflow

This repo keeps the **training loop in C**. Python is optional and used for:

- launching the C binary (`python/train.py`)
- calling the C env API (`python/env.py`)
- bridging external envs (Gymnasium/PettingZoo/Retro) via sockets

## Option A: C-only (no Python)

Run the C binary directly:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 2000
```

## Option B: Python wrapper (still C loop)

`python/train.py` shells out to the C binary:

```bash
python3 -m python.train
TFRL_BIN=build/tinyfin-rl python3 -m python.train
```

This does **not** run Gymnasium/PettingZoo/Retro unless you enable the bridge.

## Option C: Python env bridge (external envs)

Start the bridge server (Python owns the env, C owns training):

```bash
python3 python/bridge_server.py --socket /tmp/tfrl_py_bridge.sock
export TFRL_PY_BRIDGE=/tmp/tfrl_py_bridge.sock
./build/tinyfin-rl train --algo dqn --env py:gymnasium:CartPole-v1 --steps 2000
```

Examples:

```bash
./build/tinyfin-rl train --algo dqn --env py:pettingzoo:butterfly.pistonball_v6 --steps 2000 --share-policy
./build/tinyfin-rl train --algo dqn --env py:retro:Airstriker-Genesis --steps 2000
```

## Why the bridge is needed

- The C runner only knows the C env API.
- Gymnasium/PettingZoo/Retro live in Python.
- The bridge forwards `reset/step` calls over a UNIX socket.

This keeps the **C training loop** while letting you plug in external envs.
