# tinyfin-rl

<p align="center">
  <img src="resources/logo/logo.png" alt="tinyfin logo" />
</p>


Tinyfin-RL is a modular reinforcement learning library intended to be built around a tensor backend (Tinyfin or another library). The core goal is a C-first, embeddable architecture with clean interfaces, optional backends, and raylib rendering inside C.

## Status

This repository provides foundational building blocks (policies, agents, replay buffer, trainer) plus a C env/plugin ABI and C render ABI. Python is a thin binding layer to load C environments and renderers.

## Goals

- Minimal, readable RL primitives that translate to C modules.
- Optional tensor backend integration without hard coupling.
- Training visualization via raylib in C renderers.

## Direction

This library is intended to be used as a project foundation rather than a small Python-only teaching repo. C environments are the source of truth; Python bindings are optional.

## Structure

- `tinyfin_rl/` core library + C headers
- `tinyfin/` Tinyfin submodule (tensor core + C backend source)
- `environments/` standalone environment projects (raylib, etc.)
- `examples/` C-first training examples
- `raylib/` raylib-quickstart submodule for C visualization and env prototypes
- `raylib-src/` raylib source submodule for direct builds without premake
- `docs/` project notes and usage guides
- `roadmap.md` milestone plan
 - `tests/` backend parity checks (CPU/CUDA) when Tinyfin is available

## Docs

- `docs/overview.md` project overview
- `docs/install.md` setup and optional dependencies
- `docs/python_bindings.md` Python bindings for C env plugins + renderers
- `docs/api_conventions.md` API expectations
- `docs/c_api.md` C API overview
- `docs/trpo.md` minimal TRPO placeholder
- `docs/ppo_walkthrough.md` PPO walkthrough
- `docs/dqn_walkthrough.md` DQN walkthrough
- `docs/sac_walkthrough.md` SAC walkthrough
- `docs/adapters.md` Gymnasium + PettingZoo adapters
- `environments/first-env/README.md` raylib gridworld + training notes
- `environments/lineworld/README.md` lineworld + Q-learning
- `environments/maze_rooms/README.md` four-room maze
- `environments/cliffwalk/README.md` cliff-walking grid
- `environments/cartpole_simple/README.md` cartpole with box observation
- `environments/grid_soccer/README.md` single-agent grid soccer
- `examples/README.md` C-first training examples

## Quickstart (C examples)

Build the C examples and run the deterministic trainer smoke test:

```bash
cd tinyfin-rl
make
./build/trainer_smoke
```

## C training (DQN/PPO)

```bash
cd tinyfin-rl
make train_dqn train_ppo
./build/train_dqn --env environments/lineworld/liblineworld.so --obs-n 7 --action-n 2
./build/train_ppo --env environments/first-env/libfirst_env.so --obs-n 100 --action-n 4
```

## Raylib env (train + autoplay)

Train the Q-learning policy and watch it in the raylib viewer:

```bash
cd tinyfin-rl/environments/first-env
make train_q
./train_q --episodes 2000 --report 200 --out q_table.csv
make
./first_env
```

Press `P` to toggle autoplay and `L` to reload `q_table.csv`.

## Headless env (lineworld)

Train a Q-learning policy in a simple 1D lineworld:

```bash
cd tinyfin-rl/environments/lineworld
make
./train_q --episodes 2000 --report 200 --out q_table.csv
```

## Python bindings (C envs)

Use Python to drive C environments via the plugin ABI and train via the C trainer ABI. See `docs/python_bindings.md` for full examples.

## Gymnasium examples (Python)

The `examples/python` folder includes CartPole DQN/PPO runs using Gymnasium for clarity.


## C backend and Tinyfin

Tinyfin lives in a separate repository and is included here as a submodule. The key requirement is a stable C-facing ABI. The `CBackend` loader is a placeholder for that integration.

Tinyfin provides the C backend used by the C trainers. Python can call the C trainers via bindings when Tinyfin is available.

## License

See `LICENSE`.
