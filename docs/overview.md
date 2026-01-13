# tinyfin-rl overview

Tinyfin-RL is a modular reinforcement learning toolkit aimed at C-first usage with a tensor backend (Tinyfin or an alternative). The API layers are intentionally small so they can map cleanly to C modules and Python bindings.

## Core API

- `Env`: minimal environment interface with `reset()` and `step()`.
- `Space`: action/observation space helpers (`Discrete`, `Box`).
- `Policy`: inference interface for action selection.
- `Agent`: glue for policy + replay buffer.
- `ReplayBuffer`: simple FIFO buffer for off-policy methods.
- `Trainer`: minimal training loop with optional visualization callbacks.

## Visualization

Raylib visualization is implemented in C. Python can call the C renderers via `ctypes`, but there are no Python-side raylib bindings.

## Env metadata

Environment plugins may expose `tfrl_env_metadata_get()` (see `tinyfin_rl/rl_env_info.h`) for action/observation space introspection in Python.

## C backend integration

The `CBackend` class in `tinyfin_rl/backend_c.py` is a placeholder for loading Tinyfin or another tensor backend via a shared library. The intent is to keep the core library backend-agnostic and modular.

Tinyfin is included as a submodule in `tinyfin/` for local builds and C backend integration.

Python bindings for C environments live in `tinyfin_rl/bindings`. See `docs/python_bindings.md` for loading env plugins and raylib renderers from Python.

## C trainers

The C trainer ABI (`tinyfin_rl/rl_train.h`) provides DQN and PPO implementations backed by Tinyfin.

For usage examples, see `docs/training.md`.

## Usability helpers

- Checkpoint helpers for PPO/DQN live in `tinyfin_rl/checkpoint.py`.
- Deterministic evaluation is available via `tinyfin_rl/eval.py`.
- Experiment metrics can be logged with `tinyfin_rl/experiment.py`.

## Next steps

Check `roadmap.md` for the planned milestones and upcoming algorithms.
