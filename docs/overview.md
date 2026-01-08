# tinyfin-rl overview

Tinyfin-RL is a modular reinforcement learning toolkit aimed at C-first usage with a tensor backend (Tinyfin or an alternative). The API layers are intentionally small so they can map cleanly to C modules and bindings.

## Core API

- `Env`: minimal environment interface with `reset()` and `step()`.
- `Space`: action/observation space helpers (`Discrete`, `Box`).
- `Policy`: inference interface for action selection.
- `Agent`: glue for policy + replay buffer.
- `ReplayBuffer`: simple FIFO buffer for off-policy methods.
- `Trainer`: minimal training loop with optional visualization callbacks.

## Visualization

Raylib visualization is expected to be implemented in C. The current Python visualizer is a temporary placeholder so the training loop can stay modular while the C modules are built.

## C backend integration

The `CBackend` class in `tinyfin_rl/backend_c.py` is a placeholder for loading Tinyfin or another tensor backend via a shared library. The intent is to keep the core library backend-agnostic and modular.

For now, a minimal Tinyfin snapshot lives in `csrc/tinyfin/` so the tensor backend can be embedded directly in this repository.

## Next steps

Check `roadmap.md` for the planned milestones and upcoming algorithms.
