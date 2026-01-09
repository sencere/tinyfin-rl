# tinyfin-rl

<p align="center">
  <img src="resources/logo/logo.png" alt="tinyfin logo" />
</p>


Tinyfin-RL is a modular reinforcement learning library intended to be built around a tensor backend (Tinyfin or another library). The core goal is a C-first, embeddable architecture with clean interfaces, optional backends, and room for higher-performance training loops and visualization.

## Status

This repository currently provides foundational building blocks (envs, spaces, policies, agents, replay buffer, trainer) plus a minimal Tinyfin backend adapter and REINFORCE scaffolding. Python examples are temporary scaffolding and will be replaced by C modules and bindings.

## Goals

- Minimal, readable RL primitives that translate to C modules.
- Optional tensor backend integration without hard coupling.
- Training visualization via Raylib (C integration).

## Direction

This library is intended to be used as a project foundation rather than a small Python-only teaching repo. The Python files in `examples/` are placeholders that will be superseded by C implementations and project templates.

## Structure

- `tinyfin_rl/` core library
- `examples/` temporary scaffolding (will be replaced by C-based samples)
- `tinyfin/` Tinyfin submodule (tensor core + C backend source)
- `env/` standalone environment projects (Raylib, etc.)
- `raylib/` raylib-quickstart submodule for C visualization and env prototypes
- `raylib-src/` raylib source submodule for direct builds without premake
- `docs/` project notes and usage guides
- `roadmap.md` milestone plan
 - `tests/` backend parity checks (CPU/CUDA) when Tinyfin is available

## Docs

- `docs/overview.md` project overview
- `docs/install.md` setup and optional dependencies
- `docs/first_algorithm.md` REINFORCE sketch
- `docs/api_conventions.md` API expectations
- `docs/c_api.md` C API overview
- `env/first-env/README.md` raylib gridworld + training notes

## C backend and Tinyfin

Tinyfin lives in a separate repository and is included here as a submodule. The key requirement is a stable C-facing ABI. The `CBackend` loader is a placeholder for that integration.

For Python scaffolding, `TinyfinBackend` wires the Tinyfin Python bindings into a minimal policy/optimizer loop. Run examples from this repo with:

```bash
PYTHONPATH=/path/to/tinyfin/python python examples/reinforce_bandit.py
PYTHONPATH=/path/to/tinyfin/python python examples/ppo_bandit.py
PYTHONPATH=/path/to/tinyfin/python python examples/ppo_bandit_vector.py
PYTHONPATH=/path/to/tinyfin/python python examples/a2c_bandit.py
PYTHONPATH=/path/to/tinyfin/python python examples/benchmark_rollout.py
```

## Raylib visualization

Visualization is intended to be a C/Raylib integration. The current Python hooks only exist to keep the training loop modular until the C modules land.

## License

See `LICENSE`.
