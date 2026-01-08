# tinyfin-rl

Tinyfin-RL is a modular reinforcement learning library intended to be built around a tensor backend (Tinyfin or another library). The core goal is a C-first, embeddable architecture with clean interfaces, optional backends, and room for higher-performance training loops and visualization.

## Status

This repository currently provides foundational building blocks (envs, spaces, policies, agents, replay buffer, trainer) and placeholders for backend and visualization integration. Python examples are temporary scaffolding and will be replaced by C modules and bindings.

## Goals

- Minimal, readable RL primitives that translate to C modules.
- Optional tensor backend integration without hard coupling.
- Training visualization via Raylib (C integration).

## Direction

This library is intended to be used as a project foundation rather than a small Python-only teaching repo. The Python files in `examples/` are placeholders that will be superseded by C implementations and project templates.

## Structure

- `tinyfin_rl/` core library
- `examples/` temporary scaffolding (will be replaced by C-based samples)
- `csrc/tinyfin/` embedded minimal Tinyfin subset (C tensor core snapshot)
- `docs/` project notes and usage guides
- `roadmap.md` milestone plan

## Docs

- `docs/overview.md` project overview
- `docs/install.md` setup and optional dependencies
- `docs/first_algorithm.md` REINFORCE sketch
- `docs/api_conventions.md` API expectations

## C backend and Tinyfin

Tinyfin lives in a separate repository and may be partially copied or replaced. The key requirement here is a stable C-facing ABI. The `CBackend` loader is a placeholder for that integration.

## Raylib visualization

Visualization is intended to be a C/Raylib integration. The current Python hooks only exist to keep the training loop modular until the C modules land.

## License

See `LICENSE`.
