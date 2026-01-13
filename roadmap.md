# tinyfin-rl Roadmap

**Goal:** A modular, C-first RL library with a pluggable tensor backend (Tinyfin or alternatives), fast simulation, and solid algorithm coverage. Python is a thin binding layer, not the source of truth.

## Immediate Focus (1 sprint)
- Lock C API & ABI (`Env`, plugin ABI, renderer ABI, metadata ABI).
- Stabilize the raylib build pipeline across environments.
- Provide a C trainer + renderer sample per env.
- Finalize Python bindings for C envs (introspection + adapter).

## Core Milestones

### 0 — Foundations (In Progress)
- C-first project skeleton, build, logging, deterministic seeding.
- Core C APIs: envs, agents, policies, buffers, trainers.
- Tensor backend ABI (Tinyfin-compatible).
- Env plugin ABI (`rl_plugin.h`) and renderer ABI (`rl_render.h`).
- **Result:** C env plugins build and render via raylib.

### 1 — Python Bindings (In Progress)
- `ctypes` bindings for env plugins and renderers.
- Env metadata introspection (`rl_env_info.h`).
- **Result:** Python can load C envs and render via C-only raylib.

### 2 — C Trainers (Planned)
- Minimal C trainers for on-policy and off-policy loops.
- Reference trainer targets each env plugin.
- **Result:** End-to-end C training without Python.

### 3 — Algorithms (Planned)
- Move core algorithm implementations to C or provide clear C bindings.
- Keep Python trainers as optional wrappers.
- **Result:** C-first training is the default path.

## Product & Performance

### 4 — Environments & Simulation (In Progress)
- Expand in-repo C environments.
- Pair each env with a renderer + viewer app.
- **Result:** Env catalog with C viewer + plugin build.

### 5 — Performance & Vectorization (Planned)
- Vectorized rollout core in C.
- Benchmarks for env + inference throughput.
- **Result:** Steps/sec targets met and benchmark-gated.

### 6 — Usability & Reliability (Planned)
- Reproducible training via C trainers.
- Core docs and walkthroughs updated to C-first flow.
- **Result:** C-first workflows documented end-to-end.

## Current State (Summary)
- C env plugins and C renderers are the primary path.
- C trainer ABI exists with DQN/PPO backed by Tinyfin.
- Python can load C envs and call C trainers via bindings.
- C viewers exist for each in-repo environment.

## Next Steps
- Add C trainer loop examples and migrate algorithm walkthroughs.
- Expand env metadata and space support beyond discrete.
- Stabilize raylib toolchain on all supported platforms.
