# Tinyfin-RL Abstract Architecture
**C-first simulation, optional rendering, optional Python**

This document describes an abstract architecture for Tinyfin-RL that keeps
**fast C simulation as the source of truth**, makes **raylib rendering optional
and non-invasive**, and treats **Python as an optional host layer**, not a
dependency.

---

## Core Principle

**Simulation and rendering are separate concerns.**

- **Simulation (C)**  
  Deterministic, fast, headless by default, and parallelizable.

- **Rendering (raylib in C)**  
  A *viewer* that can attach to a running environment instance or replay a
  recorded trace.

- **Python**  
  Optional orchestration layer for training and analysis.  
  Never owns raylib, never opens windows.

This guarantees:
- Maximum simulation speed when rendering is disabled.
- Safe, debuggable visualization when rendering is enabled.
- No raylib lifecycle issues inside training loops or Python processes.

---

## Layer Overview

```

+-------------------+
|   Python Host     |  (optional)
|  (training, eval) |
+-------------------+
|
v
+-------------------+
|     C Core        |  <-- source of truth
|  Env ABI +        |
|  Env Metadata     |
+-------------------+
|
v
+-------------------+
| Renderer ABI      |  <-- renderer-agnostic
+-------------------+
|
v
+-------------------+
| Renderer Plugin   |  (raylib)
|   / Viewer        |
+-------------------+

```

---

## 1. C Core: Environment ABI (Source of Truth)

The Environment ABI is minimal and **contains no rendering concepts**.

Typical responsibilities:
- Environment lifecycle
- State transitions
- Rewards and termination
- Observation and action specs

### Key properties

- Stable
- Small
- Renderer-agnostic
- Suitable for high-performance training

Example responsibilities (current):
- `create / reset / step / destroy`
- Optional `tfrl_env_metadata_get()` for obs/action metadata (`rl_env_info.h`)
- Optional: `seed`, `clone`, `serialize`

## 1b. C Core: Trainer ABI (Source of Truth)

Algorithms live in C and are exposed via `rl_train.h` (DQN, PPO). Python bindings call these C trainers; Python does not re-implement the algorithms.

---

## 2. C Core: Renderer ABI (No raylib in headers)

Rendering is handled through a small, renderer-agnostic ABI (`rl_render.h`).

The environment provides *data*, not graphics.

### Goal

Allow any renderer to visualize the environment without the environment knowing
anything about raylib, windows, or GPUs.

### Supported patterns

#### A) Renderer ABI (current)

Renderers export a small set of functions:

- `tfrl_renderer_init(...)`
- `tfrl_renderer_draw(...)`
- `tfrl_renderer_should_close()`
- `tfrl_renderer_close()`

These renderers can use raylib internally while keeping Python and training code free of raylib bindings.

#### B) Render Data Snapshot (planned)

Future snapshot ABI for renderer-agnostic state dumps:

- Entity created / destroyed
- Agent moved
- Episode ended
- Custom debug signals

These snapshots can be recorded to disk and replayed later.

This enables:
- Offline visualization
- Deterministic debugging
- Frame-by-frame inspection

---

## 3. Renderer Plugins (C + raylib)

Renderers are **separate shared libraries or binaries**.

They:
- Own raylib initialization and shutdown
- Own the window and input loop
- Translate render data into graphics

They do **not**:
- Step the environment during training
- Affect simulation speed unless explicitly attached

### Renderer modes

- **Live attach**  
  Visualize a running environment instance.

- **Trace replay**  
  Load a trace file and render it offline.

---

## 4. Hosts

### 4.1 C Host (Always Available)

A native C executable acts as the backbone:

Examples:
- `tfrl-run --env cartpole --steps 1e6 --trace out.tft`
- `tfrl-view --trace out.tft`
- `tfrl-view --env cartpole --interactive`

Benefits:
- No Python required
- Ideal for debugging and CI
- Clean separation from training logic

---

### 4.2 Python Host (Optional)

Python is a *consumer* of the C core.

It can:
- Load environment plugins
- Run training algorithms
- Collect metrics
- Write trace files

It cannot:
- Call raylib
- Open windows
- Own rendering logic

This keeps Python safe, portable, and optional.

---

## Rendering Workflow (Pufferlib-style)

### Workflow 1: Attach Viewer to Live Env

- Training runs headless across many environments.
- One environment exposes render snapshots.
- Viewer periodically reads and displays that snapshot.

Advantages:
- Minimal overhead
- No renderer inside training workers
- Safe for multiprocessing

---

### Workflow 2: Record Trace, Replay Later

- Training records render events or snapshots for selected episodes.
- Viewer replays the trace offline.

Advantages:
- Deterministic visualization
- Debuggable frame-by-frame
- No impact on training speed

---

## Recommended Minimal ABI Set (v1)

### Environment ABI
- `env_create(config)`
- `env_reset(seed)`
- `env_step(action)`
- `env_destroy()`
- `env_get_spec()`

### Render Data ABI
- `env_render_bytes_needed(env)`
- `env_render_write(env, void* dst, size_t len)`

(Render packet is versioned and renderer-agnostic.)

### Renderer ABI
- `tfrl_renderer_init(...)`
- `tfrl_renderer_draw(...)` (accepts a `tfrl_value`)
- `tfrl_renderer_should_close()`
- `tfrl_renderer_close()`

---

## Why This Architecture Works

- Raylib is isolated and optional
- Simulation stays fast and deterministic
- Rendering never blocks training
- Python is not a dependency
- Offline replay enables serious debugging

This mirrors the architecture of professional simulators and game engines:
**core simulation first, viewer second, scripting optional**.

---

## Recommended v1 Implementation Plan

1. Freeze the Environment ABI.
2. Implement a versioned Render Snapshot format.
3. Build a standalone viewer (`tfrl-view`) using raylib.
4. Add Python bindings only for the Environment ABI.
5. Add trace recording and replay.

---

## Non-Goals

- Python-owned rendering
- Raylib bindings in Python
- Rendering inside training loops
- Python-only environments

---

**Design mantra:**  
> _Fast, headless simulation by default.  
> Rendering is a tool, not a requirement._
```
