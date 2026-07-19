# Tinyfin-RL v2 Roadmap

## Goal

Tinyfin-RL v2 should be a small, inspectable C-first RL library and CLI for
training, evaluation, replay, rendering, and production inference.

CPU must work by default. GPU support should come through Tinyfin. Raylib should
remain optional and isolated to viewer code.

## Current State

Implemented:

- Single `tinyfin-rl` CLI for `train`, `eval`, and `replay`.
- Public v2 headers under `include/tfrl/`.
- Compatibility shims for old internal header paths.
- Deterministic C environment API.
- Batch stepping and multi-agent helpers.
- Optional render snapshots and raylib viewer.
- Combined env shared library: `build/libtfrl_env.so`.
- Separated core env libraries:
  - `build/libtfrl_env_lineworld.so`
  - `build/libtfrl_env_maze_rooms.so`
  - `build/libtfrl_env_coin_maze.so`
- Dynamic env loading by shared library path.
- Dynamic env loading by manifest path.
- Core env manifests:
  - `envs/lineworld/env.toml`
  - `envs/maze_rooms/env.toml`
  - `envs/coin_maze/env.toml`
- v2 algorithm scope enforced in the factory: `dqn`, `ppo`, `nca`, and `random`.
- Removed placeholder algorithm names fail fast.
- Build/setup wrappers:
  - `scripts/build.sh`
  - `scripts/setup_env.sh`

In transition:

- `dqn` and `ppo` still use compatibility baseline implementations.
- `nca` is implemented as a first-pass trainable cellular policy, but not yet
  Tinyfin autograd-backed or integrated with PPO.
- Env source files still live under `src/envs/`, and automatic env discovery
  without central source edits is not implemented yet.
- Checkpoint manifests are defined as a public type but not fully persisted as
  v2 bundles.
- Production ABI and `libtfrl_runtime.so` are still planned.

## Milestone 1: Clean Skeleton

Status: mostly implemented.

- Public headers under `include/tfrl`.
- Runner, env, algo, metrics, checkpoint, and render snapshot APIs defined.
- Existing internal includes route through compatibility shims.
- Normal builds use the public include path.
- Docs describe the current v2 usage and limitations.

Remaining:

- Remove obsolete internal-only API assumptions as later milestones land.
- Decide whether `reinforce` remains internal only or returns as a documented
  baseline.

## Milestone 2: Env Modules

Status: partially implemented.

Target:

- Build each core env as its own library:
  - `libtfrl_env_lineworld.so`
  - `libtfrl_env_maze_rooms.so`
  - `libtfrl_env_coin_maze.so`
- Load envs by shared library path:

```bash
tinyfin-rl train --env ./build/libtfrl_env_lineworld.so
```

- Load envs by manifest path:

```bash
tinyfin-rl train --env envs/lineworld/env.toml
```

Remaining:

- Move env source into isolated `envs/<name>/` folders or provide thin module
  wrappers there.
- Add automatic manifest-backed discovery for new env names.
- Support:

```bash
tinyfin-rl train --env lineworld
tinyfin-rl train --env ./build/libtfrl_env_lineworld.so
```

Acceptance:

- New envs do not require central source edits for discovery.
- Each env has reset, step, terminal, invalid-action, render snapshot, and
  determinism tests.

## Milestone 3: Real DQN

Status: planned.

Target:

- Tinyfin-backed policy network.
- Replay buffer.
- Target network.
- Epsilon schedule.
- Save/load checkpoint bundle.
- Batch inference.
- Golden proof on `lineworld`.

## Milestone 4: Raylib Workflow

Status: partially implemented.

Implemented:

- Optional raylib viewer path.
- Trace replay.
- `scripts/build.sh --with-raylib`.

Remaining:

- Robust vendored/system raylib selection.
- Viewer smoke tests.
- Replay tests that validate snapshot rendering.

## Milestone 5: PPO

Status: planned.

Target:

- Rollout buffer.
- GAE.
- Clipped objective.
- Entropy bonus.
- Value head.
- Save/load checkpoint bundle.
- Golden proof on `maze_rooms`.

## Milestone 6: Tinyfin-NCA

Status: partially implemented.

Implemented:

- `--algo nca`.
- Public `tfrl/nca.h`.
- Deterministic 1D cellular rollout.
- Online TD-style training updates.
- Diagnostics and save/load.
- `lineworld` smoke test.

Remaining:

- Tinyfin tensor/autograd-backed optimization.
- Diversity metrics.
- PPO integration.

## Milestone 7: Production

Status: planned.

Target:

- `tinyfin-rl-prod` inference binary.
- `libtfrl_runtime.so` embeddable runtime.
- Stable checkpoint compatibility.
- Packaging script and deployment docs.

## Milestone 8: Optional Env Pack

Status: planned.

Target:

- Optional env modules such as `snake`, `breakout_tiny`, and `warehouse_bot`.
- Same deterministic API and tests as core envs.
- No direct renderer dependency.
