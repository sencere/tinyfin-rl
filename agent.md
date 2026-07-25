# Tinyfin-RL v2 Agent Plan

## Implementation Checklist

Legend:

- `[x]` implemented in the current working tree
- `[~]` partially implemented
- `[ ]` not implemented yet

### Milestone 1: Clean Skeleton

- `[x]` Add public v2 headers under `include/tfrl/`.
- `[x]` Add `include/tfrl/env.h`.
- `[x]` Add `include/tfrl/algo.h`.
- `[x]` Add `include/tfrl/runner.h`.
- `[x]` Add `include/tfrl/metrics.h`.
- `[x]` Add `include/tfrl/checkpoint.h`.
- `[x]` Add `include/tfrl/render_snapshot.h`.
- `[x]` Add umbrella header `include/tfrl/tfrl.h`.
- `[x]` Keep old internal header paths as compatibility shims.
- `[x]` Update Make/CMake include paths for the public header tree.
- `[x]` Add a public-header compile smoke test.
- `[~]` Split runner, env API, algorithm API, checkpoint API, and render snapshot API.
  Public API boundaries exist; source files are not fully moved into the target
  v2 folder layout yet.

### Milestone 2: Env Modules

- `[x]` Add env manifests for core envs:
  - `envs/lineworld/env.toml`
  - `envs/maze_rooms/env.toml`
  - `envs/coin_maze/env.toml`
- `[~]` Keep only core envs in docs and v2 guidance.
  The code still contains extra envs while the split is in progress.
- `[x]` Move each environment implementation into its own top-level `envs/<name>/`
  folder.
- `[x]` Build `lineworld` as `libtfrl_env_lineworld.so`.
- `[x]` Build `maze_rooms` as `libtfrl_env_maze_rooms.so`.
- `[x]` Build `coin_maze` as `libtfrl_env_coin_maze.so`.
- `[x]` Add dynamic env loading by shared library path.
- `[~]` Add manifest-backed env discovery that avoids central source edits.
  Loading by manifest path and `envs/<name>/env.toml` name discovery works.
  Adding a new C env module still needs build-system and module-registry wiring.
- `[x]` Add per-env reset/step/terminal/invalid-action tests.
  Core environment tests now cover known-path reachability and terminal reward
  for `maze_rooms` and `lineworld`, plus invalid-action behavior for `coin_maze`.
- `[x]` Add per-env replayable trace tests.
  Deterministic render traces are recorded and compared for repeated seeded
  `lineworld` runs.

### Milestone 3: DQN

- `[x]` Expose `--algo dqn`.
- `[x]` Provide a trainable discrete-action baseline behind the DQN interface.
- `[~]` Add replay storage, a synchronized target table, and epsilon decay to
  the tabular DQN baseline. Neural Tinyfin-backed DQN remains planned.
- `[~]` Implement Tinyfin-backed DQN policy network.
  A Tinyfin `Linear` feature network and Q head now provide neural inference
  and a trainable head; full autograd minibatch training remains planned.
- `[~]` Implement target network (tabular and neural target heads now exist;
  full synchronized minibatch training remains planned).
- `[~]` Implement full replay-buffer integration for DQN updates.
  Neural DQN now stores transitions and trains its Q head from replay entries;
  prioritized/random minibatch sampling remains planned.
- `[~]` Implement epsilon schedule (linear/advanced schedules remain planned).
- `[x]` Implement DQN checkpoint save/load bundle.
  Both tabular and neural DQN paths save/load their model values, target
  values, counters, RNG state, and hyperparameters.
- `[~]` Support batch inference through the algorithm API.
- `[ ]` Prove DQN learning on `lineworld` with a golden training test.

### Milestone 4: PPO

- `[x]` Expose `--algo ppo`.
- `[~]` Provide an existing reinforce-style trainable baseline behind the PPO name.
- `[ ]` Implement true PPO rollout buffer.
- `[~]` Implement GAE.
  The PPO baseline now computes normalized GAE(lambda)-style advantages over
  each collected trajectory; value bootstrapping and true rollout batching
  remain planned.
- `[ ]` Implement clipped PPO objective.
- `[ ]` Implement entropy bonus metrics.
- `[ ]` Implement value loss metrics.
- `[ ]` Implement PPO checkpoint save/load bundle.
- `[ ]` Prove PPO learning on `maze_rooms`.

### Milestone 5: Tinyfin-NCA

- `[x]` Expose `--algo nca`.
- `[x]` Add public NCA policy interface in `include/tfrl/nca.h`.
- `[x]` Implement deterministic NCA rollout.
- `[x]` Implement trainable NCA parameters.
- `[~]` Integrate NCA training with Tinyfin.
  Current implementation uses the Tinyfin-RL algorithm API and a C TD-style
  update. Full Tinyfin tensor/autograd-backed NCA optimization is still needed.
- `[~]` Add NCA diagnostics and diversity metrics.
  Diagnostics exist; diversity metrics are not implemented yet.
- `[x]` Add NCA checkpoint save/load.
- `[x]` Add an NCA smoke training test on `lineworld`.
- `[~]` Integrate NCA with PPO.
  `nca-ppo` now stores the action-time log probability, computes a clipped
  PPO-style probability ratio during updates, and applies entropy pressure to
  the NCA readout. It now maintains four in-memory candidate policies, scores
  them by episode return, mutates replacements from the current elite, and
  saves only the best candidate. Full trajectory-level PPO optimization and
  configurable population size remain planned.

### Milestone 6: Rendering Workflow

- `[x]` Keep rendering snapshot-based.
- `[x]` Keep raylib isolated to viewer code.
- `[x]` Support `--render off|live`.
- `[x]` Support trace replay.
- `[x]` Add `scripts/build.sh --with-raylib`.
- `[x]` Add `scripts/setup_env.sh`.
- `[x]` Add robust system-raylib detection with automatic vendored fallback.
- `[ ]` Add viewer smoke tests.
- `[x]` Add render trace replay tests.
  Snapshot trace frames are replayed and compared deterministically in the
  core trace regression test.

### Milestone 7: Production

- `[~]` Keep `build/tinyfin-prod` inference-oriented binary.
- `[ ]` Add `tinyfin-rl-prod` target name.
- `[ ]` Add `libtfrl_runtime.so`.
- `[ ]` Define stable production inference ABI.
- `[ ]` Implement versioned checkpoint compatibility.
- `[ ]` Add production packaging script.
- `[ ]` Add client embedding docs.

### Documentation And Tooling

- `[x]` Add current v2 usage guide in `docs/usage.md`.
- `[x]` Update README for v2 build and algorithm scope.
- `[x]` Update architecture documentation for current v2 state.
- `[x]` Update roadmap to distinguish implemented, partial, and planned work.
- `[x]` Update CLI/training/perf/replay docs to avoid stale removed-algo commands.
- `[x]` Update smoke scripts away from removed placeholder algorithms.
- `[x]` Make removed v1 placeholder algorithms fail fast instead of silently
  falling back.
- `[ ]` Commit the v2 work after separating it from pre-existing unrelated
  working-tree changes.

## Vision

Tinyfin-RL v2 should become a small, professional reinforcement learning
library for C-first training, evaluation, rendering, and production inference.
The library should be easy to download, compile, and run out of the box on CPU,
with optional GPU acceleration and optional raylib visualization.

The new version should focus on correctness, clean boundaries, reliable
training workflows, and production deployment. It should not try to include
many half-finished algorithms. A smaller set of strong, tested components is
the goal.

## Product Goals

- Train agents from the CLI with clear real-time metrics.
- Evaluate trained agents deterministically.
- Render training, evaluation, and replay with raylib.
- Compile each environment as an isolated module.
- Use Tinyfin as the primary compute, tensor, autograd, optimizer, and backend
  library.
- Support CPU by default and GPU when Tinyfin is built with GPU support.
- Save and load trained policies reliably.
- Provide production binaries and embeddable libraries for clients.
- Keep the codebase simple enough to audit, extend, and maintain.

## Non-Goals

- Do not keep a long list of algorithm names that all fall back to a placeholder.
- Do not make environments depend on raylib, UI code, or the training runner.
- Do not require Python for normal training or production inference.
- Do not make users manually discover library paths or linker variables.
- Do not treat demos as proof of training correctness.
- Do not introduce heavyweight ML dependencies beside Tinyfin.

## Core Architecture

The project should be split into independent layers:

```text
CLI
  -> runner
    -> algorithm
    -> environment module
    -> checkpoint writer
    -> metric stream
    -> optional trace writer
    -> optional viewer

production runner
  -> checkpoint loader
  -> algorithm inference
  -> environment or client integration
```

Each layer should have a narrow public API. The runner should coordinate work,
but it should not own environment internals, renderer internals, or algorithm
state formats.

## Repository Layout

Target layout:

```text
tinyfin-rl/
  CMakeLists.txt
  Makefile
  README.md
  agent.md
  docs/
    architecture.md
    build.md
    envs.md
    algorithms.md
    checkpoints.md
    rendering.md
    production.md
  include/tfrl/
    env.h
    algo.h
    runner.h
    metrics.h
    checkpoint.h
    render_snapshot.h
  src/
    cli/
    runner/
    algo/
      dqn/
      ppo/
      nca/
    core/
    render/
      raylib_viewer.c
      trace.c
    production/
  envs/
    lineworld/
      env.c
      render.c
      CMakeLists.txt
      env.toml
    maze_rooms/
      env.c
      render.c
      CMakeLists.txt
      env.toml
    coin_maze/
      env.c
      render.c
      CMakeLists.txt
      env.toml
  scripts/
    setup_env.sh
    build.sh
    build_raylib.sh
    train.sh
    eval.sh
    replay.sh
    package_prod.sh
  third_party/
    raylib/
    tinyfin/
  examples/
```

## Environment Isolation

Every environment should live in its own folder and compile as its own library.
This is the most important structural change.

Each environment module should expose the same stable ABI:

```c
const tfrl_env_desc *tfrl_env_desc(void);
tfrl_env *tfrl_env_create(const tfrl_env_config *cfg);
void tfrl_env_destroy(tfrl_env *env);
tfrl_obs tfrl_env_reset(tfrl_env *env, uint64_t seed);
tfrl_step_result tfrl_env_step(tfrl_env *env, tfrl_action action);
size_t tfrl_env_render_bytes_needed(const tfrl_env *env);
size_t tfrl_env_render_write(tfrl_env *env, void *dst, size_t dst_len);
```

Each env folder should build one artifact:

```text
build/libtfrl_env_maze_rooms.so
build/libtfrl_env_lineworld.so
build/libtfrl_env_coin_maze.so
```

The runner should load environments either statically or dynamically:

```bash
tinyfin-rl train --env maze_rooms
tinyfin-rl train --env ./build/libtfrl_env_maze_rooms.so
```

The registry should not require editing central source files for every new env.
New envs should be discoverable from an env manifest:

```toml
name = "maze_rooms"
kind = "grid"
library = "libtfrl_env_maze_rooms.so"
obs_type = "discrete"
action_type = "discrete"
```

## Environment Quality Contract

Every environment must provide:

- Deterministic reset with seed.
- Deterministic stepping.
- Clear observation and action specs.
- A render snapshot.
- Unit tests for reset, step, terminal state, and invalid actions.
- A smoke training config.
- A replayable trace test.

The environment must not:

- Include raylib.
- Allocate per step unless explicitly documented.
- Use global mutable state.
- Read files during stepping.
- Depend on the training algorithm.

## Environment Roadmap

Tinyfin-RL should have a small core set of serious benchmark environments and a
separate set of exciting environments that make the library fun to use and easy
to demonstrate.

Core environments:

- `lineworld`: smallest deterministic training sanity check.
- `maze_rooms`: canonical grid-world environment.
- `coin_maze`: sparse-reward navigation and collection.

Exciting environments:

- `snake`: discrete control, survival, food collection, easy to render.
- `floppy`: continuous-feeling arcade timing with simple observations.
- `breakout_tiny`: compact Breakout-style paddle and brick environment.
- `tetris_tiny`: planning-heavy discrete environment.
- `warehouse_bot`: grid logistics with pickup/dropoff tasks.
- `traffic_grid`: multi-agent coordination at intersections.
- `resource_gather`: small RTS-like gathering and base-delivery task.
- `arena_duel`: two-agent competitive grid arena.
- `market_maker_tiny`: toy financial inventory-control environment.

Each exciting env must still obey the same production rules:

- Isolated folder.
- Independent build target.
- Deterministic seed behavior.
- Render snapshot.
- Unit tests.
- Small smoke training config.
- No direct renderer dependency.

Exciting envs should not block the core library. They should be optional modules
that prove the architecture is flexible.

## Algorithms

Tinyfin-RL v2 should ship only three algorithm families initially:

### DQN

For discrete action environments.

Required features:

- Replay buffer.
- Target network.
- Epsilon schedule.
- Checkpoint save/load.
- CPU backend through Tinyfin.
- GPU backend through Tinyfin when available.
- Batch inference.

### PPO

For discrete and box observations with discrete or continuous actions.

Required features:

- Rollout buffer.
- GAE.
- Clipped objective.
- Entropy bonus.
- Value function.
- Checkpoint save/load.
- CPU and GPU through Tinyfin.

### Tinyfin-NCA

Neural cellular automata based agent diversity module.

Purpose:

- Train diverse policies or policy populations.
- Support self-play or population sampling later.
- Provide agent diversity without adding many separate RL algorithms.

Initial scope:

- Define the NCA policy interface.
- Implement deterministic NCA rollout.
- Train NCA parameters with Tinyfin.
- Integrate with PPO first, then DQN if useful.

## Tinyfin-First Rule

Tinyfin-RL should primarily use Tinyfin for numerical work.

Tinyfin should own:

- Tensors.
- Autograd.
- Neural network layers.
- Optimizers.
- Model serialization.
- CPU backend execution.
- GPU backend execution.

Tinyfin-RL should own:

- Environment APIs.
- Runner control flow.
- Rollout and replay storage.
- RL losses and algorithm orchestration.
- Metrics.
- Checkpoint manifests.
- Render snapshots.
- CLI and production entrypoints.

Acceptable external dependencies:

- C standard library and POSIX APIs for runtime basics.
- pthreads for CPU parallelism.
- raylib for optional visualization only.
- CMake/pkg-config for build discovery.

Avoid:

- PyTorch, TensorFlow, JAX, ONNX Runtime, or similar training dependencies in
  the core library.
- CUDA calls outside Tinyfin backend code.
- Renderer dependencies inside environments.
- Python as a required runtime.

## Removed Algorithm Scope

The following should not exist in v2 until they are real and tested:

- Rainbow
- QR-DQN
- IQN
- IQL
- A2C
- A3C
- TRPO
- SAC
- TD3
- IMPALA
- MCTS

They can return later as plugins or experimental modules, not as advertised core
features.

## Training Screen

The current `fish_once`-style dashboard is the desired direction. It should show
real metrics only.

Required metrics:

- Step progress.
- Environment steps.
- Samples per second.
- Updates per second.
- Episodes completed.
- Last return.
- Mean return over recent N episodes.
- Mean episode length.
- Loss values.
- Policy entropy.
- Value loss.
- Q loss for DQN.
- Exploration rate.
- Learning rate.
- Replay buffer size.
- Device/backend.
- Env step time.
- Inference time.
- Update time.
- Render time.

The dashboard should be optional:

```bash
tinyfin-rl train --dashboard live
tinyfin-rl train --dashboard off
tinyfin-rl train --metrics-json runs/metrics.jsonl
```

Training logs should be machine-readable when dashboard is disabled:

```text
step=1000 episodes=42 mean_return=0.71 sps=15820 loss=0.032
```

## Checkpoints

Checkpointing is mandatory for v2.

Every algorithm checkpoint should include:

- Algorithm name and version.
- Environment name and spec hash.
- Tinyfin model tensors.
- Optimizer state.
- Normalization state.
- RNG state.
- Training step.
- Metric summary.

Recommended format:

```text
runs/maze_dqn/
  manifest.json
  policy.tftensor
  target_policy.tftensor
  optimizer.json
  replay.meta.json
```

Commands:

```bash
tinyfin-rl train --algo dqn --env maze_rooms --steps 100000 --save runs/maze_dqn
tinyfin-rl eval --algo dqn --env maze_rooms --load runs/maze_dqn --episodes 20
tinyfin-rl eval --algo dqn --env maze_rooms --load runs/maze_dqn --render live
```

## Rendering

Rendering should stay snapshot-based:

- Environments write render snapshots.
- Raylib consumes snapshots.
- Training never depends on raylib.
- Replay uses trace snapshots.

Raylib should be integrated in a way that is easy but optional.

Preferred approach:

- Use system raylib if available.
- Otherwise build vendored raylib automatically.
- Hide library path setup behind scripts.
- Provide `tinyfin-rl-raylib` as a separate binary if needed.

Commands should be simple:

```bash
./scripts/build.sh
./scripts/build.sh --with-raylib
./scripts/setup_env.sh
tinyfin-rl replay --trace-in runs/run.tft
```

`setup_env.sh` should export:

```bash
export TFRL_HOME="/path/to/tinyfin-rl"
export LD_LIBRARY_PATH="$TFRL_HOME/build/lib:$TFRL_HOME/third_party/raylib/lib:$LD_LIBRARY_PATH"
export PATH="$TFRL_HOME/build/bin:$PATH"
```

The user should not need to remember `LD_LIBRARY_PATH` details.

## CPU And GPU

CPU must be the default path and always work.

GPU should be enabled through Tinyfin:

```bash
./scripts/build.sh --backend cpu
./scripts/build.sh --backend cuda
```

Runtime selection:

```bash
tinyfin-rl train --backend cpu --device cpu
tinyfin-rl train --backend cuda --device gpu
```

All algorithm code should use Tinyfin backend APIs rather than CUDA-specific
logic directly.

If Tinyfin does not expose a needed operation yet, prefer adding the operation to
Tinyfin instead of adding a second math library to Tinyfin-RL.

## Production Build

Production should be a first-class target.

Deliverables:

- `tinyfin-rl` for train/eval/replay.
- `tinyfin-rl-prod` for inference only.
- `libtfrl_runtime.so` for embedding in client systems.
- Per-env shared libraries.
- Optional `tinyfin-rl-raylib` viewer binary.

Production command:

```bash
tinyfin-rl-prod \
  --env maze_rooms \
  --load runs/maze_dqn \
  --steps 10000 \
  --deterministic
```

Production rules:

- No training code required in production binary.
- No raylib dependency unless explicitly requested.
- Stable C ABI.
- Versioned checkpoint compatibility.
- Clear error messages for missing envs or incompatible checkpoints.

## Build System

CMake should be the primary build system. Make can stay as a convenience wrapper.

Targets:

```text
tfrl_core
tfrl_runner
tfrl_algo_dqn
tfrl_algo_ppo
tfrl_algo_nca
tfrl_env_lineworld
tfrl_env_maze_rooms
tfrl_env_coin_maze
tfrl_viewer_raylib
tinyfin-rl
tinyfin-rl-prod
```

Recommended commands:

```bash
./scripts/build.sh
./scripts/build.sh --with-raylib
./scripts/build.sh --backend cuda --with-raylib
./scripts/test.sh
```

## Testing

Minimum test gates:

- Unit tests for each env.
- Unit tests for replay buffer.
- Unit tests for checkpoint save/load.
- Unit tests for render snapshot encoding.
- CLI smoke tests.
- Determinism tests.
- Golden training tests for lineworld.
- Render trace replay test.
- CPU/GPU parity tests where GPU is available.

Do not mark an algorithm complete until:

- It improves on at least one deterministic env.
- It saves and reloads correctly.
- Evaluation reproduces expected behavior.
- Metrics expose losses and returns.
- CPU tests pass.
- GPU tests pass or are explicitly skipped.

## First Implementation Milestones

### Milestone 1: Clean Skeleton

- Create new folder layout.
- Move public headers to `include/tfrl`.
- Split core runner, env API, algorithm API, checkpoint API, and render snapshot API.
- Keep only `lineworld` and `maze_rooms`.

### Milestone 2: Env Modules

- Build `lineworld`, `maze_rooms`, and `coin_maze` as separate libraries.
- Add env manifests.
- Add static and dynamic env loading.
- Add env-specific tests.

### Milestone 3: Real DQN

- Implement DQN with Tinyfin model, replay buffer, target network, save/load.
- Prove it on `lineworld`.
- Add `eval --load --render live`.

### Milestone 4: Raylib Workflow

- Add robust raylib build script.
- Add setup script for environment variables.
- Add replay and live render smoke tests.
- Make render display agent, goal, walls, reward, and done state correctly.

### Milestone 5: PPO

- Implement PPO with rollout buffer, GAE, value head, entropy, save/load.
- Prove it on `maze_rooms`.

### Milestone 6: Tinyfin-NCA

- Define NCA agent/policy interface.
- Implement first NCA policy module.
- Add diversity metrics.
- Integrate with PPO training.

### Milestone 7: Production

- Add production binary.
- Add stable inference ABI.
- Add packaging script.
- Add docs for client embedding.

### Milestone 8: Exciting Env Pack

- Add `snake`, `breakout_tiny`, and `warehouse_bot` as optional env modules.
- Add render snapshots and replay tests for each.
- Add short demo configs.
- Keep these separate from the core correctness suite.

## Recommended Training Targets

Start with:

```bash
tinyfin-rl train --algo dqn --env lineworld --steps 50000
```

Then:

```bash
tinyfin-rl train --algo ppo --env maze_rooms --steps 250000
```

Do not use `maze_rooms` as the main target until its render snapshot and goal
logic have tests and visual verification.

## Quality Bar

Tinyfin-RL v2 should feel like a library, not a demo.

The standard for merging features:

- Clear API.
- Tests.
- Docs.
- Deterministic behavior.
- Production story.
- Useful errors.
- No fake algorithm names.
- No hidden runtime dependencies.
- No environment-renderer coupling.
