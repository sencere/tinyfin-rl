# Performance

## Batch stepping

`tfrl_env_step_batch` provides a simple vectorized stepping API that loops over
multiple env instances. This is the starting point for true SIMD or parallel
env stepping.

`tfrl_env_reset_batch` and `tfrl_env_reset_batch_seeds` provide fast reset
paths for single-agent env arrays (initial resets and sparse episode resets).

Threaded stepping is available via `--threads N` to split env stepping across
worker threads.

Aggressive local build flags (use for benchmarks only):

```bash
make PERF=1
cmake -DTFRL_PERF_FLAGS=ON -S . -B build
cmake --build build
```

For deterministic baselines, use `--deterministic` (or `--threads 1`) to avoid
thread scheduling variability.

## Benchmark

```bash
./scripts/bench_steps.sh 200000
```

Run baseline profiles (CPU by default, CUDA when `RUN_CUDA=1`):

```bash
./scripts/perf_baselines.sh 20000
RUN_BLAS=1 ./scripts/perf_baselines.sh 20000
RUN_CUDA=1 ./scripts/perf_baselines.sh 20000
```

BLAS baseline requires rebuilding Tinyfin with BLAS:

```bash
make -C tinyfin ENABLE_BLAS=1 libtinyfin.so
```

Summarize baseline results:

```bash
python3 scripts/perf_compare.py runs/perf_baselines
```

Smoke checks (mp DQN + IMPALA telemetry):

```bash
./scripts/perf_smoke.sh
```

Use vectorized env stepping for throughput:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --envs 32 --steps 200000 --render off
```

With threads:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --envs 32 --threads 4 --steps 200000 --render off
```

Batched inference is enabled for DQN when `--envs > 1` via `act_batch`.

## Profiling

Enable the training profiler to get a simple breakdown of time spent in
environment stepping, algorithm updates, and rendering.

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 20000 --profile
```

Write a JSON report:

```bash
./build/tinyfin-rl train --algo dqn --env maze_rooms --steps 20000 --profile-json runs/profile.json
```

IMPALA telemetry (queue depth + SPS):

```bash
./build/tinyfin-rl train --algo impala --env maze_rooms --steps 20000 --actor-count 4 --learner-batch 8 --learner-batch-auto --log-every 200 --profile
```

## Tuning Knobs

- `--envs N` increases parallel environment stepping per iteration.
- `--threads N` splits env stepping across threads (agent_count == 1).
- `--train-every N` reduces update frequency to boost SPS.
- `--batch-size N` trades compute intensity vs. update cost.
- `--learning-starts N` delays updates until replay has warm data.
- `--grad-steps N` performs multiple updates per training tick.
