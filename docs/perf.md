# Performance

## Batch stepping

`tfrl_env_step_batch` provides a simple vectorized stepping API that loops over
multiple env instances. This is the starting point for true SIMD or parallel
env stepping.

Threaded stepping is available via `--threads N` to split env stepping across
worker threads.

## Benchmark

```bash
./scripts/bench_steps.sh 200000
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
