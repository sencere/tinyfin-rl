# Performance

## Batch stepping

`tfrl_env_step_batch` provides a simple vectorized stepping API that loops over
multiple env instances. This is the starting point for true SIMD or parallel
env stepping.

## Benchmark

```bash
./scripts/bench_steps.sh 200000
```

Use vectorized env stepping for throughput:

```bash
./build/tinyfin-rl train --algo dqn --env lineworld --envs 32 --steps 200000 --render off
```
