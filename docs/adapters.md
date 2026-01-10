# Env Adapters

## Gymnasium

Use Gymnasium environments via a thin wrapper:

```bash
python - <<'PY'
from tinyfin_rl.adapters.gymnasium_adapter import make_gymnasium
env = make_gymnasium("CartPole-v1")
obs = env.reset()
print(obs)
PY
```

## PettingZoo (parallel)

Wrap a PettingZoo parallel env:

```bash
python - <<'PY'
from tinyfin_rl.adapters.pettingzoo_adapter import make_pettingzoo_parallel
env = make_pettingzoo_parallel("butterfly/coop_pong_v5")
obs, info = env.reset()
print(list(obs.keys())[:2])
PY
```
