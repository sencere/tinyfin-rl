# Env Adapters

## Gymnasium

Use Gymnasium environments via a thin wrapper (Python-only):

```bash
python - <<'PY'
from tinyfin_rl.adapters.gymnasium_adapter import make_gymnasium
env = make_gymnasium("CartPole-v1")
obs = env.reset()
print(obs)
PY
```

You can also use the factory helper (Python-only):

```bash
python - <<'PY'
from tinyfin_rl.factory import make
env = make("gym:CartPole-v1")
obs = env.reset()
print(obs)
PY
```

## PettingZoo (parallel)

Wrap a PettingZoo parallel env (Python-only):

```bash
python - <<'PY'
from tinyfin_rl.adapters.pettingzoo_adapter import make_pettingzoo_parallel
env = make_pettingzoo_parallel("butterfly/cooperative_pong_v5")
obs, info = env.reset()
print(list(obs.keys())[:2])
PY
```

Factory helper (Python-only):

```bash
python - <<'PY'
from tinyfin_rl.factory import make
env = make("pettingzoo:butterfly/coop_pong_v5")
obs, _ = env.reset()
print(list(obs.keys())[:2])
PY
```
