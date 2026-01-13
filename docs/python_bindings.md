# Python bindings (C-first environments)

Tinyfin-RL keeps environments in C and exposes them to Python via a minimal plugin ABI (`tinyfin_rl/rl_plugin.h`). The Python layer uses `ctypes` to load shared libraries, call `reset/step`, and render via a small renderer ABI (`tinyfin_rl/rl_render.h`). Raylib stays entirely inside C.

## Build the C shared libraries

From the gridworld environment:

```bash
cd tinyfin-rl/environments/first-env
make libfirst_env.so libfirst_env_render.so
```

This builds:
- `libfirst_env.so` (env plugin)
- `libfirst_env_render.so` (renderer implementing `rl_render.h`)

If you do not have a system raylib, the build uses `raylib-src` and produces `libraylib.so` there.
Use `make RAYLIB_DIR=../../raylib-src` in each env if you want to point at the bundled raylib.

## Minimal Python usage

```python
from tinyfin_rl.bindings import EnvPlugin, EnvPluginAdapter, GridRaylibRenderer

plugin = EnvPlugin("environments/first-env/libfirst_env.so")
env = EnvPluginAdapter(plugin)
renderer = GridRaylibRenderer(
    "environments/first-env/libfirst_env_render.so",
    grid_size=10,
    cell_size=48,
    title="tinyfin-rl grid",
)

obs = env.reset()
reward = 0.0
done = False

while not renderer.should_close():
    renderer.draw(obs, reward, done)
    action = 0  # TODO: replace with a policy
    obs, reward, done, _ = env.step(action)
    if done:
        obs = env.reset()
        done = False

renderer.close()
env.close()
```

## Train from Python (C trainers)

```python
from tinyfin_rl.bindings import EnvPlugin, TrainLib, DqnConfig

plugin = EnvPlugin("environments/lineworld/liblineworld.so")
train = TrainLib()
cfg = DqnConfig(
    obs_n=7,
    action_n=2,
    steps=1000,
    gamma=0.99,
    lr=0.05,
    epsilon=0.1,
    save_path=b"runs/lineworld_dqn",
    load_path=b"runs/lineworld_dqn",
)
status, metrics = train.train_dqn(plugin, cfg)
print(status, metrics.return_mean, metrics.loss)
```

## Notes

- The renderer is C-only and uses raylib for all drawing.
- The Python binding is intentionally small: it only deals with `tfrl_env` and basic value types.
- Override paths with `TFRL_ENV_PLUGIN_LIB` and `TFRL_GRID_RENDER_LIB` if needed.
- If a plugin provides `tfrl_env_metadata_get`, the adapter can auto-create `Discrete` or `Box` spaces.
- `TrainLib` loads `build/libtfrl_train.so` by default; override with `TFRL_TRAIN_LIB`.
- `save_path`/`load_path` in the configs are `c_char_p` fields; pass `bytes` or `None`.
