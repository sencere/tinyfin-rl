# SAC Walkthrough

SAC training in Python expects an `Env` interface. Use `EnvPluginAdapter` with a C env plugin (see `docs/python_bindings.md`) and ensure Tinyfin is on `PYTHONPATH`. The current SAC implementation assumes scalar obs/action.
