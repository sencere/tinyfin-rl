# Environment Template

Use this as a starting point for new C-first environments.

Files to provide:
- `plugin_<env>.c` (implements `rl_plugin.h` + optional metadata)
- `render_<env>.c` (implements `rl_render.h`)
- `view.c` (viewer executable)
- `Makefile`
- `viewer.conf` (defaults for plugin_viewer)

## Build checklist

```bash
make
./<env>
```

## plugin_viewer

```bash
cd ../common
make
./plugin_viewer ../<env> --fps 10
```

`viewer.conf` can include `fps=10` to set a default.
