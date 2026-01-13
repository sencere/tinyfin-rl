# Trace Format

Traces are a sequence of render snapshots stored in a file. The file header:

```
magic: "TFT1"
uint32 version (1)
```

Then each frame is written as:

```
tfrl_render_snapshot_header
payload bytes
```

Replay reads frames and feeds them to the viewer. Current snapshots use v2 with
`reward`, `done`, and `env_kind` in the payload header (see `docs/render_snapshot.md`).
