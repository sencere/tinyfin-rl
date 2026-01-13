# Render Snapshot Format

Render snapshots let the environment describe its state as a byte buffer that
the viewer can interpret. The environment never links against raylib.

## Header

```c
typedef struct {
    uint32_t api_version;
    uint32_t frame_index;
    uint32_t payload_bytes;
} tfrl_render_snapshot_header;
```

## maze_rooms payload

The canonical environment emits:

```
header
grid_snapshot struct
wall bytes (width * height)
```

`grid_snapshot` fields:

```
width, height
agent_x, agent_y
goal_x, goal_y
step, max_steps
```

The viewer reads the payload and draws a grid.

`lineworld` uses the same payload but sets `height=1` and has no walls.
