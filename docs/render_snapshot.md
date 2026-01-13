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

## Payload (v1)

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

## Payload (v2)

The current format uses v2 with extra metadata:

```
width, height
agent_x, agent_y
goal_x, goal_y
step, max_steps
reward, done, env_kind
```

The walls array follows the v2 grid header.

## Payload (v3)

v3 adds an entity list for richer environments:

```
width, height
agent_x, agent_y
goal_x, goal_y
step, max_steps
reward, done, env_kind
entity_count
```

After the walls array, v3 appends `entity_count` entries:

```
type, x, y, value
```

`type` is application-defined (e.g., 1 = coin, 2 = agent).
