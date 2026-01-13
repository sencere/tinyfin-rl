#ifndef TINYFIN_RL_RENDER_SNAPSHOT_H
#define TINYFIN_RL_RENDER_SNAPSHOT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define TFRL_SNAPSHOT_API_VERSION 0x0001

typedef struct {
    uint32_t api_version;
    uint32_t frame_index;
    uint32_t payload_bytes;
} tfrl_render_snapshot_header;

// Optional: query required bytes for a render snapshot.
// Return 0 if unsupported.
size_t tfrl_render_snapshot_size(void *env_ctx);

// Optional: write snapshot into the caller-provided buffer.
// Returns bytes written, or 0 if unsupported.
size_t tfrl_render_snapshot_write(void *env_ctx, void *buffer, size_t buffer_len);

#ifdef __cplusplus
}
#endif

#endif
