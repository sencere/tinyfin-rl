#ifndef TFRL_RENDER_SNAPSHOT_H
#define TFRL_RENDER_SNAPSHOT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

#define TFRL_SNAPSHOT_API_VERSION 0x0002

typedef struct {
    uint32_t api_version;
    uint32_t frame_index;
    uint32_t payload_bytes;
} tfrl_render_snapshot_header;

#ifdef __cplusplus
}
#endif

#endif
