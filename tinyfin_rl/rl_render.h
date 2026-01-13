#ifndef TINYFIN_RL_RENDER_H
#define TINYFIN_RL_RENDER_H

#ifdef __cplusplus
extern "C" {
#endif

// Renderer ABI: implement these symbols in a shared library.
// The renderer can use raylib internally, but callers only link to this ABI.

#include "rl.h"

#ifdef _WIN32
#define TFRL_RENDER_EXPORT __declspec(dllexport)
#else
#define TFRL_RENDER_EXPORT __attribute__((visibility("default")))
#endif

TFRL_RENDER_EXPORT int tfrl_renderer_init(int grid_size, int cell_size, const char *title);
TFRL_RENDER_EXPORT void tfrl_renderer_draw(const tfrl_value *obs, double reward, int done);
TFRL_RENDER_EXPORT int tfrl_renderer_should_close(void);
TFRL_RENDER_EXPORT void tfrl_renderer_close(void);

#ifdef __cplusplus
}
#endif

#endif
