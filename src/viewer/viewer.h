#ifndef TFRL_VIEWER_H
#define TFRL_VIEWER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct {
    int fps;
    const char *title;
} tfrl_viewer_config;

typedef struct tfrl_viewer tfrl_viewer;

tfrl_viewer *tfrl_viewer_create(const tfrl_viewer_config *cfg);
void tfrl_viewer_draw(tfrl_viewer *viewer, const void *snapshot, size_t len);
int tfrl_viewer_should_close(tfrl_viewer *viewer);
void tfrl_viewer_destroy(tfrl_viewer *viewer);

#ifdef __cplusplus
}
#endif

#endif
