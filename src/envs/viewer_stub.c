#include "viewer.h"

#include <stdlib.h>

struct tfrl_viewer {
    int dummy;
};

tfrl_viewer *tfrl_viewer_create(const tfrl_viewer_config *cfg) {
    (void)cfg;
    return (tfrl_viewer *)calloc(1, sizeof(tfrl_viewer));
}

void tfrl_viewer_destroy(tfrl_viewer *viewer) {
    free(viewer);
}

int tfrl_viewer_should_close(tfrl_viewer *viewer) {
    (void)viewer;
    return 0;
}

void tfrl_viewer_draw(tfrl_viewer *viewer, const void *snapshot, size_t len) {
    (void)viewer;
    (void)snapshot;
    (void)len;
}
