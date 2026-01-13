#include <stdio.h>
#include <stdlib.h>

#include "core/replay_buffer.h"

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_replay_buffer: %s\n", msg);
        return 0;
    }
    return 1;
}

int main(void) {
    srand(0);
    tfrl_replay_buffer *buf = tfrl_replay_create(10, 0.6f);
    if (!expect(buf != NULL, "create")) return 1;

    for (int i = 0; i < 8; i++) {
        tfrl_transition tr = {0};
        tr.obs.index = i;
        tr.action.index = i % 2;
        tr.reward = (double)i;
        tr.next_obs.index = i + 1;
        tr.done = 0;
        tfrl_replay_push(buf, &tr, (float)(i + 1));
    }
    if (!expect(tfrl_replay_size(buf) == 8, "size after push")) return 1;

    int idx[4] = {0};
    float weights[4] = {0};
    int got = tfrl_replay_sample(buf, 4, idx, weights, 0.4f);
    if (!expect(got == 4, "sample count")) return 1;
    for (int i = 0; i < 4; i++) {
        if (!expect(idx[i] >= 0 && idx[i] < 8, "sample index range")) return 1;
        if (!expect(weights[i] > 0.0f && weights[i] <= 1.0f, "sample weight range")) return 1;
    }

    tfrl_replay_update_priority(buf, idx[0], 5.0f);
    const tfrl_transition *tr = tfrl_replay_get(buf, idx[0]);
    if (!expect(tr != NULL, "get transition")) return 1;

    tfrl_replay_free(buf);
    return 0;
}
