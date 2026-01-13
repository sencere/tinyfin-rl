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

    tfrl_replay_buffer *ring = tfrl_replay_create(4, 0.0f);
    if (!expect(ring != NULL, "create ring")) return 1;
    for (int i = 0; i < 6; i++) {
        tfrl_transition tr2 = {0};
        tr2.obs.index = i;
        tfrl_replay_push(ring, &tr2, 1.0f);
    }
    if (!expect(tfrl_replay_size(ring) == 4, "ring size after wrap")) return 1;
    int seen[6] = {0};
    for (int i = 0; i < 4; i++) {
        const tfrl_transition *cur = tfrl_replay_get(ring, i);
        if (!expect(cur != NULL, "ring get")) return 1;
        if (cur->obs.index >= 0 && cur->obs.index < 6) {
            seen[cur->obs.index] = 1;
        }
    }
    for (int i = 2; i < 6; i++) {
        if (!expect(seen[i] == 1, "ring contains last entries")) return 1;
    }
    int idx2[4] = {0};
    float weights2[4] = {0};
    int got2 = tfrl_replay_sample(ring, 4, idx2, weights2, 0.0f);
    if (!expect(got2 == 4, "ring sample count")) return 1;
    for (int i = 0; i < 4; i++) {
        if (!expect(weights2[i] == 1.0f, "uniform weight")) return 1;
    }
    tfrl_replay_free(ring);

    tfrl_replay_buffer *prio = tfrl_replay_create(8, 0.6f);
    if (!expect(prio != NULL, "create prio")) return 1;
    for (int i = 0; i < 8; i++) {
        tfrl_transition tr3 = {0};
        tr3.obs.index = i;
        float p = i == 7 ? 10.0f : 1.0f;
        tfrl_replay_push(prio, &tr3, p);
    }
    int counts[8] = {0};
    srand(42);
    for (int i = 0; i < 500; i++) {
        int idx3[1] = {0};
        float w3[1] = {0};
        int got3 = tfrl_replay_sample(prio, 1, idx3, w3, 0.6f);
        if (!expect(got3 == 1, "prio sample count")) return 1;
        counts[idx3[0]]++;
        if (!expect(w3[0] > 0.0f && w3[0] <= 1.0f, "prio weight range")) return 1;
    }
    if (!expect(counts[7] > counts[0], "prio bias toward high priority")) return 1;

    int idx_a[4] = {0};
    float w_a[4] = {0};
    srand(123);
    if (!expect(tfrl_replay_sample(prio, 4, idx_a, w_a, 0.4f) == 4, "det sample a")) return 1;
    int idx_b[4] = {0};
    float w_b[4] = {0};
    srand(123);
    if (!expect(tfrl_replay_sample(prio, 4, idx_b, w_b, 0.4f) == 4, "det sample b")) return 1;
    for (int i = 0; i < 4; i++) {
        if (!expect(idx_a[i] == idx_b[i], "det idx match")) return 1;
        if (!expect(w_a[i] == w_b[i], "det weight match")) return 1;
    }

    tfrl_replay_free(prio);
    return 0;
}
