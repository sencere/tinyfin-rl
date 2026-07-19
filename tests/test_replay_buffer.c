#include <stdio.h>

#include "core/replay_buffer.h"

static int expect(int cond, const char *msg) {
    if (!cond) {
        fprintf(stderr, "test_replay_buffer: %s\n", msg);
        return 0;
    }
    return 1;
}

int main(void) {
    tfrl_replay_profile_stats stats = {-1.0, -1};

    tfrl_replay_profile_reset();
    tfrl_replay_profile_get(&stats);
    if (!expect(stats.sample_seconds == 0.0, "reset sample seconds")) return 1;
    if (!expect(stats.sample_calls == 0, "reset sample calls")) return 1;

    tfrl_replay_profile_add_sample(0.25);
    tfrl_replay_profile_add_sample(0.75);
    tfrl_replay_profile_add_sample(0.0);
    tfrl_replay_profile_add_sample(-1.0);

    tfrl_replay_profile_get(&stats);
    if (!expect(stats.sample_seconds == 1.0, "accumulated sample seconds")) return 1;
    if (!expect(stats.sample_calls == 2, "positive sample call count")) return 1;

    tfrl_replay_profile_get(NULL);

    tfrl_replay_profile_reset();
    tfrl_replay_profile_get(&stats);
    if (!expect(stats.sample_seconds == 0.0, "final reset sample seconds")) return 1;
    if (!expect(stats.sample_calls == 0, "final reset sample calls")) return 1;

    return 0;
}
