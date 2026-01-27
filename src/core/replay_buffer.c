#include "replay_buffer.h"

#include <pthread.h>

static pthread_mutex_t g_replay_lock = PTHREAD_MUTEX_INITIALIZER;
static tfrl_replay_profile_stats g_stats = {0};

void tfrl_replay_profile_reset(void) {
    pthread_mutex_lock(&g_replay_lock);
    g_stats.sample_seconds = 0.0;
    g_stats.sample_calls = 0;
    pthread_mutex_unlock(&g_replay_lock);
}

void tfrl_replay_profile_get(tfrl_replay_profile_stats *out_stats) {
    if (!out_stats) return;
    pthread_mutex_lock(&g_replay_lock);
    *out_stats = g_stats;
    pthread_mutex_unlock(&g_replay_lock);
}

void tfrl_replay_profile_add_sample(double seconds) {
    if (seconds <= 0.0) return;
    pthread_mutex_lock(&g_replay_lock);
    g_stats.sample_seconds += seconds;
    g_stats.sample_calls += 1;
    pthread_mutex_unlock(&g_replay_lock);
}
