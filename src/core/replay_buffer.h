#ifndef TFRL_REPLAY_BUFFER_H
#define TFRL_REPLAY_BUFFER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double sample_seconds;
    long long sample_calls;
} tfrl_replay_profile_stats;

void tfrl_replay_profile_reset(void);
void tfrl_replay_profile_get(tfrl_replay_profile_stats *out_stats);
void tfrl_replay_profile_add_sample(double seconds);

#ifdef __cplusplus
}
#endif

#endif
