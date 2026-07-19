#ifndef TFRL_PUBLIC_METRICS_H
#define TFRL_PUBLIC_METRICS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int step;
    long long env_steps;
    double samples_per_second;
    double updates_per_second;
    int episodes;
    double last_return;
    double mean_return;
    double mean_episode_length;
    double loss;
    double policy_entropy;
    double value_loss;
    double q_loss;
    double exploration_rate;
    double learning_rate;
    int replay_size;
    const char *backend;
    const char *device;
    double env_seconds;
    double inference_seconds;
    double update_seconds;
    double render_seconds;
} tfrl_metrics;

#ifdef __cplusplus
}
#endif

#endif
