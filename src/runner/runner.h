#ifndef TFRL_RUNNER_H
#define TFRL_RUNNER_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TFRL_MODE_TRAIN = 0,
    TFRL_MODE_EVAL = 1,
    TFRL_MODE_REPLAY = 2,
} tfrl_mode;

typedef struct {
    tfrl_mode mode;
    const char *algo;
    const char *env_name;
    int envs;
    int steps;
    int episodes;
    int seed;
    float gamma;
    float lr;
    float epsilon;
    float entropy_coef;
    float clip_eps;
    float kl_target;
    float gae_lambda;
    int replay_size;
    int batch_size;
    float per_alpha;
    float per_beta;
    int mcts_sims;
    int mcts_depth;
    int steps_per_batch;
    int epochs;
    int log_every;
    int render;
    int render_every;
    int render_fps;
    int render_env;
    const char *trace_out;
    const char *trace_in;
    int dump_meta_json;
    int actor_count;
    int learner_batch;
    int queue_capacity;
    const char *save_path;
    const char *load_path;
    int threads;
    int deterministic;
    int agents_expected;
    int share_policy;
} tfrl_runner_config;

int tfrl_runner_run(const tfrl_runner_config *cfg);

#ifdef __cplusplus
}
#endif

#endif
