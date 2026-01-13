#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "runner/runner.h"
#include "core/algo_api.h"
#include "core/env_api.h"
#include "core/trace.h"
#include "viewer/viewer.h"

static void seed_rng(int seed) {
    if (seed <= 0) {
        seed = (int)time(NULL);
    }
    srand((unsigned)seed);
}

static int should_render(int step, int render_every) {
    if (render_every <= 1) return 1;
    return (step % render_every) == 0;
}

typedef struct {
    tfrl_env **envs;
    const tfrl_action *actions;
    tfrl_step_result *steps;
    int start;
    int count;
} tfrl_step_job;

static void *step_worker(void *arg) {
    tfrl_step_job *job = (tfrl_step_job *)arg;
    int end = job->start + job->count;
    for (int i = job->start; i < end; i++) {
        job->steps[i] = tfrl_env_step(job->envs[i], job->actions[i]);
    }
    return NULL;
}

static void step_envs_threaded(tfrl_env **envs, int env_count, const tfrl_action *actions, tfrl_step_result *steps, int threads) {
    if (env_count <= 0) return;
    if (threads <= 1 || env_count == 1) {
        tfrl_env_step_batch(envs, env_count, actions, steps);
        return;
    }
    if (threads > env_count) threads = env_count;
    pthread_t *tids = (pthread_t *)calloc((size_t)threads, sizeof(pthread_t));
    tfrl_step_job *jobs = (tfrl_step_job *)calloc((size_t)threads, sizeof(tfrl_step_job));
    if (!tids || !jobs) {
        free(tids);
        free(jobs);
        tfrl_env_step_batch(envs, env_count, actions, steps);
        return;
    }

    int base = env_count / threads;
    int extra = env_count % threads;
    int start = 0;
    for (int t = 0; t < threads; t++) {
        int count = base + (t < extra ? 1 : 0);
        jobs[t].envs = envs;
        jobs[t].actions = actions;
        jobs[t].steps = steps;
        jobs[t].start = start;
        jobs[t].count = count;
        pthread_create(&tids[t], NULL, step_worker, &jobs[t]);
        start += count;
    }
    for (int t = 0; t < threads; t++) {
        pthread_join(tids[t], NULL);
    }
    free(tids);
    free(jobs);
}

static int run_replay(const tfrl_runner_config *cfg) {
    tfrl_trace_reader *reader = tfrl_trace_reader_open(cfg->trace_in);
    if (!reader) {
        fprintf(stderr, "failed to open trace: %s\n", cfg->trace_in);
        return 1;
    }
    const char *meta = tfrl_trace_reader_meta(reader);
    tfrl_viewer_config vcfg = {.fps = cfg->render_fps, .title = "tinyfin-rl replay", .meta = meta};
    tfrl_viewer *viewer = tfrl_viewer_create(&vcfg);
    if (!viewer) {
        fprintf(stderr, "viewer not available (build without raylib?)\n");
        tfrl_trace_reader_close(reader);
        return 1;
    }

    size_t buf_cap = 1024 * 1024;
    void *buffer = calloc(1, buf_cap);
    if (!buffer) {
        tfrl_viewer_destroy(viewer);
        tfrl_trace_reader_close(reader);
        return 1;
    }

    size_t len = 0;
    while (tfrl_trace_reader_next(reader, buffer, buf_cap, &len)) {
        tfrl_viewer_draw(viewer, buffer, len);
        if (tfrl_viewer_should_close(viewer)) break;
    }
    free(buffer);
    tfrl_viewer_destroy(viewer);
    tfrl_trace_reader_close(reader);
    return 0;
}

int tfrl_runner_run(const tfrl_runner_config *cfg) {
    if (!cfg) return 1;
    if (cfg->mode == TFRL_MODE_REPLAY) {
        if (!cfg->trace_in) {
            fprintf(stderr, "replay requires --trace-in\n");
            return 1;
        }
        return run_replay(cfg);
    }

    int seed = cfg->seed;
    if (cfg->deterministic && seed == 0) {
        seed = 1;
        fprintf(stdout, "deterministic mode: seed set to %d\n", seed);
    }
    seed_rng(seed);
    int env_count = cfg->envs > 0 ? cfg->envs : 1;
    int render_idx = cfg->render_env;
    if (render_idx < 0) render_idx = 0;
    if (render_idx >= env_count) {
        fprintf(stderr, "render-env out of range (0..%d)\n", env_count - 1);
        return 1;
    }
    tfrl_env **envs = (tfrl_env **)calloc((size_t)env_count, sizeof(tfrl_env *));
    if (!envs) return 1;
    for (int i = 0; i < env_count; i++) {
        tfrl_env_config env_cfg = {.name = cfg->env_name, .seed = (uint64_t)(seed + i)};
        envs[i] = tfrl_env_create(&env_cfg);
        if (!envs[i]) {
            fprintf(stderr, "env creation failed\n");
            for (int j = 0; j < i; j++) {
                tfrl_env_destroy(envs[j]);
            }
            free(envs);
            return 1;
        }
    }
    const tfrl_env_spec *spec = tfrl_env_get_spec(envs[0]);
    tfrl_algo_config algo_cfg = {
        .name = cfg->algo ? cfg->algo : "dqn",
        .obs_n = spec->obs_n,
        .action_n = spec->action_n,
        .obs_type = spec->obs_type,
        .action_type = spec->action_type,
        .obs_dims = spec->obs_dims,
        .action_dims = spec->action_dims,
        .obs_shape = {spec->obs_shape[0], spec->obs_shape[1]},
        .action_shape = {spec->action_shape[0], spec->action_shape[1]},
        .obs_dtype = spec->obs_dtype,
        .action_dtype = spec->action_dtype,
        .obs_low = spec->obs_low,
        .obs_high = spec->obs_high,
        .action_low = spec->action_low,
        .action_high = spec->action_high,
        .gamma = cfg->gamma,
        .lr = cfg->lr,
        .epsilon = cfg->mode == TFRL_MODE_TRAIN ? cfg->epsilon : 0.0f,
        .entropy_coef = cfg->entropy_coef,
        .clip_eps = cfg->clip_eps,
        .gae_lambda = cfg->gae_lambda,
        .replay_size = cfg->replay_size,
        .batch_size = cfg->batch_size,
        .per_alpha = cfg->per_alpha,
        .per_beta = cfg->per_beta,
        .mcts_sims = cfg->mcts_sims,
        .mcts_depth = cfg->mcts_depth,
        .steps_per_batch = cfg->steps_per_batch,
        .epochs = cfg->epochs,
        .save_path = cfg->save_path,
        .load_path = cfg->load_path,
        .env_name = cfg->env_name,
        .seed = seed,
        .deterministic = cfg->deterministic,
    };
    tfrl_algo_config_apply_defaults(&algo_cfg);
    tfrl_algo algo = tfrl_algo_create(&algo_cfg, spec);
    if (!algo.ctx) {
        fprintf(stderr, "algo creation failed\n");
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }

    tfrl_viewer *viewer = NULL;
    void *render_buf = NULL;
    size_t render_buf_len = 0;
    if (cfg->render) {
        tfrl_viewer_config vcfg = {.fps = cfg->render_fps, .title = "tinyfin-rl"};
        viewer = tfrl_viewer_create(&vcfg);
        if (!viewer) {
            fprintf(stderr, "viewer not available (build without raylib?)\n");
        } else {
            render_buf_len = tfrl_env_render_bytes_needed(envs[render_idx]);
            render_buf = calloc(1, render_buf_len);
        }
    }

    tfrl_trace_writer *writer = NULL;
    char trace_meta[512];
    snprintf(trace_meta, sizeof(trace_meta),
             "algo=%s defaults=v%d env=%s seed=%d deterministic=%d gamma=%.3f lr=%.6f epsilon=%.3f entropy=%.3f clip_eps=%.3f gae_lambda=%.3f "
             "replay=%d batch=%d per_alpha=%.3f per_beta=%.3f steps_per_batch=%d epochs=%d mcts_sims=%d mcts_depth=%d",
             algo_cfg.name ? algo_cfg.name : "null",
             algo_cfg.defaults_version,
             algo_cfg.env_name ? algo_cfg.env_name : "null",
             algo_cfg.seed,
             algo_cfg.deterministic,
             algo_cfg.gamma,
             algo_cfg.lr,
             algo_cfg.epsilon,
             algo_cfg.entropy_coef,
             algo_cfg.clip_eps,
             algo_cfg.gae_lambda,
             algo_cfg.replay_size,
             algo_cfg.batch_size,
             algo_cfg.per_alpha,
             algo_cfg.per_beta,
             algo_cfg.steps_per_batch,
             algo_cfg.epochs,
             algo_cfg.mcts_sims,
             algo_cfg.mcts_depth);
    if (cfg->trace_out) {
        writer = tfrl_trace_writer_open_with_meta(cfg->trace_out, trace_meta);
        if (!writer) {
            fprintf(stderr, "failed to open trace out: %s\n", cfg->trace_out);
        } else {
            if (!render_buf) {
                render_buf_len = tfrl_env_render_bytes_needed(envs[render_idx]);
                render_buf = calloc(1, render_buf_len);
            }
        }
    }

    tfrl_obs *obs = (tfrl_obs *)calloc((size_t)env_count, sizeof(tfrl_obs));
    tfrl_action *actions = (tfrl_action *)calloc((size_t)env_count, sizeof(tfrl_action));
    tfrl_step_result *steps = (tfrl_step_result *)calloc((size_t)env_count, sizeof(tfrl_step_result));
    double *episode_returns = (double *)calloc((size_t)env_count, sizeof(double));
    int *episode_counts = (int *)calloc((size_t)env_count, sizeof(int));
    if (!obs || !actions || !steps || !episode_returns || !episode_counts) {
        fprintf(stderr, "allocation failed\n");
        free(obs);
        free(actions);
        free(steps);
        free(episode_returns);
        free(episode_counts);
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }
    for (int i = 0; i < env_count; i++) {
        obs[i] = tfrl_env_reset(envs[i], (uint64_t)(seed + i));
    }

    if (cfg->mode == TFRL_MODE_TRAIN) {
        int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
        int thread_count = cfg->threads > 0 ? cfg->threads : env_count;
        if (cfg->deterministic) thread_count = 1;
        for (int step = 0; step < total_steps; step++) {
            tfrl_algo_act_batch(&algo, obs, env_count, actions);
            if (env_count == 1) {
                steps[0] = tfrl_env_step(envs[0], actions[0]);
            } else {
                step_envs_threaded(envs, env_count, actions, steps, thread_count);
            }
            for (int i = 0; i < env_count; i++) {
                tfrl_transition transition = {
                    .obs = obs[i],
                    .action = actions[i],
                    .reward = steps[i].reward,
                    .next_obs = steps[i].observation,
                    .done = steps[i].done,
                };
                algo.vtable->update(algo.ctx, &transition);
                obs[i] = steps[i].observation;
                episode_returns[i] += steps[i].reward;
            }

            if ((viewer || writer) && should_render(step, cfg->render_every)) {
                size_t written = tfrl_env_render_write(envs[render_idx], render_buf, render_buf_len);
                if (writer && written > 0) {
                    tfrl_trace_writer_write(writer, render_buf, written);
                }
                if (viewer && written > 0) {
                    tfrl_viewer_draw(viewer, render_buf, written);
                }
            }

            for (int i = 0; i < env_count; i++) {
                if (steps[i].done) {
                    episode_counts[i] += 1;
                    if (cfg->log_every > 0 && (episode_counts[i] % cfg->log_every) == 0) {
                        fprintf(stdout, "env=%d episode=%d return=%.4f\n", i, episode_counts[i], episode_returns[i]);
                    }
                    obs[i] = tfrl_env_reset(envs[i], (uint64_t)(cfg->seed + i));
                    episode_returns[i] = 0.0;
                }
            }
            if (viewer && tfrl_viewer_should_close(viewer)) {
                break;
            }
        }
        if (env_count > 1) {
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < env_count; i++) {
                sum += episode_returns[i];
                count += episode_counts[i] > 0 ? episode_counts[i] : 0;
            }
            if (count > 0) {
                fprintf(stdout, "avg_return=%.4f episodes=%d\n", sum / (double)count, count);
            }
        }
    } else if (cfg->mode == TFRL_MODE_EVAL) {
        int episodes = cfg->episodes > 0 ? cfg->episodes : 10;
        for (int ep = 0; ep < episodes; ep++) {
            obs[0] = tfrl_env_reset(envs[0], (uint64_t)seed);
            int done = 0;
            int step = 0;
            while (!done) {
                actions[0] = algo.vtable->act(algo.ctx, obs[0]);
                steps[0] = tfrl_env_step(envs[0], actions[0]);
                obs[0] = steps[0].observation;
                done = steps[0].done;

                if ((viewer || writer) && should_render(step, cfg->render_every)) {
                    size_t written = tfrl_env_render_write(envs[render_idx], render_buf, render_buf_len);
                    if (writer && written > 0) {
                        tfrl_trace_writer_write(writer, render_buf, written);
                    }
                    if (viewer && written > 0) {
                        tfrl_viewer_draw(viewer, render_buf, written);
                    }
                }
                step++;
                if (viewer && tfrl_viewer_should_close(viewer)) {
                    ep = episodes;
                    break;
                }
            }
        }
    }

    if (writer) tfrl_trace_writer_close(writer);
    if (viewer) tfrl_viewer_destroy(viewer);
    free(render_buf);
    if (cfg->save_path) {
        tfrl_algo_save(&algo, cfg->save_path);
    }
    tfrl_algo_destroy(&algo);
    for (int i = 0; i < env_count; i++) {
        tfrl_env_destroy(envs[i]);
    }
    free(envs);
    free(obs);
    free(actions);
    free(steps);
    free(episode_returns);
    free(episode_counts);
    return 0;
}
