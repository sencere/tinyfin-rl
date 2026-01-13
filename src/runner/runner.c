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

static int run_replay(const tfrl_runner_config *cfg) {
    tfrl_trace_reader *reader = tfrl_trace_reader_open(cfg->trace_in);
    if (!reader) {
        fprintf(stderr, "failed to open trace: %s\n", cfg->trace_in);
        return 1;
    }
    tfrl_viewer_config vcfg = {.fps = cfg->render_fps, .title = "tinyfin-rl replay"};
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

    seed_rng(cfg->seed);
    int env_count = cfg->envs > 0 ? cfg->envs : 1;
    if (env_count > 1 && cfg->render) {
        fprintf(stderr, "render disabled for envs > 1\n");
    }
    tfrl_env **envs = (tfrl_env **)calloc((size_t)env_count, sizeof(tfrl_env *));
    if (!envs) return 1;
    for (int i = 0; i < env_count; i++) {
        tfrl_env_config env_cfg = {.name = cfg->env_name, .seed = (uint64_t)(cfg->seed + i)};
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
        .gamma = cfg->gamma,
        .lr = cfg->lr,
        .epsilon = cfg->mode == TFRL_MODE_TRAIN ? cfg->epsilon : 0.0f,
        .clip_eps = cfg->clip_eps,
        .steps_per_batch = cfg->steps_per_batch,
        .epochs = cfg->epochs,
        .save_path = cfg->save_path,
        .load_path = cfg->load_path,
    };
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
    if (cfg->render && env_count == 1) {
        tfrl_viewer_config vcfg = {.fps = cfg->render_fps, .title = "tinyfin-rl"};
        viewer = tfrl_viewer_create(&vcfg);
        if (!viewer) {
            fprintf(stderr, "viewer not available (build without raylib?)\n");
        } else {
            render_buf_len = tfrl_env_render_bytes_needed(envs[0]);
            render_buf = calloc(1, render_buf_len);
        }
    }

    tfrl_trace_writer *writer = NULL;
    if (cfg->trace_out) {
        writer = tfrl_trace_writer_open(cfg->trace_out);
        if (!writer) {
            fprintf(stderr, "failed to open trace out: %s\n", cfg->trace_out);
        } else {
            if (!render_buf) {
                render_buf_len = tfrl_env_render_bytes_needed(envs[0]);
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
        obs[i] = tfrl_env_reset(envs[i], (uint64_t)(cfg->seed + i));
    }

    if (cfg->mode == TFRL_MODE_TRAIN) {
        int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
        for (int step = 0; step < total_steps; step++) {
            for (int i = 0; i < env_count; i++) {
                actions[i] = algo.vtable->act(algo.ctx, obs[i]);
            }
            if (env_count == 1) {
                steps[0] = tfrl_env_step(envs[0], actions[0]);
            } else {
                tfrl_env_step_batch(envs, env_count, actions, steps);
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
                size_t written = tfrl_env_render_write(envs[0], render_buf, render_buf_len);
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
            obs[0] = tfrl_env_reset(envs[0], (uint64_t)cfg->seed);
            int done = 0;
            int step = 0;
            while (!done) {
                actions[0] = algo.vtable->act(algo.ctx, obs[0]);
                steps[0] = tfrl_env_step(envs[0], actions[0]);
                obs[0] = steps[0].observation;
                done = steps[0].done;

                if ((viewer || writer) && should_render(step, cfg->render_every)) {
                    size_t written = tfrl_env_render_write(envs[0], render_buf, render_buf_len);
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
