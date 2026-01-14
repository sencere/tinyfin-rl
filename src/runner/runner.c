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
    if (meta && meta[0] != '\0') {
        fprintf(stdout, "trace_meta: %s\n", meta);
    }
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
    int agent_count = tfrl_env_agent_count(envs[0]);
    if (agent_count <= 0) {
        fprintf(stderr, "env agent count invalid\n");
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }
    for (int i = 1; i < env_count; i++) {
        if (tfrl_env_agent_count(envs[i]) != agent_count) {
            fprintf(stderr, "env agent count mismatch\n");
            for (int j = 0; j < env_count; j++) {
                tfrl_env_destroy(envs[j]);
            }
            free(envs);
            return 1;
        }
    }
    if (cfg->agents_expected > 0 && cfg->agents_expected != agent_count) {
        fprintf(stderr, "env agent count mismatch (expected %d got %d)\n", cfg->agents_expected, agent_count);
        for (int i = 0; i < env_count; i++) {
            tfrl_env_destroy(envs[i]);
        }
        free(envs);
        return 1;
    }
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
        .kl_target = cfg->kl_target,
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
    int policy_count = cfg->share_policy ? 1 : agent_count;
    tfrl_algo algos[policy_count];
    char load_paths[policy_count][256];
    for (int a = 0; a < policy_count; a++) {
        tfrl_algo_config cfg_agent = algo_cfg;
        if (cfg->load_path && policy_count > 1) {
            snprintf(load_paths[a], sizeof(load_paths[a]), "%s.agent%d", cfg->load_path, a);
            cfg_agent.load_path = load_paths[a];
        }
        algos[a] = tfrl_algo_create(&cfg_agent, spec);
        if (!algos[a].ctx) {
            fprintf(stderr, "algo creation failed\n");
            for (int k = 0; k < a; k++) {
                tfrl_algo_destroy(&algos[k]);
            }
            for (int i = 0; i < env_count; i++) {
                tfrl_env_destroy(envs[i]);
            }
            free(envs);
            return 1;
        }
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
             "algo=%s defaults=v%d env=%s agents=%d share_policy=%d seed=%d deterministic=%d gamma=%.3f lr=%.6f epsilon=%.3f entropy=%.3f clip_eps=%.3f gae_lambda=%.3f "
             "replay=%d batch=%d per_alpha=%.3f per_beta=%.3f steps_per_batch=%d epochs=%d mcts_sims=%d mcts_depth=%d",
             algo_cfg.name ? algo_cfg.name : "null",
             algo_cfg.defaults_version,
             algo_cfg.env_name ? algo_cfg.env_name : "null",
             agent_count,
             cfg->share_policy ? 1 : 0,
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

    int total_agents = env_count * agent_count;
    tfrl_obs *obs = (tfrl_obs *)calloc((size_t)total_agents, sizeof(tfrl_obs));
    tfrl_action *actions = (tfrl_action *)calloc((size_t)total_agents, sizeof(tfrl_action));
    tfrl_step_result *steps = (tfrl_step_result *)calloc((size_t)total_agents, sizeof(tfrl_step_result));
    double *episode_returns = (double *)calloc((size_t)total_agents, sizeof(double));
    int *episode_counts = (int *)calloc((size_t)total_agents, sizeof(int));
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
        int got = tfrl_env_reset_multi(envs[i], (uint64_t)(seed + i), &obs[i * agent_count], agent_count);
        if (got != agent_count) {
            fprintf(stderr, "env reset failed\n");
            for (int a = 0; a < agent_count; a++) {
                tfrl_algo_destroy(&algos[a]);
            }
            for (int j = 0; j < env_count; j++) {
                tfrl_env_destroy(envs[j]);
            }
            free(envs);
            free(obs);
            free(actions);
            free(steps);
            free(episode_returns);
            free(episode_counts);
            return 1;
        }
    }

    if (cfg->mode == TFRL_MODE_TRAIN) {
        int total_steps = cfg->steps > 0 ? cfg->steps : 1000;
        int thread_count = cfg->threads > 0 ? cfg->threads : env_count;
        if (cfg->deterministic) thread_count = 1;
        tfrl_obs *obs_batch = (tfrl_obs *)calloc((size_t)env_count, sizeof(tfrl_obs));
        tfrl_action *act_batch = (tfrl_action *)calloc((size_t)env_count, sizeof(tfrl_action));
        if (!obs_batch || !act_batch) {
            fprintf(stderr, "allocation failed\n");
            free(obs_batch);
            free(act_batch);
            for (int a = 0; a < agent_count; a++) {
                tfrl_algo_destroy(&algos[a]);
            }
            for (int i = 0; i < env_count; i++) {
                tfrl_env_destroy(envs[i]);
            }
            free(envs);
            free(obs);
            free(actions);
            free(steps);
            free(episode_returns);
            free(episode_counts);
            return 1;
        }
        for (int step = 0; step < total_steps; step++) {
            for (int a = 0; a < agent_count; a++) {
                for (int i = 0; i < env_count; i++) {
                    obs_batch[i] = obs[i * agent_count + a];
                }
                int algo_idx = cfg->share_policy ? 0 : a;
                tfrl_algo_act_batch(&algos[algo_idx], obs_batch, env_count, act_batch);
                for (int i = 0; i < env_count; i++) {
                    actions[i * agent_count + a] = act_batch[i];
                }
            }
            for (int i = 0; i < env_count; i++) {
                int got = tfrl_env_step_multi(envs[i], &actions[i * agent_count], agent_count, &steps[i * agent_count], agent_count);
                if (got != agent_count) {
                    fprintf(stderr, "env step failed\n");
                    step = total_steps;
                    break;
                }
            }
            for (int i = 0; i < env_count; i++) {
                for (int a = 0; a < agent_count; a++) {
                    int idx = i * agent_count + a;
                    tfrl_transition transition = {
                        .obs = obs[idx],
                        .action = actions[idx],
                        .reward = steps[idx].reward,
                        .next_obs = steps[idx].observation,
                        .done = steps[idx].done,
                    };
                    int algo_idx = cfg->share_policy ? 0 : a;
                    algos[algo_idx].vtable->update(algos[algo_idx].ctx, &transition);
                    obs[idx] = steps[idx].observation;
                    episode_returns[idx] += steps[idx].reward;
                }
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
                int env_done = 0;
                for (int a = 0; a < agent_count; a++) {
                    int idx = i * agent_count + a;
                    if (steps[idx].done) env_done = 1;
                }
                if (env_done) {
                    for (int a = 0; a < agent_count; a++) {
                        int idx = i * agent_count + a;
                        episode_counts[idx] += 1;
                        if (cfg->log_every > 0 && (episode_counts[idx] % cfg->log_every) == 0) {
                            fprintf(stdout, "env=%d agent=%d episode=%d return=%.4f\n", i, a, episode_counts[idx], episode_returns[idx]);
                        }
                        episode_returns[idx] = 0.0;
                    }
                    int got = tfrl_env_reset_multi(envs[i], (uint64_t)(cfg->seed + i), &obs[i * agent_count], agent_count);
                    if (got != agent_count) {
                        fprintf(stderr, "env reset failed\n");
                        break;
                    }
                }
            }
            if (viewer && tfrl_viewer_should_close(viewer)) {
                break;
            }
        }
        if (env_count > 1 || agent_count > 1) {
            double sum = 0.0;
            int count = 0;
            for (int i = 0; i < total_agents; i++) {
                sum += episode_returns[i];
                count += episode_counts[i] > 0 ? episode_counts[i] : 0;
            }
            if (count > 0) {
                fprintf(stdout, "avg_return=%.4f episodes=%d\n", sum / (double)count, count);
            }
        }
        free(obs_batch);
        free(act_batch);
    } else if (cfg->mode == TFRL_MODE_EVAL) {
        int episodes = cfg->episodes > 0 ? cfg->episodes : 10;
        for (int ep = 0; ep < episodes; ep++) {
            if (tfrl_env_reset_multi(envs[0], (uint64_t)seed, obs, agent_count) != agent_count) {
                fprintf(stderr, "env reset failed\n");
                break;
            }
            int done = 0;
            int step = 0;
            while (!done) {
                for (int a = 0; a < agent_count; a++) {
                    int algo_idx = cfg->share_policy ? 0 : a;
                    actions[a] = algos[algo_idx].vtable->act(algos[algo_idx].ctx, obs[a]);
                }
                if (tfrl_env_step_multi(envs[0], actions, agent_count, steps, agent_count) != agent_count) {
                    fprintf(stderr, "env step failed\n");
                    done = 1;
                } else {
                    for (int a = 0; a < agent_count; a++) {
                        obs[a] = steps[a].observation;
                        if (steps[a].done) done = 1;
                    }
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
        if (policy_count == 1) {
            tfrl_algo_save(&algos[0], cfg->save_path);
        } else {
            for (int a = 0; a < policy_count; a++) {
                char path[256];
                snprintf(path, sizeof(path), "%s.agent%d", cfg->save_path, a);
                tfrl_algo_save(&algos[a], path);
            }
        }
    }
    for (int a = 0; a < policy_count; a++) {
        tfrl_algo_destroy(&algos[a]);
    }
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
