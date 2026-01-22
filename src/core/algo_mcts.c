#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "algo_api.h"

typedef enum {
    MCTS_ENV_NONE = 0,
    MCTS_ENV_MAZE = 1,
    MCTS_ENV_LINEWORLD = 2,
    MCTS_ENV_SIM = 3,
} mcts_env_kind;

typedef struct {
    int action_n;
    int obs_n;
    float gamma;
    int sims;
    int max_depth;
    float c_uct;
    mcts_env_kind env;
    char env_name[64];
    tfrl_env *sim_env;
    void *state_buf;
    size_t state_size;
    int *visits;
    float *value_sum;
    int *child_visits;
    float *child_value_sum;
} tfrl_mcts_algo;

static int is_wall_maze(int x, int y) {
    if (x < 0 || x >= 10 || y < 0 || y >= 10) return 1;
    if (x == 4 && y != 2 && y != 7) return 1;
    if (y == 4 && x != 2 && x != 7) return 1;
    return 0;
}

static int maze_step(int state, int action, float *reward, int *done) {
    int x = state % 10;
    int y = state / 10;
    int nx = x;
    int ny = y;
    if (action == 0) ny -= 1;
    else if (action == 1) ny += 1;
    else if (action == 2) nx -= 1;
    else if (action == 3) nx += 1;
    if (!is_wall_maze(nx, ny)) {
        x = nx;
        y = ny;
    }
    int reached = (x == 9 && y == 9);
    if (reward) *reward = reached ? 1.0f : -0.01f;
    if (done) *done = reached;
    return y * 10 + x;
}

static int lineworld_step(int state, int action, float *reward, int *done) {
    int x = state;
    if (action == 0) x -= 1;
    else if (action == 1) x += 1;
    if (x < 0) x = 0;
    if (x > 6) x = 6;
    int reached = (x == 6);
    if (reward) *reward = reached ? 1.0f : -0.01f;
    if (done) *done = reached;
    return x;
}

static int env_step(const tfrl_mcts_algo *algo, int state, int action, float *reward, int *done) {
    if (algo->env == MCTS_ENV_MAZE) return maze_step(state, action, reward, done);
    if (algo->env == MCTS_ENV_LINEWORLD) return lineworld_step(state, action, reward, done);
    if (reward) *reward = 0.0f;
    if (done) *done = 1;
    return state;
}

static float rollout(tfrl_mcts_algo *algo, int state, int depth) {
    float total = 0.0f;
    float discount = 1.0f;
    for (int d = depth; d < algo->max_depth; d++) {
        int action = rand() % algo->action_n;
        float r = 0.0f;
        int done = 0;
        state = env_step(algo, state, action, &r, &done);
        total += discount * r;
        if (done) break;
        discount *= algo->gamma;
    }
    return total;
}

static float rollout_env(tfrl_mcts_algo *algo, tfrl_env *env, int depth) {
    float total = 0.0f;
    float discount = 1.0f;
    for (int d = depth; d < algo->max_depth; d++) {
        int action = rand() % algo->action_n;
        tfrl_action act = {.index = action};
        tfrl_step_result step = tfrl_env_step(env, act);
        total += discount * (float)step.reward;
        if (step.done) break;
        discount *= algo->gamma;
    }
    return total;
}

static int select_action(tfrl_mcts_algo *algo, int state) {
    int offset = state * algo->action_n;
    int best = 0;
    float best_score = -1e30f;
    float log_n = logf((float)(algo->visits[state] + 1));
    for (int a = 0; a < algo->action_n; a++) {
        int n = algo->child_visits[offset + a];
        if (n == 0) return a;
        float q = algo->child_value_sum[offset + a] / (float)n;
        float u = algo->c_uct * sqrtf(log_n / (float)n);
        float score = q + u;
        if (score > best_score) {
            best_score = score;
            best = a;
        }
    }
    return best;
}

static float simulate(tfrl_mcts_algo *algo, int state, int depth) {
    if (depth >= algo->max_depth) return 0.0f;
    if (algo->visits[state] == 0) {
        float r = rollout(algo, state, depth);
        algo->visits[state] += 1;
        algo->value_sum[state] += r;
        return r;
    }

    int action = select_action(algo, state);
    float r = 0.0f;
    int done = 0;
    int next = env_step(algo, state, action, &r, &done);
    float total = r;
    if (!done) {
        total += algo->gamma * simulate(algo, next, depth + 1);
    }

    int offset = state * algo->action_n + action;
    algo->child_visits[offset] += 1;
    algo->child_value_sum[offset] += total;
    algo->visits[state] += 1;
    algo->value_sum[state] += total;
    return total;
}

static float simulate_env(tfrl_mcts_algo *algo, tfrl_env *env, int state, int depth) {
    if (depth >= algo->max_depth) return 0.0f;
    if (state < 0 || state >= algo->obs_n) return 0.0f;
    if (algo->visits[state] == 0) {
        float r = rollout_env(algo, env, depth);
        algo->visits[state] += 1;
        algo->value_sum[state] += r;
        return r;
    }

    int action = select_action(algo, state);
    tfrl_action act = {.index = action};
    tfrl_step_result step = tfrl_env_step(env, act);
    int next = step.observation.index;
    float total = (float)step.reward;
    if (!step.done && next >= 0 && next < algo->obs_n) {
        total += algo->gamma * simulate_env(algo, env, next, depth + 1);
    }

    int offset = state * algo->action_n + action;
    algo->child_visits[offset] += 1;
    algo->child_value_sum[offset] += total;
    algo->visits[state] += 1;
    algo->value_sum[state] += total;
    return total;
}

static void reset_tree(tfrl_mcts_algo *algo) {
    size_t state_count = (size_t)algo->obs_n;
    size_t child_count = state_count * (size_t)algo->action_n;
    memset(algo->visits, 0, state_count * sizeof(int));
    memset(algo->value_sum, 0, state_count * sizeof(float));
    memset(algo->child_visits, 0, child_count * sizeof(int));
    memset(algo->child_value_sum, 0, child_count * sizeof(float));
}

static tfrl_action mcts_act(void *ctx, tfrl_obs obs) {
    tfrl_mcts_algo *algo = (tfrl_mcts_algo *)ctx;
    tfrl_action action = {0};
    if (!algo || algo->env == MCTS_ENV_NONE) return action;
    int state = obs.index;
    if (state < 0 || state >= algo->obs_n) return action;

    reset_tree(algo);
    for (int i = 0; i < algo->sims; i++) {
        simulate(algo, state, 0);
    }

    int offset = state * algo->action_n;
    int best = 0;
    int best_visits = -1;
    for (int a = 0; a < algo->action_n; a++) {
        int n = algo->child_visits[offset + a];
        if (n > best_visits) {
            best_visits = n;
            best = a;
        }
    }
    action.index = best;
    return action;
}

static tfrl_action mcts_act_env(void *ctx, tfrl_env *env, tfrl_obs obs) {
    tfrl_mcts_algo *algo = (tfrl_mcts_algo *)ctx;
    tfrl_action action = {0};
    if (!algo || !env) return action;
    if (algo->env != MCTS_ENV_SIM) return mcts_act(ctx, obs);
    int state = obs.index;
    if (state < 0 || state >= algo->obs_n) return action;

    if (!algo->sim_env) {
        tfrl_env_config cfg = {.name = algo->env_name, .seed = 0};
        algo->sim_env = tfrl_env_create(&cfg);
        if (!algo->sim_env) return action;
    }
    size_t size = tfrl_env_state_size(env);
    if (size == 0) return action;
    if (algo->state_size != size || !algo->state_buf) {
        free(algo->state_buf);
        algo->state_buf = calloc(1, size);
        algo->state_size = size;
        if (!algo->state_buf) return action;
    }
    if (!tfrl_env_state_save(env, algo->state_buf, algo->state_size)) return action;

    reset_tree(algo);
    for (int i = 0; i < algo->sims; i++) {
        if (!tfrl_env_state_load(algo->sim_env, algo->state_buf, algo->state_size)) break;
        simulate_env(algo, algo->sim_env, state, 0);
    }

    int offset = state * algo->action_n;
    int best = 0;
    int best_visits = -1;
    for (int a = 0; a < algo->action_n; a++) {
        int n = algo->child_visits[offset + a];
        if (n > best_visits) {
            best_visits = n;
            best = a;
        }
    }
    action.index = best;
    return action;
}

static void mcts_update(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
}

static void mcts_destroy(void *ctx) {
    tfrl_mcts_algo *algo = (tfrl_mcts_algo *)ctx;
    if (!algo) return;
    free(algo->visits);
    free(algo->value_sum);
    free(algo->child_visits);
    free(algo->child_value_sum);
    free(algo->state_buf);
    if (algo->sim_env) {
        tfrl_env_destroy(algo->sim_env);
    }
    free(algo);
}

static const tfrl_algo_vtable MCTS_VTABLE = {
    .act = mcts_act,
    .act_env = mcts_act_env,
    .act_batch = NULL,
    .update = mcts_update,
    .save = NULL,
    .destroy = mcts_destroy,
};

tfrl_algo tfrl_algo_mcts_create(const tfrl_algo_config *cfg) {
    tfrl_algo out = {0};
    if (!cfg || cfg->obs_n <= 0 || cfg->action_n <= 0) return out;
    if (cfg->obs_type != TFRL_SPACE_DISCRETE || cfg->action_type != TFRL_SPACE_DISCRETE) return out;

    tfrl_mcts_algo *algo = (tfrl_mcts_algo *)calloc(1, sizeof(tfrl_mcts_algo));
    if (!algo) return out;
    algo->obs_n = cfg->obs_n;
    algo->action_n = cfg->action_n;
    algo->gamma = cfg->gamma > 0.0f ? cfg->gamma : 0.99f;
    algo->sims = cfg->mcts_sims > 0 ? cfg->mcts_sims : 200;
    algo->max_depth = cfg->mcts_depth > 0 ? cfg->mcts_depth : 50;
    algo->c_uct = 1.4f;
    algo->env = MCTS_ENV_NONE;
    if (cfg->env_name && strcmp(cfg->env_name, "maze_rooms") == 0) {
        algo->env = MCTS_ENV_MAZE;
    } else if (cfg->env_name && strcmp(cfg->env_name, "lineworld") == 0) {
        algo->env = MCTS_ENV_LINEWORLD;
    } else if (cfg->env_name && strcmp(cfg->env_name, "arkanoid_disc") == 0) {
        algo->env = MCTS_ENV_SIM;
        snprintf(algo->env_name, sizeof(algo->env_name), "%s", cfg->env_name);
    }
    if (algo->env == MCTS_ENV_NONE) {
        free(algo);
        return out;
    }

    size_t state_count = (size_t)algo->obs_n;
    size_t child_count = state_count * (size_t)algo->action_n;
    algo->visits = (int *)calloc(state_count, sizeof(int));
    algo->value_sum = (float *)calloc(state_count, sizeof(float));
    algo->child_visits = (int *)calloc(child_count, sizeof(int));
    algo->child_value_sum = (float *)calloc(child_count, sizeof(float));
    if (!algo->visits || !algo->value_sum || !algo->child_visits || !algo->child_value_sum) {
        mcts_destroy(algo);
        return out;
    }

    out.ctx = algo;
    out.vtable = &MCTS_VTABLE;
    return out;
}
