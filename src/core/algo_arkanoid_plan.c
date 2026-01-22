#include <stdlib.h>
#include <string.h>

#include "algo_api.h"

typedef struct {
    float align_eps;
    float launch_target;
} tfrl_arkanoid_plan_algo;

static int env_is_arkanoid(const tfrl_env_spec *spec) {
    if (!spec || !spec->name) return 0;
    return strncmp(spec->name, "arkanoid", 8) == 0;
}

static void arkanoid_extract_obs(const tfrl_env_spec *spec, const tfrl_obs *obs,
                                 float *player_x, float *landing_x,
                                 float *ball_active, float *ball_x,
                                 float *ball_y, float *ball_vx, float *ball_vy) {
    float px = 0.5f;
    float lx = 0.5f;
    float active = 1.0f;
    float bx = 0.5f;
    float by = 0.5f;
    float vx = 0.5f;
    float vy = 0.5f;

    if (spec && spec->obs_layout && obs) {
        int len = 0;
        const float *field = tfrl_obs_unflatten_field(spec->obs_layout, obs, 0, &len);
        if (field && len >= 1) px = field[0];
        field = tfrl_obs_unflatten_field(spec->obs_layout, obs, 1, &len);
        if (field && len >= 1) {
            bx = field[0];
            if (len >= 2) by = field[1];
        }
        field = tfrl_obs_unflatten_field(spec->obs_layout, obs, 2, &len);
        if (field && len >= 1) {
            vx = field[0];
            if (len >= 2) vy = field[1];
        }
        field = tfrl_obs_unflatten_field(spec->obs_layout, obs, 4, &len);
        if (field && len >= 1) active = field[0];
        field = tfrl_obs_unflatten_field(spec->obs_layout, obs, 7, &len);
        if (field && len >= 1) lx = field[0];
    } else if (obs) {
        if (obs->data_len >= 1) px = obs->data[0];
        if (obs->data_len >= 2) bx = obs->data[1];
        if (obs->data_len >= 3) by = obs->data[2];
        if (obs->data_len >= 4) vx = obs->data[3];
        if (obs->data_len >= 5) vy = obs->data[4];
        if (obs->data_len >= 7) active = obs->data[6];
        if (obs->data_len >= 10) lx = obs->data[9];
    }

    if (player_x) *player_x = px;
    if (landing_x) *landing_x = lx;
    if (ball_active) *ball_active = active;
    if (ball_x) *ball_x = bx;
    if (ball_y) *ball_y = by;
    if (ball_vx) *ball_vx = vx;
    if (ball_vy) *ball_vy = vy;
}

static tfrl_action arkanoid_plan_act_env(void *ctx, tfrl_env *env, tfrl_obs obs) {
    tfrl_arkanoid_plan_algo *algo = (tfrl_arkanoid_plan_algo *)ctx;
    tfrl_action action = {0};
    action.index = 0;
    const tfrl_env_spec *spec = env ? tfrl_env_get_spec(env) : NULL;
    if (!env_is_arkanoid(spec)) return action;

    float player_x = 0.5f;
    float landing_x = 0.5f;
    float ball_active = 1.0f;
    float ball_x = 0.5f;
    float ball_y = 0.5f;
    float ball_vx = 0.5f;
    float ball_vy = 0.5f;
    arkanoid_extract_obs(spec, &obs, &player_x, &landing_x, &ball_active, &ball_x,
                         &ball_y, &ball_vx, &ball_vy);

    float target_x = (ball_active > 0.5f) ? landing_x : ball_x;
    if (ball_active < 0.5f) {
        if (algo->launch_target <= 0.0f) {
            algo->launch_target = (player_x < 0.5f) ? 0.8f : 0.2f;
        }
        target_x = algo->launch_target;
    } else {
        float vx_mag = ball_vx - 0.5f;
        float vy_mag = ball_vy - 0.5f;
        if (vy_mag > 0.05f && ball_y > 0.55f) {
            target_x = ball_x;
        } else if ((vx_mag < 0.02f && vx_mag > -0.02f) && ball_y > 0.6f) {
            target_x = ball_x + (ball_x < 0.5f ? 0.1f : -0.1f);
        }
    }
    if (target_x < 0.0f || target_x > 1.0f) target_x = player_x;
    float delta = target_x - player_x;
    float eps = algo->align_eps;
    if (delta > eps) {
        action.index = 2;
    } else if (delta < -eps) {
        action.index = 1;
    } else if (ball_active < 0.5f) {
        action.index = 3;
        algo->launch_target = (algo->launch_target < 0.5f) ? 0.8f : 0.2f;
    }
    return action;
}

static void arkanoid_plan_update(void *ctx, const tfrl_transition *transition) {
    (void)ctx;
    (void)transition;
}

static void arkanoid_plan_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_algo_vtable ARKANOID_PLAN_VTABLE = {
    .act = NULL,
    .act_env = arkanoid_plan_act_env,
    .act_batch = NULL,
    .update = arkanoid_plan_update,
    .save = NULL,
    .destroy = arkanoid_plan_destroy,
    .diagnostics = NULL,
};

tfrl_algo tfrl_algo_arkanoid_plan_create(const tfrl_algo_config *cfg) {
    (void)cfg;
    tfrl_arkanoid_plan_algo *algo = (tfrl_arkanoid_plan_algo *)calloc(1, sizeof(tfrl_arkanoid_plan_algo));
    if (!algo) {
        tfrl_algo out = {0};
        return out;
    }
    algo->align_eps = 0.02f;
    algo->launch_target = 0.0f;
    tfrl_algo out = {algo, &ARKANOID_PLAN_VTABLE};
    return out;
}
