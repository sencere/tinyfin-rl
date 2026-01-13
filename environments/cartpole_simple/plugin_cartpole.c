#include <math.h>
#include <stdlib.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

#define MAX_STEPS 200

typedef struct {
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    int steps;
    float obs[4];
} cart_env;

static void fill_obs(cart_env *env, tfrl_value *out_obs) {
    env->obs[0] = env->x;
    env->obs[1] = env->x_dot;
    env->obs[2] = env->theta;
    env->obs[3] = env->theta_dot;
    out_obs->type = TFRL_VALUE_BUFFER;
    out_obs->as.buffer.data = env->obs;
    out_obs->as.buffer.len = 4;
    out_obs->as.buffer.elem_size = sizeof(float);
}

static int cart_reset(void *ctx, tfrl_value *out_obs) {
    cart_env *env = (cart_env *)ctx;
    env->x = 0.0f;
    env->x_dot = 0.0f;
    env->theta = 0.0f;
    env->theta_dot = 0.0f;
    env->steps = 0;
    fill_obs(env, out_obs);
    return TFRL_OK;
}

static int cart_step(void *ctx, const tfrl_value *action, tfrl_step *out_step) {
    cart_env *env = (cart_env *)ctx;
    int act = (int)action->as.i64;
    float force = act == 1 ? 10.0f : -10.0f;
    float gravity = 9.8f;
    float masscart = 1.0f;
    float masspole = 0.1f;
    float total_mass = masscart + masspole;
    float length = 0.5f;
    float polemass_length = masspole * length;
    float tau = 0.02f;

    float cos_theta = cosf(env->theta);
    float sin_theta = sinf(env->theta);
    float temp = (force + polemass_length * env->theta_dot * env->theta_dot * sin_theta) / total_mass;
    float theta_acc = (gravity * sin_theta - cos_theta * temp) /
        (length * (4.0f / 3.0f - masspole * cos_theta * cos_theta / total_mass));
    float x_acc = temp - polemass_length * theta_acc * cos_theta / total_mass;

    env->x += tau * env->x_dot;
    env->x_dot += tau * x_acc;
    env->theta += tau * env->theta_dot;
    env->theta_dot += tau * theta_acc;
    env->steps += 1;

    int done = (env->x < -2.4f || env->x > 2.4f || env->theta < -0.209f || env->theta > 0.209f || env->steps >= MAX_STEPS);

    fill_obs(env, &out_step->observation);
    out_step->reward = done ? 0.0 : 1.0;
    out_step->terminated = done;
    out_step->truncated = env->steps >= MAX_STEPS;
    return TFRL_OK;
}

static void cart_destroy(void *ctx) {
    free(ctx);
}

static const tfrl_env_vtable CART_VTABLE = {
    .reset = cart_reset,
    .step = cart_step,
    .seed = NULL,
    .destroy = cart_destroy
};

static tfrl_env cart_create(void) {
    cart_env *env = (cart_env *)calloc(1, sizeof(cart_env));
    tfrl_env out = {env, &CART_VTABLE};
    return out;
}

static void cart_plugin_destroy(tfrl_env *env) {
    if (env && env->vtable && env->vtable->destroy) {
        env->vtable->destroy(env->ctx);
    }
}

static const tfrl_env_plugin CART_PLUGIN = {
    .api_version = TFRL_ENV_PLUGIN_API_VERSION,
    .name = "cartpole_simple",
    .create = cart_create,
    .destroy = cart_plugin_destroy
};

static const tfrl_env_metadata CART_METADATA = {
    .api_version = TFRL_ENV_INFO_API_VERSION,
    .name = "cartpole_simple",
    .observation_type = TFRL_SPACE_BOX,
    .action_type = TFRL_SPACE_DISCRETE,
    .observation_n = 0,
    .action_n = 2,
    .observation_dims = 4,
    .action_dims = 1,
    .observation_low = -2.4,
    .observation_high = 2.4,
    .action_low = 0.0,
    .action_high = 1.0
};

const tfrl_env_plugin *tfrl_env_plugin_get(void) {
    return &CART_PLUGIN;
}

const tfrl_env_metadata *tfrl_env_metadata_get(void) {
    return &CART_METADATA;
}
