#include "tfrl/env.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef tfrl_env *(*create_fn)(const tfrl_env_config *cfg);
typedef void (*destroy_fn)(tfrl_env *env);
typedef tfrl_obs (*reset_fn)(tfrl_env *env, uint64_t seed);
typedef tfrl_step_result (*step_fn)(tfrl_env *env, tfrl_action action);
typedef const tfrl_env_spec *(*spec_fn)(const tfrl_env *env);
typedef size_t (*render_bytes_fn)(const tfrl_env *env);
typedef size_t (*render_write_fn)(tfrl_env *env, void *buffer, size_t buffer_len);

static int fail(const char *msg) {
    fprintf(stderr, "test_env_modules: %s\n", msg);
    return 1;
}

static void *load_symbol(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    const char *err = dlerror();
    return err ? NULL : sym;
}

static int check_module(const char *path, const char *env_name) {
    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        fprintf(stderr, "test_env_modules: dlopen(%s): %s\n", path, dlerror());
        return 1;
    }

    create_fn create_env = (create_fn)load_symbol(handle, "tfrl_env_create");
    destroy_fn destroy_env = (destroy_fn)load_symbol(handle, "tfrl_env_destroy");
    reset_fn reset_env = (reset_fn)load_symbol(handle, "tfrl_env_reset");
    step_fn step_env = (step_fn)load_symbol(handle, "tfrl_env_step");
    spec_fn get_spec = (spec_fn)load_symbol(handle, "tfrl_env_get_spec");
    render_bytes_fn render_bytes = (render_bytes_fn)load_symbol(handle, "tfrl_env_render_bytes_needed");
    render_write_fn render_write = (render_write_fn)load_symbol(handle, "tfrl_env_render_write");
    if (!create_env || !destroy_env || !reset_env || !step_env || !get_spec ||
        !render_bytes || !render_write) {
        dlclose(handle);
        return fail("missing exported env ABI symbol");
    }

    tfrl_env_config cfg = {.name = env_name, .seed = 7};
    tfrl_env *env = create_env(&cfg);
    if (!env) {
        dlclose(handle);
        return fail("module could not create env");
    }

    const tfrl_env_spec *spec = get_spec(env);
    if (!spec || strcmp(spec->name, env_name) != 0) {
        destroy_env(env);
        dlclose(handle);
        return fail("module returned wrong env spec");
    }

    tfrl_obs obs = reset_env(env, 7);
    if (obs.data_len < 0 || obs.data_len > TFRL_MAX_BOX_DIMS) {
        destroy_env(env);
        dlclose(handle);
        return fail("module reset returned invalid observation");
    }

    tfrl_action action = {0};
    action.index = 0;
    tfrl_step_result step = step_env(env, action);
    if (step.observation.data_len < 0 || step.observation.data_len > TFRL_MAX_BOX_DIMS) {
        destroy_env(env);
        dlclose(handle);
        return fail("module step returned invalid observation");
    }

    size_t bytes = render_bytes(env);
    if (bytes == 0 || bytes > 1024 * 1024) {
        destroy_env(env);
        dlclose(handle);
        return fail("module render byte count is invalid");
    }
    void *buffer = calloc(1, bytes);
    if (!buffer) {
        destroy_env(env);
        dlclose(handle);
        return fail("could not allocate render buffer");
    }
    size_t written = render_write(env, buffer, bytes);
    free(buffer);
    destroy_env(env);
    dlclose(handle);
    if (written != bytes) return fail("module render write byte count mismatch");
    return 0;
}

int main(void) {
    if (check_module("build/libtfrl_env_lineworld.so", "lineworld")) return 1;
    if (check_module("build/libtfrl_env_maze_rooms.so", "maze_rooms")) return 1;
    if (check_module("build/libtfrl_env_coin_maze.so", "coin_maze")) return 1;
    return 0;
}
