#define _POSIX_C_SOURCE 199309L

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tinyfin/tensor.h"
#include "tinyfin/nn.h"
#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

typedef int (*renderer_init_fn)(int, int, const char *);
typedef void (*renderer_draw_fn)(const tfrl_value *, double, int);
typedef int (*renderer_should_close_fn)(void);
typedef void (*renderer_close_fn)(void);
typedef const tfrl_env_metadata *(*env_metadata_get_fn)(void);
typedef const tfrl_env_plugin *(*plugin_get_fn)(void);

static const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i < argc - 1; i++) {
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

static int arg_int(int argc, char **argv, const char *name, int def) {
    const char *v = arg_str(argc, argv, name, NULL);
    return v ? atoi(v) : def;
}

static int file_exists(const char *path) {
    return path && access(path, R_OK) == 0;
}

static void sleep_us(int us) {
    if (us <= 0) return;
    struct timespec ts;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (long)(us % 1000000) * 1000L;
    nanosleep(&ts, NULL);
}

static void strip_newline(char *s) {
    if (!s) return;
    char *p = strchr(s, '\n');
    if (p) *p = '\0';
}

static void trim(char *s) {
    if (!s) return;
    char *end = s + strlen(s);
    while (end > s && (*(end - 1) == ' ' || *(end - 1) == '\t')) {
        *(--end) = '\0';
    }
    while (*s == ' ' || *s == '\t') {
        memmove(s, s + 1, strlen(s));
    }
}

static void load_viewer_conf(const char *path, int *grid_size, int *cell_size, int *action_n, const char **title) {
    FILE *f = fopen(path, "r");
    if (!f) return;
    static char title_buf[256];
    char line[256];
    while (fgets(line, sizeof(line), f)) {
        strip_newline(line);
        trim(line);
        if (line[0] == '\0' || line[0] == '#') continue;
        char *eq = strchr(line, '=');
        if (!eq) continue;
        *eq = '\0';
        char *key = line;
        char *val = eq + 1;
        trim(key);
        trim(val);
        if (strcmp(key, "grid_size") == 0) {
            *grid_size = atoi(val);
        } else if (strcmp(key, "cell_size") == 0) {
            *cell_size = atoi(val);
        } else if (strcmp(key, "action_n") == 0) {
            *action_n = atoi(val);
        } else if (strcmp(key, "title") == 0) {
            strncpy(title_buf, val, sizeof(title_buf) - 1);
            title_buf[sizeof(title_buf) - 1] = '\0';
            *title = title_buf;
        }
    }
    fclose(f);
}

static void derive_libs_from_env_dir(const char *env_arg, char *env_lib, size_t env_len, char *render_lib, size_t render_len, char *conf_path, size_t conf_len) {
    const char *base = strrchr(env_arg, '/');
    const char *name = base ? base + 1 : env_arg;
    char name_buf[128];
    strncpy(name_buf, name, sizeof(name_buf) - 1);
    name_buf[sizeof(name_buf) - 1] = '\0';
    for (char *p = name_buf; *p; p++) {
        if (*p == '-') *p = '_';
    }
    if (base) {
        snprintf(env_lib, env_len, "%s/lib%s.so", env_arg, name_buf);
        snprintf(render_lib, render_len, "%s/lib%s_render.so", env_arg, name_buf);
        snprintf(conf_path, conf_len, "%s/viewer.conf", env_arg);
    } else {
        snprintf(env_lib, env_len, "../%s/lib%s.so", env_arg, name_buf);
        snprintf(render_lib, render_len, "../%s/lib%s_render.so", env_arg, name_buf);
        snprintf(conf_path, conf_len, "../%s/viewer.conf", env_arg);
    }
}

static Tensor *one_hot(int n, int idx) {
    int shape[2] = {1, n};
    Tensor *t = tensor_zeros(2, shape);
    if (!t) return NULL;
    if (idx >= 0 && idx < n) {
        tensor_set_f32_at(t, (size_t)idx, 1.0f);
    }
    return t;
}

static int argmax_tensor(Tensor *t) {
    int best = 0;
    float best_v = tensor_get_f32_at(t, 0);
    for (size_t i = 1; i < t->size; i++) {
        float v = tensor_get_f32_at(t, i);
        if (v > best_v) {
            best_v = v;
            best = (int)i;
        }
    }
    return best;
}

static int load_linear(Linear *layer, const char *prefix, const char *tag) {
    if (!prefix || !tag) return 0;
    char path_w[256];
    char path_b[256];
    snprintf(path_w, sizeof(path_w), "%s.%s.w.tensor", prefix, tag);
    snprintf(path_b, sizeof(path_b), "%s.%s.b.tensor", prefix, tag);
    Tensor *w = tensor_load(path_w);
    Tensor *b = tensor_load(path_b);
    if (!w || !b) {
        if (w) tensor_free(w);
        if (b) tensor_free(b);
        return 0;
    }
    if (w->size == layer->weight->size) {
        memcpy(layer->weight->data, w->data, w->size * sizeof(float));
    }
    if (b->size == layer->bias->size) {
        memcpy(layer->bias->data, b->data, b->size * sizeof(float));
    }
    tensor_free(w);
    tensor_free(b);
    return 1;
}

int main(int argc, char **argv) {
    const char *env_lib = getenv("TFRL_ENV_PLUGIN_LIB");
    const char *render_lib = getenv("TFRL_RENDER_LIB");
    const char *title = getenv("TFRL_RENDER_TITLE");
    int grid_size = 10;
    int cell_size = 48;
    int action_n = 0;
    const char *grid_env = getenv("TFRL_RENDER_GRID_SIZE");
    const char *cell_env = getenv("TFRL_RENDER_CELL_SIZE");
    const char *action_env = getenv("TFRL_ACTION_N");
    if (grid_env) grid_size = atoi(grid_env);
    if (cell_env) cell_size = atoi(cell_env);
    if (action_env) action_n = atoi(action_env);

    const char *model_prefix = arg_str(argc, argv, "--load", NULL);
    int obs_n = arg_int(argc, argv, "--obs-n", 0);
    int steps = arg_int(argc, argv, "--steps", 0);
    int fps = arg_int(argc, argv, "--fps", 60);

    const char *env_dir_arg = NULL;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--env") == 0 && i + 1 < argc) {
            env_lib = argv[++i];
        } else if (strcmp(argv[i], "--render") == 0 && i + 1 < argc) {
            render_lib = argv[++i];
        } else if (strcmp(argv[i], "--grid-size") == 0 && i + 1 < argc) {
            grid_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cell-size") == 0 && i + 1 < argc) {
            cell_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--title") == 0 && i + 1 < argc) {
            title = argv[++i];
        } else if (strcmp(argv[i], "--action-n") == 0 && i + 1 < argc) {
            action_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            model_prefix = argv[++i];
        } else if (strcmp(argv[i], "--obs-n") == 0 && i + 1 < argc) {
            obs_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--steps") == 0 && i + 1 < argc) {
            steps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            fps = atoi(argv[++i]);
        } else if (argv[i][0] != '-' && !env_dir_arg) {
            env_dir_arg = argv[i];
        }
    }

    char env_buf[256];
    char render_buf[256];
    char conf_buf[256];
    if (!env_lib && !render_lib && env_dir_arg) {
        derive_libs_from_env_dir(env_dir_arg, env_buf, sizeof(env_buf), render_buf, sizeof(render_buf), conf_buf, sizeof(conf_buf));
        env_lib = env_buf;
        render_lib = render_buf;
        load_viewer_conf(conf_buf, &grid_size, &cell_size, &action_n, &title);
    }

    if (!env_lib || !render_lib || !model_prefix) {
        fprintf(stderr, "usage: %s <env_dir_or_name> --load PATH [--obs-n N] [--action-n N] [--steps N] [--fps N]\n", argv[0]);
        fprintf(stderr, "   or: %s --env <env_plugin.so> --render <render.so> --load PATH\n", argv[0]);
        return 1;
    }
    if (fps <= 0) fps = 60;
    int frame_us = 1000000 / fps;
    if (!file_exists(env_lib) || !file_exists(render_lib)) {
        fprintf(stderr, "missing shared libraries:\n");
        if (!file_exists(env_lib)) {
            fprintf(stderr, "  env plugin: %s\n", env_lib);
        }
        if (!file_exists(render_lib)) {
            fprintf(stderr, "  renderer:  %s\n", render_lib);
        }
        return 1;
    }

    void *env_handle = dlopen(env_lib, RTLD_NOW);
    if (!env_handle) {
        fprintf(stderr, "dlopen env failed: %s\n", dlerror());
        return 1;
    }
    plugin_get_fn plugin_get = (plugin_get_fn)dlsym(env_handle, "tfrl_env_plugin_get");
    if (!plugin_get) {
        fprintf(stderr, "missing tfrl_env_plugin_get\n");
        dlclose(env_handle);
        return 1;
    }
    const tfrl_env_plugin *plugin = plugin_get();
    tfrl_env env = plugin->create();

    env_metadata_get_fn meta_get = (env_metadata_get_fn)dlsym(env_handle, "tfrl_env_metadata_get");
    if (meta_get) {
        const tfrl_env_metadata *meta = meta_get();
        if (meta) {
            if (obs_n <= 0) obs_n = (int)meta->observation_n;
            if (action_n <= 0) action_n = (int)meta->action_n;
        }
    }
    if (obs_n <= 0 || action_n <= 0) {
        fprintf(stderr, "obs_n/action_n required\n");
        return 1;
    }

    void *render_handle = dlopen(render_lib, RTLD_NOW);
    if (!render_handle) {
        fprintf(stderr, "dlopen renderer failed: %s\n", dlerror());
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(env_handle);
        return 1;
    }
    renderer_init_fn renderer_init = (renderer_init_fn)dlsym(render_handle, "tfrl_renderer_init");
    renderer_draw_fn renderer_draw = (renderer_draw_fn)dlsym(render_handle, "tfrl_renderer_draw");
    renderer_should_close_fn renderer_should_close = (renderer_should_close_fn)dlsym(render_handle, "tfrl_renderer_should_close");
    renderer_close_fn renderer_close = (renderer_close_fn)dlsym(render_handle, "tfrl_renderer_close");
    if (!renderer_init || !renderer_draw || !renderer_should_close || !renderer_close) {
        fprintf(stderr, "renderer ABI missing required symbols\n");
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(render_handle);
        dlclose(env_handle);
        return 1;
    }

    if (!renderer_init(grid_size, cell_size, title ? title : "tinyfin-rl play")) {
        fprintf(stderr, "renderer_init failed\n");
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(render_handle);
        dlclose(env_handle);
        return 1;
    }

    Linear *q = linear_create(obs_n, action_n);
    if (!q) {
        fprintf(stderr, "failed to create model\n");
        renderer_close();
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(render_handle);
        dlclose(env_handle);
        return 1;
    }
    tensor_set_requires_grad(q->weight, 0);
    tensor_set_requires_grad(q->bias, 0);
    if (!load_linear(q, model_prefix, "q")) {
        fprintf(stderr, "failed to load model from %s\n", model_prefix);
    }

    tfrl_value obs = {0};
    tfrl_step step = {0};
    if (tfrl_env_reset(&env, &obs) != TFRL_OK) {
        fprintf(stderr, "env reset failed\n");
        renderer_close();
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(render_handle);
        dlclose(env_handle);
        return 1;
    }

    int step_count = 0;
    double reward = 0.0;
    int done = 0;
    while (!renderer_should_close()) {
        renderer_draw(&obs, reward, done);

        Tensor *x = one_hot(obs_n, (int)obs.as.i64);
        Tensor *q_values = linear_forward(q, x);
        int action = argmax_tensor(q_values);
        tensor_free(q_values);
        tensor_free(x);

        tfrl_value act = {.type = TFRL_VALUE_I64, .as.i64 = action};
        if (tfrl_env_step(&env, &act, &step) != TFRL_OK) {
            break;
        }
        obs = step.observation;
        reward = step.reward;
        done = tfrl_step_done(&step);
        if (done) {
            tfrl_env_reset(&env, &obs);
            reward = 0.0;
            done = 0;
        }
        if (steps > 0 && ++step_count >= steps) {
            break;
        }
        if (frame_us > 0) {
            sleep_us(frame_us);
        }
    }

    renderer_close();
    if (plugin->destroy) {
        plugin->destroy(&env);
    } else {
        tfrl_env_destroy(&env);
    }
    dlclose(render_handle);
    dlclose(env_handle);
    tensor_free(q->weight);
    tensor_free(q->bias);
    free(q);
    return 0;
}
