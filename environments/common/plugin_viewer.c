#define _POSIX_C_SOURCE 199309L

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "tinyfin_rl/rl.h"
#include "tinyfin_rl/rl_env_info.h"
#include "tinyfin_rl/rl_plugin.h"

typedef int (*renderer_init_fn)(int, int, const char *);
typedef void (*renderer_draw_fn)(const tfrl_value *, double, int);
typedef int (*renderer_should_close_fn)(void);
typedef void (*renderer_close_fn)(void);
typedef const tfrl_env_metadata *(*env_metadata_get_fn)(void);

static void usage(const char *name) {
    fprintf(stderr, "usage:\n");
    fprintf(stderr, "  %s <env_dir_or_name> [--fps N]\n", name);
    fprintf(stderr, "  %s --env <env_plugin.so> --render <render.so> [--grid-size N] [--cell-size N] [--title STR] [--fps N]\n", name);
    fprintf(stderr, "\n");
    fprintf(stderr, "examples:\n");
    fprintf(stderr, "  %s ../first-env\n", name);
    fprintf(stderr, "  %s --env ../first-env/libfirst_env.so --render ../first-env/libfirst_env_render.so\n", name);
    fprintf(stderr, "\n");
    fprintf(stderr, "env vars:\n");
    fprintf(stderr, "  TFRL_ENV_PLUGIN_LIB, TFRL_RENDER_LIB, TFRL_RENDER_GRID_SIZE,\n");
    fprintf(stderr, "  TFRL_RENDER_CELL_SIZE, TFRL_RENDER_TITLE, TFRL_ACTION_N, TFRL_RENDER_FPS\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "config:\n");
    fprintf(stderr, "  viewer.conf is read from the env dir when using <env_dir_or_name>\n");
}

static int file_exists(const char *path) {
    return path && access(path, R_OK) == 0;
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

static void load_viewer_conf(const char *path, int *grid_size, int *cell_size, int *action_n, int *fps, const char **title) {
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
        } else if (strcmp(key, "fps") == 0) {
            *fps = atoi(val);
        } else if (strcmp(key, "title") == 0) {
            strncpy(title_buf, val, sizeof(title_buf) - 1);
            title_buf[sizeof(title_buf) - 1] = '\0';
            *title = title_buf;
        }
    }
    fclose(f);
}

static void sleep_us(int us) {
    if (us <= 0) return;
    struct timespec ts;
    ts.tv_sec = us / 1000000;
    ts.tv_nsec = (long)(us % 1000000) * 1000L;
    nanosleep(&ts, NULL);
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

int main(int argc, char **argv) {
    const char *env_lib = getenv("TFRL_ENV_PLUGIN_LIB");
    const char *render_lib = getenv("TFRL_RENDER_LIB");
    const char *title = getenv("TFRL_RENDER_TITLE");
    int grid_size = 10;
    int cell_size = 48;
    int action_n = 4;
    int fps = 60;
    const char *grid_env = getenv("TFRL_RENDER_GRID_SIZE");
    const char *cell_env = getenv("TFRL_RENDER_CELL_SIZE");
    const char *action_env = getenv("TFRL_ACTION_N");
    const char *fps_env = getenv("TFRL_RENDER_FPS");
    if (grid_env) {
        grid_size = atoi(grid_env);
    }
    if (cell_env) {
        cell_size = atoi(cell_env);
    }
    if (action_env) {
        action_n = atoi(action_env);
    }
    if (fps_env) {
        fps = atoi(fps_env);
    }

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
        } else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) {
            fps = atoi(argv[++i]);
        } else if (argv[i][0] != '-' && !env_dir_arg) {
            env_dir_arg = argv[i];
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            return 0;
        }
    }

    char env_buf[256];
    char render_buf[256];
    char conf_buf[256];
    if (!env_lib && !render_lib && env_dir_arg) {
        derive_libs_from_env_dir(env_dir_arg, env_buf, sizeof(env_buf), render_buf, sizeof(render_buf), conf_buf, sizeof(conf_buf));
        env_lib = env_buf;
        render_lib = render_buf;
        load_viewer_conf(conf_buf, &grid_size, &cell_size, &action_n, &fps, &title);
    }

    if (!env_lib || !render_lib) {
        usage(argv[0]);
        return 1;
    }
    if (!file_exists(env_lib) || !file_exists(render_lib)) {
        fprintf(stderr, "missing shared libraries:\n");
        if (!file_exists(env_lib)) {
            fprintf(stderr, "  env plugin: %s\n", env_lib);
        }
        if (!file_exists(render_lib)) {
            fprintf(stderr, "  renderer:  %s\n", render_lib);
        }
        if (env_dir_arg) {
            fprintf(stderr, "hint: build with `make` in %s\n", env_dir_arg);
        }
        return 1;
    }

    void *env_handle = dlopen(env_lib, RTLD_NOW);
    if (!env_handle) {
        fprintf(stderr, "dlopen env failed: %s\n", dlerror());
        return 1;
    }
    tfrl_env_plugin_get_fn plugin_get = (tfrl_env_plugin_get_fn)dlsym(env_handle, "tfrl_env_plugin_get");
    if (!plugin_get) {
        fprintf(stderr, "dlsym plugin_get failed\n");
        dlclose(env_handle);
        return 1;
    }
    const tfrl_env_plugin *plugin = plugin_get();
    if (!plugin || !plugin->create) {
        fprintf(stderr, "invalid plugin\n");
        dlclose(env_handle);
        return 1;
    }
    tfrl_env env = plugin->create();
    env_metadata_get_fn meta_get = (env_metadata_get_fn)dlsym(env_handle, "tfrl_env_metadata_get");
    if (meta_get) {
        const tfrl_env_metadata *meta = meta_get();
        if (meta && meta->action_type == TFRL_SPACE_DISCRETE && meta->action_n > 0) {
            action_n = (int)meta->action_n;
        }
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

    if (!renderer_init(grid_size, cell_size, title ? title : "tinyfin-rl viewer")) {
        fprintf(stderr, "renderer_init failed\n");
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(render_handle);
        dlclose(env_handle);
        return 1;
    }

    srand((unsigned)time(NULL));
    tfrl_value obs = {0};
    tfrl_step step = {0};
    if (tfrl_env_reset(&env, &obs) != TFRL_OK) {
        renderer_close();
        if (plugin->destroy) {
            plugin->destroy(&env);
        }
        dlclose(render_handle);
        dlclose(env_handle);
        return 1;
    }

    if (fps <= 0) fps = 60;
    int frame_us = 1000000 / fps;
    double reward = 0.0;
    int done = 0;
    while (!renderer_should_close()) {
        renderer_draw(&obs, reward, done);
        int action = rand() % (action_n > 0 ? action_n : 4);
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
    return 0;
}
