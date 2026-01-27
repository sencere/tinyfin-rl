#include "env_api.h"

int tfrl_obs_layout_total_len(const tfrl_obs_layout *layout) {
    if (!layout || layout->field_count <= 0 || !layout->fields) return 0;
    int total = 0;
    for (int i = 0; i < layout->field_count; i++) {
        int len = layout->fields[i].len;
        if (len < 0) return 0;
        total += len;
    }
    return total;
}

int tfrl_obs_layout_validate(const tfrl_obs_layout *layout) {
    if (!layout || layout->field_count <= 0 || !layout->fields) return 0;
    int total = 0;
    for (int i = 0; i < layout->field_count; i++) {
        const tfrl_obs_field *field = &layout->fields[i];
        if (field->len <= 0) return 0;
        if (field->dims < 0 || field->dims > 2) return 0;
        if (field->dims > 0) {
            int expected = 1;
            for (int d = 0; d < field->dims; d++) {
                if (field->shape[d] <= 0) return 0;
                expected *= field->shape[d];
            }
            if (expected != field->len) return 0;
        }
        total += field->len;
    }
    return total > 0 && total <= TFRL_MAX_BOX_DIMS;
}

int tfrl_obs_flatten(const tfrl_obs_layout *layout, const float *const *fields,
                     tfrl_obs *out_obs) {
    if (!layout || !fields || !out_obs) return 0;
    int total = tfrl_obs_layout_total_len(layout);
    if (total <= 0 || total > TFRL_MAX_BOX_DIMS) return 0;
    out_obs->index = 0;
    out_obs->data_len = total;
    int offset = 0;
    for (int i = 0; i < layout->field_count; i++) {
        const tfrl_obs_field *field = &layout->fields[i];
        const float *src = fields[i];
        if (!src) return 0;
        for (int j = 0; j < field->len; j++) {
            out_obs->data[offset + j] = src[j];
        }
        offset += field->len;
    }
    return 1;
}

int tfrl_obs_unflatten_copy(const tfrl_obs_layout *layout, const tfrl_obs *obs,
                            float *const *out_fields) {
    if (!layout || !obs || !out_fields) return 0;
    int total = tfrl_obs_layout_total_len(layout);
    if (total <= 0 || obs->data_len < total) return 0;
    int offset = 0;
    for (int i = 0; i < layout->field_count; i++) {
        const tfrl_obs_field *field = &layout->fields[i];
        float *dst = out_fields[i];
        if (!dst) return 0;
        for (int j = 0; j < field->len; j++) {
            dst[j] = obs->data[offset + j];
        }
        offset += field->len;
    }
    return 1;
}

const float *tfrl_obs_unflatten_field(const tfrl_obs_layout *layout, const tfrl_obs *obs,
                                      int field_index, int *out_len) {
    if (!layout || !obs || field_index < 0 || field_index >= layout->field_count) return NULL;
    int total = tfrl_obs_layout_total_len(layout);
    if (total <= 0 || obs->data_len < total) return NULL;
    int offset = 0;
    for (int i = 0; i < layout->field_count; i++) {
        const tfrl_obs_field *field = &layout->fields[i];
        if (i == field_index) {
            if (out_len) *out_len = field->len;
            return &obs->data[offset];
        }
        offset += field->len;
    }
    return NULL;
}
