#ifndef TFRL_TRACE_H
#define TFRL_TRACE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef struct tfrl_trace_writer tfrl_trace_writer;
typedef struct tfrl_trace_reader tfrl_trace_reader;

tfrl_trace_writer *tfrl_trace_writer_open(const char *path);
void tfrl_trace_writer_close(tfrl_trace_writer *writer);
int tfrl_trace_writer_write(tfrl_trace_writer *writer, const void *buffer, size_t len);

tfrl_trace_reader *tfrl_trace_reader_open(const char *path);
void tfrl_trace_reader_close(tfrl_trace_reader *reader);
int tfrl_trace_reader_next(tfrl_trace_reader *reader, void *buffer, size_t buffer_len, size_t *out_len);

#ifdef __cplusplus
}
#endif

#endif
