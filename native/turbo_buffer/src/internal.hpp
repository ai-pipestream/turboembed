// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "turbo_buffer.h"

#include <cstddef>
#include <string>

namespace turbo_buffer {
namespace impl {

void set_tls_error(const std::string &msg);
const char *tls_error();

void *cpu_alloc(size_t bytes, turbo_buffer_status *status);
void cpu_free(void *ptr);

turbo_buffer_status cuda_probe(turbo_buffer_placement placement);
void *cuda_alloc(
    turbo_buffer_placement placement,
    size_t bytes,
    turbo_buffer_status *status
);
void cuda_free(turbo_buffer_placement placement, void *ptr);

/** Increment the CUDA-forward malloc counter when a forward window is open. */
void note_cuda_forward_alloc();

/** True while TurboRerank CUDA forward has entered the alloc window. */
bool cuda_forward_window_open();

/**
 * If a CUDA forward window is open, count this runtime alloc on both
 * the process alloc counter and the forward-only counter.
 */
void note_cuda_runtime_alloc_if_forward();

turbo_buffer_status ze_probe(turbo_buffer_placement placement);
void *ze_alloc(
    turbo_buffer_placement placement,
    size_t bytes,
    turbo_buffer_status *status
);
void ze_free(void *ptr);
turbo_buffer_status ze_query(const void *ptr, turbo_buffer_placement *out);
turbo_buffer_status ze_memcpy(void *dst, const void *src, size_t bytes);

turbo_buffer_status metal_probe(turbo_buffer_placement placement);
void *metal_alloc(size_t bytes, turbo_buffer_status *status);
void metal_free(void *ptr);

uint32_t elem_size(turbo_buffer_dtype dtype);
uint32_t aligned_row_stride(turbo_buffer_dtype dtype, uint32_t cols);

bool placement_ok(turbo_buffer_device device, turbo_buffer_placement placement);

} // namespace impl
} // namespace turbo_buffer
