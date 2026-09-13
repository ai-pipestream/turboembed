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

turbo_buffer_status ze_probe(turbo_buffer_placement placement);
void *ze_alloc(
    turbo_buffer_placement placement,
    size_t bytes,
    turbo_buffer_status *status
);
void ze_free(void *ptr);

turbo_buffer_status metal_probe(turbo_buffer_placement placement);
void *metal_alloc(size_t bytes, turbo_buffer_status *status);
void metal_free(void *ptr);

uint32_t elem_size(turbo_buffer_dtype dtype);
uint32_t aligned_row_stride(turbo_buffer_dtype dtype, uint32_t cols);

bool placement_ok(turbo_buffer_device device, turbo_buffer_placement placement);

} // namespace impl
} // namespace turbo_buffer
