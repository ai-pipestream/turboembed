// SPDX-License-Identifier: Apache-2.0
//
// Intercept cudaMalloc / cudaHostAlloc in CUDA TUs that must not grow
// PINNED or DEVICE slabs during forward. Implement the real calls in
// cuda.cpp — do not include this header from that file.

#pragma once

#ifdef TURBO_BUFFER_CUDA
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t turbo_buffer_cuda_malloc(void **ptr, size_t bytes);

cudaError_t turbo_buffer_cuda_host_alloc(
    void **ptr,
    size_t bytes,
    unsigned int flags
);

#ifdef __cplusplus
}
#endif

#ifdef TURBO_BUFFER_CUDA_INTERCEPT
#undef cudaMalloc
#undef cudaHostAlloc
#define cudaMalloc(ptr, bytes) turbo_buffer_cuda_malloc((ptr), (bytes))
#define cudaHostAlloc(ptr, bytes, flags) \
    turbo_buffer_cuda_host_alloc((ptr), (bytes), (flags))
#endif

#endif
