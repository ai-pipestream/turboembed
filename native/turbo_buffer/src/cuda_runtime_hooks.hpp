// SPDX-License-Identifier: Apache-2.0
//
// Intercept cudaMalloc / cudaHostAlloc / cudaMemcpy* in CUDA TUs that
// must not grow PINNED or DEVICE slabs, or H2D token rows, during
// forward. Implement the real calls in cuda.cpp — do not include this
// header from that file.

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

cudaError_t turbo_buffer_cuda_memcpy(
    void *dst,
    const void *src,
    size_t count,
    enum cudaMemcpyKind kind
);

cudaError_t turbo_buffer_cuda_memcpy_async(
    void *dst,
    const void *src,
    size_t count,
    enum cudaMemcpyKind kind,
    cudaStream_t stream
);

cudaError_t turbo_buffer_cuda_memcpy2d(
    void *dst,
    size_t dpitch,
    const void *src,
    size_t spitch,
    size_t width,
    size_t height,
    enum cudaMemcpyKind kind
);

#ifdef __cplusplus
}
#endif

#ifdef TURBO_BUFFER_CUDA_INTERCEPT
#undef cudaMalloc
#undef cudaHostAlloc
#undef cudaMemcpy
#undef cudaMemcpyAsync
#undef cudaMemcpy2D
#define cudaMalloc(ptr, bytes) turbo_buffer_cuda_malloc((ptr), (bytes))
#define cudaHostAlloc(ptr, bytes, flags) \
    turbo_buffer_cuda_host_alloc((ptr), (bytes), (flags))
#define cudaMemcpy(dst, src, count, kind) \
    turbo_buffer_cuda_memcpy((dst), (src), (count), (kind))
#define cudaMemcpyAsync(dst, src, count, kind, stream) \
    turbo_buffer_cuda_memcpy_async((dst), (src), (count), (kind), (stream))
#define cudaMemcpy2D(dst, dpitch, src, spitch, width, height, kind) \
    turbo_buffer_cuda_memcpy2d(                                     \
        (dst), (dpitch), (src), (spitch), (width), (height), (kind) \
    )
#endif

#endif
