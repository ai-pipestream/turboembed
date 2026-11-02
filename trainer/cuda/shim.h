/* SPDX-License-Identifier: Apache-2.0
 *
 * The training kernels (train.cu) compiled as plain C++ for the CPU, for
 * the `cuda-shim` feature: the same source, each launch run as nested
 * loops over its grid and its blocks' threads in order, device memory
 * as host memory. The kernels are written for it: no shared memory, no
 * barriers, no atomics, every output owned by one thread, so running
 * them one thread after another is running them. It is what CI, and a
 * machine without a GPU, test the kernels and the host path on; the
 * numbers on a device come from the same source under nvcc.
 */

#ifndef TURBO_TRAIN_SHIM_H
#define TURBO_TRAIN_SHIM_H

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __launch_bounds__(...)
#define __restrict__

struct dim3 {
    unsigned x, y, z;
    dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

/* The running launch's indices, set by shim_launch for each thread. */
inline dim3 gridDim, blockDim, blockIdx, threadIdx;

typedef int cudaError_t;
enum : int { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };

inline cudaError_t cudaMalloc(void **p, size_t bytes) {
    *p = std::malloc(bytes ? bytes : 1);
    return *p ? cudaSuccess : cudaErrorMemoryAllocation;
}
inline cudaError_t cudaFree(void *p) {
    std::free(p);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void *dst, const void *src, size_t bytes, cudaMemcpyKind) {
    std::memmove(dst, src, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaMemset(void *dst, int v, size_t bytes) {
    std::memset(dst, v, bytes);
    return cudaSuccess;
}
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline cudaError_t cudaSetDevice(int d) { return d == 0 ? cudaSuccess : cudaErrorInvalidValue; }
inline cudaError_t cudaGetDeviceCount(int *n) {
    *n = 1;
    return cudaSuccess;
}
inline const char *cudaGetErrorString(cudaError_t e) {
    return e == cudaSuccess ? "no error" : e == cudaErrorMemoryAllocation ? "out of memory" : "invalid value";
}

/* Runs `body` once per thread of a grid x block launch, in order. */
template <typename F> inline void shim_launch(dim3 grid, dim3 block, F body) {
    gridDim = grid;
    blockDim = block;
    for (blockIdx.z = 0; blockIdx.z < grid.z; blockIdx.z++)
        for (blockIdx.y = 0; blockIdx.y < grid.y; blockIdx.y++)
            for (blockIdx.x = 0; blockIdx.x < grid.x; blockIdx.x++)
                for (threadIdx.z = 0; threadIdx.z < block.z; threadIdx.z++)
                    for (threadIdx.y = 0; threadIdx.y < block.y; threadIdx.y++)
                        for (threadIdx.x = 0; threadIdx.x < block.x; threadIdx.x++) body();
}

#define TT_LAUNCH(kernel, grid, block, ...) shim_launch((grid), (block), [&] { kernel(__VA_ARGS__); })

#endif
