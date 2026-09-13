// SPDX-License-Identifier: Apache-2.0
//
// Mask-weighted mean (or CLS) + optional L2 on DEVICE.
// Matches crates/backend-ort pool::{mean_pool, cls_pool, l2_normalize}
// in f32 sequential order so MiniLM goldens stay cosine ~1.0.
//
// Do not include turbo_buffer cuda_runtime_hooks.hpp here. This TU
// does not cudaMalloc or memcpy activations.

#include "pool_cuda.h"

#include <cuda_runtime.h>

namespace {

__global__ void mean_l2_kernel(
    const float *__restrict__ hidden,
    const int64_t *__restrict__ mask,
    float *__restrict__ out,
    int batch,
    int seq,
    int dim,
    int normalize
) {
    const int b = static_cast<int>(blockIdx.x);
    if (b >= batch) {
        return;
    }

    __shared__ float inv_count;
    if (threadIdx.x == 0) {
        float count = 0.f;
        for (int s = 0; s < seq; ++s) {
            if (mask[b * seq + s] != 0) {
                count += 1.f;
            }
        }
        inv_count = (count > 0.f) ? (1.f / count) : 0.f;
    }
    __syncthreads();

    for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
        float acc = 0.f;
        for (int s = 0; s < seq; ++s) {
            if (mask[b * seq + s] != 0) {
                acc += hidden[(b * seq + s) * dim + d];
            }
        }
        out[b * dim + d] = acc * inv_count;
    }
    __syncthreads();

    if (!normalize) {
        return;
    }

    __shared__ float inv_norm;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int d = 0; d < dim; ++d) {
            const float v = out[b * dim + d];
            ss += v * v;
        }
        const float norm = sqrtf(ss);
        inv_norm = (norm > 0.f) ? (1.f / norm) : 0.f;
    }
    __syncthreads();

    for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
        out[b * dim + d] *= inv_norm;
    }
}

__global__ void cls_l2_kernel(
    const float *__restrict__ hidden,
    float *__restrict__ out,
    int batch,
    int seq,
    int dim,
    int normalize
) {
    (void)seq;
    const int b = static_cast<int>(blockIdx.x);
    if (b >= batch) {
        return;
    }

    const float *src = hidden + static_cast<size_t>(b) * static_cast<size_t>(seq) * static_cast<size_t>(dim);
    for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
        out[b * dim + d] = src[d];
    }
    __syncthreads();

    if (!normalize) {
        return;
    }

    __shared__ float inv_norm;
    if (threadIdx.x == 0) {
        float ss = 0.f;
        for (int d = 0; d < dim; ++d) {
            const float v = out[b * dim + d];
            ss += v * v;
        }
        const float norm = sqrtf(ss);
        inv_norm = (norm > 0.f) ? (1.f / norm) : 0.f;
    }
    __syncthreads();

    for (int d = static_cast<int>(threadIdx.x); d < dim; d += static_cast<int>(blockDim.x)) {
        out[b * dim + d] *= inv_norm;
    }
}

int launch_rows(int batch, int dim) {
    int threads = 256;
    if (dim < threads) {
        threads = dim;
    }
    if (threads < 32) {
        threads = 32;
    }
    return threads;
}

} // namespace

extern "C" int turboembed_cuda_pool_mean_l2(
    const float *hidden_dev,
    const int64_t *mask_dev,
    float *out_dev,
    int batch,
    int seq,
    int dim,
    int normalize
) {
    if (hidden_dev == nullptr || mask_dev == nullptr || out_dev == nullptr ||
        batch <= 0 || seq <= 0 || dim <= 0) {
        return 1;
    }
    const int threads = launch_rows(batch, dim);
    mean_l2_kernel<<<batch, threads>>>(
        hidden_dev, mask_dev, out_dev, batch, seq, dim, normalize ? 1 : 0
    );
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        return static_cast<int>(launch);
    }
    return static_cast<int>(cudaDeviceSynchronize());
}

extern "C" int turboembed_cuda_pool_cls_l2(
    const float *hidden_dev,
    float *out_dev,
    int batch,
    int seq,
    int dim,
    int normalize
) {
    if (hidden_dev == nullptr || out_dev == nullptr || batch <= 0 || seq <= 0 ||
        dim <= 0) {
        return 1;
    }
    const int threads = launch_rows(batch, dim);
    cls_l2_kernel<<<batch, threads>>>(
        hidden_dev, out_dev, batch, seq, dim, normalize ? 1 : 0
    );
    const cudaError_t launch = cudaGetLastError();
    if (launch != cudaSuccess) {
        return static_cast<int>(launch);
    }
    return static_cast<int>(cudaDeviceSynchronize());
}
