// SPDX-License-Identifier: Apache-2.0
//
// Device-side post-processing for the CUDA provider. Everything here runs on
// the caller's stream and never synchronizes; the host decides when to wait.
//
// Layouts: hidden [batch, seq, dim] f32, mask [batch, seq] i32 or i64 (the
// model's input width), out [batch, out_dim] with out_dim <= dim. Rows are
// summed in sequence order in one thread per dimension, matching the f32
// sequential reference so goldens hold to cosine ~1.0.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ void normalize_row(float *row, int dim) {
    // Block-wide sum of squares via shared memory reduction.
    __shared__ float partial[256];
    float ss = 0.f;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        ss += row[d] * row[d];
    }
    partial[threadIdx.x] = ss;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            partial[threadIdx.x] += partial[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float norm = sqrtf(partial[0]);
    const float inv = norm > 1e-12f ? 1.f / norm : 0.f;
    __syncthreads();
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        row[d] *= inv;
    }
}

template <typename M>
__global__ void pool_kernel(const float *__restrict__ hidden, const M *__restrict__ mask, float *__restrict__ out,
                            int batch, int seq, int dim, int out_dim, int mode, int normalize) {
    const int b = blockIdx.x;
    if (b >= batch) {
        return;
    }
    __shared__ int count;
    __shared__ int last;
    if (threadIdx.x == 0) {
        int c = 0, l = 0;
        for (int s = 0; s < seq; ++s) {
            if (mask[b * seq + s] != 0) {
                ++c;
                l = s;
            }
        }
        count = c;
        last = l;
    }
    __syncthreads();
    float *row = out + static_cast<size_t>(b) * out_dim;
    for (int d = threadIdx.x; d < out_dim; d += blockDim.x) {
        float v = 0.f;
        if (mode == 0) { // mean
            float acc = 0.f;
            for (int s = 0; s < seq; ++s) {
                if (mask[b * seq + s] != 0) {
                    acc += hidden[(static_cast<size_t>(b) * seq + s) * dim + d];
                }
            }
            v = count > 0 ? acc / static_cast<float>(count) : 0.f;
        } else if (mode == 1) { // cls
            v = hidden[(static_cast<size_t>(b) * seq) * dim + d];
        } else { // last
            v = hidden[(static_cast<size_t>(b) * seq + last) * dim + d];
        }
        row[d] = v;
    }
    __syncthreads();
    if (normalize) {
        normalize_row(row, out_dim);
    }
}

__global__ void sigmoid_kernel(const float *__restrict__ in, float *__restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = 1.f / (1.f + expf(-in[i]));
    }
}

// One block per row of `width` logits; width <= 4096.
__global__ void softmax_rows_kernel(const float *__restrict__ in, float *__restrict__ out, int rows, int width) {
    const int r = blockIdx.x;
    if (r >= rows) {
        return;
    }
    __shared__ float red[256];
    const float *x = in + static_cast<size_t>(r) * width;
    float *y = out + static_cast<size_t>(r) * width;
    float m = -INFINITY;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        m = fmaxf(m, x[i]);
    }
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    const float rowmax = red[0];
    __syncthreads();
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        const float e = expf(x[i] - rowmax);
        y[i] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            red[threadIdx.x] += red[threadIdx.x + s];
        }
        __syncthreads();
    }
    const float inv = 1.f / red[0];
    __syncthreads();
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        y[i] *= inv;
    }
}

int threads_for(int width) {
    int t = 32;
    while (t < width && t < 256) {
        t <<= 1;
    }
    return t;
}

} // namespace

extern "C" {

/// mode: 0 mean, 1 cls, 2 last. mask_width: 4 (i32) or 8 (i64). out_dim <= dim.
/// Returns cudaError_t as int (0 = success).
int turbo_cuda_pool(const float *hidden, const void *mask, int mask_width, float *out, int batch, int seq, int dim,
                    int out_dim, int mode, int normalize, cudaStream_t stream) {
    if (hidden == nullptr || mask == nullptr || out == nullptr || batch <= 0 || seq <= 0 || dim <= 0 || out_dim <= 0 ||
        out_dim > dim || mode < 0 || mode > 2 || (mask_width != 4 && mask_width != 8)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    if (mask_width == 8) {
        pool_kernel<int64_t><<<batch, 256, 0, stream>>>(hidden, static_cast<const int64_t *>(mask), out, batch, seq, dim,
                                                       out_dim, mode, normalize ? 1 : 0);
    } else {
        pool_kernel<int32_t><<<batch, 256, 0, stream>>>(hidden, static_cast<const int32_t *>(mask), out, batch, seq, dim,
                                                       out_dim, mode, normalize ? 1 : 0);
    }
    return static_cast<int>(cudaGetLastError());
}

int turbo_cuda_sigmoid(const float *in, float *out, int n, cudaStream_t stream) {
    if (in == nullptr || out == nullptr || n <= 0) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    const int threads = 256;
    sigmoid_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(in, out, n);
    return static_cast<int>(cudaGetLastError());
}

int turbo_cuda_softmax_rows(const float *in, float *out, int rows, int width, cudaStream_t stream) {
    if (in == nullptr || out == nullptr || rows <= 0 || width <= 0 || width > 4096) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    softmax_rows_kernel<<<rows, threads_for(width), 0, stream>>>(in, out, rows, width);
    return static_cast<int>(cudaGetLastError());
}

} // extern "C"
