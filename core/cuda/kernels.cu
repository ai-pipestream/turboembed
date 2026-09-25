/* SPDX-License-Identifier: Apache-2.0
 *
 * The BERT encoder's kernels: the embedding lookup, LayerNorm, GELU,
 * attention and pooling. The linear layers are cuBLAS's, called from the
 * host side. The arithmetic follows the CPU encoder (core/src/cpu/encoder.rs)
 * step for step where the order matters: LayerNorm's mean and variance are
 * summed in F64 and its scale and shift are rounded as two F32 operations;
 * softmax subtracts the largest live score; mean pooling sums each
 * dimension over the row's positions in order, then scales by 1 / count;
 * the L2 norm is summed in F64 and floored at 1e-12.
 */

#include "kernels.h"

#include <turbo/turbo.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>

namespace turbo_cuda {

namespace {

/* Threads per block for the row kernels: four warps. */
constexpr int BLOCK = 128;
constexpr int WARPS = BLOCK / 32;

__device__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;
}

__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
    return v;
}

/* The block's sum, in every thread. red holds WARPS values. */
template <typename T> __device__ T block_sum(T v, T *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    T total = 0;
    for (int w = 0; w < WARPS; w++) total += red[w];
    return total;
}

__device__ float block_max(float v, float *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float m = -INFINITY;
    for (int w = 0; w < WARPS; w++) m = fmaxf(m, red[w]);
    return m;
}

/* row = (row - mean) / sqrt(var + eps) * w + b, with the mean and the
 * biased variance summed in F64. Each thread touches only the columns it
 * wrote, so no barrier is needed before this. */
__device__ void layer_norm_row(float *row, int n, const float *w, const float *b, float eps) {
    __shared__ double red[WARPS];
    double s = 0;
    for (int d = threadIdx.x; d < n; d += BLOCK) s += row[d];
    const double mean = block_sum(s, red) / n;
    double v = 0;
    for (int d = threadIdx.x; d < n; d += BLOCK) {
        const double c = row[d] - mean;
        v += c * c;
    }
    const double var = block_sum(v, red) / n;
    const double inv = 1.0 / sqrt(var + (double)eps);
    for (int d = threadIdx.x; d < n; d += BLOCK) {
        const float x = (float)((row[d] - mean) * inv);
        row[d] = __fadd_rn(__fmul_rn(x, w[d]), b[d]);
    }
}

__global__ void __launch_bounds__(BLOCK)
    embed_layer_norm_kernel(const int32_t *ids, const int32_t *types, const float *word, const float *position,
                            const float *type, const float *ln_w, const float *ln_b, float eps, int seq, int hidden,
                            float *x) {
    const size_t t = blockIdx.x;
    const int p = (int)(t % seq);
    const float *w = word + (size_t)ids[t] * hidden;
    const float *ps = position + (size_t)p * hidden;
    const float *tt = type + (size_t)(types ? types[t] : 0) * hidden;
    float *row = x + t * hidden;
    for (int d = threadIdx.x; d < hidden; d += BLOCK) row[d] = w[d] + ps[d] + tt[d];
    layer_norm_row(row, hidden, ln_w, ln_b, eps);
}

__global__ void __launch_bounds__(BLOCK)
    add_layer_norm_kernel(float *x, const float *y, const float *bias, const float *ln_w, const float *ln_b, float eps,
                          int hidden) {
    const size_t t = blockIdx.x;
    float *row = x + t * hidden;
    const float *yr = y + t * hidden;
    for (int d = threadIdx.x; d < hidden; d += BLOCK) row[d] = row[d] + (yr[d] + bias[d]);
    layer_norm_row(row, hidden, ln_w, ln_b, eps);
}

__global__ void bias_gelu_kernel(float *y, const float *bias, size_t n, int width) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = y[i] + bias[i % width];
        y[i] = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
    }
}

/* One block per (query, head, row). Shared memory: the query's head, the
 * row's scores, and one partial context per group of threads. */
__global__ void __launch_bounds__(BLOCK)
    attention_kernel(const float *q, const float *k, const float *v, const float *bq, const float *bk, const float *bv,
                     const int32_t *mask, int seq, int hidden, int head_dim, float scale, float *ctx) {
    extern __shared__ float sm[];
    float *qi = sm;
    float *score = qi + head_dim;
    float *part = score + seq;
    __shared__ float red[WARPS];
    __shared__ float fred[WARPS];

    const int i = blockIdx.x, head = blockIdx.y, b = blockIdx.z;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = head * head_dim;
    const int32_t *m = mask + (size_t)b * seq;
    const size_t base = (size_t)b * seq;

    const float *qrow = q + (base + i) * hidden + col;
    for (int d = threadIdx.x; d < head_dim; d += BLOCK) qi[d] = qrow[d] + bq[col + d];
    __syncthreads();

    // A warp per key: its lanes split the head's width.
    for (int j = warp; j < seq; j += WARPS) {
        if (m[j] == 0) continue;
        const float *krow = k + (base + j) * hidden + col;
        float s = 0.0f;
        for (int d = lane; d < head_dim; d += 32) s += qi[d] * (krow[d] + bk[col + d]);
        s = warp_sum(s);
        if (lane == 0) score[j] = s * scale;
    }
    __syncthreads();

    float mx = -INFINITY;
    for (int j = threadIdx.x; j < seq; j += BLOCK)
        if (m[j] != 0) mx = fmaxf(mx, score[j]);
    mx = block_max(mx, red);
    float sum = 0.0f;
    for (int j = threadIdx.x; j < seq; j += BLOCK) {
        float e = 0.0f;
        if (m[j] != 0) {
            e = expf(score[j] - mx);
            sum += e;
        }
        score[j] = e;
    }
    sum = block_sum(sum, fred);
    const float inv = 1.0f / sum;
    __syncthreads();

    float *out = ctx + (base + i) * hidden + col;
    if (head_dim <= BLOCK) {
        // Groups of head_dim threads each take every groups-th key; the
        // first group adds the partial sums in group order.
        const int groups = BLOCK / head_dim;
        const int d = threadIdx.x % head_dim, g = threadIdx.x / head_dim;
        if (g < groups) {
            float c = 0.0f;
            for (int j = g; j < seq; j += groups) {
                if (m[j] == 0) continue;
                const float p = score[j] * inv;
                c += p * (v[(base + j) * hidden + col + d] + bv[col + d]);
            }
            part[g * head_dim + d] = c;
        }
        __syncthreads();
        if (threadIdx.x < head_dim) {
            float c = 0.0f;
            for (int gg = 0; gg < groups; gg++) c += part[gg * head_dim + threadIdx.x];
            out[threadIdx.x] = c;
        }
    } else {
        for (int d = threadIdx.x; d < head_dim; d += BLOCK) {
            float c = 0.0f;
            for (int j = 0; j < seq; j++) {
                if (m[j] == 0) continue;
                const float p = score[j] * inv;
                c += p * (v[(base + j) * hidden + col + d] + bv[col + d]);
            }
            out[d] = c;
        }
    }
}

__global__ void __launch_bounds__(BLOCK) pool_kernel(const float *x, const int32_t *mask, int seq, int hidden,
                                                     int output_dim, uint32_t pooling, int l2, float *out) {
    __shared__ double red[WARPS];
    __shared__ int last;
    const int b = blockIdx.x;
    const int32_t *m = mask + (size_t)b * seq;
    const float *rows = x + (size_t)b * seq * hidden;
    float *dst = out + (size_t)b * output_dim;
    if (threadIdx.x == 0) {
        int l = 0;
        for (int p = 0; p < seq; p++)
            if (m[p] != 0) l = p;
        last = l;
    }
    __syncthreads();
    double ss = 0;
    for (int d = threadIdx.x; d < output_dim; d += BLOCK) {
        float val;
        if (pooling == TURBO_POOLING_CLS) {
            val = rows[d];
        } else if (pooling == TURBO_POOLING_LAST) {
            val = rows[(size_t)last * hidden + d];
        } else {
            // Mean over the tokens whose mask is 1, summed in position order.
            float s = 0.0f;
            unsigned n = 0;
            for (int p = 0; p < seq; p++) {
                if (m[p] == 0) continue;
                s += rows[(size_t)p * hidden + d];
                n++;
            }
            val = s * (1.0f / (float)n);
        }
        dst[d] = val;
        ss += (double)val * (double)val;
    }
    if (!l2) return;
    const double norm = fmax(sqrt(block_sum(ss, red)), 1e-12);
    const float inv = (float)(1.0 / norm);
    for (int d = threadIdx.x; d < output_dim; d += BLOCK) dst[d] *= inv;
}

__global__ void widen_f16_kernel(const uint16_t *src, size_t n, float *dst) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __half2float(__ushort_as_half(src[i]));
}

__global__ void widen_bf16_kernel(const uint16_t *src, size_t n, float *dst) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __uint_as_float((unsigned)src[i] << 16);
}

/* Blocks for an elementwise pass over n values: enough to fill the
 * device, each thread striding over the rest. */
unsigned elementwise_blocks(size_t n) {
    const size_t want = (n + 255) / 256;
    return (unsigned)(want < 65535 ? (want ? want : 1) : 65535);
}

} // namespace

cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *ids, const int32_t *types, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             int tokens, int seq, int hidden, float *x) {
    embed_layer_norm_kernel<<<tokens, BLOCK, 0, s>>>(ids, types, word, position, type, ln_w, ln_b, eps, seq, hidden, x);
    return cudaGetLastError();
}

cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *y, const float *bias, const float *ln_w,
                           const float *ln_b, float eps, int tokens, int hidden) {
    add_layer_norm_kernel<<<tokens, BLOCK, 0, s>>>(x, y, bias, ln_w, ln_b, eps, hidden);
    return cudaGetLastError();
}

cudaError_t bias_gelu(cudaStream_t s, float *y, const float *bias, int tokens, int width) {
    const size_t n = (size_t)tokens * width;
    bias_gelu_kernel<<<elementwise_blocks(n), 256, 0, s>>>(y, bias, n, width);
    return cudaGetLastError();
}

size_t attention_shared_bytes(int seq, int hidden, int heads) {
    const int head_dim = hidden / heads;
    const int part = head_dim <= BLOCK ? (BLOCK / head_dim) * head_dim : 0;
    return sizeof(float) * ((size_t)head_dim + (size_t)seq + (size_t)part);
}

cudaError_t attention_max_shared(size_t *bytes) {
    int device = 0, optin = 0;
    cudaFuncAttributes a;
    cudaError_t e = cudaGetDevice(&device);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (e == cudaSuccess) e = cudaFuncGetAttributes(&a, attention_kernel);
    if (e == cudaSuccess) *bytes = (size_t)optin > a.sharedSizeBytes ? (size_t)optin - a.sharedSizeBytes : 0;
    return e;
}

cudaError_t attention_allow_shared() {
    size_t most = 0;
    const cudaError_t e = attention_max_shared(&most);
    if (e != cudaSuccess) return e;
    return cudaFuncSetAttribute(attention_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most);
}

cudaError_t attention(cudaStream_t s, const float *q, const float *k, const float *v, const float *bq, const float *bk,
                      const float *bv, const int32_t *mask, int batch, int seq, int hidden, int heads, float *ctx) {
    const int head_dim = hidden / heads;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const dim3 grid((unsigned)seq, (unsigned)heads, (unsigned)batch);
    attention_kernel<<<grid, BLOCK, attention_shared_bytes(seq, hidden, heads), s>>>(q, k, v, bq, bk, bv, mask, seq,
                                                                                     hidden, head_dim, scale, ctx);
    return cudaGetLastError();
}

cudaError_t pool(cudaStream_t s, const float *x, const int32_t *mask, int batch, int seq, int hidden, int output_dim,
                 uint32_t pooling, int l2, float *out) {
    pool_kernel<<<batch, BLOCK, 0, s>>>(x, mask, seq, hidden, output_dim, pooling, l2, out);
    return cudaGetLastError();
}

cudaError_t widen_f16(cudaStream_t s, const uint16_t *src, size_t n, float *dst) {
    widen_f16_kernel<<<elementwise_blocks(n), 256, 0, s>>>(src, n, dst);
    return cudaGetLastError();
}

cudaError_t widen_bf16(cudaStream_t s, const uint16_t *src, size_t n, float *dst) {
    widen_bf16_kernel<<<elementwise_blocks(n), 256, 0, s>>>(src, n, dst);
    return cudaGetLastError();
}

cudaError_t kernels_run_here() {
    cudaFuncAttributes a;
    return cudaFuncGetAttributes(&a, attention_kernel);
}

} // namespace turbo_cuda
