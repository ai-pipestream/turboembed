/* SPDX-License-Identifier: Apache-2.0
 *
 * The BERT encoder's kernels: the packing, the embedding lookup,
 * LayerNorm, GELU, attention and pooling. The linear layers are cuBLAS's,
 * called from the host side. The arithmetic follows the CPU encoder
 * (core/src/cpu/encoder.rs) step for step where the order matters:
 * LayerNorm takes the mean, then the biased variance about it, and its
 * scale and shift are rounded as two F32 operations (its sums are F32
 * here, F64 on the CPU: F64 runs at a small fraction of F32's rate on
 * most NVIDIA GPUs, and a row's F32 sum is far inside the F32 bound);
 * softmax subtracts the largest live
 * score, and each head's context sums its keys in position order; mean
 * pooling sums each dimension over the row's positions in order, then
 * scales by 1 / count; the L2 norm is summed in F64 and floored at 1e-12.
 * No reduction uses atomics, so the same rows give the same bits.
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
constexpr unsigned FULL = 0xffffffffu;

__device__ double warp_sum(double v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
    return v;
}

/* The warp's sum and largest value, in every lane. */
__device__ float warp_sum_all(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ float warp_max_all(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(FULL, v, o));
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

/* row = (row - mean) / sqrt(var + eps) * w + b, by one warp, with the
 * mean, then the biased variance about it, summed in F32 (two passes over
 * the row, which the lane wrote and reads back from its cache), and the
 * result in F16 into row16 too when it is not NULL. Each lane touches
 * only the columns it wrote, so no barrier is needed before this. */
__device__ void layer_norm_row(float *row, int n, const float *w, const float *b, float eps, __half *row16) {
    const int lane = threadIdx.x & 31;
    float s = 0.0f;
    for (int d = lane; d < n; d += 32) s += row[d];
    const float mean = warp_sum_all(s) / (float)n;
    float v = 0.0f;
    for (int d = lane; d < n; d += 32) {
        const float c = row[d] - mean;
        v += c * c;
    }
    const float var = warp_sum_all(v) / (float)n;
    const float inv = 1.0f / sqrtf(var + eps);
    for (int d = lane; d < n; d += 32) {
        const float x = (row[d] - mean) * inv;
        const float y = __fadd_rn(__fmul_rn(x, w[d]), b[d]);
        row[d] = y;
        if (row16) row16[d] = __float2half_rn(y);
    }
}

constexpr int PACK_BLOCK = 256;

/* One block: a warp per row finds its last live position, then each
 * thread sums the lengths of a run of rows, thread 0 scans the sums, and
 * each thread writes its rows' starts. Integer sums, in a fixed order. */
__global__ void __launch_bounds__(PACK_BLOCK)
    pack_rows_kernel(const int32_t *mask, int batch, int seq, int32_t *start, int32_t *len) {
    __shared__ int32_t part[PACK_BLOCK];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = warp; b < batch; b += PACK_BLOCK / 32) {
        const int32_t *m = mask + (size_t)b * seq;
        int last = -1;
        for (int p = lane; p < seq; p += 32)
            if (m[p] != 0) last = p;
        for (int o = 16; o > 0; o >>= 1) last = max(last, __shfl_xor_sync(FULL, last, o));
        if (lane == 0) len[b] = last + 1;
    }
    __syncthreads();
    const int chunk = (batch + PACK_BLOCK - 1) / PACK_BLOCK;
    const int lo = min(batch, (int)threadIdx.x * chunk), hi = min(batch, lo + chunk);
    int32_t sum = 0;
    for (int b = lo; b < hi; b++) sum += len[b];
    part[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t run = 0;
        for (int i = 0; i < PACK_BLOCK; i++) {
            const int32_t n = part[i];
            part[i] = run;
            run += n;
        }
    }
    __syncthreads();
    int32_t at = part[threadIdx.x];
    for (int b = lo; b < hi; b++) {
        start[b] = at;
        at += len[b];
    }
}

/* A warp per token: WARPS positions of one row per block. A warp past
 * the row's length leaves. */
__global__ void __launch_bounds__(BLOCK)
    embed_layer_norm_kernel(const int32_t *ids, const int32_t *types, const float *word, const float *position,
                            const float *type, const float *ln_w, const float *ln_b, float eps, const int32_t *start,
                            const int32_t *len, int seq, int hidden, float *x, __half *x16) {
    const int p = blockIdx.x * WARPS + (threadIdx.x >> 5), b = blockIdx.y, lane = threadIdx.x & 31;
    if (p >= len[b]) return;
    const size_t at = (size_t)b * seq + p;
    const size_t t = (size_t)start[b] + p;
    const float *w = word + (size_t)ids[at] * hidden;
    const float *ps = position + (size_t)p * hidden;
    const float *tt = type + (size_t)(types ? types[at] : 0) * hidden;
    float *row = x + t * hidden;
    for (int d = lane; d < hidden; d += 32) row[d] = w[d] + ps[d] + tt[d];
    layer_norm_row(row, hidden, ln_w, ln_b, eps, x16 ? x16 + t * hidden : nullptr);
}

/* A warp per token, WARPS tokens per block. */
__global__ void __launch_bounds__(BLOCK)
    add_layer_norm_kernel(float *x, const float *y, const float *bias, const float *ln_w, const float *ln_b, float eps,
                          int tokens, int hidden, __half *x16) {
    const size_t t = (size_t)blockIdx.x * WARPS + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (t >= (size_t)tokens) return;
    float *row = x + t * hidden;
    const float *yr = y + t * hidden;
    for (int d = lane; d < hidden; d += 32) row[d] = row[d] + (yr[d] + bias[d]);
    layer_norm_row(row, hidden, ln_w, ln_b, eps, x16 ? x16 + t * hidden : nullptr);
}

__global__ void bias_gelu_kernel(float *y, const float *bias, size_t n, int width, __half *y16) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = y[i] + bias[i % width];
        const float g = 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f));
        if (y16)
            y16[i] = __float2half_rn(g);
        else
            y[i] = g;
    }
}

/* Keys per chunk of K or V held in shared memory: one per lane. */
constexpr int KC = 32;
/* A head's width in values per lane, at the widest. */
constexpr int DPL = ATTENTION_MAX_HEAD_DIM / 32;

size_t round_to(size_t n, size_t a) { return (n + a - 1) / a * a; }

/* Dynamic shared memory for QPW queries per warp: the queries, one chunk
 * of keys or values padded by one column against bank conflicts, and a
 * row of scores per query. */
size_t attention_bytes(int qpw, int max_len, int head_dim) {
    const size_t qt = (size_t)qpw * WARPS;
    return sizeof(float) *
           (qt * head_dim + (size_t)KC * (head_dim + 1) + qt * round_to((size_t)(max_len > 0 ? max_len : 1), KC));
}

/* One block per (tile of QPW * WARPS queries, head, row), each warp
 * taking QPW of the queries. Keys and values go through shared memory a
 * chunk of KC at a time, up to the row's length: a lane scores one key of
 * the chunk against each of its warp's queries, then each warp takes its
 * queries' softmax, then a lane sums the chunk's values for a column of
 * the head, key by key in position order. */
template <int QPW>
__global__ void __launch_bounds__(BLOCK)
    attention_kernel(const float *q, const float *k, const float *v, int ld, const float *bq, const float *bk,
                     const float *bv, const int32_t *mask, const int32_t *start, const int32_t *len, int seq,
                     int hidden, int head_dim, int lpad, float scale, float *ctx, __half *ctx16) {
    constexpr int QT = QPW * WARPS;
    extern __shared__ float sm[];
    const int d = head_dim, dp = head_dim + 1;
    float *qs = sm;
    float *kv = qs + QT * d;
    float *sc = kv + KC * dp;

    const int b = blockIdx.z, head = blockIdx.y, q0 = blockIdx.x * QT;
    const int n = len[b];
    if (q0 >= n) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int col = head * d;
    const size_t base = (size_t)start[b];
    const int32_t *m = mask + (size_t)b * seq;
    const int nq = min(QT, n - q0);

    for (int i = threadIdx.x; i < QT * d; i += BLOCK) {
        const int qi = i / d, c = i - qi * d;
        qs[i] = qi < nq ? q[(base + q0 + qi) * ld + col + c] + bq[col + c] : 0.0f;
    }

    // Scores, a chunk of keys at a time.
    for (int c0 = 0; c0 < n; c0 += KC) {
        __syncthreads();
        for (int i = threadIdx.x; i < KC * d; i += BLOCK) {
            const int j = i / d, c = i - j * d;
            kv[j * dp + c] = c0 + j < n ? k[(base + c0 + j) * ld + col + c] + bk[col + c] : 0.0f;
        }
        __syncthreads();
        const float *kr = kv + lane * dp;
#pragma unroll
        for (int w = 0; w < QPW; w++) {
            const int qi = warp * QPW + w;
            const float *qr = qs + qi * d;
            float s = 0.0f;
            for (int c = 0; c < d; c++) s += qr[c] * kr[c];
            sc[qi * lpad + c0 + lane] = s * scale;
        }
    }
    __syncwarp();

    // Softmax over the live keys, from the largest score; each warp its
    // own queries. The probabilities are e * inv, as on the CPU.
    float inv[QPW];
#pragma unroll
    for (int w = 0; w < QPW; w++) {
        float *r = sc + (warp * QPW + w) * lpad;
        float mx = -INFINITY;
        for (int j = lane; j < n; j += 32)
            if (m[j] != 0) mx = fmaxf(mx, r[j]);
        mx = warp_max_all(mx);
        float sum = 0.0f;
        for (int j = lane; j < n; j += 32) {
            float e = 0.0f;
            if (m[j] != 0) {
                e = expf(r[j] - mx);
                sum += e;
            }
            r[j] = e;
        }
        inv[w] = 1.0f / warp_sum_all(sum);
    }
    __syncwarp();

    // The context, a chunk of values at a time, keys in position order.
    float acc[QPW][DPL];
#pragma unroll
    for (int w = 0; w < QPW; w++)
#pragma unroll
        for (int u = 0; u < DPL; u++) acc[w][u] = 0.0f;
    for (int c0 = 0; c0 < n; c0 += KC) {
        __syncthreads();
        for (int i = threadIdx.x; i < KC * d; i += BLOCK) {
            const int j = i / d, c = i - j * d;
            kv[j * dp + c] = c0 + j < n ? v[(base + c0 + j) * ld + col + c] + bv[col + c] : 0.0f;
        }
        __syncthreads();
        const int cn = min(KC, n - c0);
        for (int j = 0; j < cn; j++) {
            if (m[c0 + j] == 0) continue;
            const float *vr = kv + j * dp;
#pragma unroll
            for (int w = 0; w < QPW; w++) {
                const float p = sc[(warp * QPW + w) * lpad + c0 + j] * inv[w];
#pragma unroll
                for (int u = 0; u < DPL; u++) {
                    const int c = lane + 32 * u;
                    if (c < d) acc[w][u] += p * vr[c];
                }
            }
        }
    }

#pragma unroll
    for (int w = 0; w < QPW; w++) {
        const int qi = warp * QPW + w;
        if (qi >= nq) continue;
        const size_t at = (base + q0 + qi) * hidden + col;
#pragma unroll
        for (int u = 0; u < DPL; u++) {
            const int c = lane + 32 * u;
            if (c >= d) continue;
            if (ctx16)
                ctx16[at + c] = __float2half_rn(acc[w][u]);
            else
                ctx[at + c] = acc[w][u];
        }
    }
}

/* Queries per warp: four where their scores fit the shared memory, else one. */
constexpr int WIDE = 4;
constexpr int NARROW = 1;

__global__ void __launch_bounds__(BLOCK)
    pool_kernel(const float *x, const int32_t *mask, const int32_t *start, const int32_t *len, int seq, int hidden,
                int output_dim, uint32_t pooling, int l2, float *out) {
    __shared__ double red[WARPS];
    const int b = blockIdx.x;
    const int32_t *m = mask + (size_t)b * seq;
    const int n = len[b];
    const float *rows = x + (size_t)start[b] * hidden;
    float *dst = out + (size_t)b * output_dim;
    double ss = 0;
    for (int d = threadIdx.x; d < output_dim; d += BLOCK) {
        float val;
        if (pooling == TURBO_POOLING_CLS) {
            val = rows[d];
        } else if (pooling == TURBO_POOLING_LAST) {
            // The row's length ends at its last live token.
            val = rows[(size_t)(n - 1) * hidden + d];
        } else {
            // Mean over the tokens whose mask is 1, summed in position order.
            float s = 0.0f;
            unsigned c = 0;
            for (int p = 0; p < n; p++) {
                if (m[p] == 0) continue;
                s += rows[(size_t)p * hidden + d];
                c++;
            }
            val = s * (1.0f / (float)c);
        }
        dst[d] = val;
        ss += (double)val * (double)val;
    }
    if (!l2) return;
    const double norm = fmax(sqrt(block_sum(ss, red)), 1e-12);
    const float scale = (float)(1.0 / norm);
    for (int d = threadIdx.x; d < output_dim; d += BLOCK) dst[d] *= scale;
}

__global__ void widen_f16_kernel(const uint16_t *src, size_t n, float *dst) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __half2float(__ushort_as_half(src[i]));
}

__global__ void widen_bf16_kernel(const uint16_t *src, size_t n, float *dst) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __uint_as_float((unsigned)src[i] << 16);
}

/* Every thread that sees an overflow writes the same 1. */
__global__ void narrow_f16_kernel(const float *src, size_t n, uint16_t *dst, int32_t *overflow) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float f = src[i];
        const __half h = __float2half_rn(f);
        if (__hisinf(h) && isfinite(f)) *overflow = 1;
        dst[i] = __half_as_ushort(h);
    }
}

/* Blocks for an elementwise pass over n values: enough to fill the
 * device, each thread striding over the rest. */
unsigned elementwise_blocks(size_t n) {
    const size_t want = (n + 255) / 256;
    return (unsigned)(want < 65535 ? (want ? want : 1) : 65535);
}

__half *as_half(uint16_t *p) { return reinterpret_cast<__half *>(p); }

} // namespace

cudaError_t pack_rows(cudaStream_t s, const int32_t *mask, int batch, int seq, int32_t *start, int32_t *len) {
    pack_rows_kernel<<<1, PACK_BLOCK, 0, s>>>(mask, batch, seq, start, len);
    return cudaGetLastError();
}

cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *ids, const int32_t *types, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             const int32_t *start, const int32_t *len, int batch, int max_len, int seq, int hidden,
                             float *x, uint16_t *x16) {
    const dim3 grid((unsigned)((max_len + WARPS - 1) / WARPS), (unsigned)batch);
    embed_layer_norm_kernel<<<grid, BLOCK, 0, s>>>(ids, types, word, position, type, ln_w, ln_b, eps, start, len, seq,
                                                   hidden, x, as_half(x16));
    return cudaGetLastError();
}

cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *y, const float *bias, const float *ln_w,
                           const float *ln_b, float eps, int tokens, int hidden, uint16_t *x16) {
    add_layer_norm_kernel<<<(tokens + WARPS - 1) / WARPS, BLOCK, 0, s>>>(x, y, bias, ln_w, ln_b, eps, tokens, hidden,
                                                                        as_half(x16));
    return cudaGetLastError();
}

cudaError_t bias_gelu(cudaStream_t s, float *y, const float *bias, int tokens, int width, uint16_t *y16) {
    const size_t n = (size_t)tokens * width;
    bias_gelu_kernel<<<elementwise_blocks(n), 256, 0, s>>>(y, bias, n, width, as_half(y16));
    return cudaGetLastError();
}

size_t attention_shared_bytes(int seq, int hidden, int heads) { return attention_bytes(NARROW, seq, hidden / heads); }

cudaError_t attention_max_shared(size_t *bytes) {
    int device = 0, optin = 0;
    cudaFuncAttributes wide, narrow;
    cudaError_t e = cudaGetDevice(&device);
    if (e == cudaSuccess) e = cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (e == cudaSuccess) e = cudaFuncGetAttributes(&wide, attention_kernel<WIDE>);
    if (e == cudaSuccess) e = cudaFuncGetAttributes(&narrow, attention_kernel<NARROW>);
    if (e == cudaSuccess) {
        const size_t fixed = wide.sharedSizeBytes > narrow.sharedSizeBytes ? wide.sharedSizeBytes
                                                                           : narrow.sharedSizeBytes;
        *bytes = (size_t)optin > fixed ? (size_t)optin - fixed : 0;
    }
    return e;
}

cudaError_t attention_allow_shared() {
    size_t most = 0;
    cudaError_t e = attention_max_shared(&most);
    if (e == cudaSuccess)
        e = cudaFuncSetAttribute(attention_kernel<WIDE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most);
    if (e == cudaSuccess)
        e = cudaFuncSetAttribute(attention_kernel<NARROW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most);
    return e;
}

cudaError_t attention(cudaStream_t s, const float *q, const float *k, const float *v, int ld, const float *bq,
                      const float *bk, const float *bv, const int32_t *mask, const int32_t *start, const int32_t *len,
                      int batch, int max_len, int seq, int hidden, int heads, size_t shared_most, float *ctx,
                      uint16_t *ctx16) {
    const int head_dim = hidden / heads;
    if (head_dim > ATTENTION_MAX_HEAD_DIM) return cudaErrorInvalidValue;
    const float scale = 1.0f / sqrtf((float)head_dim);
    const int lpad = (int)round_to((size_t)max_len, KC);
    const size_t wide = attention_bytes(WIDE, max_len, head_dim);
    if (wide <= shared_most) {
        const int qt = WIDE * WARPS;
        const dim3 grid((unsigned)((max_len + qt - 1) / qt), (unsigned)heads, (unsigned)batch);
        attention_kernel<WIDE><<<grid, BLOCK, wide, s>>>(q, k, v, ld, bq, bk, bv, mask, start, len, seq, hidden,
                                                         head_dim, lpad, scale, ctx, as_half(ctx16));
    } else {
        const int qt = NARROW * WARPS;
        const dim3 grid((unsigned)((max_len + qt - 1) / qt), (unsigned)heads, (unsigned)batch);
        attention_kernel<NARROW><<<grid, BLOCK, attention_bytes(NARROW, max_len, head_dim), s>>>(
            q, k, v, ld, bq, bk, bv, mask, start, len, seq, hidden, head_dim, lpad, scale, ctx, as_half(ctx16));
    }
    return cudaGetLastError();
}

cudaError_t pool(cudaStream_t s, const float *x, const int32_t *mask, const int32_t *start, const int32_t *len,
                 int batch, int seq, int hidden, int output_dim, uint32_t pooling, int l2, float *out) {
    pool_kernel<<<batch, BLOCK, 0, s>>>(x, mask, start, len, seq, hidden, output_dim, pooling, l2, out);
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

cudaError_t narrow_f16(cudaStream_t s, const float *src, size_t n, uint16_t *dst, int32_t *overflow) {
    narrow_f16_kernel<<<elementwise_blocks(n), 256, 0, s>>>(src, n, dst, overflow);
    return cudaGetLastError();
}

cudaError_t kernels_run_here() {
    cudaFuncAttributes a;
    return cudaFuncGetAttributes(&a, attention_kernel<WIDE>);
}

} // namespace turbo_cuda
