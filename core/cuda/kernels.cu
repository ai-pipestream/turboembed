/* SPDX-License-Identifier: Apache-2.0
 *
 * The BERT encoder's kernels: the packing, the embedding lookup, the
 * linear layers with what follows each fused into them, attention,
 * LayerNorm and pooling. The arithmetic follows the CPU encoder
 * (core/src/cpu/encoder.rs) step for step where the order matters:
 * LayerNorm takes the mean, then the biased variance about it, and its
 * scale and shift are rounded as two F32 operations (its sums are F32
 * here, F64 on the CPU: F64 runs at a small fraction of F32's rate on
 * most NVIDIA GPUs, and a row's F32 sum is far inside the F32 bound);
 * softmax subtracts the largest live score (a running one, carried from
 * one block of keys to the next and rescaled, as flash attention does);
 * mean pooling sums each dimension over the row's positions in order,
 * then scales by 1 / count; the L2 norm is summed in F32 and floored at
 * 1e-12. No reduction uses atomics, and every sum is taken in an order
 * fixed by the shapes, the run's token count and the blocks the device
 * holds at once, so the same rows in the same batch on the same device
 * give the same bits.
 */

#include "kernels.h"

#include <turbo/turbo.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <type_traits>
#include <utility>

/* mma.sync m16n8k16 with F16 inputs and cp.async are sm_80's. A build for
 * an older architecture carries the kernels that use them as stubs, which
 * the host side never launches there. */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#define TURBO_NO_MMA 1
#endif

namespace turbo_cuda {

namespace {

constexpr unsigned FULL = 0xffffffffu;

/* The warp's sum and largest value, in every lane. */
__device__ float warp_sum_all(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(FULL, v, o);
    return v;
}

__device__ int warp_max_all(int v) {
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(FULL, v, o));
    return v;
}

__device__ inline float to_float(float v) { return v; }
__device__ inline float to_float(__half v) { return __half2float(v); }

__device__ inline void put(float *p, float v) { *p = v; }
__device__ inline void put(__half *p, float v) { *p = __float2half_rn(v); }

/* Four consecutive values, 16 bytes of F32 or 8 of F16, widened. */
__device__ inline void load4(const float *p, float (&v)[4]) {
    const float4 x = __ldg(reinterpret_cast<const float4 *>(p));
    v[0] = x.x;
    v[1] = x.y;
    v[2] = x.z;
    v[3] = x.w;
}

__device__ inline void load4(const __half *p, float (&v)[4]) {
    const uint2 u = __ldg(reinterpret_cast<const uint2 *>(p));
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
    v[0] = a.x;
    v[1] = a.y;
    v[2] = b.x;
    v[3] = b.y;
}

/* W consecutive values stored at once: 2 or 4, F32 or rounded to F16. */
template <int W> __device__ inline void store_vec(float *p, const float (&v)[W]) {
    if constexpr (W == 4)
        *reinterpret_cast<float4 *>(p) = make_float4(v[0], v[1], v[2], v[3]);
    else
        *reinterpret_cast<float2 *>(p) = make_float2(v[0], v[1]);
}

template <int W> __device__ inline void store_vec(__half *p, const float (&v)[W]) {
    const __half2 a = __floats2half2_rn(v[0], v[1]);
    if constexpr (W == 4) {
        const __half2 b = __floats2half2_rn(v[2], v[3]);
        uint2 u;
        u.x = *reinterpret_cast<const unsigned *>(&a);
        u.y = *reinterpret_cast<const unsigned *>(&b);
        *reinterpret_cast<uint2 *>(p) = u;
    } else {
        *reinterpret_cast<__half2 *>(p) = a;
    }
}

__device__ inline float gelu(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678118654752440f)); }

/* Where column n of the QKV product goes in the head-major output, less
 * the token's own offset (t * head_dim). */
__device__ inline size_t qkv_column(const GemmArgs &g, int n) {
    const int which = n / g.hidden, hc = n - which * g.hidden;
    const int head = hc / g.head_dim, d = hc - head * g.head_dim;
    return ((size_t)(which * g.heads + head) * g.tcap) * g.head_dim + d;
}

__half *as_half(uint16_t *p) { return reinterpret_cast<__half *>(p); }


/* Four values from shared memory, 16 bytes of F32 or 8 of F16, widened. */
__device__ inline void lds4(const float *p, float (&v)[4]) {
    const float4 x = *reinterpret_cast<const float4 *>(p);
    v[0] = x.x;
    v[1] = x.y;
    v[2] = x.z;
    v[3] = x.w;
}

__device__ inline void lds4(const __half *p, float (&v)[4]) {
    const uint2 u = *reinterpret_cast<const uint2 *>(p);
    const float2 a = __half22float2(*reinterpret_cast<const __half2 *>(&u.x));
    const float2 b = __half22float2(*reinterpret_cast<const __half2 *>(&u.y));
    v[0] = a.x;
    v[1] = a.y;
    v[2] = b.x;
    v[3] = b.y;
}

/* The written rows' arrays, from the run's shape. */
__device__ inline const int32_t *ids_of(const int32_t *rows, const Info &) { return rows; }
__device__ inline const int32_t *mask_of(const int32_t *rows, const Info &in) {
    return rows + (size_t)in.batch * in.seq;
}
__device__ inline const int32_t *types_of(const int32_t *rows, const Info &in) {
    return rows + 2 * (size_t)in.batch * in.seq;
}

// ---- The packing -----------------------------------------------------------------

constexpr int PACK_BLOCK = 512;
constexpr int PACK_WARPS = PACK_BLOCK / 32;

/* Query tiles in a row of n tokens. */
__device__ __host__ inline int query_tiles(int n, int q) { return (n + q - 1) / q; }

/* The exclusive prefix sum of one value per thread over the block, and
 * the total in *total: a shuffle scan in each warp, then the warps' sums
 * in turn. Integer sums, so the order does not matter. */
__device__ int32_t block_exclusive_scan(int32_t v, int32_t *warp_sums, int32_t *total) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int32_t inc = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int32_t y = __shfl_up_sync(FULL, inc, o);
        if (lane >= o) inc += y;
    }
    if (lane == 31) warp_sums[warp] = inc;
    __syncthreads();
    if (warp == 0) {
        int32_t w = lane < PACK_WARPS ? warp_sums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int32_t y = __shfl_up_sync(FULL, w, o);
            if (lane >= o) w += y;
        }
        if (lane < PACK_WARPS) warp_sums[lane] = w;
    }
    __syncthreads();
    *total = warp_sums[PACK_WARPS - 1];
    return inc - v + (warp > 0 ? warp_sums[warp - 1] : 0);
}

/* The rows from the page-locked staging, which the device reads over
 * the bus, into device memory: 16 bytes at a time, each thread with
 * FETCH_LOADS reads over the bus in flight before it stores, then the
 * tail. */
constexpr int FETCH_BLOCK = 256, FETCH_LOADS = 4;

__global__ void __launch_bounds__(FETCH_BLOCK) fetch_rows_kernel(FetchArgs a) {
    const int quads = a.n / 4;
    const int4 *s = reinterpret_cast<const int4 *>(a.src);
    int4 *d = reinterpret_cast<int4 *>(a.dst);
    constexpr int SPAN = FETCH_BLOCK * FETCH_LOADS;
    for (int base = blockIdx.x * SPAN + threadIdx.x; base < quads; base += gridDim.x * SPAN) {
        int4 v[FETCH_LOADS];
#pragma unroll
        for (int l = 0; l < FETCH_LOADS; l++) {
            const int i = base + l * FETCH_BLOCK;
            if (i < quads) v[l] = s[i];
        }
#pragma unroll
        for (int l = 0; l < FETCH_LOADS; l++) {
            const int i = base + l * FETCH_BLOCK;
            if (i < quads) d[i] = v[l];
        }
    }
    const int i = quads * 4 + blockIdx.x * FETCH_BLOCK + threadIdx.x;
    if (i < a.n) a.dst[i] = a.src[i];
}

/* One block. A warp per row finds its last live position and whether a
 * masked one comes before it; each thread sums the lengths of a run of
 * rows and a block scan gives each run's start: integer sums in a fixed
 * order. Then the rows are binned by query tiles, most first, for
 * attention's schedule: the counts are integer atomics, and the order
 * within a bin, the only thing the atomics leave open, changes which
 * block computes a row, never what it computes. Each bin's rows have the
 * same number of work items, so item_start does not depend on that order
 * either. Last, a warp per row writes each of its tokens' row and key
 * bias. */
__global__ void __launch_bounds__(PACK_BLOCK) pack_rows_kernel(PackArgs a) {
    extern __shared__ int32_t pack_sm[];
    __shared__ int32_t warp_sums[PACK_WARPS];
    const int q = a.queries;
    const int batch = a.run.batch, seq = a.run.seq;
    const int nb = query_tiles(seq, q) + 1; // bin k holds rows of nb - k tiles
    int32_t *bin_row = pack_sm, *bin_cur = pack_sm + nb, *bin_item = pack_sm + 2 * nb;
    const Packing &p = a.p;
    const int32_t *mask = a.rows + (size_t)batch * seq;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int k = threadIdx.x; k < nb; k += PACK_BLOCK) bin_row[k] = 0;
    for (int b = warp; b < batch; b += PACK_WARPS) {
        const int32_t *m = mask + (size_t)b * seq;
        int last = -1;
        for (int j = lane; j < seq; j += 32)
            if (m[j] != 0) last = j;
        last = warp_max_all(last);
        bool hole = false;
        for (int j = lane; j < last; j += 32) hole |= m[j] == 0;
        hole = __any_sync(FULL, hole);
        if (lane == 0) {
            p.len[b] = last + 1;
            p.holes[b] = hole ? 1 : 0;
        }
    }
    __syncthreads();

    const int chunk = (batch + PACK_BLOCK - 1) / PACK_BLOCK;
    const int lo = min(batch, (int)threadIdx.x * chunk), hi = min(batch, lo + chunk);
    int32_t sum = 0;
    for (int b = lo; b < hi; b++) {
        sum += p.len[b];
        atomicAdd(&bin_row[nb - query_tiles(p.len[b], q)], 1);
    }
    int32_t tokens = 0;
    int32_t at = block_exclusive_scan(sum, warp_sums, &tokens);
    if (threadIdx.x == 0) {
        int32_t rows = 0, items = 0;
        for (int k = 0; k < nb; k++) {
            const int32_t n = bin_row[k];
            bin_row[k] = rows;
            bin_cur[k] = rows;
            bin_item[k] = items;
            rows += n;
            items += n * (nb - k) * a.heads;
        }
        p.item_start[batch] = items;
        Info in;
        in.tokens = tokens;
        in.batch = batch;
        in.seq = seq;
        in.pooling = a.run.pooling;
        in.l2 = a.run.l2;
        in.output_dim = a.run.output_dim;
        in.has_types = a.run.has_types;
        in.items = items;
        *p.info = in;
    }
    __syncthreads();
    for (int b = lo; b < hi; b++) {
        p.start[b] = at;
        at += p.len[b];
        const int tiles = query_tiles(p.len[b], q), k = nb - tiles;
        const int slot = atomicAdd(&bin_cur[k], 1);
        p.order[slot] = b;
        p.item_start[slot] = bin_item[k] + (slot - bin_row[k]) * tiles * a.heads;
    }
    __syncthreads();

    for (int b = warp; b < batch; b += PACK_WARPS) {
        const int32_t *m = mask + (size_t)b * seq;
        const int n = p.len[b], s0 = p.start[b];
        for (int j = lane; j < n; j += 32) {
            p.tok_row[s0 + j] = b;
            p.key_bias[s0 + j] = m[j] != 0 ? 0.0f : -1e30f;
        }
    }
}

// ---- Row kernels -------------------------------------------------------------------

/* Threads per block for the warp-per-token kernels: eight warps. */
constexpr int ROW_BLOCK = 256;
constexpr int ROW_WARPS = ROW_BLOCK / 32;

/* v = LayerNorm(v) over the n values a warp holds, four at a time: lane l
 * holds columns 4 l + 128 i to 4 l + 128 i + 3 in v[i], those under n (a
 * multiple of 4). The mean, then the biased variance about it, are summed
 * in F32; the result is written to row, and in F16 to row16 when it is
 * not NULL. */
template <int V>
__device__ void layer_norm_regs(float (&v)[V][4], int n, const float *w, const float *b, float eps, float *row,
                                __half *row16) {
    const int lane = threadIdx.x & 31;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < V; i++)
        if (4 * lane + 128 * i < n) s += (v[i][0] + v[i][1]) + (v[i][2] + v[i][3]);
    const float mean = warp_sum_all(s) / (float)n;
    float q = 0.0f;
#pragma unroll
    for (int i = 0; i < V; i++)
        if (4 * lane + 128 * i < n)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float c = v[i][j] - mean;
                q += c * c;
            }
    const float var = warp_sum_all(q) / (float)n;
    const float inv = 1.0f / sqrtf(var + eps);
#pragma unroll
    for (int i = 0; i < V; i++) {
        const int d = 4 * lane + 128 * i;
        if (d >= n) continue;
        float wv[4], bv[4], y[4];
        load4(w + d, wv);
        load4(b + d, bv);
#pragma unroll
        for (int j = 0; j < 4; j++) y[j] = __fadd_rn(__fmul_rn((v[i][j] - mean) * inv, wv[j]), bv[j]);
        store_vec<4>(row + d, y);
        if (row16) store_vec<4>(row16 + d, y);
    }
}

template <int V>
__global__ void __launch_bounds__(ROW_BLOCK)
    embed_layer_norm_kernel(const int32_t *rows, const float *word, const float *position, const float *type,
                            const float *ln_w, const float *ln_b, float eps, Packing p, int hidden, float *x,
                            __half *x16) {
    const Info in = *p.info;
    const int32_t *ids = ids_of(rows, in), *types = types_of(rows, in);
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < in.tokens; t += gridDim.x * ROW_WARPS) {
        const int b = p.tok_row[t], pos = t - p.start[b];
        const size_t at = (size_t)b * in.seq + pos;
        const float *w = word + (size_t)ids[at] * hidden;
        const float *ps = position + (size_t)pos * hidden;
        const float *tt = type + (size_t)(in.has_types ? types[at] : 0) * hidden;
        float v[V][4];
#pragma unroll
        for (int i = 0; i < V; i++) {
            const int d = 4 * lane + 128 * i;
            if (d < hidden) {
                float a[4], c[4], e[4];
                load4(w + d, a);
                load4(ps + d, c);
                load4(tt + d, e);
#pragma unroll
                for (int j = 0; j < 4; j++) v[i][j] = a[j] + c[j] + e[j];
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++) v[i][j] = 0.0f;
            }
        }
        layer_norm_regs(v, hidden, ln_w, ln_b, eps, x + (size_t)t * hidden, x16 ? x16 + (size_t)t * hidden : nullptr);
    }
}

template <int V>
__global__ void __launch_bounds__(ROW_BLOCK)
    add_layer_norm_kernel(float *x, const float *y, const float *bias, const float *ln_w, const float *ln_b,
                          float eps, const Info *info, int hidden, __half *x16) {
    const int tokens = info->tokens;
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < tokens; t += gridDim.x * ROW_WARPS) {
        float *row = x + (size_t)t * hidden;
        const float *pr = y + (size_t)t * hidden;
        float v[V][4];
#pragma unroll
        for (int i = 0; i < V; i++) {
            const int d = 4 * lane + 128 * i;
            if (d >= hidden) {
#pragma unroll
                for (int j = 0; j < 4; j++) v[i][j] = 0.0f;
                continue;
            }
            float p[4], r[4], bv[4];
            load4(pr + d, p);
            const float4 xr = *reinterpret_cast<const float4 *>(row + d);
            r[0] = xr.x;
            r[1] = xr.y;
            r[2] = xr.z;
            r[3] = xr.w;
            load4(bias + d, bv);
#pragma unroll
            for (int j = 0; j < 4; j++) v[i][j] = r[j] + (p[j] + bv[j]);
        }
        layer_norm_regs(v, hidden, ln_w, ln_b, eps, row, x16 ? x16 + (size_t)t * hidden : nullptr);
    }
}

/* A thread per column of the pooled vector, a block per row. */
constexpr int POOL_BLOCK = 384;
constexpr int POOL_WARPS = POOL_BLOCK / 32;
/* Tokens a thread's loads have in flight at once, for the mean. */
constexpr int POOL_AHEAD = 16;

/* The block's sum of one value per thread, in every thread, in a fixed
 * order: each warp's shuffle tree, then the warps in turn. */
__device__ __forceinline__ float block_sum(float v, float *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum_all(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < POOL_WARPS; w++) total += red[w];
    return total;
}

/* A block per row, looping over the run's rows. The mean issues the loads
 * of POOL_AHEAD tokens before it adds any, then adds them in position
 * order. */
__global__ void __launch_bounds__(POOL_BLOCK)
    pool_kernel(const float *x, const int32_t *rows, Packing p, int hidden, float *out) {
    __shared__ float red[POOL_WARPS];
    const Info in = *p.info;
    const int32_t *mask = mask_of(rows, in);
    for (int b = blockIdx.x; b < in.batch; b += gridDim.x) {
        const int32_t *m = mask + (size_t)b * in.seq;
        const int n = p.len[b];
        const float *rw = x + (size_t)p.start[b] * hidden;
        float *dst = out + (size_t)b * in.output_dim;
        float ss = 0.0f;
        for (int d = threadIdx.x; d < in.output_dim; d += POOL_BLOCK) {
            float val;
            if (in.pooling == TURBO_POOLING_CLS) {
                val = rw[d];
            } else if (in.pooling == TURBO_POOLING_LAST) {
                // The row's length ends at its last live token.
                val = rw[(size_t)(n - 1) * hidden + d];
            } else {
                // Mean over the tokens whose mask is 1, summed in position order.
                float s = 0.0f;
                unsigned c = 0;
                for (int q0 = 0; q0 < n; q0 += POOL_AHEAD) {
                    float v[POOL_AHEAD];
                    int k[POOL_AHEAD];
#pragma unroll
                    for (int u = 0; u < POOL_AHEAD; u++) {
                        const bool in_row = q0 + u < n;
                        k[u] = in_row ? m[q0 + u] : 0;
                        v[u] = in_row ? rw[(size_t)(q0 + u) * hidden + d] : 0.0f;
                    }
#pragma unroll
                    for (int u = 0; u < POOL_AHEAD; u++)
                        if (k[u] != 0) {
                            s += v[u];
                            c++;
                        }
                }
                val = s * (1.0f / (float)c);
            }
            dst[d] = val;
            ss += val * val;
        }
        if (in.l2) {
            const float norm = fmaxf(sqrtf(block_sum(ss, red)), 1e-12f);
            const float scale = 1.0f / norm;
            for (int d = threadIdx.x; d < in.output_dim; d += POOL_BLOCK) dst[d] *= scale;
        }
    }
}

/* Groups of the row's tokens the split pooling sums apart, at most. */
constexpr int POOL_GROUPS = 8;
/* A thread's float4 loads in flight at once. */
constexpr int POOL_QUADS_AHEAD = 8;

/* A block per row, looping over the run's rows: a thread per four
 * columns of the pooled vector, and for the mean, as many groups of
 * such threads as the block holds (at most POOL_GROUPS), each summing a
 * contiguous run of the row's tokens in position order with
 * POOL_QUADS_AHEAD loads in flight; the first group adds the groups'
 * sums in group order. The vector stays in registers through the L2
 * normalization and is written once. */
__global__ void __launch_bounds__(POOL_BLOCK)
    pool_split_kernel(const float *x, const int32_t *rows, Packing p, int hidden, float *out) {
    __shared__ float4 part[POOL_BLOCK];
    __shared__ unsigned counts[POOL_GROUPS];
    __shared__ float red[POOL_WARPS];
    const Info in = *p.info;
    const int32_t *mask = mask_of(rows, in);
    const int quads = (in.output_dim + 3) / 4;
    const bool mean = in.pooling != TURBO_POOLING_CLS && in.pooling != TURBO_POOLING_LAST;
    int groups = POOL_BLOCK / quads;
    groups = !mean || groups < 1 ? 1 : groups > POOL_GROUPS ? POOL_GROUPS : groups;
    const int grp = threadIdx.x / quads, q = threadIdx.x % quads, d = 4 * q;
    for (int b = blockIdx.x; b < in.batch; b += gridDim.x) {
        const int32_t *m = mask + (size_t)b * in.seq;
        const int n = p.len[b];
        const float *rw = x + (size_t)p.start[b] * hidden + d;
        float4 val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (!mean) {
            if (threadIdx.x < quads) {
                const int at = in.pooling == TURBO_POOLING_CLS ? 0 : n - 1;
                val = *reinterpret_cast<const float4 *>(rw + (size_t)at * hidden);
            }
        } else {
            if (grp < groups) {
                const int per = (n + groups - 1) / groups, lo = grp * per, hi = min(n, lo + per);
                float4 sum = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                unsigned c = 0;
                for (int q0 = lo; q0 < hi; q0 += POOL_QUADS_AHEAD) {
                    float4 v[POOL_QUADS_AHEAD];
                    int k[POOL_QUADS_AHEAD];
#pragma unroll
                    for (int u = 0; u < POOL_QUADS_AHEAD; u++) {
                        k[u] = 0;
                        v[u] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                        if (q0 + u < hi) {
                            k[u] = m[q0 + u];
                            v[u] = __ldg(reinterpret_cast<const float4 *>(rw + (size_t)(q0 + u) * hidden));
                        }
                    }
#pragma unroll
                    for (int u = 0; u < POOL_QUADS_AHEAD; u++)
                        if (k[u] != 0) {
                            sum.x += v[u].x;
                            sum.y += v[u].y;
                            sum.z += v[u].z;
                            sum.w += v[u].w;
                            c++;
                        }
                }
                part[threadIdx.x] = sum;
                if (q == 0) counts[grp] = c;
            }
            __syncthreads();
            if (threadIdx.x < quads) {
                unsigned c = 0;
                for (int g2 = 0; g2 < groups; g2++) {
                    const float4 s = part[g2 * quads + threadIdx.x];
                    val.x += s.x;
                    val.y += s.y;
                    val.z += s.z;
                    val.w += s.w;
                    c += counts[g2];
                }
                const float inv = 1.0f / (float)c;
                val = make_float4(val.x * inv, val.y * inv, val.z * inv, val.w * inv);
            }
        }
        float o[4] = {val.x, val.y, val.z, val.w};
        float ss = 0.0f;
        if (threadIdx.x < quads)
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (d + j < in.output_dim) ss += o[j] * o[j];
        if (in.l2) {
            const float scale = 1.0f / fmaxf(sqrtf(block_sum(ss, red)), 1e-12f);
#pragma unroll
            for (int j = 0; j < 4; j++) o[j] *= scale;
        }
        float *dst = out + (size_t)b * in.output_dim;
        if (threadIdx.x < quads)
#pragma unroll
            for (int j = 0; j < 4; j++)
                if (d + j < in.output_dim) dst[d + j] = o[j];
        // part and counts are read before the next row writes them.
        __syncthreads();
    }
}

#ifndef TURBO_NO_MMA
__device__ inline unsigned smem_addr(const void *p) { return (unsigned)__cvta_generic_to_shared(p); }

__device__ inline void cp_async16(void *dst, const void *src, bool ok) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(smem_addr(dst)), "l"(src),
                 "r"(ok ? 16 : 0));
}

__device__ inline void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N> __device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ inline void ldsm_x4(uint32_t (&r)[4], const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

__device__ inline void ldsm_x4_trans(uint32_t (&r)[4], const void *p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_addr(p)));
}

/* d += a b for one m16n8k16 tile. */
__device__ inline void mma16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* d += a b for one m16n8k16 tile, d F16 (a half2 of each row's two
 * columns in each register, rows g and g + 8, as mma16816's floats). */
__device__ inline void mma16816_f16(uint32_t (&d)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* d += a b for one m16n8k8 tile of TF32 operands. */
__device__ inline void mma1688_tf32(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* x rounded to TF32, to the nearest with ties away from zero. */
__device__ inline uint32_t to_tf32(float x) {
    uint32_t u;
    asm("cvt.rna.tf32.f32 %0, %1;\n" : "=r"(u) : "f"(x));
    return u;
}

__device__ inline uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}
#endif

// ---- GEMMs ---------------------------------------------------------------------------

/* 16-byte copies from global to shared memory: cp.async where there is
 * one, else a load and a store, for the SIMT kernel's F16 build before
 * sm_80. A copy of !ok writes zeros. */
__device__ inline void copy16(void *dst, const void *src, bool ok) {
#ifndef TURBO_NO_MMA
    cp_async16(dst, src, ok);
#else
    *reinterpret_cast<uint4 *>(dst) = ok ? *reinterpret_cast<const uint4 *>(src) : make_uint4(0, 0, 0, 0);
#endif
}

__device__ inline void copy_commit() {
#ifndef TURBO_NO_MMA
    cp_async_commit();
#endif
}

template <int N> __device__ inline void copy_wait() {
#ifndef TURBO_NO_MMA
    cp_async_wait<N>();
#endif
}

// -- Stream-K ------------------------------------------------------------------------
//
// A GEMM's work is its tiles times the k steps of each, counted in one
// line, tile after tile. The launch is the blocks the device holds at
// once, and each takes an equal, contiguous share of the line, so every
// SM has the same work whatever the token count, with no wave left
// partly empty. A block works through its share one tile's run of k
// steps (a segment) at a time, last segment first. The block holding a
// tile's last k step finishes the tile: it adds the partial products of
// the earlier blocks that hold the rest of the tile's k steps, nearest
// first, then runs the epilogue. An earlier block's run of such a tile is
// the last segment of its share, which it does first, so the finisher's
// wait is short, and a block only ever waits on blocks before it, which
// the device started first. Partial products go through a workspace slot
// per block and a flag the finisher clears. The split points depend only
// on the token count, the shape and the launch, so the same rows on the
// same device give the same bits; every output is a fixed sum of fixed
// FMA chains.

struct Share {
    long long lo, hi, work;
    int blocks, steps;
};

/* The first step of block b's share. With min_steps SK_WHOLE_TILES,
 * whole tiles to a block, so no tile is split and no block waits on
 * another; min_steps is the kernel's argument, read where it is needed
 * rather than held in the share. */
__device__ inline long long share_start(const Share &s, int b, int min_steps) {
    if (min_steps == SK_WHOLE_TILES) return s.work / s.steps * b / s.blocks * s.steps;
    return s.work * b / s.blocks;
}

/* This block's share of tiles x steps of work, or an empty one. */
__device__ inline Share share_of(int tiles, int steps, int min_steps) {
    Share s;
    s.work = (long long)tiles * steps;
    const int least = min_steps > 0 ? min_steps : SK_MIN_STEPS;
    const long long want = min_steps == SK_WHOLE_TILES ? tiles : s.work / least;
    const long long most = want > 0 ? want : 1;
    s.blocks = (int)(most < (long long)gridDim.x ? most : (long long)gridDim.x);
    s.steps = steps;
    if ((int)blockIdx.x >= s.blocks) {
        s.lo = s.hi = 0;
    } else {
        s.lo = share_start(s, (int)blockIdx.x, min_steps);
        s.hi = share_start(s, (int)blockIdx.x + 1, min_steps);
    }
    return s;
}

/* A partial product stored: once every thread's values are, one thread
 * raises the flag with a release at device scope, which orders the
 * block's stores, seen by that thread through the barrier, before it. */
__device__ inline void sk_raise(int *flag) {
    __syncthreads();
    if (threadIdx.x == 0) {
#if __CUDA_ARCH__ >= 700
        asm volatile("st.release.gpu.global.b32 [%0], %1;" ::"l"(flag), "r"(1) : "memory");
#else
        __threadfence();
        *reinterpret_cast<volatile int *>(flag) = 1;
#endif
    }
}

/* The flag read with an acquire at device scope: what the raising block
 * stored before it is then visible to this thread, and through the
 * barrier after the wait to the block. */
__device__ inline int sk_flag(const int *flag) {
    int v;
#if __CUDA_ARCH__ >= 700
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(flag) : "memory");
#else
    v = *reinterpret_cast<const volatile int *>(flag);
    __threadfence();
#endif
    return v;
}

/* Spins before a wait gives up, each with a short sleep between reads:
 * about a second, far past any launch. */
constexpr long long SK_MOST_SPINS = 1LL << 24;

/* Waits for a partial product's flag and clears it for the next launch.
 * The block raising it has a lower index and does it before any wait of
 * its own, so this relies only on the device starting blocks in index
 * order, as it does: a block waited on has started before its waiter.
 * Should the flag never come, the wait gives up rather than hang the
 * run, and sets *fault, which the host reads after the run and reports
 * as an error. */
__device__ inline void sk_wait(int *flag, int *fault) {
    if (threadIdx.x == 0) {
        long long spins = 0;
        while (sk_flag(flag) == 0) {
            if (++spins == SK_MOST_SPINS) {
                *reinterpret_cast<volatile int *>(fault) = 1;
                break;
            }
#if __CUDA_ARCH__ >= 700
            __nanosleep(48);
#endif
        }
        *reinterpret_cast<volatile int *>(flag) = 0;
    }
    __syncthreads();
}

/* ADD_LN's end of a tile: once every tile of the block of rows from m0
 * is in out, the block that finished the last of them normalizes the
 * rows, R to a warp at a time, each as add_layer_norm's warp does (the
 * same function, so the same sums), reading what other blocks wrote
 * from L2. Every thread of the block calls it. */
template <int NT>
__device__ void ln_rows_when_done(const GemmArgs &g, int m0, int rows, int *counter, int tiles) {
    constexpr int R = 2, W = NT / 32, V = LN_FUSED_MAX_HIDDEN / 128;
    __shared__ int last;
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        last = atomicAdd(counter, 1) == tiles - 1;
        if (last) *counter = 0;
    }
    __syncthreads();
    if (!last) return;
    __threadfence();
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, n = g.n;
    float *x = static_cast<float *>(g.out);
    __half *x16 = reinterpret_cast<__half *>(g.x16);
    for (int r0 = warp * R; r0 < rows; r0 += W * R) {
        float v[R][V][4];
#pragma unroll
        for (int u = 0; u < R; u++)
#pragma unroll
            for (int i = 0; i < V; i++) {
                const int d = 4 * lane + 128 * i;
                if (r0 + u < rows && d < n) {
                    const float4 q = __ldcg(reinterpret_cast<const float4 *>(x + (size_t)(m0 + r0 + u) * n + d));
                    v[u][i][0] = q.x;
                    v[u][i][1] = q.y;
                    v[u][i][2] = q.z;
                    v[u][i][3] = q.w;
                } else {
#pragma unroll
                    for (int j = 0; j < 4; j++) v[u][i][j] = 0.0f;
                }
            }
#pragma unroll
        for (int u = 0; u < R; u++) {
            if (r0 + u >= rows) break;
            const size_t t = (size_t)(m0 + r0 + u);
            layer_norm_regs(v[u], n, g.ln_w, g.ln_b, g.eps, x + t * n, x16 ? x16 + t * n : nullptr);
        }
    }
}

/* Where token t's column c goes, for the row-major epilogues. */
template <typename TOut> __device__ inline TOut *out_at(const GemmArgs &g, int t, int c) {
    return static_cast<TOut *>(g.out) + (size_t)t * g.n + c;
}

// -- The SIMT GEMM: F32 (and F16 before sm_80), FMAs ----------------------------------
//
// A BM x BN tile per block of (BM / TM) x (BN / 8) threads, each thread
// TM x 8 outputs: rows ty + (BM / TM) i and columns tx + (BN / 8) j, so
// a quarter warp's shared memory reads are one row of A, which they
// share, or 8 consecutive rows of B, in distinct banks for rows of 20
// floats. k goes 16 at a time through a 3-stage cp.async pipeline of
// rows of A and of the weight, k contiguous in each; each thread reads
// four k values of each of its rows and columns, TM + 8 LDS.128 per
// 32 TM FMAs: 16 per 256 at 8 x 8, 24 per 512 at 16 x 8 (128 x 128
// over 128 threads, which reads less but runs one block of four warps
// to an SM, and measured slower than 128 x 64 at 8 x 8 on an sm_89).
// Each output's FMAs run in k order at every tile.

constexpr int SIMT_BK = 16, SIMT_STAGES = 3;

template <typename TIn> __host__ __device__ constexpr int simt_ld() { return SIMT_BK + 16 / (int)sizeof(TIn); }

/* Four blocks to an SM at 64 x 64, two at 128 x 64, one at 128 x 128. */
template <int BM, int BN> constexpr int simt_min_blocks() {
    return BM * BN <= 64 * 64 ? 4 : BM * BN <= 128 * 64 ? 2 : 1;
}

template <int BM, int BN, typename TIn> constexpr size_t simt_gemm_smem() {
    return (size_t)SIMT_STAGES * (BM + BN) * simt_ld<TIn>() * sizeof(TIn);
}

template <int BM, int BN, int TM, typename TIn, int EPI, typename TOut>
__global__ void __launch_bounds__((BM / TM) * (BN / 8), (simt_min_blocks<BM, BN>()))
    gemm_simt_kernel(GemmArgs g) {
    constexpr int TX = BN / 8, TY = BM / TM, NT = TX * TY, LD = simt_ld<TIn>(), E = 16 / (int)sizeof(TIn);
    constexpr int CH = SIMT_BK / E; // 16-byte chunks per row of a stage
    constexpr int SLOT = BM * BN;
    extern __shared__ __align__(16) unsigned char gemm_sm[];
    TIn *As = reinterpret_cast<TIn *>(gemm_sm);
    TIn *Bs = As + SIMT_STAGES * BM * LD;
    const TIn *A = static_cast<const TIn *>(g.a), *B = static_cast<const TIn *>(g.w);
    const int M = g.info->tokens, N = g.n, K = g.k;
    const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
    const Share sh = share_of(mt * nt, (K + SIMT_BK - 1) / SIMT_BK, g.min_steps);
    const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;

    // The share's segments last to first: a tile the next block finishes
    // comes first, so its partial product is ready early.
    for (long long at = sh.hi; at > sh.lo;) {
        const int tile = (int)((at - 1) / sh.steps);
        const long long first = (long long)tile * sh.steps, end = first + sh.steps;
        const long long begin = sh.lo > first ? sh.lo : first;
        const int n0 = (tile % nt) * BN, m0 = (tile / nt) * BM;
        const int kb = (int)(begin - first) * SIMT_BK, ke = min(K, (int)(at - first) * SIMT_BK);
        auto load_stage = [&](int st, int k0) {
            TIn *as = As + st * BM * LD, *bs = Bs + st * BN * LD;
            for (int i = threadIdx.x; i < BM * CH; i += NT) {
                const int r = i / CH, c = (i % CH) * E, gm = m0 + r, gk = k0 + c;
                const bool ok = gm < M && gk < ke;
                copy16(as + r * LD + c, ok ? A + (size_t)gm * K + gk : A, ok);
            }
            for (int i = threadIdx.x; i < BN * CH; i += NT) {
                const int r = i / CH, c = (i % CH) * E, gn = n0 + r, gk = k0 + c;
                const bool ok = gn < N && gk < ke;
                copy16(bs + r * LD + c, ok ? B + (size_t)gn * K + gk : B, ok);
            }
        };

        float acc[TM][8];
#pragma unroll
        for (int i = 0; i < TM; i++)
#pragma unroll
            for (int j = 0; j < 8; j++) acc[i][j] = 0.0f;

        const int ktiles = (ke - kb + SIMT_BK - 1) / SIMT_BK;
#pragma unroll
        for (int s = 0; s < SIMT_STAGES - 1; s++) {
            if (s < ktiles) load_stage(s, kb + s * SIMT_BK);
            copy_commit();
        }
        for (int kt = 0; kt < ktiles; kt++) {
            copy_wait<SIMT_STAGES - 2>();
            __syncthreads();
            const int next = kt + SIMT_STAGES - 1;
            if (next < ktiles) load_stage(next % SIMT_STAGES, kb + next * SIMT_BK);
            copy_commit();
            const TIn *as = As + (kt % SIMT_STAGES) * BM * LD + ty * LD;
            const TIn *bs = Bs + (kt % SIMT_STAGES) * BN * LD + tx * LD;
#pragma unroll(TM == 16 ? 1 : SIMT_BK / 4)
            for (int kk = 0; kk < SIMT_BK; kk += 4) {
                float b[8][4];
#pragma unroll
                for (int j = 0; j < 8; j++) lds4(bs + j * TX * LD + kk, b[j]);
#pragma unroll
                for (int i = 0; i < TM; i++) {
                    float a[4];
                    lds4(as + i * TY * LD + kk, a);
#pragma unroll
                    for (int q = 0; q < 4; q++)
#pragma unroll
                        for (int j = 0; j < 8; j++) acc[i][j] = fmaf(a[q], b[j][q], acc[i][j]);
                }
            }
        }
        copy_wait<0>();
        __syncthreads();

        if (at != end) {
            // The start of a tile a later block finishes.
            float *slot = g.ws + (size_t)blockIdx.x * SLOT + threadIdx.x;
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) __stcg(slot + (i * 8 + j) * NT, acc[i][j]);
            sk_raise(g.flags + blockIdx.x);
            at = begin;
            continue;
        }
        for (int b = (int)blockIdx.x - 1; b >= 0 && share_start(sh, b + 1, g.min_steps) > first; b--) {
            sk_wait(g.flags + b, g.fault);
            const float *slot = g.ws + (size_t)b * SLOT + threadIdx.x;
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < 8; j++) acc[i][j] += __ldcg(slot + (i * 8 + j) * NT);
        }
        at = begin;

        size_t col[8];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int c = n0 + tx + TX * j;
            col[j] = EPI == EPI_QKV && c < N ? qkv_column(g, c) : 0;
        }
#pragma unroll
        for (int i = 0; i < TM; i++) {
            const int t = m0 + ty + TY * i;
            if (t >= M) continue;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const int c = n0 + tx + TX * j;
                if (c >= N) continue;
                if constexpr (EPI == EPI_QKV)
                    put(static_cast<TOut *>(g.out) + col[j] + (size_t)t * g.head_dim, acc[i][j] + g.bias[c]);
                else if constexpr (EPI == EPI_GELU)
                    put(out_at<TOut>(g, t, c), gelu(acc[i][j] + g.bias[c]));
                else if constexpr (EPI == EPI_ADD_LN) {
                    float *o = out_at<float>(g, t, c);
                    *o = *o + (acc[i][j] + g.bias[c]);
                } else
                    put(out_at<TOut>(g, t, c), acc[i][j]);
            }
        }
        if constexpr (EPI == EPI_ADD_LN) ln_rows_when_done<NT>(g, m0, min(BM, M - m0), g.rows_done + m0 / BM, nt);
    }
}

// -- The F16 GEMM on the tensor cores --------------------------------------------------
//
// mma.sync m16n8k16, F16 operands, F32 accumulators; or, at MODEL for
// an F32 model (TIn float), m16n8k8 with F32 operands rounded to TF32
// as each fragment is read from shared memory, F32 accumulators. A
// block of WM x WN warps, each a (BM / WM) x (BN / WN) accumulator tile,
// computes a BM x BN tile over k in steps of 64 bytes of each row (32
// F16 values, 16 F32), through a STAGES-deep cp.async pipeline in
// shared memory, rows padded to 80 bytes, which keeps ldmatrix's reads
// of F16 and the TF32 fragments' reads of F32 (8 rows by 4 values to a
// warp) free of bank conflicts. The finished tile goes through
// shared memory, so every store to global memory is 16 contiguous
// bytes. The wide GEMMs (QKV and the first feed-forward) take 128 x 128
// over eight warps of 32 x 64: each operand byte read from L2 feeds
// twice the MMAs of 128 x 64, and each A fragment four B fragments'
// worth. The N = 384 GEMMs keep 128 x 64, whose three N tiles already
// leave few tiles to share among the SMs.

/* A stage's row of k: 64 bytes, padded to 80. */
constexpr int MMA_LD_BYTES = 80;

template <typename TIn> __host__ __device__ constexpr int mma_k() { return 64 / (int)sizeof(TIn); }
template <typename TIn> __host__ __device__ constexpr int mma_ld() { return MMA_LD_BYTES / (int)sizeof(TIn); }

template <int BM, int BN, int STAGES, typename TOut> constexpr size_t mma_gemm_smem() {
    const size_t pipe = (size_t)STAGES * (BM + BN) * MMA_LD_BYTES;
    const size_t out = (size_t)BM * (BN + 16 / sizeof(TOut)) * sizeof(TOut);
    return pipe > out ? pipe : out;
}

#ifndef TURBO_NO_MMA
__device__ inline void put2(float *p, float a, float b) { *reinterpret_cast<float2 *>(p) = make_float2(a, b); }
__device__ inline void put2(__half *p, float a, float b) { *reinterpret_cast<__half2 *>(p) = __floats2half2_rn(a, b); }
#endif

/* Two blocks to an SM when the shared memory fits two (128 x 64 at three
 * stages, 128 x 128 at two with an F16 output), else one. */
template <int BM, int BN, int STAGES, typename TOut> constexpr int mma_min_blocks() {
    return mma_gemm_smem<BM, BN, STAGES, TOut>() <= 48 * 1024 ? 2 : 1;
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI, typename TOut, typename TIn, bool ACC16 = false>
__global__ void __launch_bounds__(WM *WN * 32, (mma_min_blocks<BM, BN, STAGES, TOut>()))
    gemm_mma_kernel(GemmArgs g) {
#ifndef TURBO_NO_MMA
    constexpr int NT = WM * WN * 32, WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;
    constexpr int E = 16 / (int)sizeof(TOut), OLD = BN + E; // the output tile's row, in TOut
    constexpr int SLOT = BM * BN;
    constexpr int MMA_K = mma_k<TIn>(), MMA_LD = mma_ld<TIn>(), CE = 16 / (int)sizeof(TIn);
    constexpr bool TF32 = sizeof(TIn) == 4;
    static_assert(TF32 || NI % 2 == 0, "B fragments load two n8 tiles at once");
    static_assert(!(TF32 && ACC16), "F16 accumulators take F16 operands");
    // F16 accumulators, with ACC16, over each ACC16_STEPS k steps (64
    // terms), then added into acc; the chunks are whole k steps from k 0
    // on, so a tile's sum is the same whichever blocks share it.
    constexpr int ACC16_STEPS = 64 / MMA_K;
    extern __shared__ __align__(16) unsigned char gemm_sm[];
    TIn *As = reinterpret_cast<TIn *>(gemm_sm);
    TIn *Bs = As + STAGES * BM * MMA_LD;
    TOut *Cs = reinterpret_cast<TOut *>(gemm_sm);
    const TIn *A = static_cast<const TIn *>(g.a), *B = static_cast<const TIn *>(g.w);
    const int M = g.info->tokens, N = g.n, K = g.k;
    const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
    const Share sh = share_of(mt * nt, (K + MMA_K - 1) / MMA_K, g.min_steps);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp / WN, wn = warp % WN;

    // The share's segments last to first: a tile the next block finishes
    // comes first, so its partial product is ready early.
    for (long long at = sh.hi; at > sh.lo;) {
        const int tile = (int)((at - 1) / sh.steps);
        const long long first = (long long)tile * sh.steps, end = first + sh.steps;
        const long long begin = sh.lo > first ? sh.lo : first;
        const int n0 = (tile % nt) * BN, m0 = (tile / nt) * BM;
        const int kb = (int)(begin - first) * MMA_K, ke = min(K, (int)(at - first) * MMA_K);
        auto load_stage = [&](int st, int k0) {
            TIn *as = As + st * BM * MMA_LD, *bs = Bs + st * BN * MMA_LD;
            for (int i = threadIdx.x; i < BM * (MMA_K / CE); i += NT) {
                const int r = i >> 2, ch = (i & 3) * CE, gm = m0 + r, gk = k0 + ch;
                const bool ok = gm < M && gk < ke;
                cp_async16(as + r * MMA_LD + ch, ok ? A + (size_t)gm * K + gk : A, ok);
            }
            for (int i = threadIdx.x; i < BN * (MMA_K / CE); i += NT) {
                const int r = i >> 2, ch = (i & 3) * CE, gn = n0 + r, gk = k0 + ch;
                const bool ok = gn < N && gk < ke;
                cp_async16(bs + r * MMA_LD + ch, ok ? B + (size_t)gn * K + gk : B, ok);
            }
        };

        float acc[MI][NI][4];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NI; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

        uint32_t hacc[ACC16 ? MI : 1][ACC16 ? NI : 1][2];
        if constexpr (ACC16) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++) hacc[i][j][0] = hacc[i][j][1] = 0u;
        }

        const int ktiles = (ke - kb + MMA_K - 1) / MMA_K;
#pragma unroll
        for (int s = 0; s < STAGES - 1; s++) {
            if (s < ktiles) load_stage(s, kb + s * MMA_K);
            cp_async_commit();
        }
        for (int kt = 0; kt < ktiles; kt++) {
            cp_async_wait<STAGES - 2>();
            __syncthreads();
            const int next = kt + STAGES - 1;
            if (next < ktiles) load_stage(next % STAGES, kb + next * MMA_K);
            cp_async_commit();
            const TIn *as = As + (kt % STAGES) * BM * MMA_LD, *bs = Bs + (kt % STAGES) * BN * MMA_LD;
            if constexpr (TF32) {
                // A's a0..a3 at (g, t), (g + 8, t), (g, t + 4), (g + 8, t + 4)
                // of the 16 x 8 tile, B's b0, b1 at (k t, n g), (k t + 4, n g).
                const int gq = lane >> 2, tq = lane & 3;
#pragma unroll
                for (int kk = 0; kk < MMA_K; kk += 8) {
                    uint32_t af[MI][4], bf[NI][2];
#pragma unroll
                    for (int i = 0; i < MI; i++) {
                        const TIn *p = as + (wm * WTM + i * 16 + gq) * MMA_LD + kk + tq;
                        af[i][0] = to_tf32(p[0]);
                        af[i][1] = to_tf32(p[8 * MMA_LD]);
                        af[i][2] = to_tf32(p[4]);
                        af[i][3] = to_tf32(p[8 * MMA_LD + 4]);
                    }
#pragma unroll
                    for (int j = 0; j < NI; j++) {
                        const TIn *q = bs + (wn * WTN + j * 8 + gq) * MMA_LD + kk + tq;
                        bf[j][0] = to_tf32(q[0]);
                        bf[j][1] = to_tf32(q[4]);
                    }
#pragma unroll
                    for (int i = 0; i < MI; i++)
#pragma unroll
                        for (int j = 0; j < NI; j++) mma1688_tf32(acc[i][j], af[i], bf[j][0], bf[j][1]);
                }
            } else {
#pragma unroll
                for (int kk = 0; kk < MMA_K; kk += 16) {
                    uint32_t af[MI][4], bf[NI][2];
#pragma unroll
                    for (int i = 0; i < MI; i++)
                        ldsm_x4(af[i], as + (wm * WTM + i * 16 + (lane & 15)) * MMA_LD + kk + (lane >> 4) * 8);
#pragma unroll
                    for (int j = 0; j < NI / 2; j++) {
                        uint32_t r[4];
                        ldsm_x4(r, bs + (wn * WTN + j * 16 + (lane >> 4) * 8 + (lane & 7)) * MMA_LD + kk +
                                       ((lane >> 3) & 1) * 8);
                        bf[2 * j][0] = r[0];
                        bf[2 * j][1] = r[1];
                        bf[2 * j + 1][0] = r[2];
                        bf[2 * j + 1][1] = r[3];
                    }
#pragma unroll
                    for (int i = 0; i < MI; i++)
#pragma unroll
                        for (int j = 0; j < NI; j++) {
                            if constexpr (ACC16)
                                mma16816_f16(hacc[i][j], af[i], bf[j][0], bf[j][1]);
                            else
                                mma16816(acc[i][j], af[i], bf[j][0], bf[j][1]);
                        }
                }
                if constexpr (ACC16) {
                    if ((kb / MMA_K + kt + 1) % ACC16_STEPS == 0 || kt == ktiles - 1) {
#pragma unroll
                        for (int i = 0; i < MI; i++)
#pragma unroll
                            for (int j = 0; j < NI; j++)
#pragma unroll
                                for (int h = 0; h < 2; h++) {
                                    const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[i][j][h]));
                                    acc[i][j][2 * h] += v.x;
                                    acc[i][j][2 * h + 1] += v.y;
                                    hacc[i][j][h] = 0u;
                                }
                    }
                }
            }
        }
        cp_async_wait<0>();
        __syncthreads();

        if (at != end) {
            // The start of a tile a later block finishes.
            float *slot = g.ws + (size_t)blockIdx.x * SLOT + threadIdx.x;
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) __stcg(slot + ((i * NI + j) * 4 + e) * NT, acc[i][j][e]);
            sk_raise(g.flags + blockIdx.x);
            at = begin;
            continue;
        }
        for (int b = (int)blockIdx.x - 1; b >= 0 && share_start(sh, b + 1, g.min_steps) > first; b--) {
            sk_wait(g.flags + b, g.fault);
            const float *slot = g.ws + (size_t)b * SLOT + threadIdx.x;
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] += __ldcg(slot + ((i * NI + j) * 4 + e) * NT);
        }
        at = begin;

        // The tile, finished, into shared memory.
#pragma unroll
        for (int j = 0; j < NI; j++) {
            const int c = wn * WTN + j * 8 + (lane & 3) * 2;
            float b0 = 0.0f, b1 = 0.0f;
            if constexpr (EPI != EPI_PLAIN)
                if (n0 + c < N) {
                    b0 = g.bias[n0 + c];
                    b1 = g.bias[n0 + c + 1];
                }
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int r = wm * WTM + i * 16 + (lane >> 2) + h * 8;
                    float v0 = acc[i][j][2 * h] + b0, v1 = acc[i][j][2 * h + 1] + b1;
                    if constexpr (EPI == EPI_GELU) {
                        v0 = gelu(v0);
                        v1 = gelu(v1);
                    }
                    put2(Cs + r * OLD + c, v0, v1);
                }
        }
        __syncthreads();
        // Out, 16 bytes at a time.
        for (int i = threadIdx.x; i < BM * (BN / E); i += NT) {
            const int r = i / (BN / E), cc = (i % (BN / E)) * E, t = m0 + r, c = n0 + cc;
            if (t >= M || c >= N) continue;
            const uint4 v = *reinterpret_cast<const uint4 *>(Cs + r * OLD + cc);
            if constexpr (EPI == EPI_QKV) {
                TOut *o = static_cast<TOut *>(g.out) + (size_t)t * g.head_dim;
                if (g.head_dim % E == 0) {
                    *reinterpret_cast<uint4 *>(o + qkv_column(g, c)) = v;
                } else {
                    const TOut *e = reinterpret_cast<const TOut *>(&v);
                    for (int u = 0; u < E; u++) o[qkv_column(g, c + u)] = e[u];
                }
            } else if constexpr (EPI == EPI_ADD_LN) {
                float4 *o = reinterpret_cast<float4 *>(out_at<float>(g, t, c));
                const float4 p = *reinterpret_cast<const float4 *>(&v), r = *o;
                *o = make_float4(r.x + p.x, r.y + p.y, r.z + p.z, r.w + p.w);
            } else {
                *reinterpret_cast<uint4 *>(out_at<TOut>(g, t, c)) = v;
            }
        }
        __syncthreads();
        if constexpr (EPI == EPI_ADD_LN) ln_rows_when_done<NT>(g, m0, min(BM, M - m0), g.rows_done + m0 / BM, nt);
    }
#else
    (void)g;
    __trap();
#endif
}

// -- The F16 GEMM on the tensor cores, swizzled -----------------------------------------
//
// The same product as gemm_mma_kernel at F16 operands, laid out so more
// of it overlaps. A stage's rows are 64 bytes, unpadded: 16-byte chunk c
// of row r is stored at c ^ ((r >> 1) & 3), so the 8 rows an ldmatrix
// reads at one chunk fall in 8 distinct 16-byte bank groups, and 3
// stages of 128 x 128 take 48 KB, which lets two blocks share an SM.
// The block's share of k steps, over all its tiles, is one stream: the
// loads run STAGES - 1 steps ahead of the MMAs across the end of a tile,
// so the next tile's first stages are in flight while a tile is finished.
// The finished tile goes from registers to global memory: F32 as a
// float2 a lane (a quad of lanes writes 32 contiguous bytes), F16
// gathered by shuffles within the quad, 8 columns to a lane, into one
// 16-byte store. With DB a k step's two sets of fragments are both read
// before its MMAs are issued. With ACC16 each 64 terms of k are summed in
// F16, as in gemm_mma_kernel.

#ifndef TURBO_NO_MMA
/* Where 16-byte chunk c of row r of a swizzled stage is, in chunks. */
__device__ constexpr int swz_chunk(int r, int c) { return r * 4 + (c ^ ((r >> 1) & 3)); }

#endif

template <int BM, int BN, int STAGES> constexpr size_t swz_gemm_smem() {
    return (size_t)STAGES * (BM + BN) * 64;
}

/* Three blocks to an SM when three fit in 100 KB (sm_86 and sm_89's
 * most), two when two do. */
template <int BM, int BN, int STAGES> constexpr int swz_min_blocks() {
    constexpr size_t smem = swz_gemm_smem<BM, BN, STAGES>();
    return smem <= 32 * 1024 ? 3 : smem <= 49 * 1024 ? 2 : 1;
}

#ifndef TURBO_NO_MMA
/* A block's place in its share, walked as gemm_mma_kernel walks it: the
 * segments last to first, each segment's k steps first to last. */
struct SwzCursor {
    int tile, k, kb, ke; /* tile -1 past the share's end; k steps within the tile */
    __device__ void start(const Share &s) { segment(s, s.hi); }
    /* The segment ending before step at of the work. */
    __device__ void segment(const Share &s, long long at) {
        if (at <= s.lo) {
            tile = -1;
            return;
        }
        tile = (int)((at - 1) / s.steps);
        const long long first = (long long)tile * s.steps;
        kb = s.lo > first ? (int)(s.lo - first) : 0;
        ke = (int)(at - first);
        k = kb;
    }
    __device__ bool live() const { return tile >= 0; }
    /* share() gives the Share, made again only at a segment's end. */
    template <typename F> __device__ void next(F share) {
        if (++k == ke) {
            const Share s = share();
            segment(s, (long long)tile * s.steps + kb);
        }
    }
};

/* A stage's k step and its flags: the first of its segment, the last,
 * and the tile's last. */
constexpr int STEP_K = 0xffffff, STEP_OPENS = 1 << 24, STEP_CLOSES = 1 << 25, STEP_ENDS = 1 << 26;

/* f(std::integral_constant<int, i>) for i from 0 to N - 1: a loop the
 * compiler cannot leave rolled, for loops whose index picks registers
 * (an index known only at run time puts the array in local memory). */
template <typename F, int... I> __device__ __forceinline__ void static_for_(F &&f, std::integer_sequence<int, I...>) {
    (f(std::integral_constant<int, I>{}), ...);
}
template <int N, typename F> __device__ __forceinline__ void static_for(F &&f) {
    static_for_(f, std::make_integer_sequence<int, N>{});
}

/* A quad's F16 row of four n8 tiles, a half2 of each from each lane
 * (mine[u] is tile u's), gathered so lane tq holds tile tq's 8 values in
 * column order: a 4 x 4 transpose in two exchanges, across bit 0 of tq,
 * then across bit 1. */
__device__ inline uint4 quad_gather(const uint32_t (&mine)[4], int tq) {
    const bool b = tq & 1, c1 = tq & 2;
    const uint32_t y0 = __shfl_xor_sync(0xffffffffu, b ? mine[0] : mine[1], 1);
    const uint32_t y1 = __shfl_xor_sync(0xffffffffu, b ? mine[2] : mine[3], 1);
    // Tiles b and 2 + b: this lane's piece, then its partner's.
    const uint32_t k0 = b ? mine[1] : mine[0], k2 = b ? mine[3] : mine[2];
    const uint32_t w0 = __shfl_xor_sync(0xffffffffu, c1 ? k0 : k2, 2);
    const uint32_t w1 = __shfl_xor_sync(0xffffffffu, c1 ? y0 : y1, 2);
    const uint32_t own = c1 ? k2 : k0, other = c1 ? y1 : y0;
    // In lane order: this pair of lanes, then the other.
    const uint32_t plo = b ? other : own, phi = b ? own : other;
    const uint32_t wlo = b ? w1 : w0, whi = b ? w0 : w1;
    return make_uint4(c1 ? wlo : plo, c1 ? whi : phi, c1 ? plo : wlo, c1 ? phi : whi);
}

/* The eight halves of w to eight head-major columns, the first at col
 * (qkv_column's) and d within its head, for a head width not a multiple
 * of 8: the halves by shifts, so w stays in registers, and each next
 * column by a step, a head's end stepping to the next head's start
 * (the next head, or the next of Q, K and V, is the next head-major
 * block), with no division. */
__device__ inline void qkv_store8(const GemmArgs &g, __half *o, size_t col, int d, uint4 w) {
    const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
    const size_t jump = (size_t)g.tcap * g.head_dim - (size_t)(g.head_dim - 1);
#pragma unroll
    for (int u = 0; u < 8; u++) {
        o[col] = __ushort_as_half((unsigned short)(wv[u >> 1] >> ((u & 1) * 16)));
        if (++d == g.head_dim) {
            d = 0;
            col += jump;
        } else {
            col++;
        }
    }
}
#endif

template <int BM, int BN, int WM, int WN, int STAGES, int EPI, typename TOut, bool ACC16, bool DB,
          bool WHOLE = false>
__global__ void __launch_bounds__(WM *WN * 32, (swz_min_blocks<BM, BN, STAGES>()))
    gemm_swz_kernel(GemmArgs g) {
#ifndef TURBO_NO_MMA
    constexpr int NT = WM * WN * 32, WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;
    constexpr int MMA_K = 32, SLOT = BM * BN, KK = MMA_K / 16;
    constexpr int ACC16_STEPS = 64 / MMA_K;
    // WHOLE: the F16 accumulators over the whole of a segment's k, with
    // no F32 sums but the stream-K partial products and their total.
    static_assert(!WHOLE || ACC16, "whole-k F16 sums are F16 accumulators");
    static_assert(NI % 4 == 0, "F16 stores gather four n8 tiles");
    static_assert(STAGES >= 2, "a pipeline");
    // ADD_LN on a tile of whole rows (N <= BN, which the plan sees to):
    // the LayerNorm in the epilogue, from registers.
    constexpr bool ROW_LN = EPI == EPI_ADD_LN && BN == ROW_LN_WIDTH;
    __shared__ float row_sums[ROW_LN ? 2 : 1][ROW_LN ? BM : 1][WN];
    static_assert(!DB || KK == 2, "the fragments alternate between two sets, the next step's first in set 0");
    extern __shared__ __align__(16) unsigned char gemm_sm[];
    __half *As = reinterpret_cast<__half *>(gemm_sm);
    __half *Bs = As + STAGES * BM * MMA_K;
    const __half *A = static_cast<const __half *>(g.a), *B = static_cast<const __half *>(g.w);
    const int M = g.info->tokens, N = g.n, K = g.k;
    const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN;
    // The Share made again where it is needed, not held in registers.
    const int steps = (K + MMA_K - 1) / MMA_K;
    auto share = [&]() { return share_of(mt * nt, steps, g.min_steps); };
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp / WN, wn = warp % WN, gq = lane >> 2, tq = lane & 3;

    // Each stage's k step, for the MMAs: its tile, and its k step with
    // flags, written with its loads and read STAGES - 1 barriers later.
    __shared__ int2 stage_step[STAGES];
    auto load_stage = [&](int st, const SwzCursor &c) {
        if (threadIdx.x == 0)
            stage_step[st] = make_int2(c.tile, c.k | (c.k == c.kb ? STEP_OPENS : 0) |
                                                   (c.k + 1 == c.ke ? STEP_CLOSES : 0) |
                                                   (c.ke == steps ? STEP_ENDS : 0));
        const int n0 = (c.tile % nt) * BN, m0 = (c.tile / nt) * BM, k0 = c.k * MMA_K;
        __half *as = As + st * BM * MMA_K, *bs = Bs + st * BN * MMA_K;
        for (int i = threadIdx.x; i < BM * 4; i += NT) {
            const int r = i >> 2, ch = i & 3, gm = m0 + r, gk = k0 + ch * 8;
            const bool ok = gm < M && gk < K;
            cp_async16(as + swz_chunk(r, ch) * 8, ok ? A + (size_t)gm * K + gk : A, ok);
        }
        for (int i = threadIdx.x; i < BN * 4; i += NT) {
            const int r = i >> 2, ch = i & 3, gn = n0 + r, gk = k0 + ch * 8;
            const bool ok = gn < N && gk < K;
            cp_async16(bs + swz_chunk(r, ch) * 8, ok ? B + (size_t)gn * K + gk : B, ok);
        }
    };

    SwzCursor ld;
    ld.start(share());
#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (ld.live()) {
            load_stage(s, ld);
            ld.next(share);
        }
        cp_async_commit();
    }
    int st = 0, lst = STAGES - 1; // the stage computed, the stage loaded next
    auto after = [](int x) { return x + 1 == STAGES ? 0 : x + 1; };

    // With DB, the fragments of two halves of k, 16 each: those of the
    // half whose MMAs run and those of the next, read while they do. The
    // next after a k step's last half is the next step's first, so each
    // step's barrier comes before its last half's MMAs, not before its
    // first ldmatrix: the pipeline of cutlass's multistage mainloop.
    // Without (the eight-warp mix's shapes), each step waits at
    // its barrier, then reads A's fragments and B's a pair of n8 tiles at
    // a time, each pair's MMAs issued as it arrives.
    uint32_t af[DB ? 2 : 1][MI][4], bf[DB ? 2 : 1][NI][2];
    auto fragments = [&](int f, int stage, int kk) {
        f %= DB ? 2 : 1;
        const __half *as = As + stage * BM * MMA_K, *bs = Bs + stage * BN * MMA_K;
#pragma unroll
        for (int i = 0; i < MI; i++)
            ldsm_x4(af[f][i], as + swz_chunk(wm * WTM + i * 16 + (lane & 15), kk * 2 + (lane >> 4)) * 8);
#pragma unroll
        for (int j = 0; j < NI / 2; j++) {
            uint32_t r[4];
            ldsm_x4(r, bs + swz_chunk(wn * WTN + j * 16 + (lane >> 4) * 8 + (lane & 7), kk * 2 + ((lane >> 3) & 1)) * 8);
            bf[f][2 * j][0] = r[0];
            bf[f][2 * j][1] = r[1];
            bf[f][2 * j + 1][0] = r[2];
            bf[f][2 * j + 1][1] = r[3];
        }
    };
    // The barrier before a step's MMAs: its stage has landed, the stage
    // the step before it read is free for the loads STAGES - 1 steps on.
    int2 step = make_int2(0, 0); // the flags of the step whose stage landed last
    auto next_stage = [&]() {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        step = stage_step[st];
        if (ld.live()) {
            load_stage(lst, ld);
            ld.next(share);
        }
        cp_async_commit();
        lst = after(lst);
    };

    // The F32 sums; with WHOLE, not written until the tile closes, from
    // the F16 sums, so no F32 array is live beside them and the fragments
    // in the mainloop.
    float acc[MI][NI][4];
    int left;
    {
        const Share sh = share();
        left = (int)(sh.hi - sh.lo);
    }
    // With DB, the first fragments of the step that comes next, read
    // where nothing else needs the registers: here, at the end of a
    // step that does not close its segment, and after the epilogue of
    // one that does (its stage landed at the barrier before the last
    // MMAs). Read before the epilogue, they would be live across it.
    auto first_fragments = [&]() {
        if constexpr (DB)
            if (left > 0) fragments(0, st, 0);
    };
    if (DB && left > 0) {
        next_stage();
        first_fragments();
    }
    while (left > 0) {
        if constexpr (!DB) next_stage();
        const int tile = step.x;
        const long long first = (long long)tile * steps;
        if (!WHOLE && (step.y & STEP_OPENS)) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;
        }
        // One k step, or with ACC16 the steps to the end of a chunk of
        // ACC16_STEPS (or of the segment), summed in F16 and then added.
        uint32_t hacc[ACC16 ? MI : 1][ACC16 ? NI : 1][2];
        if constexpr (ACC16) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++) hacc[i][j][0] = hacc[i][j][1] = 0u;
        }
        bool closes, chunk_done;
        int flags;
        for (bool again = false;; again = true) {
            if (!DB && again) {
                next_stage();
            }
            flags = step.y;
            const int k = flags & STEP_K;
            closes = flags & STEP_CLOSES;
            chunk_done = !ACC16 || closes || (!WHOLE && (k + 1) % ACC16_STEPS == 0);
            const int cur = st;
            if constexpr (!DB) {
                const __half *as = As + cur * BM * MMA_K, *bs = Bs + cur * BN * MMA_K;
#pragma unroll
                for (int kk = 0; kk < KK; kk++) {
#pragma unroll
                    for (int i = 0; i < MI; i++)
                        ldsm_x4(af[0][i], as + swz_chunk(wm * WTM + i * 16 + (lane & 15), kk * 2 + (lane >> 4)) * 8);
#pragma unroll
                    for (int j = 0; j < NI / 2; j++) {
                        uint32_t r[4];
                        ldsm_x4(r, bs + swz_chunk(wn * WTN + j * 16 + (lane >> 4) * 8 + (lane & 7),
                                                  kk * 2 + ((lane >> 3) & 1)) *
                                            8);
#pragma unroll
                        for (int i = 0; i < MI; i++) {
                            if constexpr (ACC16) {
                                mma16816_f16(hacc[i][2 * j], af[0][i], r[0], r[1]);
                                mma16816_f16(hacc[i][2 * j + 1], af[0][i], r[2], r[3]);
                            } else {
                                mma16816(acc[i][2 * j], af[0][i], r[0], r[1]);
                                mma16816(acc[i][2 * j + 1], af[0][i], r[2], r[3]);
                            }
                        }
                    }
                }
                left--;
                st = after(st);
            } else {
#pragma unroll
            for (int kk = 0; kk < KK; kk++) {
                if (kk + 1 < KK) {
                    fragments((kk + 1) & 1, cur, kk + 1);
                } else {
                    left--;
                    st = after(st);
                    if (left > 0) {
                        next_stage();
                        // At a segment's end they wait for the epilogue.
                        if (!closes) fragments((kk + 1) & 1, st, 0);
                    }
                }
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NI; j++) {
                        if constexpr (ACC16)
                            mma16816_f16(hacc[i][j], af[(kk & 1) % (DB ? 2 : 1)][i], bf[(kk & 1) % (DB ? 2 : 1)][j][0],
                                         bf[(kk & 1) % (DB ? 2 : 1)][j][1]);
                        else
                            mma16816(acc[i][j], af[(kk & 1) % (DB ? 2 : 1)][i], bf[(kk & 1) % (DB ? 2 : 1)][j][0],
                                     bf[(kk & 1) % (DB ? 2 : 1)][j][1]);
                    }
            }
            }
            if (chunk_done) break;
        }
        if constexpr (ACC16 && !WHOLE) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int h = 0; h < 2; h++) {
                        const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[i][j][h]));
                        acc[i][j][2 * h] += v.x;
                        acc[i][j][2 * h + 1] += v.y;
                    }
        }
        if (!closes) continue;
        // This block's product of the tile. WHOLE: a segment is one chunk,
        // and its F16 sum, in F32, the product.
        auto product = [&](int i, int j, int e) -> float {
            if constexpr (WHOLE) {
                const float2 v = __half22float2(*reinterpret_cast<const __half2 *>(&hacc[i][j][e >> 1]));
                return e & 1 ? v.y : v.x;
            } else {
                return acc[i][j][e];
            }
        };

        if (!(flags & STEP_ENDS)) {
            // The start of a tile a later block finishes.
            float *slot = g.ws + (size_t)blockIdx.x * SLOT + threadIdx.x;
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) __stcg(slot + ((i * NI + j) * 4 + e) * NT, product(i, j, e));
            sk_raise(g.flags + blockIdx.x);
            first_fragments();
            continue;
        }
        // WHOLE: the F32 sums first written here, from the F16 sums, once
        // the fragments are dead, so none is live in the mainloop.
        if constexpr (WHOLE) {
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] = product(i, j, e);
        }
        // The earlier blocks' partial products, added in F32 from the
        // last block to the first.
        const Share sh = share();
        for (int b = (int)blockIdx.x - 1; b >= 0 && share_start(sh, b + 1, g.min_steps) > first; b--) {
            sk_wait(g.flags + b, g.fault);
            const float *slot = g.ws + (size_t)b * SLOT + threadIdx.x;
#pragma unroll
            for (int i = 0; i < MI; i++)
#pragma unroll
                for (int j = 0; j < NI; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) acc[i][j][e] += __ldcg(slot + ((i * NI + j) * 4 + e) * NT);
        }
        auto total = [&](int i, int j, int e) -> float { return acc[i][j][e]; };

        // The finished tile, from registers.
        const int n0 = (tile % nt) * BN, m0 = (tile / nt) * BM;
        const int cw = n0 + wn * WTN;
        // Every index into acc below is a template constant (static_for),
        // never a loop the compiler may leave rolled.
        // Bias, and GELU, in place first. An F16 output is packed as each
        // n8 tile is done, two half2 of its four values, and its floats
        // not read again: half the registers live into the stores.
        constexpr bool f16_out = sizeof(TOut) == 2;
        static_assert(!f16_out || EPI != EPI_PLAIN, "F16 outputs have a bias, which packs them");
        [[maybe_unused]] uint32_t packed[f16_out ? MI : 1][f16_out ? NI : 1][2];
        // WHOLE: an F16 output's values taken from the F16 sums as they
        // are packed; an F32 output's bias added as it is stored, or with
        // the residual.
        if constexpr (EPI != EPI_PLAIN && (!WHOLE || f16_out))
            static_for<NI>([&](auto J) {
                constexpr int j = J.value;
                const int c = cw + j * 8 + tq * 2;
                const float b0 = c < N ? __ldg(g.bias + c) : 0.0f, b1 = c < N ? __ldg(g.bias + c + 1) : 0.0f;
                static_for<MI>([&](auto I) {
                    constexpr int i = I.value;
                    if constexpr (WHOLE) {
                        static_for<2>([&](auto H) {
                            constexpr int h = H.value;
                            float v0 = total(i, j, 2 * h) + b0, v1 = total(i, j, 2 * h + 1) + b1;
                            if constexpr (EPI == EPI_GELU) {
                                v0 = gelu(v0);
                                v1 = gelu(v1);
                            }
                            const __half2 p = __floats2half2_rn(v0, v1);
                            packed[i][j][h] = *reinterpret_cast<const uint32_t *>(&p);
                        });
                    } else {
#pragma unroll
                        for (int e = 0; e < 4; e++) {
                            float &v = acc[i][j][e];
                            v += e & 1 ? b1 : b0;
                            if constexpr (EPI == EPI_GELU) v = gelu(v);
                        }
                        if constexpr (f16_out)
                            static_for<2>([&](auto H) {
                                constexpr int h = H.value;
                                const __half2 p = __floats2half2_rn(acc[i][j][2 * h], acc[i][j][2 * h + 1]);
                                packed[i][j][h] = *reinterpret_cast<const uint32_t *>(&p);
                            });
                    }
                });
            });
        if constexpr (ROW_LN) {
            // The residual, then the LayerNorm of each row, which this
            // tile holds whole: a lane's 2 x NI values of a row summed, a
            // quad's four by shuffles, then the WN warps across the row
            // in warp order through shared memory; the mean, then the
            // variance about it, each over the row's N columns.
            float *x = static_cast<float *>(g.out);
            static_for<MI>([&](auto I) {
                static_for<2>([&](auto H) {
                    constexpr int i = I.value, h = H.value;
                    const int t = m0 + wm * WTM + i * 16 + gq + h * 8;
                    static_for<NI>([&](auto J) {
                        constexpr int j = J.value;
                        const int c = cw + j * 8 + tq * 2;
                        float2 r = make_float2(0.0f, 0.0f);
                        if (t < M && c < N) r = __ldcg(reinterpret_cast<const float2 *>(x + (size_t)t * N + c));
                        // WHOLE: the F32 sums made here, where the F16
                        // ones end.
                        float v0 = total(i, j, 2 * h), v1 = total(i, j, 2 * h + 1);
                        if constexpr (WHOLE) {
                            v0 += c < N ? __ldg(g.bias + c) : 0.0f;
                            v1 += c < N ? __ldg(g.bias + c + 1) : 0.0f;
                        }
                        acc[i][j][2 * h] = c < N ? v0 + r.x : 0.0f;
                        acc[i][j][2 * h + 1] = c < N ? v1 + r.y : 0.0f;
                    });
                });
            });
            float mean[MI][2], inv[MI][2];
            static_for<2>([&](auto PASS) {
                constexpr int pass = PASS.value;
                static_for<MI>([&](auto I) {
                    static_for<2>([&](auto H) {
                        constexpr int i = I.value, h = H.value;
                        float q = 0.0f;
                        static_for<NI>([&](auto J) {
                            constexpr int j = J.value;
#pragma unroll
                            for (int e = 0; e < 2; e++) {
                                const float v = acc[i][j][2 * h + e];
                                if constexpr (pass == 0) {
                                    q += v;
                                } else if (cw + j * 8 + tq * 2 < N) {
                                    const float d = v - mean[i][h];
                                    q += d * d;
                                }
                            }
                        });
                        q += __shfl_xor_sync(0xffffffffu, q, 1);
                        q += __shfl_xor_sync(0xffffffffu, q, 2);
                        if (tq == 0) row_sums[pass][wm * WTM + i * 16 + gq + h * 8][wn] = q;
                    });
                });
                __syncthreads();
                static_for<MI>([&](auto I) {
                    static_for<2>([&](auto H) {
                        constexpr int i = I.value, h = H.value;
                        const float *rs = row_sums[pass][wm * WTM + i * 16 + gq + h * 8];
                        float q = rs[0];
#pragma unroll
                        for (int w = 1; w < WN; w++) q += rs[w];
                        if constexpr (pass == 0)
                            mean[i][h] = q / (float)N;
                        else
                            inv[i][h] = 1.0f / sqrtf(q / (float)N + g.eps);
                    });
                });
            });
            __half *x16 = reinterpret_cast<__half *>(g.x16);
            static_for<NI / 4>([&](auto JQ) {
                static_for<MI>([&](auto I) {
                    static_for<2>([&](auto H) {
                        constexpr int jq = JQ.value, i = I.value, h = H.value;
                        const int t = m0 + wm * WTM + i * 16 + gq + h * 8;
                        uint32_t mine[4];
                        static_for<4>([&](auto U) {
                            constexpr int u = U.value;
                            const int c = cw + (jq * 4 + u) * 8 + tq * 2;
                            float y0 = 0.0f, y1 = 0.0f;
                            if (c < N) {
                                const float2 wv = __ldg(reinterpret_cast<const float2 *>(g.ln_w + c));
                                const float2 bv = __ldg(reinterpret_cast<const float2 *>(g.ln_b + c));
                                const float a0 = acc[i][jq * 4 + u][2 * h], a1 = acc[i][jq * 4 + u][2 * h + 1];
                                y0 = __fadd_rn(__fmul_rn((a0 - mean[i][h]) * inv[i][h], wv.x), bv.x);
                                y1 = __fadd_rn(__fmul_rn((a1 - mean[i][h]) * inv[i][h], wv.y), bv.y);
                                if (t < M) *reinterpret_cast<float2 *>(x + (size_t)t * N + c) = make_float2(y0, y1);
                            }
                            const __half2 p = __floats2half2_rn(y0, y1);
                            mine[u] = *reinterpret_cast<const uint32_t *>(&p);
                        });
                        if (x16) {
                            const uint4 w = quad_gather(mine, tq);
                            const int c = cw + (jq * 4 + tq) * 8;
                            if (t < M && c < N) *reinterpret_cast<uint4 *>(x16 + (size_t)t * N + c) = w;
                        }
                    });
                });
            });
            first_fragments();
            continue;
        }
        if constexpr (sizeof(TOut) == 4) {
            // F32: a float2 a lane, 32 contiguous bytes a quad.
            static_for<MI>([&](auto I) {
                static_for<2>([&](auto H) {
                    constexpr int i = I.value, h = H.value;
                    const int t = m0 + wm * WTM + i * 16 + gq + h * 8;
                    if (t >= M) return;
                    float2 *row = reinterpret_cast<float2 *>(out_at<float>(g, t, cw + tq * 2));
                    static_for<NI>([&](auto J) {
                        constexpr int j = J.value;
                        if (cw + j * 8 >= N) return;
                        float2 v = make_float2(total(i, j, 2 * h), total(i, j, 2 * h + 1));
                        if constexpr (WHOLE && EPI != EPI_PLAIN) {
                            const int c = cw + j * 8 + tq * 2;
                            v.x += c < N ? __ldg(g.bias + c) : 0.0f;
                            v.y += c < N ? __ldg(g.bias + c + 1) : 0.0f;
                            if constexpr (EPI == EPI_GELU) {
                                v.x = gelu(v.x);
                                v.y = gelu(v.y);
                            }
                        }
                        if constexpr (EPI == EPI_ADD_LN) {
                            const float2 r = row[j * 4];
                            v.x += r.x;
                            v.y += r.y;
                        }
                        row[j * 4] = v;
                    });
                });
            });
        } else {
            // F16: lane tq of a quad stores n8 tile jq * 4 + tq's 8 columns.
            // Their head-major place (QKV) is worked out once for all the
            // rows, outside the loops over them: its divisions compile to
            // a call, which the loops must not hold.
            const bool whole = EPI != EPI_QKV || g.head_dim % 8 == 0;
            static_for<NI / 4>([&](auto JQ) {
                const int c = cw + (JQ.value * 4 + tq) * 8;
                [[maybe_unused]] size_t col = 0;
                [[maybe_unused]] int d = 0;
                if constexpr (EPI == EPI_QKV)
                    if (c < N) {
                        const int which = c / g.hidden, hc = c - which * g.hidden;
                        const int head = hc / g.head_dim;
                        d = hc - head * g.head_dim;
                        col = ((size_t)(which * g.heads + head) * g.tcap) * g.head_dim + d;
                    }
                static_for<MI>([&](auto I) {
                    static_for<2>([&](auto H) {
                        constexpr int jq = JQ.value, i = I.value, h = H.value;
                        const int t = m0 + wm * WTM + i * 16 + gq + h * 8;
                        uint32_t mine[4];
                        static_for<4>([&](auto U) { mine[U.value] = packed[i][jq * 4 + U.value][h]; });
                        const uint4 w = quad_gather(mine, tq);
                        if (t >= M || c >= N) return;
                        if constexpr (EPI == EPI_QKV) {
                            __half *o = static_cast<__half *>(g.out) + (size_t)t * g.head_dim;
                            if (whole)
                                *reinterpret_cast<uint4 *>(o + col) = w;
                            else
                                qkv_store8(g, o, col, d, w);
                        } else {
                            *reinterpret_cast<uint4 *>(out_at<TOut>(g, t, c)) = w;
                        }
                    });
                });
            });
        }
        if constexpr (EPI == EPI_ADD_LN) ln_rows_when_done<NT>(g, m0, min(BM, M - m0), g.rows_done + m0 / BM, nt);
        first_fragments();
    }
    cp_async_wait<0>();
#else
    (void)g;
    __trap();
#endif
}

/* The tile a GEMM runs: TURBO_CUDA_TILE's, or TILE_DEFAULT, which each
 * kernel resolves to its own; the tensor cores take the FMA kernel's
 * 16 x 8 micro-tile as plain 128 x 128. */
Tile resolve_tile(bool mma, Tile t) {
    if (t == TILE_128x128_16x8 && mma) return TILE_128x128;
    return t;
}

/* A GEMM kernel, its threads and its dynamic shared memory. */
struct GemmKernel {
    void (*fn)(GemmArgs);
    int threads;
    size_t smem;
    int bm, bn;
    int per_sm; /* the blocks to an SM its launch bounds ask for */
};

template <int BM, int BN, int TM, typename TIn, int EPI, typename TOut> GemmKernel simt_kernel() {
    return {gemm_simt_kernel<BM, BN, TM, TIn, EPI, TOut>,
            (BM / TM) * (BN / 8),
            simt_gemm_smem<BM, BN, TIn>(),
            BM,
            BN,
            simt_min_blocks<BM, BN>()};
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI, typename TOut, typename TIn, bool ACC16 = false>
GemmKernel mma_kernel() {
    return {gemm_mma_kernel<BM, BN, WM, WN, STAGES, EPI, TOut, TIn, ACC16>,
            WM * WN * 32,
            mma_gemm_smem<BM, BN, STAGES, TOut>(),
            BM,
            BN,
            mma_min_blocks<BM, BN, STAGES, TOut>()};
}

template <int BM, int BN, int WM, int WN, int STAGES, int EPI, typename TOut, bool ACC16, bool DB,
          bool WHOLE = false>
GemmKernel swz_kernel() {
    return {gemm_swz_kernel<BM, BN, WM, WN, STAGES, EPI, TOut, ACC16, DB, WHOLE>,
            WM * WN * 32,
            swz_gemm_smem<BM, BN, STAGES>(),
            BM,
            BN,
            swz_min_blocks<BM, BN, STAGES>()};
}

/* The swizzled kernel's tiles (F16 operands), three stages each:
 * TILE_SWIZZLED 128 x 128 over four warps of 64 x 64, two blocks to an
 * SM; TILE_SWIZZLED_256x128 the same but 256 x 128 over eight such warps,
 * one block to an SM, for GELU; TILE_SWIZZLED_8W the eight-warp mix. */
template <typename TOut, int EPI, bool ACC16> GemmKernel swz_for(Tile t) {
    constexpr bool wide = EPI == EPI_QKV || EPI == EPI_GELU;
    // F16 sums over the whole of k: 128 x 128 over four warps of 64 x 64,
    // TensorRT's shape, at four stages (one block to an SM), three (two)
    // or two (three); 64 x 384, whole rows, over eight warps of 32 x 96
    // at three stages for the GEMMs with the LayerNorm in their epilogue;
    // and 256 x 128 over eight warps of 64 x 64 at three stages for QKV
    // and GELU, their F16 outputs. The others take 128 x 128 at three.
    if (t == TILE_F16_WHOLE_K) return swz_kernel<128, 128, 2, 2, 4, EPI, TOut, true, true, true>();
    if (t == TILE_F16_WHOLE_K_2) return swz_kernel<128, 128, 2, 2, 2, EPI, TOut, true, true, true>();
    if constexpr (EPI == EPI_ADD_LN)
        if (t == TILE_F16_WHOLE_K_ROWS) return swz_kernel<64, ROW_LN_WIDTH, 2, 4, 3, EPI, TOut, true, true, true>();
    if constexpr (wide && sizeof(TOut) == 2)
        if (t == TILE_F16_WHOLE_K_256) return swz_kernel<256, 128, 4, 2, 3, EPI, TOut, true, true, true>();
    if (t == TILE_F16_WHOLE_K_3 || t == TILE_F16_WHOLE_K_ROWS || t == TILE_F16_WHOLE_K_256)
        return swz_kernel<128, 128, 2, 2, 3, EPI, TOut, true, true, true>();
    if constexpr (EPI == EPI_ADD_LN && !ACC16)
        if (t == TILE_SWIZZLED_ROWS) return swz_kernel<64, ROW_LN_WIDTH, 2, 4, 3, EPI, TOut, false, true>();
    if (t == TILE_SWIZZLED_ROWS) t = TILE_SWIZZLED_8W;
    // F16 accumulators only at the eight-warp mix's shapes.
    if constexpr (wide) {
        // Warps of 32 x 64 with F16 accumulators, 64 x 32 without: the
        // layouts ptxas fits in 128 registers.
        if (ACC16) return swz_kernel<128, 128, 4, 2, 3, EPI, TOut, true, false>();
        if (t == TILE_SWIZZLED_8W) return swz_kernel<128, 128, 2, 4, 3, EPI, TOut, false, false>();
    } else {
        // (The plain product fits in 128 registers only as warps of 16 x 64.)
        if constexpr (ACC16 && EPI == EPI_PLAIN) return swz_kernel<128, 64, 8, 1, 3, EPI, TOut, true, false>();
        if constexpr (ACC16 && EPI != EPI_PLAIN) return swz_kernel<128, 64, 4, 2, 3, EPI, TOut, true, false>();
        // Warps of 32 x 32 without the pipelined mainloop, like the other
        // eight-warp shapes: on eight warps the cross-step pipeline itself
        // costs time, not the registers (the plain product on an RTX 4080,
        // dense rows: 961 us pipelined, 960 with one block to an SM and so
        // no register cap, 899 without).
        if (t == TILE_SWIZZLED_8W) return swz_kernel<128, 64, 4, 2, 4, EPI, TOut, false, false>();
    }
    if constexpr (!ACC16) {
        if constexpr (EPI == EPI_GELU)
            if (t == TILE_SWIZZLED_256x128) return swz_kernel<256, 128, 4, 2, 3, EPI, TOut, false, true>();
        return swz_kernel<128, 128, 2, 2, 3, EPI, TOut, false, true>();
    }
    __builtin_unreachable();
}

/* The FMA kernel's tiles, 8 x 8 outputs to a thread but the 16 x 8 of
 * 128 x 128 over 128 threads; 128 x 64 by default. */
template <typename TIn, typename TOut, int EPI> GemmKernel simt_for(Tile t) {
    if (t == TILE_DEFAULT) t = TILE_128x64;
    switch (t) {
    case TILE_64x64: return simt_kernel<64, 64, 8, TIn, EPI, TOut>();
    case TILE_128x128: return simt_kernel<128, 128, 8, TIn, EPI, TOut>();
    case TILE_128x128_16x8: return simt_kernel<128, 128, 16, TIn, EPI, TOut>();
    default: return simt_kernel<128, 64, 8, TIn, EPI, TOut>();
    }
}

/* The tensor cores' tiles. The F16 default is TILE_EIGHT_WARPS: 128 x 128
 * at two stages over eight warps of 32 x 64 for QKV and GELU, 128 x 64 at
 * three over eight warps of 32 x 32 for the others (128 x 64 is also
 * TF32's default, for every GEMM). Cutlass's shape, warps of 64 x 64 so
 * each ldmatrix feeds four MMAs, one block to an SM at three stages of
 * k 32, is selectable: 256 x 128 over eight warps and 128 x 128 over
 * four; on an RTX 4080 it is slower than the eight-warp tiles. 64 x 64, four warps of 32 x 32, at
 * four stages. An F32 output tile of 256 x 128 does not fit shared
 * memory, so such a GEMM takes 128 x 128 over four warps instead. */
template <typename TOut, int EPI, typename TIn> GemmKernel mma_for(Tile t) {
    constexpr bool f16 = sizeof(TIn) == 2, wide = EPI == EPI_QKV || EPI == EPI_GELU;
    if constexpr (f16) {
        if (t == TILE_EIGHT_WARPS_F16_ACCUMULATE) {
            if (wide) return mma_kernel<128, 128, 4, 2, 2, EPI, TOut, TIn, true>();
            return mma_kernel<128, 64, 4, 2, 3, EPI, TOut, TIn, true>();
        }
        switch (t) {
        case TILE_SWIZZLED:
        case TILE_SWIZZLED_8W:
        case TILE_SWIZZLED_256x128:
        case TILE_SWIZZLED_ROWS:
        case TILE_F16_WHOLE_K:
        case TILE_F16_WHOLE_K_3:
        case TILE_F16_WHOLE_K_ROWS:
        case TILE_F16_WHOLE_K_256:
        case TILE_F16_WHOLE_K_2: return swz_for<TOut, EPI, false>(t);
        case TILE_SWIZZLED_8W_F16_ACCUMULATE: return swz_for<TOut, EPI, true>(TILE_SWIZZLED_8W);
        default: break;
        }
    }
    if (t == TILE_DEFAULT && f16) t = TILE_EIGHT_WARPS;
    if (t == TILE_EIGHT_WARPS) t = wide && f16 ? TILE_128x128 : TILE_128x64;
    if (t == TILE_256x128 && sizeof(TOut) == 4) t = TILE_128x128_4W;
    switch (t) {
    case TILE_64x64: return mma_kernel<64, 64, 2, 2, 4, EPI, TOut, TIn>();
    case TILE_128x128: return mma_kernel<128, 128, 4, 2, 2, EPI, TOut, TIn>();
    case TILE_128x128_4W: return mma_kernel<128, 128, 2, 2, 3, EPI, TOut, TIn>();
    case TILE_256x128:
        if constexpr (sizeof(TOut) == 2) return mma_kernel<256, 128, 4, 2, 3, EPI, TOut, TIn>();
        return mma_kernel<128, 128, 2, 2, 3, EPI, TOut, TIn>();
    default: return mma_kernel<128, 64, 4, 2, 3, EPI, TOut, TIn>();
    }
}

GemmKernel gemm_kernel(Epilogue e, bool half, bool tc, Tile tile) {
    const Tile t = resolve_tile(tc, tile);
    if (tc && half) {
        switch (e) {
        case EPI_QKV: return mma_for<__half, EPI_QKV, __half>(t);
        case EPI_GELU: return mma_for<__half, EPI_GELU, __half>(t);
        case EPI_ADD_LN: return mma_for<float, EPI_ADD_LN, __half>(t);
        default: return mma_for<float, EPI_PLAIN, __half>(t);
        }
    }
    if (tc) {
        switch (e) {
        case EPI_QKV: return mma_for<float, EPI_QKV, float>(t);
        case EPI_GELU: return mma_for<float, EPI_GELU, float>(t);
        case EPI_ADD_LN: return mma_for<float, EPI_ADD_LN, float>(t);
        default: return mma_for<float, EPI_PLAIN, float>(t);
        }
    }
    if (half) {
        switch (e) {
        case EPI_QKV: return simt_for<__half, __half, EPI_QKV>(t);
        case EPI_GELU: return simt_for<__half, __half, EPI_GELU>(t);
        case EPI_ADD_LN: return simt_for<__half, float, EPI_ADD_LN>(t);
        default: return simt_for<__half, float, EPI_PLAIN>(t);
        }
    }
    switch (e) {
    case EPI_QKV: return simt_for<float, float, EPI_QKV>(t);
    case EPI_GELU: return simt_for<float, float, EPI_GELU>(t);
    case EPI_ADD_LN: return simt_for<float, float, EPI_ADD_LN>(t);
    default: return simt_for<float, float, EPI_PLAIN>(t);
    }
}

/* The cuBLAS epilogues: a block per token, looping over the run's tokens,
 * its threads over the columns. */
template <typename TOut>
__global__ void __launch_bounds__(256) qkv_epilogue_kernel(const float *raw, GemmArgs g) {
    const int tokens = g.info->tokens;
    TOut *o = static_cast<TOut *>(g.out);
    for (int t = blockIdx.x; t < tokens; t += gridDim.x) {
        const float *r = raw + (size_t)t * g.n;
        for (int c = threadIdx.x; c < g.n; c += blockDim.x)
            put(o + qkv_column(g, c) + (size_t)t * g.head_dim, r[c] + g.bias[c]);
    }
}

template <typename TOut> __global__ void __launch_bounds__(256) gelu_epilogue_kernel(const float *raw, GemmArgs g) {
    const int tokens = g.info->tokens;
    TOut *o = static_cast<TOut *>(g.out);
    for (int t = blockIdx.x; t < tokens; t += gridDim.x) {
        const float *r = raw + (size_t)t * g.n;
        for (int c = threadIdx.x; c < g.n; c += blockDim.x) put(o + (size_t)t * g.n + c, gelu(r[c] + g.bias[c]));
    }
}

// ---- Attention ---------------------------------------------------------------------
//
// Work items are (row, head, query tile), rows longest first as the
// packing ordered them; a block loops over items. Keys and values of the
// row's head go through shared memory a chunk at a time (the whole row in
// one chunk up to the chunk's length, which make_plan sets), and the
// softmax is carried across chunks and key blocks by its running largest
// score, rescaling what was summed before. Keys past the row's length do
// not exist in the packed row; a masked key inside it scores -1e30 below
// the rest, from the packing's key_bias, read only for rows that have one.

/* The item's row slot: the last i with item_start[i] <= item. */
__device__ inline int item_slot(const int32_t *item_start, int batch, int item) {
    int lo = 0, hi = batch - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (item_start[mid] <= item)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

struct Item {
    int b, head, q0, n, base, holes;
};

__device__ inline Item decode(const AttnArgs &a, int item, int batch) {
    const Packing &p = a.p;
    const int slot = item_slot(p.item_start, batch, item);
    const int local = item - p.item_start[slot];
    Item it;
    it.b = p.order[slot];
    it.head = local % a.heads;
    it.q0 = (local / a.heads) * a.queries;
    it.n = p.len[it.b];
    it.base = p.start[it.b];
    it.holes = p.holes[it.b];
    return it;
}

// -- FMA attention: EXACT, and FASTEST where mma does not apply ----------------------
//
// Eight warps: two groups of 32 queries, a lane per query, each group's
// keys split four ways among its warps. A lane keeps its query and its
// context in registers, so every shared memory read is a broadcast; keys
// go four at a time, four independent dot products. The four warps'
// partial softmaxes are merged in a fixed order at the end. Splitting the
// keys keeps each lane's chain of dependent work short: most rows are far
// shorter than a block's queries.

constexpr int SIMT_ATT_THREADS = 256;
constexpr int SIMT_ATT_QUERIES = 64;
constexpr int KSPLITS = 4;
/* Keys scored at once by a lane: independent dot products. */
constexpr int G = 4;

template <typename T, int HD> constexpr size_t simt_smem(int chunk) {
    const size_t keys = (size_t)chunk * HD * 2 * sizeof(T) + (size_t)chunk * sizeof(float);
    const size_t merge = (size_t)KSPLITS * 2 * (HD + 2) * 32 * sizeof(float);
    return keys > merge ? keys : merge;
}

/* A head's rows, [rows][d] in global memory, into shared memory rows of
 * HD, zero past d. */
template <typename T, int HD> __device__ inline void load_rows(T *dst, const T *src, int rows, int d) {
    if (d == HD) {
        constexpr int PER = 16 / sizeof(T);
        const int n = rows * HD / PER;
        const uint4 *s = reinterpret_cast<const uint4 *>(src);
        uint4 *o = reinterpret_cast<uint4 *>(dst);
        for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = s[i];
    } else {
        for (int i = threadIdx.x; i < rows * HD; i += blockDim.x) {
            const int r = i / HD, c = i % HD;
            dst[i] = c < d ? src[(size_t)r * d + c] : T(0.0f);
        }
    }
}

/* The dot product of q with a shared memory row of HD, in two halves:
 * even and odd columns, each in column order. */
template <int HD, typename T> __device__ inline float dot_row(const float (&q)[HD], const T *k) {
    float s0 = 0.0f, s1 = 0.0f;
#pragma unroll
    for (int c = 0; c < HD; c += 4) {
        float v[4];
        lds4(k + c, v);
        s0 = fmaf(q[c], v[0], s0);
        s1 = fmaf(q[c + 1], v[1], s1);
        s0 = fmaf(q[c + 2], v[2], s0);
        s1 = fmaf(q[c + 3], v[3], s1);
    }
    return s0 + s1;
}

/* o += e v for a shared memory row v of HD. */
template <int HD, typename T> __device__ inline void add_row(float (&o)[HD], float e, const T *v) {
#pragma unroll
    for (int c = 0; c < HD; c += 4) {
        float x[4];
        lds4(v + c, x);
#pragma unroll
        for (int u = 0; u < 4; u++) o[c + u] = fmaf(e, x[u], o[c + u]);
    }
}

template <typename T, int HD>
__global__ void __launch_bounds__(SIMT_ATT_THREADS, HD <= 32 ? 2 : 1) attention_simt_kernel(AttnArgs a) {
    extern __shared__ __align__(16) unsigned char att_sm[];
    const int chunk = a.chunk;
    T *Ks = reinterpret_cast<T *>(att_sm);
    T *Vs = Ks + (size_t)chunk * HD;
    float *kbs = reinterpret_cast<float *>(Vs + (size_t)chunk * HD);
    float *merge = reinterpret_cast<float *>(att_sm);
    const int items = a.p.info->items, batch = a.p.info->batch;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, grp = warp & 1, ks = warp >> 1;
    const int d = a.head_dim;
    const T *qkv = static_cast<const T *>(a.qkv);
    const size_t hs = (size_t)a.tcap * d;

    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const Item it = decode(a, item, batch);
        const T *Qg = qkv + (size_t)it.head * hs + (size_t)it.base * d;
        const T *Kg = qkv + (size_t)(a.heads + it.head) * hs + (size_t)it.base * d;
        const T *Vg = qkv + (size_t)(2 * a.heads + it.head) * hs + (size_t)it.base * d;
        const int q = it.q0 + grp * 32 + lane;
        const bool live = it.q0 + grp * 32 < it.n;
        float qr[HD], o[HD];
        if (q < it.n && d == HD) {
#pragma unroll
            for (int c = 0; c < HD; c += 4) {
                float v[4];
                load4(Qg + (size_t)q * HD + c, v);
#pragma unroll
                for (int u = 0; u < 4; u++) qr[c + u] = v[u];
            }
        } else {
#pragma unroll
            for (int c = 0; c < HD; c++) qr[c] = q < it.n && c < d ? to_float(Qg[(size_t)q * d + c]) : 0.0f;
        }
#pragma unroll
        for (int c = 0; c < HD; c++) o[c] = 0.0f;
        float m = -INFINITY, l = 0.0f;
        for (int c0 = 0; c0 < it.n; c0 += chunk) {
            const int cn = min(chunk, it.n - c0);
            __syncthreads();
            load_rows<T, HD>(Ks, Kg + (size_t)c0 * d, cn, d);
            load_rows<T, HD>(Vs, Vg + (size_t)c0 * d, cn, d);
            if (it.holes)
                for (int j = threadIdx.x; j < cn; j += SIMT_ATT_THREADS) kbs[j] = a.p.key_bias[it.base + c0 + j];
            __syncthreads();
            if (!live) continue;
            const int per = ((cn + KSPLITS - 1) / KSPLITS + G - 1) / G * G;
            const int klo = ks * per, khi = min(cn, klo + per);
            for (int j0 = klo; j0 < khi; j0 += G) {
                float s[G];
#pragma unroll
                for (int u = 0; u < G; u++) {
                    const int j = j0 + u;
                    s[u] = j < khi ? dot_row<HD>(qr, Ks + (size_t)j * HD) * a.scale + (it.holes ? kbs[j] : 0.0f)
                                   : -INFINITY;
                }
                float mx = m;
#pragma unroll
                for (int u = 0; u < G; u++) mx = fmaxf(mx, s[u]);
                float sum = 0.0f;
#pragma unroll
                for (int u = 0; u < G; u++) {
                    s[u] = expf(s[u] - mx);
                    sum += s[u];
                }
                // Rescale what was summed only when a lane's largest score
                // grew: where it did not, the factor is exactly 1.
                if (__any_sync(FULL, mx > m)) {
                    const float corr = expf(m - mx);
                    l *= corr;
#pragma unroll
                    for (int c = 0; c < HD; c++) o[c] *= corr;
                }
                l += sum;
#pragma unroll
                for (int u = 0; u < G; u++)
                    if (j0 + u < khi) add_row<HD>(o, s[u], Vs + (size_t)(j0 + u) * HD);
                m = mx;
            }
        }
        // Every warp is done with the keys: the merge reuses their memory.
        __syncthreads();
        {
            float *mine = merge + (size_t)(ks * 2 + grp) * (HD + 2) * 32;
            mine[lane] = m;
            mine[32 + lane] = l;
#pragma unroll
            for (int c = 0; c < HD; c++) mine[(2 + c) * 32 + lane] = o[c];
        }
        __syncthreads();
        if (ks == 0 && q < it.n) {
            float mm = -INFINITY;
#pragma unroll
            for (int s = 0; s < KSPLITS; s++) mm = fmaxf(mm, merge[(size_t)(s * 2 + grp) * (HD + 2) * 32 + lane]);
            float L = 0.0f;
#pragma unroll
            for (int c = 0; c < HD; c++) o[c] = 0.0f;
#pragma unroll
            for (int s = 0; s < KSPLITS; s++) {
                const float *r = merge + (size_t)(s * 2 + grp) * (HD + 2) * 32;
                const float w = expf(r[lane] - mm);
                L += r[32 + lane] * w;
#pragma unroll
                for (int c = 0; c < HD; c++) o[c] += r[(2 + c) * 32 + lane] * w;
            }
            const float inv = 1.0f / L;
            T *dst = static_cast<T *>(a.ctx) + (size_t)(it.base + q) * a.hidden + it.head * d;
#pragma unroll
            for (int c = 0; c < HD; c++)
                if (c < d) put(dst + c, o[c] * inv);
        }
    }
}


constexpr int MMA_QUERIES = 64;

// -- FMA attention as two small GEMMs: EXACT and MODEL -------------------------------
//
// A block of 128 threads takes 32 queries of one head of one row and the
// row's keys 64 at a time. S = Q K^T for the chunk is a register tile,
// each thread 4 queries by 4 keys, summed over the head's values in
// order from Q and K held transposed in shared memory (a warp's reads
// are two broadcast rows of Q and 16 consecutive 16-byte pieces of K).
// The 16 threads of a query row agree on its running largest score and
// the chunk's sum by four shuffles; the probabilities replace K in shared
// memory, and O += P V is a second register tile, each thread the same 4
// queries by HD / 16 values, summed over the chunk's keys in position
// order. Keys and chunks go in position order, so every sum's order is
// fixed.

constexpr int TILED_THREADS = 128;
constexpr int TILED_QUERIES = 32;
constexpr int TILED_CHUNK = 64;
constexpr int TILED_LDQ = TILED_QUERIES + 4; // Q^T's row, and P's
constexpr int TILED_LDK = TILED_CHUNK + 4;   // K^T's row

/* W consecutive floats from shared memory in one load: 1, 2 or 4. */
template <int W> __device__ inline void lds_vec(const float *p, float (&v)[W]) {
    if constexpr (W == 4) {
        const float4 x = *reinterpret_cast<const float4 *>(p);
        v[0] = x.x;
        v[1] = x.y;
        v[2] = x.z;
        v[3] = x.w;
    } else if constexpr (W == 2) {
        const float2 x = *reinterpret_cast<const float2 *>(p);
        v[0] = x.x;
        v[1] = x.y;
    } else {
        v[0] = *p;
    }
}

template <int HD> __host__ __device__ constexpr size_t tiled_region() {
    const size_t k = (size_t)HD * TILED_LDK, p = (size_t)TILED_CHUNK * TILED_LDQ;
    return k > p ? k : p;
}

template <int HD> constexpr size_t tiled_smem_floats() {
    return (size_t)HD * TILED_LDQ + tiled_region<HD>() + (size_t)TILED_CHUNK * HD + TILED_CHUNK;
}

template <typename T, int HD> size_t tiled_smem(int) { return tiled_smem_floats<HD>() * sizeof(float); }

template <typename T, int HD>
__global__ void __launch_bounds__(TILED_THREADS, HD == 64 ? 2 : 4) attention_tiled_kernel(AttnArgs a) {
    constexpr int W = HD / 16; // values of the context per thread and query
    extern __shared__ __align__(16) unsigned char att_sm[];
    float *Qt = reinterpret_cast<float *>(att_sm); // [HD][LDQ]
    float *Kt = Qt + HD * TILED_LDQ;                // [HD][LDK], then P [CHUNK][LDQ]
    float *Ps = Kt;
    float *Vs = Kt + tiled_region<HD>(); // [CHUNK][HD]
    float *kbs = Vs + TILED_CHUNK * HD;  // [CHUNK]
    const int items = a.p.info->items, batch = a.p.info->batch;
    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    const int d = a.head_dim;
    const T *qkv = static_cast<const T *>(a.qkv);
    const size_t hs = (size_t)a.tcap * d;

    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const Item it = decode(a, item, batch);
        const T *Qg = qkv + (size_t)it.head * hs + (size_t)it.base * d;
        const T *Kg = qkv + (size_t)(a.heads + it.head) * hs + (size_t)it.base * d;
        const T *Vg = qkv + (size_t)(2 * a.heads + it.head) * hs + (size_t)it.base * d;
        const int nq = min(TILED_QUERIES, it.n - it.q0);
        __syncthreads();
        for (int i = threadIdx.x; i < TILED_QUERIES * HD; i += TILED_THREADS) {
            const int q = i / HD, c = i - q * HD;
            Qt[c * TILED_LDQ + q] = q < nq && c < d ? to_float(Qg[(size_t)(it.q0 + q) * d + c]) : 0.0f;
        }
        float o[4][W], m[4], l[4];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            m[i] = -INFINITY;
            l[i] = 0.0f;
#pragma unroll
            for (int w = 0; w < W; w++) o[i][w] = 0.0f;
        }
        for (int c0 = 0; c0 < it.n; c0 += TILED_CHUNK) {
            const int cn = min(TILED_CHUNK, it.n - c0);
            __syncthreads();
            if (d == HD) {
                // K^T four keys by four values a thread, transposed in its
                // registers, so each store is 16 bytes and a quarter warp's
                // stores are 8 consecutive ones; V as it is.
                for (int i = threadIdx.x; i < (TILED_CHUNK / 4) * (HD / 4); i += TILED_THREADS) {
                    const int k = 4 * (i % (TILED_CHUNK / 4)), c = 4 * (i / (TILED_CHUNK / 4));
                    float r[4][4];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        if (k + u < cn) {
                            load4(Kg + (size_t)(c0 + k + u) * HD + c, r[u]);
                        } else {
#pragma unroll
                            for (int v = 0; v < 4; v++) r[u][v] = 0.0f;
                        }
                    }
#pragma unroll
                    for (int v = 0; v < 4; v++)
                        *reinterpret_cast<float4 *>(Kt + (c + v) * TILED_LDK + k) =
                            make_float4(r[0][v], r[1][v], r[2][v], r[3][v]);
                }
                for (int i = threadIdx.x; i < cn * (HD / 4); i += TILED_THREADS) {
                    const int k = i / (HD / 4), c = 4 * (i % (HD / 4));
                    float r[4];
                    load4(Vg + (size_t)(c0 + k) * HD + c, r);
                    *reinterpret_cast<float4 *>(Vs + k * HD + c) = make_float4(r[0], r[1], r[2], r[3]);
                }
            } else {
                for (int i = threadIdx.x; i < cn * HD; i += TILED_THREADS) {
                    const int k = i / HD, c = i - k * HD;
                    const bool in = c < d;
                    const size_t at = (size_t)(c0 + k) * d + c;
                    Kt[c * TILED_LDK + k] = in ? to_float(Kg[at]) : 0.0f;
                    Vs[k * HD + c] = in ? to_float(Vg[at]) : 0.0f;
                }
            }
            if (it.holes)
                for (int j = threadIdx.x; j < cn; j += TILED_THREADS) kbs[j] = a.p.key_bias[it.base + c0 + j];
            __syncthreads();

            // S = Q K^T, this thread's queries 4 ty + i and keys 4 tx + j.
            float s[4][4];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) s[i][j] = 0.0f;
#pragma unroll 8
            for (int c = 0; c < HD; c++) {
                const float4 qv = *reinterpret_cast<const float4 *>(Qt + c * TILED_LDQ + 4 * ty);
                const float4 kv = *reinterpret_cast<const float4 *>(Kt + c * TILED_LDK + 4 * tx);
                const float qa[4] = {qv.x, qv.y, qv.z, qv.w}, ka[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) s[i][j] = fmaf(qa[i], ka[j], s[i][j]);
            }
            // The online softmax, a query row's 16 threads agreeing by
            // shuffles within their half warp.
            float corr[4];
#pragma unroll
            for (int i = 0; i < 4; i++) {
                float mx = -INFINITY;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int k = 4 * tx + j;
                    s[i][j] = k < cn ? s[i][j] * a.scale + (it.holes ? kbs[k] : 0.0f) : -INFINITY;
                    mx = fmaxf(mx, s[i][j]);
                }
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(FULL, mx, off));
                const float mn = fmaxf(m[i], mx);
                corr[i] = expf(m[i] - mn);
                float sum = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    s[i][j] = expf(s[i][j] - mn);
                    sum += s[i][j];
                }
#pragma unroll
                for (int off = 8; off > 0; off >>= 1) sum += __shfl_xor_sync(FULL, sum, off);
                l[i] = l[i] * corr[i] + sum;
                m[i] = mn;
#pragma unroll
                for (int w = 0; w < W; w++) o[i][w] *= corr[i];
            }
            // P over K^T, once every thread is done with K.
            __syncthreads();
#pragma unroll
            for (int j = 0; j < 4; j++)
                *reinterpret_cast<float4 *>(Ps + (4 * tx + j) * TILED_LDQ + 4 * ty) =
                    make_float4(s[0][j], s[1][j], s[2][j], s[3][j]);
            __syncthreads();

            // O += P V over the chunk's keys in position order.
#pragma unroll 4
            for (int k = 0; k < cn; k++) {
                const float4 pv = *reinterpret_cast<const float4 *>(Ps + k * TILED_LDQ + 4 * ty);
                const float pa[4] = {pv.x, pv.y, pv.z, pv.w};
                float v[W];
                lds_vec<W>(Vs + k * HD + W * tx, v);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int w = 0; w < W; w++) o[i][w] = fmaf(pa[i], v[w], o[i][w]);
            }
        }
        T *ctx = static_cast<T *>(a.ctx);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int q = 4 * ty + i;
            if (q >= nq) continue;
            const float inv = 1.0f / l[i];
            T *dst = ctx + (size_t)(it.base + it.q0 + q) * a.hidden + it.head * d;
#pragma unroll
            for (int w = 0; w < W; w++)
                if (W * tx + w < d) put(dst + W * tx + w, o[i][w] * inv);
        }
    }
}

// -- Tensor core attention: FASTEST, heads of 32 or 64 --------------------------------
//
// Four warps, 16 queries each, as flash attention 2 lays them out: S =
// Q K^T and O += P V as mma.sync m16n8k16 with F32 accumulators, 64 keys
// at a time, the softmax in F32 on the S fragments (each row's four lanes
// agree by two shuffles), P rounded to F16 in the registers the PV
// product reads. Rows padded to D + 8 halves keep ldmatrix free of bank
// conflicts.

constexpr int MMA_ATT_THREADS = 128;

template <int D> constexpr size_t mma_smem(int chunk) {
    return (size_t)(MMA_QUERIES + 2 * chunk) * (D + 8) * sizeof(__half) + (size_t)chunk * sizeof(float);
}

template <int D> __global__ void __launch_bounds__(MMA_ATT_THREADS) attention_mma_kernel(AttnArgs a) {
#ifndef TURBO_NO_MMA
    constexpr int LD = D + 8, DK = D / 16, DN = D / 8;
    extern __shared__ __align__(16) unsigned char att_sm[];
    const int chunk = a.chunk;
    __half *Qs = reinterpret_cast<__half *>(att_sm);
    __half *Ks = Qs + MMA_QUERIES * LD;
    __half *Vs = Ks + (size_t)chunk * LD;
    float *kbs = reinterpret_cast<float *>(Vs + (size_t)chunk * LD);
    const int items = a.p.info->items, batch = a.p.info->batch;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const __half *qkv = static_cast<const __half *>(a.qkv);
    const size_t hs = (size_t)a.tcap * D;
    const uint4 zero = make_uint4(0, 0, 0, 0);

    for (int item = blockIdx.x; item < items; item += gridDim.x) {
        const Item it = decode(a, item, batch);
        const __half *Qg = qkv + (size_t)it.head * hs + (size_t)it.base * D;
        const __half *Kg = qkv + (size_t)(a.heads + it.head) * hs + (size_t)it.base * D;
        const __half *Vg = qkv + (size_t)(2 * a.heads + it.head) * hs + (size_t)it.base * D;
        const bool live = it.q0 + warp * 16 < it.n;
        __syncthreads();
        for (int i = threadIdx.x; i < MMA_QUERIES * (D / 8); i += MMA_ATT_THREADS) {
            const int r = i / (D / 8), c = (i % (D / 8)) * 8;
            *reinterpret_cast<uint4 *>(Qs + r * LD + c) =
                it.q0 + r < it.n ? *reinterpret_cast<const uint4 *>(Qg + (size_t)(it.q0 + r) * D + c) : zero;
        }
        uint32_t qf[DK][4];
        float o[DN][4];
#pragma unroll
        for (int j = 0; j < DN; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) o[j][e] = 0.0f;
        float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.0f, 0.0f};

        for (int c0 = 0; c0 < it.n; c0 += chunk) {
            const int cn = min(chunk, it.n - c0), padded = (cn + 63) & ~63;
            if (c0 > 0) __syncthreads();
#pragma unroll 4
            for (int i = threadIdx.x; i < padded * (D / 8); i += MMA_ATT_THREADS) {
                const int r = i / (D / 8), c = (i % (D / 8)) * 8;
                const bool in = r < cn;
                *reinterpret_cast<uint4 *>(Ks + r * LD + c) =
                    in ? *reinterpret_cast<const uint4 *>(Kg + (size_t)(c0 + r) * D + c) : zero;
                *reinterpret_cast<uint4 *>(Vs + r * LD + c) =
                    in ? *reinterpret_cast<const uint4 *>(Vg + (size_t)(c0 + r) * D + c) : zero;
            }
            if (it.holes)
                for (int j = threadIdx.x; j < cn; j += MMA_ATT_THREADS) kbs[j] = a.p.key_bias[it.base + c0 + j];
            __syncthreads();
            if (c0 == 0)
#pragma unroll
                for (int k = 0; k < DK; k++)
                    ldsm_x4(qf[k], Qs + (warp * 16 + (lane & 15)) * LD + k * 16 + (lane >> 4) * 8);
            if (!live) continue;

            for (int k0 = 0; k0 < cn; k0 += 64) {
                float s[8][4];
#pragma unroll
                for (int j = 0; j < 8; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) s[j][e] = 0.0f;
#pragma unroll
                for (int k = 0; k < DK; k++)
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        uint32_t r[4];
                        ldsm_x4(r, Ks + (k0 + j * 16 + (lane >> 4) * 8 + (lane & 7)) * LD + k * 16 +
                                        ((lane >> 3) & 1) * 8);
                        mma16816(s[2 * j], qf[k], r[0], r[1]);
                        mma16816(s[2 * j + 1], qf[k], r[2], r[3]);
                    }
                float mx[2] = {m[0], m[1]};
#pragma unroll
                for (int j = 0; j < 8; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        const int key = k0 + j * 8 + (lane & 3) * 2 + (e & 1);
                        float v = s[j][e] * a.scale;
                        if (key >= cn)
                            v = -INFINITY;
                        else if (it.holes)
                            v += kbs[key];
                        s[j][e] = v;
                        mx[e >> 1] = fmaxf(mx[e >> 1], v);
                    }
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    mx[h] = fmaxf(mx[h], __shfl_xor_sync(FULL, mx[h], 1));
                    mx[h] = fmaxf(mx[h], __shfl_xor_sync(FULL, mx[h], 2));
                }
                const float corr[2] = {expf(m[0] - mx[0]), expf(m[1] - mx[1])};
                float sum[2] = {0.0f, 0.0f};
#pragma unroll
                for (int j = 0; j < 8; j++)
#pragma unroll
                    for (int e = 0; e < 4; e++) {
                        s[j][e] = expf(s[j][e] - mx[e >> 1]);
                        sum[e >> 1] += s[j][e];
                    }
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    l[h] = l[h] * corr[h] + sum[h];
                    m[h] = mx[h];
                }
#pragma unroll
                for (int j = 0; j < DN; j++) {
                    o[j][0] *= corr[0];
                    o[j][1] *= corr[0];
                    o[j][2] *= corr[1];
                    o[j][3] *= corr[1];
                }
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    const uint32_t pa[4] = {pack_half2(s[2 * kk][0], s[2 * kk][1]),
                                            pack_half2(s[2 * kk][2], s[2 * kk][3]),
                                            pack_half2(s[2 * kk + 1][0], s[2 * kk + 1][1]),
                                            pack_half2(s[2 * kk + 1][2], s[2 * kk + 1][3])};
#pragma unroll
                    for (int j = 0; j < D / 16; j++) {
                        uint32_t r[4];
                        ldsm_x4_trans(r, Vs + (k0 + kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + j * 16 +
                                             (lane >> 4) * 8);
                        mma16816(o[2 * j], pa, r[0], r[1]);
                        mma16816(o[2 * j + 1], pa, r[2], r[3]);
                    }
                }
            }
        }
        if (!live) continue;
#pragma unroll
        for (int h = 0; h < 2; h++) {
            l[h] += __shfl_xor_sync(FULL, l[h], 1);
            l[h] += __shfl_xor_sync(FULL, l[h], 2);
        }
        const float inv[2] = {1.0f / l[0], 1.0f / l[1]};
        __half *ctx = static_cast<__half *>(a.ctx);
#pragma unroll
        for (int h = 0; h < 2; h++) {
            const int q = it.q0 + warp * 16 + (lane >> 2) + h * 8;
            if (q >= it.n) continue;
            __half *dst = ctx + (size_t)(it.base + q) * a.hidden + it.head * D + (lane & 3) * 2;
#pragma unroll
            for (int j = 0; j < DN; j++)
                *reinterpret_cast<__half2 *>(dst + j * 8) =
                    __floats2half2_rn(o[j][2 * h] * inv[h], o[j][2 * h + 1] * inv[h]);
        }
    }
#else
    (void)a;
    __trap();
#endif
}


// The same attention, 128 queries to a block of eight warps (a warp per
// 16), so each chunk of keys and values in shared memory serves twice the
// queries. Keys and values go 64 at a time through three buffers filled
// by cp.async, the next two chunks' in flight while this one's products
// and softmax run, one barrier to a chunk; the next item's queries and
// first chunks load during an item's last. The softmax is in base 2:
// scale x log2(e) goes into each exponent's one multiply-add, and
// ex2.approx takes the place of expf (a masked key's p is 0, as the key
// bias of -1e30 makes it in the other kernels). Only rows whose items the
// pack puts first, longest first, change the schedule, not the sums: each
// query's keys go in position order, 64 at a time.
// TURBO_CUDA_ATTENTION=exact keeps the earlier softmax, bit for bit: the
// scores scaled before the largest is taken, and exp2f.

constexpr int FA_QUERIES = 128, FA_THREADS = 256, FA_KEYS = 64;

template <int D> constexpr size_t fa_smem(int) {
    return (size_t)(FA_QUERIES + 6 * FA_KEYS) * (D + 8) * sizeof(__half) + 3 * FA_KEYS * sizeof(float);
}

template <int D> constexpr int fa_min_blocks() { return fa_smem<D>(0) * 2 <= 96 * 1024 ? 2 : 1; }

/* TURBO_CUDA_ATTENTION=fa32, heads of 32: two tiles of 16 queries to a
 * warp, the 128 queries over four warps, so each key and value fragment
 * read feeds four products in place of two. Shared memory holds it to
 * two blocks to an SM, eight warps; held to 168 registers for three it
 * spills, so it takes up to 255 for two. */
constexpr int FA32_THREADS = 128;

#ifndef TURBO_NO_MMA
// A masked key's score in place of its product, unscaled, as the key
// bias's -1e30 is scaled: a power of two, so its product with the scale
// is exact and, while a row has no live key yet, each masked key's p is
// exactly 1, as it was with the bias; the first live key's rescaling
// takes them to 0.
constexpr float FA_MASKED = -0x1p100f;

/* 2^x by ex2.approx alone: within 2 ulp, and 0 for results below 2^-126,
 * where exp2f takes four instructions to be exact. Each p is rounded to
 * F16 for P V, far coarser than 2 ulp of F32 down to F16's least normal,
 * and a result below 2^-24 is 0 in F16 either way. */
__device__ inline float ex2_approx(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}

// One chunk of keys for a warp's MT tiles of 16 queries: S = Q K^T,
// the running softmax, O += P V, each key and value fragment serving
// every tile. The largest score and m are kept unscaled (the scale is
// positive), and each p is ex2.approx of one fused multiply-add, s sl2 -
// m sl2. EXACT (TURBO_CUDA_ATTENTION=exact): the scores are scaled first
// and the key bias added, and each p is exp2f of the scaled score less
// m, the arithmetic before either. WHOLE: the chunk's 64 keys are all
// the row's and none is masked, so no score is tested; the sums are the
// same either way, and a query's are the same at any MT.
template <int D, int MT, bool WHOLE, bool EXACT>
__device__ __forceinline__ void fa_chunk(const uint32_t (&qf)[MT][D / 16][4], float (&o)[MT][D / 8][4],
                                         float (&m)[MT][2], float (&l)[MT][2], const __half *Ks, const __half *Vs,
                                         const float *kb, int cn, int holes, float sl2, int lane) {
    constexpr int LD = D + 8, DK = D / 16, DN = D / 8;
    float s[MT][8][4];
#pragma unroll
    for (int t = 0; t < MT; t++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) s[t][j][e] = 0.0f;
#pragma unroll
    for (int k = 0; k < DK; k++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t r[4];
            ldsm_x4(r, Ks + (j * 16 + (lane >> 4) * 8 + (lane & 7)) * LD + k * 16 + ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int t = 0; t < MT; t++) {
                mma16816(s[t][2 * j], qf[t][k], r[0], r[1]);
                mma16816(s[t][2 * j + 1], qf[t][k], r[2], r[3]);
            }
        }
    float corr[MT][2];
#pragma unroll
    for (int t = 0; t < MT; t++) {
        float mx[2] = {m[t][0], m[t][1]};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                float v = EXACT ? s[t][j][e] * sl2 : s[t][j][e];
                if constexpr (!WHOLE) {
                    const int key = j * 8 + (lane & 3) * 2 + (e & 1);
                    if (key >= cn)
                        v = -INFINITY;
                    else if (EXACT && holes)
                        v += kb[key];
                    else if (!EXACT && holes && kb[key] != 0.0f)
                        v = FA_MASKED;
                }
                s[t][j][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(FULL, mx[h], 1));
            mx[h] = fmaxf(mx[h], __shfl_xor_sync(FULL, mx[h], 2));
        }
        float nm[2];
#pragma unroll
        for (int h = 0; h < 2; h++) {
            corr[t][h] = EXACT ? exp2f(m[t][h] - mx[h]) : ex2_approx((m[t][h] - mx[h]) * sl2);
            nm[h] = EXACT ? -mx[h] : -mx[h] * sl2;
        }
        float sum[2] = {0.0f, 0.0f};
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                s[t][j][e] =
                    EXACT ? exp2f(s[t][j][e] + nm[e >> 1]) : ex2_approx(fmaf(s[t][j][e], sl2, nm[e >> 1]));
                sum[e >> 1] += s[t][j][e];
            }
#pragma unroll
        for (int h = 0; h < 2; h++) {
            l[t][h] = l[t][h] * corr[t][h] + sum[h];
            m[t][h] = mx[h];
        }
#pragma unroll
        for (int j = 0; j < DN; j++) {
            o[t][j][0] *= corr[t][0];
            o[t][j][1] *= corr[t][0];
            o[t][j][2] *= corr[t][1];
            o[t][j][3] *= corr[t][1];
        }
    }
#pragma unroll
    for (int kk = 0; kk < 4; kk++) {
        uint32_t pa[MT][4];
#pragma unroll
        for (int t = 0; t < MT; t++) {
            pa[t][0] = pack_half2(s[t][2 * kk][0], s[t][2 * kk][1]);
            pa[t][1] = pack_half2(s[t][2 * kk][2], s[t][2 * kk][3]);
            pa[t][2] = pack_half2(s[t][2 * kk + 1][0], s[t][2 * kk + 1][1]);
            pa[t][3] = pack_half2(s[t][2 * kk + 1][2], s[t][2 * kk + 1][3]);
        }
#pragma unroll
        for (int j = 0; j < D / 16; j++) {
            uint32_t r[4];
            ldsm_x4_trans(r, Vs + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * LD + j * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int t = 0; t < MT; t++) {
                mma16816(o[t][2 * j], pa[t], r[0], r[1]);
                mma16816(o[t][2 * j + 1], pa[t], r[2], r[3]);
            }
        }
    }
}
#endif

template <int D, bool EXACT, int MT>
__global__ void __launch_bounds__(MT == 1 ? FA_THREADS : FA32_THREADS, (MT == 1 ? fa_min_blocks<D>() : 2))
    attention_fa_kernel(AttnArgs a) {
#ifndef TURBO_NO_MMA
    constexpr int LD = D + 8, DK = D / 16, DN = D / 8, CH = D / 8; // CH: 16-byte chunks of a row
    constexpr int NT = FA_THREADS / MT, WQ = 16 * MT;               // threads, and each warp's queries
    extern __shared__ __align__(16) unsigned char att_sm[];
    __half *Qs = reinterpret_cast<__half *>(att_sm);
    __half *KV = Qs + FA_QUERIES * LD; // buffer b: K at KV + 2 b FA_KEYS LD, V after it
    float *kbs = reinterpret_cast<float *>(KV + 6 * FA_KEYS * LD);
    const int items = a.p.info->items, batch = a.p.info->batch;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const __half *qkv = static_cast<const __half *>(a.qkv);
    const size_t hs = (size_t)a.tcap * D;
    const float sl2 = a.scale * 1.4426950408889634f;
    int item = blockIdx.x;
    if (item >= items) return;

    // An item's queries into Qs; its keys and values c0 on into buffer
    // b, and the key bias with plain loads, which the barrier before the
    // chunk's use orders.
    auto load_q = [&](const Item &x) {
        const __half *Qg = qkv + (size_t)x.head * hs + (size_t)x.base * D;
        for (int i = threadIdx.x; i < FA_QUERIES * CH; i += NT) {
            const int r = i / CH, c = (i % CH) * 8;
            const bool in = x.q0 + r < x.n;
            cp_async16(Qs + r * LD + c, in ? Qg + (size_t)(x.q0 + r) * D + c : Qg, in);
        }
    };
    auto load_chunk = [&](const Item &x, int b, int c0) {
        const __half *Kg = qkv + (size_t)(a.heads + x.head) * hs + (size_t)x.base * D;
        const __half *Vg = Kg + (size_t)a.heads * hs;
        __half *Ks = KV + (size_t)(2 * b) * FA_KEYS * LD, *Vs = Ks + FA_KEYS * LD;
        for (int i = threadIdx.x; i < FA_KEYS * CH; i += NT) {
            const int r = i / CH, c = (i % CH) * 8;
            const bool in = c0 + r < x.n;
            cp_async16(Ks + r * LD + c, in ? Kg + (size_t)(c0 + r) * D + c : Kg, in);
            cp_async16(Vs + r * LD + c, in ? Vg + (size_t)(c0 + r) * D + c : Vg, in);
        }
        if (x.holes && threadIdx.x < FA_KEYS && c0 + (int)threadIdx.x < x.n)
            kbs[b * FA_KEYS + threadIdx.x] = a.p.key_bias[x.base + c0 + threadIdx.x];
    };

    // The chunks of the block's items run on as one sequence, chunk c of
    // it in buffer c mod 3, and each group of loads committed is one
    // chunk's (empty where there is none to load): the first item's
    // queries and first chunk here, and its second; then one at the top
    // of each chunk, but two at the top of an item's last, the next
    // item's queries and first chunk, and its second.
    Item it = decode(a, item, batch);
    load_q(it);
    load_chunk(it, 0, 0);
    cp_async_commit();
    if (it.n > FA_KEYS) load_chunk(it, 1, FA_KEYS);
    cp_async_commit();
    for (int b = 0;;) {
        const int next = item + gridDim.x;
        const bool more = next < items;
        const Item nx = more ? decode(a, next, batch) : it;
        const bool live = it.q0 + warp * WQ < it.n;
        const int chunks = (it.n + FA_KEYS - 1) / FA_KEYS;

        uint32_t qf[MT][DK][4];
        float o[MT][DN][4], m[MT][2], l[MT][2];
#pragma unroll
        for (int t = 0; t < MT; t++) {
#pragma unroll
            for (int j = 0; j < DN; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) o[t][j][e] = 0.0f;
            m[t][0] = m[t][1] = -INFINITY;
            l[t][0] = l[t][1] = 0.0f;
        }

        for (int c = 0; c < chunks; c++, b = b == 2 ? 0 : b + 1) {
            const int c0 = c * FA_KEYS, cn = min(FA_KEYS, it.n - c0);
            cp_async_wait<1>();
            // Chunk c is in, and every warp is done with chunk c - 1, so
            // its buffer takes chunk c + 2.
            __syncthreads();
            if (c == 0)
#pragma unroll
                for (int t = 0; t < MT; t++)
#pragma unroll
                    for (int k = 0; k < DK; k++)
                        ldsm_x4(qf[t][k], Qs + (warp * WQ + t * 16 + (lane & 15)) * LD + k * 16 + (lane >> 4) * 8);
            const int b1 = b == 2 ? 0 : b + 1, b2 = b == 0 ? 2 : b - 1;
            if (c + 2 < chunks) {
                load_chunk(it, b2, c0 + 2 * FA_KEYS);
            } else if (c + 1 == chunks && more) {
                // Qs is free once every warp has its fragments: at the
                // barrier above past chunk 0, after one more at it.
                if (chunks == 1) __syncthreads();
                load_q(nx);
                load_chunk(nx, b1, 0);
                cp_async_commit();
                if (nx.n > FA_KEYS) load_chunk(nx, b2, FA_KEYS);
            }
            cp_async_commit();
            if (live) {
                const __half *Ks = KV + (size_t)(2 * b) * FA_KEYS * LD, *Vs = Ks + FA_KEYS * LD;
                const float *kb = kbs + b * FA_KEYS;
                if (cn == FA_KEYS && !it.holes)
                    fa_chunk<D, MT, true, EXACT>(qf, o, m, l, Ks, Vs, kb, cn, it.holes, sl2, lane);
                else
                    fa_chunk<D, MT, false, EXACT>(qf, o, m, l, Ks, Vs, kb, cn, it.holes, sl2, lane);
            }
        }
        if (live) {
            __half *ctx = static_cast<__half *>(a.ctx);
#pragma unroll
            for (int t = 0; t < MT; t++) {
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    l[t][h] += __shfl_xor_sync(FULL, l[t][h], 1);
                    l[t][h] += __shfl_xor_sync(FULL, l[t][h], 2);
                }
                const float inv[2] = {1.0f / l[t][0], 1.0f / l[t][1]};
#pragma unroll
                for (int h = 0; h < 2; h++) {
                    const int q = it.q0 + warp * WQ + t * 16 + (lane >> 2) + h * 8;
                    if (q >= it.n) continue;
                    __half *dst = ctx + (size_t)(it.base + q) * a.hidden + it.head * D + (lane & 3) * 2;
#pragma unroll
                    for (int j = 0; j < DN; j++)
                        *reinterpret_cast<__half2 *>(dst + j * 8) =
                            __floats2half2_rn(o[t][j][2 * h] * inv[h], o[t][j][2 * h + 1] * inv[h]);
                }
            }
        }
        if (!more) break;
        item = next;
        it = nx;
    }
#else
    (void)a;
    __trap();
#endif
}


// ---- Weights -----------------------------------------------------------------------

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


/* The attention kernel for a session, its shared memory for a chunk of
 * keys, its block size, its queries per item and its longest chunk. */
struct AttnKernel {
    void (*fn)(AttnArgs);
    int threads;
    size_t (*smem)(int);
    int queries;
    int chunk_cap;
};

/* Keys per chunk for the FMA kernel. */
constexpr int SIMT_CHUNK = 128;

AttnKernel attention_kernel_for(const Shape &s) {
    const int d = s.hidden / s.heads;
    if (s.half && s.tensor_cores && (d == 32 || d == 64)) {
        if (s.wide_attention) {
            if (d == 32 && s.fa32)
                return {attention_fa_kernel<32, false, 2>, FA32_THREADS, fa_smem<32>, FA_QUERIES, FA_KEYS};
            if (d == 32)
                return {s.exact_exp2 ? attention_fa_kernel<32, true, 1> : attention_fa_kernel<32, false, 1>, FA_THREADS,
                        fa_smem<32>, FA_QUERIES, FA_KEYS};
            return {s.exact_exp2 ? attention_fa_kernel<64, true, 1> : attention_fa_kernel<64, false, 1>, FA_THREADS,
                    fa_smem<64>, FA_QUERIES, FA_KEYS};
        }
        if (d == 32) return {attention_mma_kernel<32>, MMA_ATT_THREADS, mma_smem<32>, MMA_QUERIES, 256};
        return {attention_mma_kernel<64>, MMA_ATT_THREADS, mma_smem<64>, MMA_QUERIES, 256};
    }
#define SIMT_ATT(T, HD)                                                                                                \
    (s.split_attention                                                                                                 \
         ? AttnKernel{attention_simt_kernel<T, HD>, SIMT_ATT_THREADS, simt_smem<T, HD>, SIMT_ATT_QUERIES, SIMT_CHUNK}  \
         : AttnKernel{attention_tiled_kernel<T, HD>, TILED_THREADS, tiled_smem<T, HD>, TILED_QUERIES, TILED_CHUNK})
    if (s.half) {
        if (d <= 16) return SIMT_ATT(__half, 16);
        if (d <= 32) return SIMT_ATT(__half, 32);
        return SIMT_ATT(__half, 64);
    }
    if (d <= 16) return SIMT_ATT(float, 16);
    if (d <= 32) return SIMT_ATT(float, 32);
    return SIMT_ATT(float, 64);
#undef SIMT_ATT
}

/* The blocks of fn that fit on the device at once, at least one. */
cudaError_t resident(const void *fn, int threads, size_t smem, int sms, int *out) {
    int per = 0;
    const cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, threads, smem);
    *out = (per > 0 ? per : 1) * sms;
    return e;
}

int cap(long long want, int most) { return (int)(want < most ? (want > 0 ? want : 1) : most); }

/* Every kernel of a run takes the same split of an SM's memory between
 * shared memory and L1, the most shared memory, so the device need not
 * change it between one kernel of the graph and the next. */
cudaError_t same_carveout(const void *fn) {
    return cudaFuncSetAttribute(fn, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
}

/* Float4s per lane of the row kernels for a hidden width. */
template <typename F> cudaError_t with_row_width(int hidden, F &&f) {
    if (hidden <= 128) return f(std::integral_constant<int, 1>{});
    if (hidden <= 256) return f(std::integral_constant<int, 2>{});
    if (hidden <= 384) return f(std::integral_constant<int, 3>{});
    if (hidden <= 512) return f(std::integral_constant<int, 4>{});
    if (hidden <= 768) return f(std::integral_constant<int, 6>{});
    if (hidden <= 1024) return f(std::integral_constant<int, 8>{});
    return f(std::integral_constant<int, 16>{});
}

} // namespace

// ---- Launchers -------------------------------------------------------------------------

namespace {

/* gemm_kernel's kernel, or, when its shared memory is more than the
 * device lets a block have, 128 x 64's. */
GemmKernel gemm_kernel_fitting(Epilogue e, bool half, bool tensor_cores, Tile tile) {
    const GemmKernel k = gemm_kernel(e, half, tensor_cores, tile);
    int dev = 0, optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        return k;
    return k.smem > (size_t)optin ? gemm_kernel(e, half, tensor_cores, TILE_128x64) : k;
}

} // namespace

cudaError_t gemm_prepare(Epilogue e, bool half, bool tensor_cores, Tile tile) {
    const GemmKernel k = gemm_kernel_fitting(e, half, tensor_cores, tile);
    const void *fn = reinterpret_cast<const void *>(k.fn);
    cudaError_t err = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)k.smem);
    if (err == cudaSuccess) err = same_carveout(fn);
    return err;
}

cudaError_t gemm_grid(Epilogue e, bool half, bool tensor_cores, Tile tile, int sms, int *grid, size_t *ws_floats,
                      bool *crowded) {
    const GemmKernel k = gemm_kernel_fitting(e, half, tensor_cores, tile);
    // As many blocks as fit, whatever the tokens: the kernel counts the
    // tiles of the run's M, read on the device, and shares them out.
    const cudaError_t err = resident(reinterpret_cast<const void *>(k.fn), k.threads, k.smem, sms, grid);
    *ws_floats = (size_t)*grid * k.bm * k.bn;
    if (crowded && *grid < k.per_sm * sms) *crowded = true;
    return err;
}

cudaError_t gemm(cudaStream_t s, Epilogue e, bool half, bool tensor_cores, Tile tile, const GemmArgs &g, int grid) {
    const GemmKernel k = gemm_kernel_fitting(e, half, tensor_cores, tile);
    void *args[] = {const_cast<GemmArgs *>(&g)};
    return cudaLaunchKernel(reinterpret_cast<const void *>(k.fn), dim3(grid), dim3(k.threads), args, k.smem, s);
}

cudaError_t qkv_epilogue(cudaStream_t s, const float *raw, const GemmArgs &g, bool half, const Plan &plan) {
    if (half)
        qkv_epilogue_kernel<__half><<<plan.epi_grid, 256, 0, s>>>(raw, g);
    else
        qkv_epilogue_kernel<float><<<plan.epi_grid, 256, 0, s>>>(raw, g);
    return cudaGetLastError();
}

cudaError_t gelu_epilogue(cudaStream_t s, const float *raw, const GemmArgs &g, bool half, const Plan &plan) {
    if (half)
        gelu_epilogue_kernel<__half><<<plan.epi_grid, 256, 0, s>>>(raw, g);
    else
        gelu_epilogue_kernel<float><<<plan.epi_grid, 256, 0, s>>>(raw, g);
    return cudaGetLastError();
}

cudaError_t make_plan(const Shape &s, Plan *p) {
    const AttnKernel ak = attention_kernel_for(s);
    const int nb = query_tiles(s.seq_cap, ak.queries) + 1;
    p->pack_smem = (size_t)3 * nb * sizeof(int32_t);
    p->rows_grid = cap(((long long)s.tcap + ROW_WARPS - 1) / ROW_WARPS, s.sms * 8);
    p->pool_grid = cap(s.batch_cap, s.sms * 4);
    p->column_pool = s.column_pool || s.hidden > 4 * POOL_BLOCK;
    p->epi_grid = cap(s.tcap, s.sms * 8);
    p->fetch_grid = cap(((long long)3 * s.tcap / 4 + FETCH_BLOCK * FETCH_LOADS - 1) / (FETCH_BLOCK * FETCH_LOADS),
                        s.sms * 2);
    // The attention output and second feed-forward GEMMs: the product
    // alone, or with the LayerNorm after it.
    p->fused_ln = s.fused_ln && s.hidden <= LN_FUSED_MAX_HIDDEN;
    p->ln_counts = p->fused_ln ? ln_counters(s.tcap) : 0;
    cudaError_t e = cudaSuccess;
    for (int i = 0; i < GEMM_COUNT && e == cudaSuccess; i++) {
        const Gemm g = (Gemm)i;
        const Epilogue ep = gemm_epilogue(g, p->fused_ln);
        const bool mma = gemm_mma(s, g);
        size_t ws = 0;
        e = gemm_prepare(ep, s.half, mma, s.gemm[g].tile);
        if (e == cudaSuccess)
            e = gemm_grid(ep, s.half, mma, s.gemm[g].tile, s.sms, &p->gemm_grid[g], &ws, &p->gemm_crowded);
        p->sk_floats = ws > p->sk_floats ? ws : p->sk_floats;
        p->sk_flags = p->gemm_grid[g] > p->sk_flags ? p->gemm_grid[g] : p->sk_flags;
    }
    if (e != cudaSuccess) return e;

    int chunk = ((s.seq_cap + 63) & ~63);
    if (chunk > ak.chunk_cap) chunk = ak.chunk_cap;
    while (chunk > 64 && ak.smem(chunk) > s.smem_optin) chunk -= 64;
    p->attn_chunk = chunk;
    p->attn_queries = ak.queries;
    p->attn_smem = ak.smem(chunk);
    if (p->attn_smem > s.smem_optin) return cudaErrorInvalidValue;
    const void *afn = reinterpret_cast<const void *>(ak.fn);
    e = cudaFuncSetAttribute(afn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)p->attn_smem);
    if (e == cudaSuccess) e = same_carveout(afn);
    int res = 0;
    if (e == cudaSuccess) e = resident(afn, ak.threads, p->attn_smem, s.sms, &res);
    const long long items = (long long)s.batch_cap * query_tiles(s.seq_cap, ak.queries) * s.heads;
    p->attn_grid = cap(items, res);
    if (e != cudaSuccess) return e;

    // The row kernels too, for this width.
    const void *rows[5] = {reinterpret_cast<const void *>(pack_rows_kernel),
                           reinterpret_cast<const void *>(pool_kernel), nullptr, nullptr,
                           reinterpret_cast<const void *>(fetch_rows_kernel)};
    with_row_width(s.hidden, [&](auto v) {
        rows[2] = reinterpret_cast<const void *>(embed_layer_norm_kernel<decltype(v)::value>);
        rows[3] = reinterpret_cast<const void *>(add_layer_norm_kernel<decltype(v)::value>);
        return cudaSuccess;
    });
    for (const void *fn : rows)
        if (e == cudaSuccess) e = same_carveout(fn);
    return e;
}

cudaError_t pack_rows(cudaStream_t s, const PackArgs &a, const Plan &plan) {
    pack_rows_kernel<<<1, PACK_BLOCK, plan.pack_smem, s>>>(a);
    return cudaGetLastError();
}

void pack_rows_node(const PackArgs *a, void **args, const Plan &plan, cudaKernelNodeParams *out) {
    args[0] = const_cast<PackArgs *>(a);
    out->func = reinterpret_cast<void *>(pack_rows_kernel);
    out->gridDim = dim3(1);
    out->blockDim = dim3(PACK_BLOCK);
    out->sharedMemBytes = (unsigned)plan.pack_smem;
    out->kernelParams = args;
    out->extra = nullptr;
}

const void *pack_rows_function() { return reinterpret_cast<const void *>(pack_rows_kernel); }

cudaError_t fetch_rows(cudaStream_t s, const FetchArgs &a, const Plan &plan) {
    fetch_rows_kernel<<<plan.fetch_grid, FETCH_BLOCK, 0, s>>>(a);
    return cudaGetLastError();
}

void fetch_rows_node(const FetchArgs *a, void **args, const Plan &plan, cudaKernelNodeParams *out) {
    args[0] = const_cast<FetchArgs *>(a);
    out->func = reinterpret_cast<void *>(fetch_rows_kernel);
    out->gridDim = dim3(plan.fetch_grid);
    out->blockDim = dim3(FETCH_BLOCK);
    out->sharedMemBytes = 0;
    out->kernelParams = args;
    out->extra = nullptr;
}

const void *fetch_rows_function() { return reinterpret_cast<const void *>(fetch_rows_kernel); }

cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *rows, const float *word, const float *position,
                             const float *type, const float *ln_w, const float *ln_b, float eps, const Packing &p,
                             int hidden, float *x, uint16_t *x16, const Plan &plan) {
    return with_row_width(hidden, [&](auto v) {
        embed_layer_norm_kernel<decltype(v)::value><<<plan.rows_grid, ROW_BLOCK, 0, s>>>(
            rows, word, position, type, ln_w, ln_b, eps, p, hidden, x, as_half(x16));
        return cudaGetLastError();
    });
}

cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *y, const float *bias,
                           const float *ln_w, const float *ln_b, float eps, const Info *info, int hidden,
                           uint16_t *x16, const Plan &plan) {
    return with_row_width(hidden, [&](auto v) {
        add_layer_norm_kernel<decltype(v)::value><<<plan.rows_grid, ROW_BLOCK, 0, s>>>(
            x, y, bias, ln_w, ln_b, eps, info, hidden, as_half(x16));
        return cudaGetLastError();
    });
}

cudaError_t pool(cudaStream_t s, const float *x, const int32_t *rows, const Packing &p, int hidden, float *out,
                 const Plan &plan) {
    if (plan.column_pool)
        pool_kernel<<<plan.pool_grid, POOL_BLOCK, 0, s>>>(x, rows, p, hidden, out);
    else
        pool_split_kernel<<<plan.pool_grid, POOL_BLOCK, 0, s>>>(x, rows, p, hidden, out);
    return cudaGetLastError();
}

cudaError_t attention(cudaStream_t s, const AttnArgs &a, const Shape &shape, const Plan &plan) {
    const AttnKernel ak = attention_kernel_for(shape);
    void *args[] = {const_cast<AttnArgs *>(&a)};
    return cudaLaunchKernel(reinterpret_cast<const void *>(ak.fn), dim3(plan.attn_grid), dim3(ak.threads), args,
                            plan.attn_smem, s);
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
    return cudaFuncGetAttributes(&a, pack_rows_kernel);
}

} // namespace turbo_cuda
