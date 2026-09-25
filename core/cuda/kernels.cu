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
 * fixed by the shapes alone, so the same rows give the same bits.
 */

#include "kernels.h"

#include <turbo/turbo.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cfloat>
#include <type_traits>

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

// ---- The packing -----------------------------------------------------------------

constexpr int PACK_BLOCK = 512;
constexpr int PACK_WARPS = PACK_BLOCK / 32;

/* Query tiles in a row of n tokens. */
__device__ __host__ inline int query_tiles(int n) { return (n + ATTENTION_QUERIES - 1) / ATTENTION_QUERIES; }

/* One block. A warp per row finds its last live position and whether a
 * masked one comes before it; each thread sums the lengths of a run of
 * rows, thread 0 scans the sums, and each thread writes its rows'
 * starts: integer sums in a fixed order. Then the rows are binned by
 * query tiles, most first, for attention's schedule: the counts are
 * integer atomics, and the order within a bin, the only thing the atomics
 * leave open, changes which block computes a row, never what it computes.
 * Each bin's rows have the same number of work items, so item_start does
 * not depend on that order either. Last, a warp per row writes each of
 * its tokens' row and key bias. */
__global__ void __launch_bounds__(PACK_BLOCK) pack_rows_kernel(PackArgs a) {
    extern __shared__ int32_t pack_sm[];
    const int nb = query_tiles(a.pitch) + 1; // bin k holds rows of nb - k tiles
    int32_t *bin_row = pack_sm, *bin_cur = pack_sm + nb, *bin_item = pack_sm + 2 * nb, *part = pack_sm + 3 * nb;
    const Packing &p = a.p;
    const int batch = a.run.batch, seq = a.run.seq, pitch = a.pitch;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

    for (int k = threadIdx.x; k < nb; k += PACK_BLOCK) bin_row[k] = 0;
    for (int b = warp; b < batch; b += PACK_WARPS) {
        const int32_t *m = a.mask + (size_t)b * pitch;
        int last = -1;
        for (int q = lane; q < seq; q += 32)
            if (m[q] != 0) last = q;
        last = warp_max_all(last);
        bool hole = false;
        for (int q = lane; q < last; q += 32) hole |= m[q] == 0;
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
        atomicAdd(&bin_row[nb - query_tiles(p.len[b])], 1);
    }
    part[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t run = 0;
        for (int i = 0; i < PACK_BLOCK; i++) {
            const int32_t n = part[i];
            part[i] = run;
            run += n;
        }
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
        in.tokens = run;
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
    int32_t at = part[threadIdx.x];
    for (int b = lo; b < hi; b++) {
        p.start[b] = at;
        at += p.len[b];
        const int tiles = query_tiles(p.len[b]), k = nb - tiles;
        const int slot = atomicAdd(&bin_cur[k], 1);
        p.order[slot] = b;
        p.item_start[slot] = bin_item[k] + (slot - bin_row[k]) * tiles * a.heads;
    }
    __syncthreads();

    for (int b = warp; b < batch; b += PACK_WARPS) {
        const int32_t *m = a.mask + (size_t)b * pitch;
        const int n = p.len[b], s0 = p.start[b];
        for (int q = lane; q < n; q += 32) {
            p.tok_row[s0 + q] = b;
            p.key_bias[s0 + q] = m[q] != 0 ? 0.0f : -1e30f;
        }
    }
}

// ---- Row kernels -------------------------------------------------------------------

/* Threads per block for the warp-per-token kernels: eight warps. */
constexpr int ROW_BLOCK = 256;
constexpr int ROW_WARPS = ROW_BLOCK / 32;

/* v = LayerNorm(v) over the n values a warp holds, V per lane (lane + 32 i
 * for i under V, those under n), with the mean, then the biased variance
 * about it, summed in F32; written to row, and in F16 to row16 when it is
 * not NULL. */
template <int V>
__device__ void layer_norm_regs(float (&v)[V], int n, const float *w, const float *b, float eps, float *row,
                                __half *row16) {
    const int lane = threadIdx.x & 31;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < V; i++)
        if (lane + 32 * i < n) s += v[i];
    const float mean = warp_sum_all(s) / (float)n;
    float q = 0.0f;
#pragma unroll
    for (int i = 0; i < V; i++)
        if (lane + 32 * i < n) {
            const float c = v[i] - mean;
            q += c * c;
        }
    const float var = warp_sum_all(q) / (float)n;
    const float inv = 1.0f / sqrtf(var + eps);
#pragma unroll
    for (int i = 0; i < V; i++) {
        const int d = lane + 32 * i;
        if (d >= n) continue;
        const float x = (v[i] - mean) * inv;
        const float y = __fadd_rn(__fmul_rn(x, w[d]), b[d]);
        row[d] = y;
        if (row16) row16[d] = __float2half_rn(y);
    }
}

template <int V>
__global__ void __launch_bounds__(ROW_BLOCK, 1)
    embed_layer_norm_kernel(const int32_t *ids, const int32_t *types, int pitch, const float *word,
                            const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                            Packing p, int hidden, float *x, __half *x16) {
    const int tokens = p.info->tokens, has_types = p.info->has_types;
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < tokens; t += gridDim.x * ROW_WARPS) {
        const int b = p.tok_row[t], pos = t - p.start[b];
        const size_t at = (size_t)b * pitch + pos;
        const float *w = word + (size_t)ids[at] * hidden;
        const float *ps = position + (size_t)pos * hidden;
        const float *tt = type + (size_t)(has_types ? types[at] : 0) * hidden;
        float v[V];
#pragma unroll
        for (int i = 0; i < V; i++) {
            const int d = lane + 32 * i;
            v[i] = d < hidden ? w[d] + ps[d] + tt[d] : 0.0f;
        }
        layer_norm_regs(v, hidden, ln_w, ln_b, eps, x + (size_t)t * hidden, x16 ? x16 + (size_t)t * hidden : nullptr);
    }
}

template <int V>
__global__ void __launch_bounds__(ROW_BLOCK, 1)
    add_layer_norm_kernel(float *x, const float *part, int splits, int tcap, const float *bias, const float *ln_w,
                          const float *ln_b, float eps, const Info *info, int hidden, __half *x16) {
    const int tokens = info->tokens;
    const int lane = threadIdx.x & 31;
    const size_t plane = (size_t)tcap * hidden;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < tokens; t += gridDim.x * ROW_WARPS) {
        float *row = x + (size_t)t * hidden;
        const float *pr = part + (size_t)t * hidden;
        float v[V];
#pragma unroll
        for (int i = 0; i < V; i++) {
            const int d = lane + 32 * i;
            if (d >= hidden) {
                v[i] = 0.0f;
                continue;
            }
            float y = pr[d];
            for (int s = 1; s < splits; s++) y += pr[s * plane + d];
            v[i] = row[d] + (y + bias[d]);
        }
        layer_norm_regs(v, hidden, ln_w, ln_b, eps, row, x16 ? x16 + (size_t)t * hidden : nullptr);
    }
}

constexpr int POOL_BLOCK = 128;

/* The block's sum of one value per thread, in every thread, in a fixed
 * order: each warp's shuffle tree, then the warps in turn. */
__device__ float block_sum(float v, float *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum_all(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float total = 0.0f;
    for (int w = 0; w < POOL_BLOCK / 32; w++) total += red[w];
    return total;
}

/* A block per row, looping over the run's rows. The mean's loads are
 * issued four positions at a time, and summed in position order. */
__global__ void __launch_bounds__(POOL_BLOCK)
    pool_kernel(const float *x, const int32_t *mask, int pitch, Packing p, int hidden, float *out) {
    __shared__ float red[POOL_BLOCK / 32];
    const Info in = *p.info;
    for (int b = blockIdx.x; b < in.batch; b += gridDim.x) {
        const int32_t *m = mask + (size_t)b * pitch;
        const int n = p.len[b];
        const float *rows = x + (size_t)p.start[b] * hidden;
        float *dst = out + (size_t)b * in.output_dim;
        float ss = 0.0f;
        for (int d = threadIdx.x; d < in.output_dim; d += POOL_BLOCK) {
            float val;
            if (in.pooling == TURBO_POOLING_CLS) {
                val = rows[d];
            } else if (in.pooling == TURBO_POOLING_LAST) {
                // The row's length ends at its last live token.
                val = rows[(size_t)(n - 1) * hidden + d];
            } else {
                // Mean over the tokens whose mask is 1, summed in position order.
                float s = 0.0f;
                unsigned c = 0;
                int q = 0;
                for (; q + 4 <= n; q += 4) {
                    float v[4];
                    int k[4];
#pragma unroll
                    for (int u = 0; u < 4; u++) {
                        v[u] = rows[(size_t)(q + u) * hidden + d];
                        k[u] = m[q + u];
                    }
#pragma unroll
                    for (int u = 0; u < 4; u++)
                        if (k[u] != 0) {
                            s += v[u];
                            c++;
                        }
                }
                for (; q < n; q++)
                    if (m[q] != 0) {
                        s += rows[(size_t)q * hidden + d];
                        c++;
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

// ---- GEMM epilogues ------------------------------------------------------------------

/* W consecutive columns c.. of token t's product, split's share of it,
 * finished and stored as the epilogue says. col holds qkv_column for each
 * of them, for EPI_QKV. */
template <int EPI, typename TOut, int W>
__device__ inline void finish(const GemmArgs &g, int split, int t, int c, float (&v)[W], const size_t *col) {
    if constexpr (EPI == EPI_PARTIAL) {
        store_vec<W>(static_cast<float *>(g.out) + ((size_t)split * g.tcap + t) * g.n + c, v);
    } else if constexpr (EPI == EPI_GELU) {
#pragma unroll
        for (int j = 0; j < W; j++) v[j] = gelu(v[j] + g.bias[c + j]);
        store_vec<W>(static_cast<TOut *>(g.out) + (size_t)t * g.n + c, v);
    } else {
        TOut *o = static_cast<TOut *>(g.out);
#pragma unroll
        for (int j = 0; j < W; j++) put(o + col[j] + (size_t)t * g.head_dim, v[j] + g.bias[c + j]);
    }
}

// ---- The F32 GEMM (and F16 before sm_80): FMAs through shared memory ---------------
//
// A block of 2 * BM threads computes a BM x 64 tile, each thread 8 rows
// by 4 columns, over k in steps of 16 through double-buffered shared
// memory, with the next step's loads in flight while this one computes.
// Every output is one F32 FMA chain over k in order.

constexpr int SK = 16;

template <int BM, typename TIn, int EPI, typename TOut>
__global__ void __launch_bounds__(BM * 2) gemm_simt_kernel(GemmArgs g) {
    constexpr int BN = 64, TX = BN / 4, NT = BM * 2;
    constexpr int LA = BM * SK / 4 / NT, LB = BN * SK / 4 / NT;
    __shared__ __align__(16) float As[2][SK][BM + 4];
    __shared__ __align__(16) float Bs[2][SK][BN + 4];
    const TIn *A = static_cast<const TIn *>(g.a), *B = static_cast<const TIn *>(g.w);
    const int M = g.info->tokens, N = g.n, K = g.k;
    const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN, tiles = mt * nt * g.splits;
    const int tx = threadIdx.x % TX, ty = threadIdx.x / TX;

    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int split = tile % g.splits, rest = tile / g.splits;
        const int n0 = (rest % nt) * BN, m0 = (rest / nt) * BM;
        const int kb = split * g.ksplit, ke = min(K, kb + g.ksplit);
        float ra[LA][4], rb[LB][4];
        auto fetch = [&](int k0) {
#pragma unroll
            for (int i = 0; i < LA; i++) {
                const int idx = threadIdx.x + i * NT, r = idx >> 2, kq = (idx & 3) * 4;
                const int gm = m0 + r, gk = k0 + kq;
                if (gm < M && gk < ke) {
                    load4(A + (size_t)gm * K + gk, ra[i]);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; j++) ra[i][j] = 0.0f;
                }
            }
#pragma unroll
            for (int i = 0; i < LB; i++) {
                const int idx = threadIdx.x + i * NT, r = idx >> 2, kq = (idx & 3) * 4;
                const int gn = n0 + r, gk = k0 + kq;
                if (gn < N && gk < ke) {
                    load4(B + (size_t)gn * K + gk, rb[i]);
                } else {
#pragma unroll
                    for (int j = 0; j < 4; j++) rb[i][j] = 0.0f;
                }
            }
        };
        auto stash = [&](int buf) {
#pragma unroll
            for (int i = 0; i < LA; i++) {
                const int idx = threadIdx.x + i * NT, r = idx >> 2, kq = (idx & 3) * 4;
#pragma unroll
                for (int j = 0; j < 4; j++) As[buf][kq + j][r] = ra[i][j];
            }
#pragma unroll
            for (int i = 0; i < LB; i++) {
                const int idx = threadIdx.x + i * NT, r = idx >> 2, kq = (idx & 3) * 4;
#pragma unroll
                for (int j = 0; j < 4; j++) Bs[buf][kq + j][r] = rb[i][j];
            }
        };

        float acc[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
        const int ktiles = (ke - kb + SK - 1) / SK;
        fetch(kb);
        stash(0);
        __syncthreads();
        for (int kt = 0; kt < ktiles; kt++) {
            const int buf = kt & 1;
            if (kt + 1 < ktiles) fetch(kb + (kt + 1) * SK);
#pragma unroll
            for (int kk = 0; kk < SK; kk++) {
                const float4 a0 = *reinterpret_cast<const float4 *>(&As[buf][kk][ty * 8]);
                const float4 a1 = *reinterpret_cast<const float4 *>(&As[buf][kk][ty * 8 + 4]);
                const float4 b = *reinterpret_cast<const float4 *>(&Bs[buf][kk][tx * 4]);
                const float av[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
                const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
                for (int i = 0; i < 8; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
            }
            if (kt + 1 < ktiles) stash(buf ^ 1);
            __syncthreads();
        }

        const int c = n0 + tx * 4;
        if (c >= N) continue;
        size_t col[4] = {0, 0, 0, 0};
        if constexpr (EPI == EPI_QKV)
#pragma unroll
            for (int j = 0; j < 4; j++) col[j] = qkv_column(g, c + j);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int t = m0 + ty * 8 + i;
            if (t < M) finish<EPI, TOut, 4>(g, split, t, c, acc[i], col);
        }
    }
}

// ---- The F16 GEMM on the tensor cores ------------------------------------------------
//
// mma.sync m16n8k16, F16 operands, F32 accumulators. A block of WM x WN
// warps computes a BM x BN tile over k in steps of 32, through a
// three-stage cp.async pipeline in shared memory (rows padded to 40
// halves, which keeps ldmatrix free of bank conflicts). Each warp holds a
// (BM / WM) x (BN / WN) accumulator tile. Every output's k order is fixed
// by the tile shape, so a run repeats its bits.

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

__device__ inline uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}
#endif

template <int BM, int BN, int WM, int WN, int EPI, typename TOut>
__global__ void __launch_bounds__(WM *WN * 32) gemm_mma_kernel(GemmArgs g) {
#ifndef TURBO_NO_MMA
    // k per stage, a stage's row in halves, and the stages.
    constexpr int MK = 32, MLD = MK + 8, MSTAGES = 3;
    constexpr int NT = WM * WN * 32, WTM = BM / WM, WTN = BN / WN, MI = WTM / 16, NI = WTN / 8;
    static_assert(NI % 2 == 0, "B fragments load two n8 tiles at once");
    __shared__ __align__(16) __half As[MSTAGES][BM][MLD];
    __shared__ __align__(16) __half Bs[MSTAGES][BN][MLD];
    const __half *A = static_cast<const __half *>(g.a), *B = static_cast<const __half *>(g.w);
    const int M = g.info->tokens, N = g.n, K = g.k;
    const int mt = (M + BM - 1) / BM, nt = (N + BN - 1) / BN, tiles = mt * nt * g.splits;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp / WN, wn = warp % WN;

    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int split = tile % g.splits, rest = tile / g.splits;
        const int n0 = (rest % nt) * BN, m0 = (rest / nt) * BM;
        const int kb = split * g.ksplit, ke = min(K, kb + g.ksplit);
        auto load_stage = [&](int st, int k0) {
            for (int i = threadIdx.x; i < BM * (MK / 8); i += NT) {
                const int r = i >> 2, ch = (i & 3) * 8, gm = m0 + r, gk = k0 + ch;
                const bool ok = gm < M && gk < ke;
                cp_async16(&As[st][r][ch], ok ? A + (size_t)gm * K + gk : A, ok);
            }
            for (int i = threadIdx.x; i < BN * (MK / 8); i += NT) {
                const int r = i >> 2, ch = (i & 3) * 8, gn = n0 + r, gk = k0 + ch;
                const bool ok = gn < N && gk < ke;
                cp_async16(&Bs[st][r][ch], ok ? B + (size_t)gn * K + gk : B, ok);
            }
        };

        float acc[MI][NI][4];
#pragma unroll
        for (int i = 0; i < MI; i++)
#pragma unroll
            for (int j = 0; j < NI; j++)
#pragma unroll
                for (int e = 0; e < 4; e++) acc[i][j][e] = 0.0f;

        const int ktiles = (ke - kb + MK - 1) / MK;
#pragma unroll
        for (int s = 0; s < MSTAGES - 1; s++) {
            if (s < ktiles) load_stage(s, kb + s * MK);
            cp_async_commit();
        }
        for (int kt = 0; kt < ktiles; kt++) {
            cp_async_wait<MSTAGES - 2>();
            __syncthreads();
            const int next = kt + MSTAGES - 1;
            if (next < ktiles) load_stage(next % MSTAGES, kb + next * MK);
            cp_async_commit();
            const int st = kt % MSTAGES;
#pragma unroll
            for (int kk = 0; kk < MK; kk += 16) {
                uint32_t af[MI][4], bf[NI][2];
#pragma unroll
                for (int i = 0; i < MI; i++)
                    ldsm_x4(af[i], &As[st][wm * WTM + i * 16 + (lane & 15)][kk + (lane >> 4) * 8]);
#pragma unroll
                for (int j = 0; j < NI / 2; j++) {
                    uint32_t r[4];
                    ldsm_x4(r, &Bs[st][wn * WTN + j * 16 + (lane >> 4) * 8 + (lane & 7)][kk + ((lane >> 3) & 1) * 8]);
                    bf[2 * j][0] = r[0];
                    bf[2 * j][1] = r[1];
                    bf[2 * j + 1][0] = r[2];
                    bf[2 * j + 1][1] = r[3];
                }
#pragma unroll
                for (int i = 0; i < MI; i++)
#pragma unroll
                    for (int j = 0; j < NI; j++) mma16816(acc[i][j], af[i], bf[j][0], bf[j][1]);
            }
        }
        cp_async_wait<0>();
        __syncthreads();

#pragma unroll
        for (int j = 0; j < NI; j++) {
            const int c = n0 + wn * WTN + j * 8 + (lane & 3) * 2;
            if (c >= N) continue;
            size_t col[2] = {0, 0};
            if constexpr (EPI == EPI_QKV) {
                col[0] = qkv_column(g, c);
                col[1] = qkv_column(g, c + 1);
            }
#pragma unroll
            for (int i = 0; i < MI; i++) {
                const int t = m0 + wm * WTM + i * 16 + (lane >> 2);
                float lo[2] = {acc[i][j][0], acc[i][j][1]}, hi[2] = {acc[i][j][2], acc[i][j][3]};
                if (t < M) finish<EPI, TOut, 2>(g, split, t, c, lo, col);
                if (t + 8 < M) finish<EPI, TOut, 2>(g, split, t + 8, c, hi, col);
            }
        }
    }
#else
    (void)g;
    __trap();
#endif
}

/* Calls f with the kernel for a GEMM and its block size: 32 x 64 tiles
 * for n up to 512 (the hidden-wide outputs), 64 x 64 above. */
template <typename F> cudaError_t with_gemm(Epilogue e, bool half, bool tc, int n, F &&f) {
    const bool narrow = n <= 512;
    if (half && tc) {
        if (narrow) {
            switch (e) {
            case EPI_QKV: return f(gemm_mma_kernel<32, 64, 1, 4, EPI_QKV, __half>, 128);
            case EPI_GELU: return f(gemm_mma_kernel<32, 64, 1, 4, EPI_GELU, __half>, 128);
            default: return f(gemm_mma_kernel<32, 64, 1, 4, EPI_PARTIAL, float>, 128);
            }
        }
        switch (e) {
        case EPI_QKV: return f(gemm_mma_kernel<64, 64, 2, 2, EPI_QKV, __half>, 128);
        case EPI_GELU: return f(gemm_mma_kernel<64, 64, 2, 2, EPI_GELU, __half>, 128);
        default: return f(gemm_mma_kernel<64, 64, 2, 2, EPI_PARTIAL, float>, 128);
        }
    }
    if (half) {
        if (narrow) {
            switch (e) {
            case EPI_QKV: return f(gemm_simt_kernel<32, __half, EPI_QKV, __half>, 64);
            case EPI_GELU: return f(gemm_simt_kernel<32, __half, EPI_GELU, __half>, 64);
            default: return f(gemm_simt_kernel<32, __half, EPI_PARTIAL, float>, 64);
            }
        }
        switch (e) {
        case EPI_QKV: return f(gemm_simt_kernel<64, __half, EPI_QKV, __half>, 128);
        case EPI_GELU: return f(gemm_simt_kernel<64, __half, EPI_GELU, __half>, 128);
        default: return f(gemm_simt_kernel<64, __half, EPI_PARTIAL, float>, 128);
        }
    }
    if (narrow) {
        switch (e) {
        case EPI_QKV: return f(gemm_simt_kernel<32, float, EPI_QKV, float>, 64);
        case EPI_GELU: return f(gemm_simt_kernel<32, float, EPI_GELU, float>, 64);
        default: return f(gemm_simt_kernel<32, float, EPI_PARTIAL, float>, 64);
        }
    }
    switch (e) {
    case EPI_QKV: return f(gemm_simt_kernel<64, float, EPI_QKV, float>, 128);
    case EPI_GELU: return f(gemm_simt_kernel<64, float, EPI_GELU, float>, 128);
    default: return f(gemm_simt_kernel<64, float, EPI_PARTIAL, float>, 128);
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
// Work items are (row, head, 64-query tile), rows longest first as the
// packing ordered them; a block loops over items. Keys and values of the
// row's head go through shared memory a chunk at a time (the whole row in
// one chunk up to the chunk's length, which make_plan sets from max_seq),
// and the softmax is carried across chunks and key blocks by its running
// largest score, rescaling what was summed before. Keys past the row's
// length do not exist in the packed row; a masked key inside it scores
// -1e30 below the rest, from the packing's key_bias, read only for rows
// that have one.

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
    it.q0 = (local / a.heads) * ATTENTION_QUERIES;
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
// go eight at a time, eight independent dot products. The four warps'
// partial softmaxes are merged in a fixed order at the end.

constexpr int ATT_THREADS = 256;
constexpr int KSPLITS = 4;
/* Keys scored at once by a lane: independent dot products. */
constexpr int G = 4;

/* HD values of row r of a head's [n][d] keys or values into shared memory
 * rows of HD, zero past d. */
template <typename T, int HD>
__device__ inline void load_rows(T *dst, const T *src, int rows, int d) {
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

/* The dot product of q with a shared memory row of HD, in column order. */
template <int HD> __device__ inline float dot_row(const float (&q)[HD], const float *k) {
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HD; c += 4) {
        const float4 v = *reinterpret_cast<const float4 *>(k + c);
        s = fmaf(q[c], v.x, s);
        s = fmaf(q[c + 1], v.y, s);
        s = fmaf(q[c + 2], v.z, s);
        s = fmaf(q[c + 3], v.w, s);
    }
    return s;
}

template <int HD> __device__ inline float dot_row(const float (&q)[HD], const __half *k) {
    float s = 0.0f;
#pragma unroll
    for (int c = 0; c < HD; c += 8) {
        const uint4 u = *reinterpret_cast<const uint4 *>(k + c);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int h = 0; h < 4; h++) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&w[h]));
            s = fmaf(q[c + 2 * h], f.x, s);
            s = fmaf(q[c + 2 * h + 1], f.y, s);
        }
    }
    return s;
}

/* o += e v for a shared memory row v of HD. */
template <int HD> __device__ inline void add_row(float (&o)[HD], float e, const float *v) {
#pragma unroll
    for (int c = 0; c < HD; c += 4) {
        const float4 x = *reinterpret_cast<const float4 *>(v + c);
        o[c] = fmaf(e, x.x, o[c]);
        o[c + 1] = fmaf(e, x.y, o[c + 1]);
        o[c + 2] = fmaf(e, x.z, o[c + 2]);
        o[c + 3] = fmaf(e, x.w, o[c + 3]);
    }
}

template <int HD> __device__ inline void add_row(float (&o)[HD], float e, const __half *v) {
#pragma unroll
    for (int c = 0; c < HD; c += 8) {
        const uint4 u = *reinterpret_cast<const uint4 *>(v + c);
        const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int h = 0; h < 4; h++) {
            const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&w[h]));
            o[c + 2 * h] = fmaf(e, f.x, o[c + 2 * h]);
            o[c + 2 * h + 1] = fmaf(e, f.y, o[c + 2 * h + 1]);
        }
    }
}

template <typename T, int HD> constexpr size_t simt_smem(int chunk) {
    const size_t keys = (size_t)chunk * HD * 2 * sizeof(T) + (size_t)chunk * sizeof(float);
    const size_t merge = (size_t)KSPLITS * 2 * (HD + 2) * 32 * sizeof(float);
    return keys > merge ? keys : merge;
}

template <typename T, int HD> __global__ void __launch_bounds__(ATT_THREADS, HD <= 32 ? 2 : 1) attention_simt_kernel(AttnArgs a) {
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
#pragma unroll
        for (int c = 0; c < HD; c++) {
            qr[c] = q < it.n && c < d ? to_float(Qg[(size_t)q * d + c]) : 0.0f;
            o[c] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;
        for (int c0 = 0; c0 < it.n; c0 += chunk) {
            const int cn = min(chunk, it.n - c0);
            __syncthreads();
            load_rows<T, HD>(Ks, Kg + (size_t)c0 * d, cn, d);
            load_rows<T, HD>(Vs, Vg + (size_t)c0 * d, cn, d);
            if (it.holes)
                for (int j = threadIdx.x; j < cn; j += ATT_THREADS) kbs[j] = a.p.key_bias[it.base + c0 + j];
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
    return (size_t)(ATTENTION_QUERIES + 2 * chunk) * (D + 8) * sizeof(__half) + (size_t)chunk * sizeof(float);
}

template <int D> __global__ void __launch_bounds__(MMA_ATT_THREADS) attention_mma_kernel(AttnArgs a) {
#ifndef TURBO_NO_MMA
    constexpr int LD = D + 8, DK = D / 16, DN = D / 8;
    extern __shared__ __align__(16) unsigned char att_sm[];
    const int chunk = a.chunk;
    __half *Qs = reinterpret_cast<__half *>(att_sm);
    __half *Ks = Qs + ATTENTION_QUERIES * LD;
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
        for (int i = threadIdx.x; i < ATTENTION_QUERIES * (D / 8); i += MMA_ATT_THREADS) {
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

/* The attention kernel for a session, its shared memory for a chunk of
 * keys, and its block size. */
struct AttnKernel {
    void (*fn)(AttnArgs);
    int threads;
    size_t (*smem)(int);
    int chunk_cap;
};

AttnKernel attention_kernel_for(const Shape &s) {
    const int d = s.hidden / s.heads;
    if (s.half && s.tensor_cores && (d == 32 || d == 64)) {
        if (d == 32) return {attention_mma_kernel<32>, MMA_ATT_THREADS, mma_smem<32>, 256};
        return {attention_mma_kernel<64>, MMA_ATT_THREADS, mma_smem<64>, 256};
    }
    if (s.half) {
        if (d <= 16) return {attention_simt_kernel<__half, 16>, ATT_THREADS, simt_smem<__half, 16>, 128};
        if (d <= 32) return {attention_simt_kernel<__half, 32>, ATT_THREADS, simt_smem<__half, 32>, 128};
        return {attention_simt_kernel<__half, 64>, ATT_THREADS, simt_smem<__half, 64>, 128};
    }
    if (d <= 16) return {attention_simt_kernel<float, 16>, ATT_THREADS, simt_smem<float, 16>, 128};
    if (d <= 32) return {attention_simt_kernel<float, 32>, ATT_THREADS, simt_smem<float, 32>, 128};
    return {attention_simt_kernel<float, 64>, ATT_THREADS, simt_smem<float, 64>, 128};
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

/* The blocks of fn that fit on the device at once, at least one. */
cudaError_t resident(const void *fn, int threads, size_t smem, int sms, int *out) {
    int per = 0;
    const cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, threads, smem);
    *out = (per > 0 ? per : 1) * sms;
    return e;
}

int cap(long long want, int most) { return (int)(want < most ? (want > 0 ? want : 1) : most); }

/* Values per lane of the row kernels for a hidden width. */
template <typename F> cudaError_t with_row_width(int hidden, F &&f) {
    if (hidden <= 128) return f(std::integral_constant<int, 4>{});
    if (hidden <= 384) return f(std::integral_constant<int, 12>{});
    return f(std::integral_constant<int, 32>{});
}

} // namespace

// ---- Launchers -------------------------------------------------------------------------

cudaError_t gemm_grid(Epilogue e, bool half, bool tensor_cores, int n, int tcap, int splits, int sms, int *grid) {
    return with_gemm(e, half, tensor_cores, n, [&](void (*fn)(GemmArgs), int threads) {
        const int bm = n <= 512 ? 32 : 64;
        const long long tiles = (long long)((tcap + bm - 1) / bm) * ((n + 63) / 64) * splits;
        int most = 0;
        const cudaError_t err = resident(reinterpret_cast<const void *>(fn), threads, 0, sms, &most);
        *grid = cap(tiles, most);
        return err;
    });
}

cudaError_t gemm(cudaStream_t s, Epilogue e, bool half, bool tensor_cores, const GemmArgs &g, int grid) {
    return with_gemm(e, half, tensor_cores, g.n, [&](void (*fn)(GemmArgs), int threads) {
        void *args[] = {const_cast<GemmArgs *>(&g)};
        return cudaLaunchKernel(reinterpret_cast<const void *>(fn), dim3(grid), dim3(threads), args, 0, s);
    });
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
    const int nb = query_tiles(s.seq_cap) + 1;
    p->pack_smem = (size_t)(3 * nb + PACK_BLOCK) * sizeof(int32_t);
    p->rows_grid = cap(((long long)s.tcap + ROW_WARPS - 1) / ROW_WARPS, s.sms * 8);
    p->pool_grid = cap(s.batch_cap, s.sms * 8);
    p->epi_grid = cap(s.tcap, s.sms * 8);
    // Split-K for a GEMM whose k is past 1024: about 768 per split, a
    // multiple of 32, at most four.
    const int k = s.inter;
    int splits = k / 768 < 1 ? 1 : (k / 768 > 4 ? 4 : k / 768);
    p->ffn2_ksplit = ((k + splits - 1) / splits + 31) & ~31;
    p->ffn2_splits = (k + p->ffn2_ksplit - 1) / p->ffn2_ksplit;
    cudaError_t e = gemm_grid(EPI_QKV, s.half, s.tensor_cores, 3 * s.hidden, s.tcap, 1, s.sms, &p->qkv_grid);
    if (e == cudaSuccess) e = gemm_grid(EPI_PARTIAL, s.half, s.tensor_cores, s.hidden, s.tcap, 1, s.sms, &p->out_grid);
    if (e == cudaSuccess) e = gemm_grid(EPI_GELU, s.half, s.tensor_cores, s.inter, s.tcap, 1, s.sms, &p->ffn1_grid);
    if (e == cudaSuccess)
        e = gemm_grid(EPI_PARTIAL, s.half, s.tensor_cores, s.hidden, s.tcap, p->ffn2_splits, s.sms, &p->ffn2_grid);
    if (e != cudaSuccess) return e;

    const AttnKernel ak = attention_kernel_for(s);
    int chunk = ((s.seq_cap + 63) & ~63);
    if (chunk > ak.chunk_cap) chunk = ak.chunk_cap;
    while (chunk > 64 && ak.smem(chunk) > s.smem_optin) chunk -= 64;
    p->attn_chunk = chunk;
    p->attn_smem = ak.smem(chunk);
    if (p->attn_smem > s.smem_optin) return cudaErrorInvalidValue;
    e = cudaFuncSetAttribute(ak.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)p->attn_smem);
    int most = 0;
    if (e == cudaSuccess) e = resident(reinterpret_cast<const void *>(ak.fn), ak.threads, p->attn_smem, s.sms, &most);
    const long long items = (long long)s.batch_cap * query_tiles(s.seq_cap) * s.heads;
    p->attn_grid = cap(items, most);
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

cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *ids, const int32_t *types, int pitch, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             const Packing &p, int hidden, float *x, uint16_t *x16, const Plan &plan) {
    return with_row_width(hidden, [&](auto v) {
        embed_layer_norm_kernel<decltype(v)::value><<<plan.rows_grid, ROW_BLOCK, 0, s>>>(
            ids, types, pitch, word, position, type, ln_w, ln_b, eps, p, hidden, x, as_half(x16));
        return cudaGetLastError();
    });
}

cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *part, int splits, int tcap, const float *bias,
                           const float *ln_w, const float *ln_b, float eps, const Info *info, int hidden,
                           uint16_t *x16, const Plan &plan) {
    return with_row_width(hidden, [&](auto v) {
        add_layer_norm_kernel<decltype(v)::value><<<plan.rows_grid, ROW_BLOCK, 0, s>>>(
            x, part, splits, tcap, bias, ln_w, ln_b, eps, info, hidden, as_half(x16));
        return cudaGetLastError();
    });
}

cudaError_t pool(cudaStream_t s, const float *x, const int32_t *mask, int pitch, const Packing &p, int hidden,
                 float *out, const Plan &plan) {
    pool_kernel<<<plan.pool_grid, POOL_BLOCK, 0, s>>>(x, mask, pitch, p, hidden, out);
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
