/* SPDX-License-Identifier: Apache-2.0
 *
 * A DeBERTa-v2 model's kernels on the CUDA backend: deberta.h says what
 * each computes. Every launch loops over the run's work, which pack_rows
 * wrote in the Info, so one graph serves every run.
 */
#include "deberta.h"

#include <atomic>
#include <cmath>
#include <cstdlib>

namespace turbo_cuda {

namespace {

constexpr int ROW_BLOCK = 256;
constexpr int ROW_WARPS = ROW_BLOCK / 32;
/* Threads of a general attention block, a query and head each. */
constexpr int GENERAL_THREADS = 128;

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

/* The block's sum or largest of v, every thread getting it; red holds a
 * value per warp. */
template <bool MAX> __device__ float block_reduce(float v, float *red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    v = MAX ? warp_max(v) : warp_sum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = red[0];
    for (int w = 1; w < warps; w++) r = MAX ? fmaxf(r, red[w]) : r + red[w];
    return r;
}

/* A row of n values LayerNormed in place by one warp: the mean and the
 * biased variance in F32. */
__device__ void warp_layer_norm(float *row, int n, const float *w, const float *b, float eps) {
    const int lane = threadIdx.x & 31;
    float s = 0.0f;
    for (int d = lane; d < n; d += 32) s += row[d];
    const float mean = warp_sum(s) / (float)n;
    float q = 0.0f;
    for (int d = lane; d < n; d += 32) {
        const float c = row[d] - mean;
        q += c * c;
    }
    const float inv = 1.0f / sqrtf(warp_sum(q) / (float)n + eps);
    __syncwarp();
    for (int d = lane; d < n; d += 32) row[d] = __fadd_rn(__fmul_rn((row[d] - mean) * inv, w[d]), b[d]);
}

__global__ void __launch_bounds__(ROW_BLOCK)
    deberta_embed_kernel(const int32_t *rows, const float *word, const float *ln_w, const float *ln_b, float eps,
                         Packing p, int hidden, float *x) {
    const Info in = *p.info;
    const int32_t *ids = rows, *mask = rows + (size_t)in.batch * in.seq;
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < in.tokens; t += gridDim.x * ROW_WARPS) {
        const int b = p.tok_row[t], pos = t - p.start[b];
        const size_t at = (size_t)b * in.seq + pos;
        float *row = x + (size_t)t * hidden;
        if (mask[at] == 0) {
            for (int d = lane; d < hidden; d += 32) row[d] = 0.0f;
            continue;
        }
        const float *w = word + (size_t)ids[at] * hidden;
        for (int d = lane; d < hidden; d += 32) row[d] = w[d];
        __syncwarp();
        warp_layer_norm(row, hidden, ln_w, ln_b, eps);
    }
}

__global__ void __launch_bounds__(32) layer_norm_rows_kernel(float *x, const float *w, const float *b, float eps,
                                                             int hidden) {
    warp_layer_norm(x + (size_t)blockIdx.x * hidden, hidden, w, b, eps);
}

__global__ void __launch_bounds__(256) relative_kernel(const float *x, const float *w, const float *bias, int rows,
                                                      int n_out, int n_in, float *out) {
    const size_t n = (size_t)rows * n_out;
    for (size_t e = (size_t)blockIdx.x * blockDim.x + threadIdx.x; e < n; e += (size_t)gridDim.x * blockDim.x) {
        const size_t r = e / n_out, o = e % n_out;
        const float *xr = x + r * n_in, *wr = w + o * n_in;
        float s = 0.0f;
        for (int i = 0; i < n_in; i++) s = fmaf(xr[i], wr[i], s);
        out[e] = s + bias[o];
    }
}

__device__ __forceinline__ const float *head_part(const DebertaArgs &a, int which, int head) {
    return a.qkv + (size_t)(which * a.heads + head) * a.tcap * a.head_dim;
}

/* A block per (query, head) of the run, as the Level Zero general kernel:
 * a warp per key, its lanes splitting the head's width. */
__global__ void __launch_bounds__(GENERAL_THREADS) attention_general_kernel(DebertaArgs a) {
    extern __shared__ float sm[];
    const Info in = *a.p.info;
    const int hd = a.head_dim, items = in.tokens * a.heads;
    float *qi = sm, *score = sm + hd, *part = score + a.seq_cap, *red = part + GENERAL_THREADS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, warps = GENERAL_THREADS / 32;
    for (int it = blockIdx.x; it < items; it += gridDim.x) {
        const int t = it / a.heads, head = it - t * a.heads;
        const int b = a.p.tok_row[t], s0 = a.p.start[b], len = a.p.len[b], i = t - s0;
        const float *Q = head_part(a, 0, head), *K = head_part(a, 1, head), *V = head_part(a, 2, head);
        const float *kb = a.p.key_bias + s0;
        const bool live = a.p.key_bias[t] == 0.0f;
        const float *pq = a.pos + head * hd, *pk = a.pos + a.hidden + head * hd;
        for (int d = tid; d < hd; d += GENERAL_THREADS) qi[d] = Q[(size_t)t * hd + d];
        __syncthreads();
        for (int j = warp; j < len; j += warps) {
            if (!live) {
                if (lane == 0) score[j] = 0.0f;
                continue;
            }
            if (kb[j] != 0.0f) continue;
            const float *kr = K + (size_t)(s0 + j) * hd;
            const size_t r = (size_t)a.buckets[i - j + a.span] * 3 * a.hidden;
            float c2c = 0.0f, c2p = 0.0f, p2c = 0.0f;
            for (int d = lane; d < hd; d += 32) {
                c2c = fmaf(qi[d], kr[d], c2c);
                c2p = fmaf(qi[d], pk[r + d], c2p);
                p2c = fmaf(kr[d], pq[r + d], p2c);
            }
            c2c = warp_sum(c2c);
            c2p = warp_sum(c2p);
            p2c = warp_sum(p2c);
            if (lane == 0) score[j] = (c2c + c2p + p2c) * a.scale;
        }
        __syncthreads();
        float mx = -INFINITY;
        for (int j = tid; j < len; j += GENERAL_THREADS)
            if (!live || kb[j] == 0.0f) mx = fmaxf(mx, score[j]);
        mx = block_reduce<true>(mx, red);
        float sum = 0.0f;
        for (int j = tid; j < len; j += GENERAL_THREADS) {
            float e = 0.0f;
            if (!live || kb[j] == 0.0f) {
                e = expf(score[j] - mx);
                sum += e;
            }
            score[j] = e;
        }
        const float inv = 1.0f / block_reduce<false>(sum, red);
        __syncthreads();
        float *out = a.ctx + (size_t)t * a.hidden + head * hd;
        if (hd <= GENERAL_THREADS) {
            const int groups = GENERAL_THREADS / hd, d = tid % hd, g = tid / hd;
            if (g < groups) {
                float c = 0.0f;
                for (int j = g; j < len; j += groups)
                    if (!live || kb[j] == 0.0f) c = fmaf(score[j] * inv, V[(size_t)(s0 + j) * hd + d], c);
                part[g * hd + d] = c;
            }
            __syncthreads();
            if (tid < hd) {
                float c = 0.0f;
                for (int gg = 0; gg < groups; gg++) c += part[gg * hd + tid];
                out[tid] = c;
            }
        } else {
            for (int d = tid; d < hd; d += GENERAL_THREADS) {
                float c = 0.0f;
                for (int j = 0; j < len; j++)
                    if (!live || kb[j] == 0.0f) c = fmaf(score[j] * inv, V[(size_t)(s0 + j) * hd + d], c);
                out[d] = c;
            }
        }
        __syncthreads();
    }
}

/* The tiled kernel: a thread per query, QD queries of a row and head a
 * block, the query (scaled) and its context in registers, an online
 * softmax. The row's keys and values go through shared memory KD at a
 * time, and with them the relative rows the block's distances reach (at
 * most QD + KD - 1), padded by four floats; the query rows scaled, so all
 * three terms are. Each score reads its bucket from the session's table. */
template <int HD, int QD, int KD> constexpr size_t tiled_smem() {
    return sizeof(float) * (2 * (size_t)KD * HD + 2 * (size_t)(QD + KD - 1) * (HD + 4) + KD);
}

template <int HD, int QD, int KD> __global__ void __launch_bounds__(QD) attention_tiled_kernel(DebertaArgs a) {
    extern __shared__ __align__(16) float sm[];
    constexpr int PR = HD + 4;
    float *ks = sm, *vs = ks + KD * HD, *pk = vs + KD * HD, *pq = pk + (QD + KD - 1) * PR, *kb = pq + (QD + KD - 1) * PR;
    const Info in = *a.p.info;
    const int tiles = (a.seq_cap + QD - 1) / QD, items = in.batch * a.heads * tiles, tid = threadIdx.x;
    const int stride = 3 * a.hidden;
    for (int it = blockIdx.x; it < items; it += gridDim.x) {
        const int b = it / (a.heads * tiles), head = (it / tiles) % a.heads, i0 = (it % tiles) * QD;
        const int s0 = a.p.start[b], len = a.p.len[b];
        if (i0 >= len) continue;
        const float *Q = head_part(a, 0, head), *K = head_part(a, 1, head), *V = head_part(a, 2, head);
        const int i = i0 + tid;
        const bool mine = i < len, alive = mine && a.p.key_bias[s0 + i] == 0.0f;
        float q[HD], acc[HD];
        const float *qr = Q + (size_t)(s0 + (mine ? i : 0)) * HD;
#pragma unroll
        for (int d = 0; d < HD; d++) {
            q[d] = qr[d] * a.scale;
            acc[d] = 0.0f;
        }
        float m = -INFINITY, l = 0.0f;
        for (int j0 = 0; j0 < len; j0 += KD) {
            const int n = min(KD, len - j0), dlo = i0 - (j0 + KD - 1);
            const int blo = a.buckets[max(-a.span, min(a.span, dlo)) + a.span];
            // The host takes this kernel only where nb fits its rows
            // (deberta_tiled_rows); never past them.
            const int nb = min(QD + KD - 1, a.buckets[max(-a.span, min(a.span, dlo + QD + KD - 2)) + a.span] - blo + 1);
            for (int e = tid; e < n * HD; e += QD) {
                const int j = e / HD, d = e % HD;
                ks[e] = K[(size_t)(s0 + j0 + j) * HD + d];
                vs[e] = V[(size_t)(s0 + j0 + j) * HD + d];
            }
            for (int e = tid; e < nb * HD; e += QD) {
                const int r = e / HD, d = e % HD;
                const float *pr = a.pos + (size_t)(blo + r) * stride + head * HD;
                pk[r * PR + d] = pr[a.hidden + d];
                pq[r * PR + d] = pr[d] * a.scale;
            }
            for (int j = tid; j < n; j += QD) kb[j] = a.p.key_bias[s0 + j0 + j];
            __syncthreads();
            if (mine) {
                for (int j = 0; j < n; j++) {
                    float s = 0.0f;
                    if (alive) {
                        if (kb[j] != 0.0f) continue;
                        const int r = a.buckets[i - (j0 + j) + a.span] - blo;
                        const float *kr = ks + j * HD, *kp = pk + r * PR, *qp = pq + r * PR;
                        float c2c = 0.0f, c2p = 0.0f, p2c = 0.0f;
#pragma unroll
                        for (int d = 0; d < HD; d++) {
                            c2c = fmaf(q[d], kr[d], c2c);
                            c2p = fmaf(q[d], kp[d], c2p);
                            p2c = fmaf(kr[d], qp[d], p2c);
                        }
                        s = c2c + c2p + p2c;
                    }
                    if (s > m) {
                        const float c = expf(m - s);
                        l *= c;
#pragma unroll
                        for (int d = 0; d < HD; d++) acc[d] *= c;
                        m = s;
                    }
                    const float pe = expf(s - m);
                    l += pe;
                    const float *vr = vs + j * HD;
#pragma unroll
                    for (int d = 0; d < HD; d++) acc[d] = fmaf(pe, vr[d], acc[d]);
                }
            }
            __syncthreads();
        }
        if (mine) {
            const float inv = 1.0f / l;
            float *out = a.ctx + (size_t)(s0 + i) * a.hidden + head * HD;
#pragma unroll
            for (int d = 0; d < HD; d++) out[d] = acc[d] * inv;
        }
    }
}

struct Tiled {
    const void *fn;
    int queries, keys;
    size_t smem;
};

Tiled tiled_for(int head_dim) {
    switch (head_dim) {
    case 32:
        return {reinterpret_cast<const void *>(attention_tiled_kernel<32, 128, 32>), 128, 32, tiled_smem<32, 128, 32>()};
    case 64: return {reinterpret_cast<const void *>(attention_tiled_kernel<64, 64, 32>), 64, 32, tiled_smem<64, 64, 32>()};
    case 128:
        return {reinterpret_cast<const void *>(attention_tiled_kernel<128, 32, 16>), 32, 16, tiled_smem<128, 32, 16>()};
    default: return {nullptr, 0, 0, 0};
    }
}

int capped(long long n, int most) { return (int)(n < 1 ? 1 : n < most ? n : most); }

} // namespace

uint32_t deberta_bucket(long long d, uint32_t buckets, uint32_t maximum) {
    const long long mid = buckets / 2, a = d < 0 ? -d : d;
    long long magnitude = a;
    if (a > mid)
        magnitude = (long long)std::ceil(std::log((float)a / (float)mid) / std::log((float)(maximum - 1) / (float)mid) *
                                         (float)(mid - 1)) +
                    mid;
    const long long v = (d < 0 ? -magnitude : magnitude) + buckets;
    return (uint32_t)(v < 0 ? 0 : v > 2 * (long long)buckets - 1 ? 2 * (long long)buckets - 1 : v);
}

cudaError_t deberta_embed(cudaStream_t s, const int32_t *rows, const float *word, const float *ln_w,
                          const float *ln_b, float eps, const Packing &p, int hidden, float *x, int grid) {
    deberta_embed_kernel<<<grid, ROW_BLOCK, 0, s>>>(rows, word, ln_w, ln_b, eps, p, hidden, x);
    return cudaGetLastError();
}

cudaError_t deberta_layer_norm_rows(cudaStream_t s, float *x, const float *w, const float *b, float eps, int rows,
                                    int hidden) {
    layer_norm_rows_kernel<<<rows, 32, 0, s>>>(x, w, b, eps, hidden);
    return cudaGetLastError();
}

cudaError_t deberta_relative(cudaStream_t s, const float *x, const float *w, const float *bias, int rows, int n_out,
                             int n_in, float *out) {
    relative_kernel<<<capped(((long long)rows * n_out + 255) / 256, 65535), 256, 0, s>>>(x, w, bias, rows, n_out,
                                                                                       n_in, out);
    return cudaGetLastError();
}

bool deberta_tiled_rows(int head_dim, int seq_cap, const int32_t *buckets, int *reach, int *room) {
    const Tiled t = tiled_for(head_dim);
    if (!t.fn) return false;
    const long long span = seq_cap - 1;
    auto at = [&](long long d) { return (long long)buckets[(d < -span ? -span : d > span ? span : d) + span]; };
    long long most = 0;
    for (long long i0 = 0; i0 < seq_cap; i0 += t.queries)
        for (long long j0 = 0; j0 < seq_cap; j0 += t.keys) {
            const long long lo = i0 - (j0 + t.keys - 1), n = at(lo + t.queries + t.keys - 2) - at(lo) + 1;
            most = n > most ? n : most;
        }
    *reach = (int)most;
    *room = t.queries + t.keys - 1;
    return true;
}

cudaError_t deberta_plan(int head_dim, int batch_cap, int seq_cap, int heads, int sms, size_t smem_optin,
                         bool general, const int32_t *buckets, DebertaPlan *out) {
    int reach = 0, room = 0;
    const bool has = deberta_tiled_rows(head_dim, seq_cap, buckets, &reach, &room), fits = has && reach <= room;
    const Tiled t = general || !fits ? Tiled{nullptr, 0, 0, 0} : tiled_for(head_dim);
    std::string why;
    if (has && !general && !fits)
        why = "rows of " + std::to_string(seq_cap) + " tokens reach " + std::to_string(reach) +
              " relative rows, the tiled kernel stages " + std::to_string(room);
    else if (t.fn && t.smem > smem_optin)
        why = "the tiled kernel takes " + std::to_string(t.smem) + " bytes of shared memory, the device gives " +
              std::to_string(smem_optin);
    if (t.fn && t.smem <= smem_optin) {
        cudaError_t e = cudaFuncSetAttribute(t.fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)t.smem);
        int per = 0;
        if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, t.fn, t.queries, t.smem);
        if (e != cudaSuccess) return e;
        const long long items = (long long)batch_cap * heads * ((seq_cap + t.queries - 1) / t.queries);
        *out = DebertaPlan{true, t.queries, capped(items, (per > 0 ? per : 1) * sms), t.smem, {}};
        return cudaSuccess;
    }
    const size_t smem = sizeof(float) * ((size_t)head_dim + seq_cap + GENERAL_THREADS + 32);
    if (smem > smem_optin) return cudaErrorInvalidValue;
    const void *fn = reinterpret_cast<const void *>(attention_general_kernel);
    // The attribute is the kernel's, for every session: never below the
    // most a session has asked for, so a smaller session made after a
    // larger one leaves the larger one's launches valid.
    static std::atomic<size_t> most{0};
    size_t was = most.load(std::memory_order_relaxed);
    while (smem > was && !most.compare_exchange_weak(was, smem, std::memory_order_relaxed)) {
    }
    cudaError_t e = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)most.load());
    int per = 0;
    if (e == cudaSuccess) e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, fn, GENERAL_THREADS, smem);
    if (e != cudaSuccess) return e;
    *out = DebertaPlan{false, GENERAL_THREADS, capped((long long)batch_cap * seq_cap * heads, (per > 0 ? per : 1) * sms),
                       smem, why};
    return cudaSuccess;
}

cudaError_t deberta_attention(cudaStream_t s, const DebertaArgs &a, const DebertaPlan &plan) {
    void *args[] = {const_cast<DebertaArgs *>(&a)};
    const void *fn = plan.tiled ? tiled_for(a.head_dim).fn : reinterpret_cast<const void *>(attention_general_kernel);
    return cudaLaunchKernel(fn, dim3(plan.grid), dim3(plan.threads), args, plan.smem, s);
}

} // namespace turbo_cuda
