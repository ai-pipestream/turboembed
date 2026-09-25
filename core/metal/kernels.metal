// SPDX-License-Identifier: Apache-2.0
//
// The BERT encoder's kernels for Apple GPUs: the embedding lookup,
// LayerNorm, the linear layers, GELU, attention and pooling. The
// arithmetic follows the CPU encoder (core/src/cpu/encoder.rs) where the
// order matters, in F32 throughout, since Apple GPUs have no F64:
// LayerNorm takes the mean, then the variance around it, in two passes;
// softmax subtracts the largest live score; mean pooling sums each
// dimension over the row's positions in order, then scales by 1 / count;
// the L2 norm is floored at 1e-12. The library is compiled without fast
// math, so exp, sqrt and division are the precise ones.

#include <metal_stdlib>

using namespace metal;

// Threads per threadgroup for the row kernels: four SIMD groups.
constant uint BLOCK = 128;
constant uint GROUPS = BLOCK / 32;

// Pooling and normalization, as turbo.h numbers them.
constant uint POOLING_CLS = 2;
constant uint POOLING_LAST = 3;

// The threadgroup's sum or maximum, in every thread. red holds GROUPS
// values; the barrier first lets it be used again right after.
float group_sum(float v, threadgroup float *red, uint lane, uint sg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < GROUPS; g++) total += red[g];
    return total;
}

float group_max(float v, threadgroup float *red, uint lane, uint sg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (uint g = 0; g < GROUPS; g++) m = max(m, red[g]);
    return m;
}

// row = (row - mean) / sqrt(var + eps) * w + b over n values, the
// variance biased and taken around the mean. Each thread touches only the
// columns it wrote, so no barrier is needed before this.
void layer_norm_row(device float *row, uint n, device const float *w, device const float *b, float eps,
                    threadgroup float *red, uint tid, uint lane, uint sg) {
    float s = 0.0f;
    for (uint d = tid; d < n; d += BLOCK) s += row[d];
    const float mean = group_sum(s, red, lane, sg) / float(n);
    float v = 0.0f;
    for (uint d = tid; d < n; d += BLOCK) {
        const float c = row[d] - mean;
        v += c * c;
    }
    const float var = group_sum(v, red, lane, sg) / float(n);
    const float inv = 1.0f / sqrt(var + eps);
    for (uint d = tid; d < n; d += BLOCK) {
        const float x = (row[d] - mean) * inv;
        row[d] = x * w[d] + b[d];
    }
}

struct RowParams {
    uint seq;
    uint hidden;
    float eps;
    uint has_types;
};

// One threadgroup per token: its three embeddings summed, then LayerNorm.
kernel void embed_layer_norm(device const int *ids [[buffer(0)]], device const int *types [[buffer(1)]],
                             device const float *word [[buffer(2)]], device const float *position [[buffer(3)]],
                             device const float *type [[buffer(4)]], device const float *ln_w [[buffer(5)]],
                             device const float *ln_b [[buffer(6)]], device float *x [[buffer(7)]],
                             constant RowParams &p [[buffer(8)]], uint t [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[GROUPS];
    const uint h = p.hidden;
    device const float *w = word + ulong(ids[t]) * h;
    device const float *ps = position + ulong(t % p.seq) * h;
    device const float *tt = type + ulong(p.has_types ? types[t] : 0) * h;
    device float *row = x + ulong(t) * h;
    for (uint d = tid; d < h; d += BLOCK) row[d] = w[d] + ps[d] + tt[d];
    layer_norm_row(row, h, ln_w, ln_b, p.eps, red, tid, lane, sg);
}

// x = LayerNorm(x + (y + bias)), a threadgroup per token.
kernel void add_layer_norm(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                           device const float *bias [[buffer(2)]], device const float *ln_w [[buffer(3)]],
                           device const float *ln_b [[buffer(4)]], constant RowParams &p [[buffer(5)]],
                           uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[GROUPS];
    const uint h = p.hidden;
    device float *row = x + ulong(t) * h;
    device const float *yr = y + ulong(t) * h;
    for (uint d = tid; d < h; d += BLOCK) row[d] = row[d] + (yr[d] + bias[d]);
    layer_norm_row(row, h, ln_w, ln_b, p.eps, red, tid, lane, sg);
}

// ---- Linear layers ------------------------------------------------------------
//
// y[t, o] = sum_i x[t, i] w[o, i], x [m, k] and w [n, k] row-major, y [m, n]:
// a threadgroup of four SIMD groups per 32 x 32 tile of y, each group 16 x
// 16 of it in 8 x 8 matrices, over k in steps of 32 staged in threadgroup
// memory. Loads past an edge read 0, and stores past one are dropped.

struct LinearParams {
    uint m;
    uint n;
    uint k;
};

constant uint TILE = 32;

kernel void linear(device const float *x [[buffer(0)]], device const float *w [[buffer(1)]],
                   device float *y [[buffer(2)]], constant LinearParams &p [[buffer(3)]],
                   uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float xs[TILE * TILE];   // [row][k]
    threadgroup float ws[TILE * TILE];   // [k][col]
    const uint m0 = tg.y * TILE, n0 = tg.x * TILE;
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.k; k0 += TILE) {
        for (uint e = tid; e < TILE * TILE; e += BLOCK) {
            const uint r = e / TILE, c = e % TILE;
            const uint kk = k0 + c;
            xs[e] = (m0 + r < p.m && kk < p.k) ? x[ulong(m0 + r) * p.k + kk] : 0.0f;
            ws[c * TILE + r] = (n0 + r < p.n && kk < p.k) ? w[ulong(n0 + r) * p.k + kk] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < TILE; kk += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], xs + (sm + i * 8) * TILE + kk, TILE);
            for (uint j = 0; j < 2; j++) simdgroup_load(b[j], ws + kk * TILE + sn + j * 8, TILE);
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Through threadgroup memory, so the tile's edges can be cut.
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++) simdgroup_store(acc[i][j], xs + (sm + i * 8) * TILE + sn + j * 8, TILE);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < TILE * TILE; e += BLOCK) {
        const uint r = e / TILE, c = e % TILE;
        if (m0 + r < p.m && n0 + c < p.n) y[ulong(m0 + r) * p.n + n0 + c] = xs[e];
    }
}

// ---- GELU ---------------------------------------------------------------------

// erf(x), from erfc's Chebyshev fit in Numerical Recipes (6.2), relative
// error under 1.2e-7 for erfc everywhere: Metal has no erf of its own.
float erf_f32(float x) {
    const float z = abs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float r =
        t * exp(-z * z - 1.26551223f +
                t * (1.00002368f +
                     t * (0.37409196f +
                          t * (0.09678418f +
                               t * (-0.18628806f +
                                    t * (0.27886807f +
                                         t * (-1.13520398f + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f)))))))));
    return x >= 0.0f ? 1.0f - r : r - 1.0f;
}

struct GeluParams {
    uint n;
    uint width;
};

// y = GELU(y + bias), the erf form, over n values of rows width wide.
kernel void bias_gelu(device float *y [[buffer(0)]], device const float *bias [[buffer(1)]],
                      constant GeluParams &p [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    const float v = y[i] + bias[i % p.width];
    y[i] = 0.5f * v * (1.0f + erf_f32(v * 0.70710678118654752440f));
}

// ---- Attention ----------------------------------------------------------------

struct AttentionParams {
    uint seq;
    uint hidden;
    uint head_dim;
    float scale;
};

// One threadgroup per (query, head, row). Threadgroup memory, given at
// dispatch: the query's head, the row's scores, and one partial context
// per group of head_dim threads.
kernel void attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                      device const float *v [[buffer(2)]], device const float *bq [[buffer(3)]],
                      device const float *bk [[buffer(4)]], device const float *bv [[buffer(5)]],
                      device const int *mask [[buffer(6)]], device float *ctx [[buffer(7)]],
                      constant AttentionParams &p [[buffer(8)]], threadgroup float *sm [[threadgroup(0)]],
                      uint3 at [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[GROUPS];
    const uint seq = p.seq, hidden = p.hidden, hd = p.head_dim;
    threadgroup float *qi = sm;
    threadgroup float *score = qi + hd;
    threadgroup float *part = score + seq;
    const uint i = at.x, head = at.y, b = at.z;
    const uint col = head * hd;
    device const int *m = mask + ulong(b) * seq;
    const ulong base = ulong(b) * seq;

    device const float *qrow = q + (base + i) * hidden + col;
    for (uint d = tid; d < hd; d += BLOCK) qi[d] = qrow[d] + bq[col + d];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // A SIMD group per key: its lanes split the head's width.
    for (uint j = sg; j < seq; j += GROUPS) {
        if (m[j] == 0) continue;
        device const float *krow = k + (base + j) * hidden + col;
        float s = 0.0f;
        for (uint d = lane; d < hd; d += 32) s += qi[d] * (krow[d] + bk[col + d]);
        s = simd_sum(s);
        if (lane == 0) score[j] = s * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mx = -INFINITY;
    for (uint j = tid; j < seq; j += BLOCK)
        if (m[j] != 0) mx = max(mx, score[j]);
    mx = group_max(mx, red, lane, sg);
    float sum = 0.0f;
    for (uint j = tid; j < seq; j += BLOCK) {
        float e = 0.0f;
        if (m[j] != 0) {
            e = exp(score[j] - mx);
            sum += e;
        }
        score[j] = e;
    }
    sum = group_sum(sum, red, lane, sg);
    const float inv = 1.0f / sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float *out = ctx + (base + i) * hidden + col;
    if (hd <= BLOCK) {
        // Groups of head_dim threads each take every groups-th key; the
        // first group adds the partial sums in group order.
        const uint groups = BLOCK / hd;
        const uint d = tid % hd, g = tid / hd;
        if (g < groups) {
            float c = 0.0f;
            for (uint j = g; j < seq; j += groups) {
                if (m[j] == 0) continue;
                c += score[j] * inv * (v[(base + j) * hidden + col + d] + bv[col + d]);
            }
            part[g * hd + d] = c;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < hd) {
            float c = 0.0f;
            for (uint gg = 0; gg < groups; gg++) c += part[gg * hd + tid];
            out[tid] = c;
        }
    } else {
        for (uint d = tid; d < hd; d += BLOCK) {
            float c = 0.0f;
            for (uint j = 0; j < seq; j++) {
                if (m[j] == 0) continue;
                c += score[j] * inv * (v[(base + j) * hidden + col + d] + bv[col + d]);
            }
            out[d] = c;
        }
    }
}

// ---- Pooling ------------------------------------------------------------------

struct PoolParams {
    uint seq;
    uint hidden;
    uint output_dim;
    uint pooling;
    uint l2;
};

// One threadgroup per row: its vector pooled, cut to output_dim, and when
// l2 is set, scaled to unit length.
kernel void pool(device const float *x [[buffer(0)]], device const int *mask [[buffer(1)]],
                 device float *out [[buffer(2)]], constant PoolParams &p [[buffer(3)]],
                 uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float red[GROUPS];
    device const int *m = mask + ulong(b) * p.seq;
    device const float *rows = x + ulong(b) * p.seq * p.hidden;
    device float *dst = out + ulong(b) * p.output_dim;
    uint last = 0;
    for (uint t = 0; t < p.seq; t++)
        if (m[t] != 0) last = t;
    float ss = 0.0f;
    for (uint d = tid; d < p.output_dim; d += BLOCK) {
        float val;
        if (p.pooling == POOLING_CLS) {
            val = rows[d];
        } else if (p.pooling == POOLING_LAST) {
            val = rows[ulong(last) * p.hidden + d];
        } else {
            // Mean over the tokens whose mask is 1, summed in position order.
            float s = 0.0f;
            uint n = 0;
            for (uint t = 0; t < p.seq; t++) {
                if (m[t] == 0) continue;
                s += rows[ulong(t) * p.hidden + d];
                n++;
            }
            val = s * (1.0f / float(n));
        }
        dst[d] = val;
        ss += val * val;
    }
    if (p.l2 == 0) return;
    const float inv = 1.0f / max(sqrt(group_sum(ss, red, lane, sg)), 1e-12f);
    for (uint d = tid; d < p.output_dim; d += BLOCK) dst[d] *= inv;
}

// ---- Weights stored in F16 or BF16 -------------------------------------------

kernel void widen_f16(device const half *src [[buffer(0)]], device float *dst [[buffer(1)]],
                      constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = float(src[i]);
}

kernel void widen_bf16(device const ushort *src [[buffer(0)]], device float *dst [[buffer(1)]],
                       constant uint &n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = as_type<float>(uint(src[i]) << 16);
}
