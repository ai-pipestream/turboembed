// SPDX-License-Identifier: Apache-2.0
//
// The BERT encoder's kernels for Apple GPUs. A run computes only the packed
// tokens: each row's positions up to its last live token, one row after
// another, T in all, and the rows past T up to a multiple of 32 that the
// matrix kernels tile over, which hold real lookups and are never read
// into an output. Every buffer indexed by token is in that packed order;
// pos gives a token's column in its row, for its position embedding.
//
// The linear layers and attention run on SIMD-group matrices, 8 x 8 F32
// tiles staged in threadgroup memory, with the next step fused into
// the linear layers' stores: the bias, or the bias and GELU. LayerNorm and
// pooling take one SIMD group per token or row.
//
// The arithmetic follows the CPU encoder (core/src/cpu/encoder.rs) where
// the order matters, in F32 throughout, since Apple GPUs have no F64:
// LayerNorm takes the mean, then the variance around it, in two passes;
// softmax subtracts the largest live score; mean pooling sums each
// dimension over the row's live tokens in order, then scales by 1 / count;
// the L2 norm is floored at 1e-12. The library is compiled without fast
// math, so exp, sqrt and division are the precise ones.

#include <metal_stdlib>

using namespace metal;

// Pooling, as turbo.h numbers it.
constant uint POOLING_CLS = 2;
constant uint POOLING_LAST = 3;

// Where a lane's two elements of an 8 x 8 SIMD-group matrix sit: row fm,
// columns fn and fn + 1.
struct Coord {
    ushort fm, fn;
};

Coord lane_coord(ushort lane) {
    const ushort qid = lane / 4;
    return Coord{ushort((qid & 4) + ((lane / 2) % 4)), ushort((qid & 2) * 2 + (lane % 2) * 2)};
}

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

float gelu(float v) { return 0.5f * v * (1.0f + erf_f32(v * 0.70710678118654752440f)); }

// ---- LayerNorm ----------------------------------------------------------------
//
// One SIMD group per token, four tokens per threadgroup: the row's values
// are summed across the group's lanes, with no threadgroup barrier.

// row = (row - mean) / sqrt(var + eps) * w + b over n values, the variance
// biased and taken around the mean.
void layer_norm_row(device float *row, uint n, device const float *w, device const float *b, float eps, ushort lane) {
    float s = 0.0f;
    for (uint d = lane; d < n; d += 32) s += row[d];
    const float mean = simd_sum(s) / float(n);
    float v = 0.0f;
    for (uint d = lane; d < n; d += 32) {
        const float c = row[d] - mean;
        v += c * c;
    }
    const float inv = 1.0f / sqrt(simd_sum(v) / float(n) + eps);
    for (uint d = lane; d < n; d += 32) row[d] = (row[d] - mean) * inv * w[d] + b[d];
}

struct RowParams {
    uint hidden;
    float eps;
};

// A token's three embeddings summed, then LayerNorm.
kernel void embed_layer_norm(device const int *ids [[buffer(0)]], device const int *types [[buffer(1)]],
                             device const int *pos [[buffer(2)]], device const float *word [[buffer(3)]],
                             device const float *position [[buffer(4)]], device const float *type [[buffer(5)]],
                             device const float *ln_w [[buffer(6)]], device const float *ln_b [[buffer(7)]],
                             device float *x [[buffer(8)]], constant RowParams &p [[buffer(9)]],
                             uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = tg * 4 + sg, h = p.hidden;
    device const float *w = word + ulong(ids[t]) * h;
    device const float *ps = position + ulong(pos[t]) * h;
    device const float *tt = type + ulong(types[t]) * h;
    device float *row = x + ulong(t) * h;
    for (uint d = lane; d < h; d += 32) row[d] = w[d] + ps[d] + tt[d];
    layer_norm_row(row, h, ln_w, ln_b, p.eps, lane);
}

// x = LayerNorm(x + (y + bias)), y summed over the splits of a linear
// layer whose k was split, each [tokens, hidden] apart.
kernel void add_layer_norm(device float *x [[buffer(0)]], device const float *y [[buffer(1)]],
                           device const float *bias [[buffer(2)]], device const float *ln_w [[buffer(3)]],
                           device const float *ln_b [[buffer(4)]], constant RowParams &p [[buffer(5)]],
                           constant uint &splits [[buffer(6)]], constant uint &tokens [[buffer(7)]],
                           uint tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                           ushort lane [[thread_index_in_simdgroup]]) {
    const uint t = tg * 4 + sg, h = p.hidden;
    device float *row = x + ulong(t) * h;
    device const float *yr = y + ulong(t) * h;
    const ulong part = ulong(tokens) * h;
    for (uint d = lane; d < h; d += 32) {
        float sum = yr[d];
        for (uint sp = 1; sp < splits; sp++) sum += yr[sp * part + d];
        row[d] = row[d] + (sum + bias[d]);
    }
    layer_norm_row(row, h, ln_w, ln_b, p.eps, lane);
}

// ---- Linear layers ------------------------------------------------------------
//
// y[t, o] = sum_i x[t, i] w[o, i] (+ bias[o], then GELU, as epilogue says),
// x [m, k] and w [n, k] row-major, y [m, n]. A threadgroup of four SIMD
// groups computes a 32 x 64 tile of y, each group 16 x 32 of it as 2 x 4
// matrices of 8 x 8, over k in steps of 16 that the threadgroup stages in
// its memory with coalesced loads. Eight accumulators per group keep them
// in registers; more spill, and cost several times the time. m is a
// multiple of 32, and n and k of 8; what lies past n or k is read as 0 and
// never stored. Several layers of one shape go in one dispatch, their
// weights, biases and outputs at the layer's binding. When k is split
// (the epilogue then none), z is layer * splits + split, and split s sums
// k from s * kchunk into its own [m, n] of y, m * n after the one before.

struct GemmParams {
    uint m;
    uint n;
    uint k;
    uint epilogue;   // 0: none; 1: + bias; 2: + bias, then GELU
    uint splits;
    uint kchunk;
};

constant uint BM = 32, BN = 64, BK = 16;
// Staged rows are padded by 4 floats, so the rows a SIMD group reads
// start in different banks.
constant uint LD = BK + 4;

kernel void gemm(device const float *x [[buffer(0)]], device const float *w0 [[buffer(1)]],
                 device const float *w1 [[buffer(2)]], device const float *w2 [[buffer(3)]],
                 device const float *b0 [[buffer(4)]], device const float *b1 [[buffer(5)]],
                 device const float *b2 [[buffer(6)]], device float *y0 [[buffer(7)]],
                 device float *y1 [[buffer(8)]], device float *y2 [[buffer(9)]],
                 constant GemmParams &p [[buffer(10)]], uint3 tg [[threadgroup_position_in_grid]],
                 ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                 ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float xs[BM * LD];
    threadgroup float ws[BN * LD];
    const uint layer = tg.z / p.splits, split = tg.z % p.splits;
    device const float *w = layer == 0 ? w0 : layer == 1 ? w1 : w2;
    device const float *bias = layer == 0 ? b0 : layer == 1 ? b1 : b2;
    device float *y = (layer == 0 ? y0 : layer == 1 ? y1 : y2) + ulong(split) * p.m * p.n;
    const uint m0 = tg.y * BM, n0 = tg.x * BN, k = p.k, n = p.n;
    const uint kbegin = split * p.kchunk, kend = min(k, kbegin + p.kchunk);
    const uint sm = (sg / 2) * 16, sn = (sg % 2) * 32;
    simdgroup_float8x8 acc[2][4];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 4; j++) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Each thread stages one float4 of x's tile and two of w's.
    const uint r = tid / 4, c = (tid % 4) * 4;
    device const float *xr = x + (m0 + r) * k + c;
    device const float *wr0 = w + (n0 + r) * k + c;
    device const float *wr1 = w + (n0 + r + 32) * k + c;
    const bool w0in = n0 + r < n, w1in = n0 + r + 32 < n;
    for (uint k0 = kbegin; k0 < kend; k0 += BK) {
        const bool kin = k0 + c < kend;
        // Weights read where the core holds them are 4-byte aligned only.
        *(threadgroup float4 *)(xs + r * LD + c) = kin ? float4(*(device const packed_float4 *)(xr + k0)) : float4(0.0f);
        *(threadgroup float4 *)(ws + r * LD + c) =
            kin && w0in ? float4(*(device const packed_float4 *)(wr0 + k0)) : float4(0.0f);
        *(threadgroup float4 *)(ws + (r + 32) * LD + c) =
            kin && w1in ? float4(*(device const packed_float4 *)(wr1 + k0)) : float4(0.0f);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; kk < BK; kk += 8) {
            simdgroup_float8x8 a[2], b[4];
            for (uint i = 0; i < 2; i++) simdgroup_load(a[i], xs + (sm + i * 8) * LD + kk, LD);
            for (uint j = 0; j < 4; j++) simdgroup_load(b[j], ws + (sn + j * 8) * LD + kk, LD, ulong2(0, 0), true);
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 4; j++) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const Coord cd = lane_coord(lane);
    for (uint i = 0; i < 2; i++) {
        device float *yr = y + ulong(m0 + sm + i * 8 + cd.fm) * n;
        for (uint j = 0; j < 4; j++) {
            const uint col = n0 + sn + j * 8 + cd.fn;
            if (col >= n) continue;
            thread auto &e = acc[i][j].thread_elements();
            float v0 = e[0], v1 = e[1];
            if (p.epilogue >= 1) {
                v0 += bias[col];
                v1 += bias[col + 1];
            }
            if (p.epilogue == 2) {
                v0 = gelu(v0);
                v1 = gelu(v1);
            }
            yr[col] = v0;
            yr[col + 1] = v1;
        }
    }
}

// ---- Attention ----------------------------------------------------------------
//
// Four SIMD groups per 32 queries of one row, for one head, 8 queries
// each. The row's keys go through in chunks of 32, their K and V staged in
// threadgroup memory for all four groups: each group's scores for the
// chunk, a running softmax over its queries' live keys (the largest score
// so far and the sum of exponentials, rescaling what came before when the
// largest grows), and the scores times the chunk's values added to the
// output. A row's keys past its end, rounded up to 32, are the next row's
// or the padding's: their scores are dropped, so their values count for
// nothing. The head width is a multiple of 8 and at most 64.

struct AttentionParams {
    uint hidden;
    uint head_dim;
    float scale;
};

constant uint KC = 32;       // keys per chunk
constant uint HD_MAX = 64;   // widest head

kernel void attention(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                      device const float *v [[buffer(2)]], device const int *mask [[buffer(3)]],
                      device const uint2 *rows [[buffer(4)]], device const uint2 *blocks [[buffer(5)]],
                      device float *ctx [[buffer(6)]], constant AttentionParams &p [[buffer(7)]],
                      threadgroup float *mem [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],
                      ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort lane [[thread_index_in_simdgroup]]) {
    // Threadgroup memory, given at dispatch: the queries, a chunk's K and V,
    // each row padded by 4 floats, and each group's scores.
    const uint LDH = p.head_dim + 4;
    threadgroup float *qs = mem;
    threadgroup float *ks = qs + 32 * LDH;
    threadgroup float *vs = ks + KC * LDH;
    const uint2 blk = blocks[tg.x];
    const uint2 row = rows[blk.x];
    const uint p0 = row.x, len = row.y;
    const uint h = p.hidden, hd = p.head_dim, col = tg.y * hd;
    const uint first = blk.y * 32;   // the group's first query, in the row
    device const int *m = mask + p0;

    // The group's 32 queries, staged once.
    for (uint e = tid; e < 32 * hd; e += 128) {
        const uint r = e / hd, d = e % hd;
        qs[r * LDH + d] = first + r < len ? q[ulong(p0 + first + r) * h + col + d] : 0.0f;
    }

    simdgroup_float8x8 o[HD_MAX / 8];
    for (uint f = 0; f < HD_MAX / 8; f++) o[f] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    // Lane l keeps the running max and sum of query row l / 4, which its
    // four lanes share.
    const uint qr = lane / 4, part = lane % 4;
    float mx = -INFINITY, sum = 0.0f;
    threadgroup float *s = vs + KC * LDH + sg * 8 * KC;

    for (uint c0 = 0; c0 < len; c0 += KC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = tid; e < KC * hd; e += 128) {
            const uint r = e / hd, d = e % hd;
            // Keys past the row's end are staged as 0, so their dropped
            // scores multiply 0, never what another row left.
            const bool in = c0 + r < len;
            const ulong at = ulong(p0 + c0 + r) * h + col + d;
            ks[r * LDH + d] = in ? k[at] : 0.0f;
            vs[r * LDH + d] = in ? v[at] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Scores: 8 queries x 32 keys.
        for (uint j = 0; j < KC / 8; j++) {
            simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (uint d = 0; d < hd; d += 8) {
                simdgroup_float8x8 qa, kt;
                simdgroup_load(qa, qs + (sg * 8) * LDH + d, LDH);
                simdgroup_load(kt, ks + (j * 8) * LDH + d, LDH, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(acc, qa, kt, acc);
            }
            simdgroup_store(acc, s + j * 8, KC);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Each lane: 8 of its query row's 32 keys.
        float cm = -INFINITY;
        for (uint t = part * 8; t < part * 8 + 8; t++) {
            const uint key = c0 + t;
            if (key < len && m[key] != 0) cm = max(cm, s[qr * KC + t] * p.scale);
        }
        cm = max(cm, simd_shuffle_xor(cm, 1));
        cm = max(cm, simd_shuffle_xor(cm, 2));
        const float nm = max(mx, cm);
        // No live key yet leaves the row's max at -inf; nothing to rescale.
        const float alpha = nm == -INFINITY ? 1.0f : exp(mx - nm);
        float cs = 0.0f;
        for (uint t = part * 8; t < part * 8 + 8; t++) {
            const uint key = c0 + t;
            float e = 0.0f;
            if (key < len && m[key] != 0) e = exp(s[qr * KC + t] * p.scale - nm);
            s[qr * KC + t] = e;
            cs += e;
        }
        cs += simd_shuffle_xor(cs, 1);
        cs += simd_shuffle_xor(cs, 2);
        sum = sum * alpha + cs;
        mx = nm;
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // o = o * alpha (per query row) + p v.
        const float row_alpha = simd_shuffle(alpha, ushort(lane_coord(lane).fm * 4));
        for (uint f = 0; f < HD_MAX / 8; f++) {
            if (f * 8 >= hd) break;
            thread auto &oe = o[f].thread_elements();
            oe[0] *= row_alpha;
            oe[1] *= row_alpha;
            for (uint j = 0; j < KC / 8; j++) {
                simdgroup_float8x8 pa, vb;
                simdgroup_load(pa, s + j * 8, KC);
                simdgroup_load(vb, vs + (j * 8) * LDH + f * 8, LDH);
                simdgroup_multiply_accumulate(o[f], pa, vb, o[f]);
            }
        }
    }

    const Coord cd = lane_coord(lane);
    const float inv = 1.0f / simd_shuffle(sum, ushort(cd.fm * 4));
    const uint qi = first + sg * 8 + cd.fm;
    if (qi >= len) return;
    device float *out = ctx + ulong(p0 + qi) * h + col;
    for (uint f = 0; f < HD_MAX / 8; f++) {
        if (f * 8 >= hd) break;
        thread auto &oe = o[f].thread_elements();
        out[f * 8 + cd.fn] = oe[0] * inv;
        out[f * 8 + cd.fn + 1] = oe[1] * inv;
    }
}

// The same, for a head width that is not a multiple of 8 or over 64: each
// query of the 32 in turn, in one SIMD group, its scores one key per lane.
kernel void attention_narrow(device const float *q [[buffer(0)]], device const float *k [[buffer(1)]],
                             device const float *v [[buffer(2)]], device const int *mask [[buffer(3)]],
                             device const uint2 *rows [[buffer(4)]], device const uint2 *blocks [[buffer(5)]],
                             device float *ctx [[buffer(6)]], constant AttentionParams &p [[buffer(7)]],
                             threadgroup float *sc [[threadgroup(0)]], uint2 tg [[threadgroup_position_in_grid]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    const uint2 blk = blocks[tg.x];
    const uint2 row = rows[blk.x];
    const uint p0 = row.x, len = row.y;
    const uint h = p.hidden, hd = p.head_dim, col = tg.y * hd;
    device const int *m = mask + p0;
    for (uint i = blk.y * 32; i < min(len, blk.y * 32 + 32); i++) {
        device const float *qi = q + ulong(p0 + i) * h + col;
        float mx = -INFINITY;
        for (uint j = lane; j < len; j += 32) {
            if (m[j] == 0) continue;
            device const float *kj = k + ulong(p0 + j) * h + col;
            float s = 0.0f;
            for (uint d = 0; d < hd; d++) s += qi[d] * kj[d];
            sc[j] = s * p.scale;
            mx = max(mx, sc[j]);
        }
        mx = simd_max(mx);
        float sum = 0.0f;
        for (uint j = lane; j < len; j += 32) {
            const float e = m[j] != 0 ? exp(sc[j] - mx) : 0.0f;
            sc[j] = e;
            sum += e;
        }
        const float inv = 1.0f / simd_sum(sum);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        device float *out = ctx + ulong(p0 + i) * h + col;
        for (uint d = lane; d < hd; d += 32) {
            float c = 0.0f;
            for (uint j = 0; j < len; j++)
                if (m[j] != 0) c += sc[j] * inv * v[ulong(p0 + j) * h + col + d];
            out[d] = c;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- Pooling ------------------------------------------------------------------

struct PoolParams {
    uint hidden;
    uint output_dim;
    uint pooling;
    uint l2;
};

// One SIMD group per row: its vector pooled, cut to output_dim, and when
// l2 is set, scaled to unit length.
kernel void pool(device const float *x [[buffer(0)]], device const int *mask [[buffer(1)]],
                 device const uint2 *rows [[buffer(2)]], device float *out [[buffer(3)]],
                 constant PoolParams &p [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
                 ushort lane [[thread_index_in_simdgroup]]) {
    const uint2 row = rows[b];
    device const int *m = mask + row.x;
    device const float *xs = x + ulong(row.x) * p.hidden;
    device float *dst = out + ulong(b) * p.output_dim;
    float ss = 0.0f;
    for (uint d = lane; d < p.output_dim; d += 32) {
        float val;
        if (p.pooling == POOLING_CLS) {
            val = xs[d];
        } else if (p.pooling == POOLING_LAST) {
            // A row's packed tokens end at its last live one.
            val = xs[ulong(row.y - 1) * p.hidden + d];
        } else {
            // Mean over the tokens whose mask is 1, summed in position order.
            float s = 0.0f;
            uint n = 0;
            for (uint t = 0; t < row.y; t++) {
                if (m[t] == 0) continue;
                s += xs[ulong(t) * p.hidden + d];
                n++;
            }
            val = s * (1.0f / float(n));
        }
        dst[d] = val;
        ss += val * val;
    }
    if (p.l2 == 0) return;
    const float inv = 1.0f / max(sqrt(simd_sum(ss)), 1e-12f);
    for (uint d = lane; d < p.output_dim; d += 32) dst[d] *= inv;
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
