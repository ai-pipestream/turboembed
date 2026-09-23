// SPDX-License-Identifier: Apache-2.0
//
// Metal Shading Language for the BERT encoder and its heads, compiled at
// runtime with newLibraryWithSource (no offline toolchain). The encoder
// kernels are the PoC's (poc-2026-09-21, native/turborerank/src/metal_api.mm):
// one sequence at a time, fp32 throughout, the same reduction order as a
// CPU reference, and an erf-form GELU with a software erf because MSL has
// none. The head kernels (pooling, L2, classifier) are new.
//
// Included by provider.mm as a raw string.
R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct EmbedParams {
    uint seq;
    uint hidden;
    uint word_rows;
    uint pos_rows;
    uint type_rows;
};

struct LnParams {
    uint seq;
    uint hidden;
    float eps;
};

struct LinearParams {
    uint seq;
    uint k;
    uint out;
    uint has_bias;
};

struct ElemParams {
    uint n;
};

struct AttnParams {
    uint seq;
    uint hidden;
    uint heads;
    uint dh;
    float scale;
    uint batch;
};

struct PoolParams {
    uint hidden;
};

kernel void embed_kernel(
    device float *x [[buffer(0)]],
    device const int *ids [[buffer(1)]],
    device const int *pos [[buffer(2)]],
    device const int *types [[buffer(3)]],
    device const float *word [[buffer(4)]],
    device const float *pos_w [[buffer(5)]],
    device const float *type_w [[buffer(6)]],
    constant EmbedParams &p [[buffer(7)]],
    uint t [[thread_position_in_grid]]
) {
    if (t >= p.seq) return;
    const uint uid = uint(ids[t]);
    const uint upos = uint(pos[t]);
    const uint utyp = uint(types[t]);
    device float *row = x + ulong(t) * p.hidden;
    if (uid < p.word_rows && upos < p.pos_rows && utyp < p.type_rows) {
        const device float *we = word + ulong(uid) * p.hidden;
        const device float *pe = pos_w + ulong(upos) * p.hidden;
        const device float *te = type_w + ulong(utyp) * p.hidden;
        for (uint h = 0; h < p.hidden; ++h) {
            row[h] = we[h] + pe[h] + te[h];
        }
    } else {
        for (uint h = 0; h < p.hidden; ++h) {
            row[h] = 0.0f;
        }
    }
}

// One thread per token: same reduction order as the CPU kernel.
kernel void layer_norm_kernel(
    device float *x [[buffer(0)]],
    device const float *gamma [[buffer(1)]],
    device const float *beta [[buffer(2)]],
    constant LnParams &p [[buffer(3)]],
    uint t [[thread_position_in_grid]]
) {
    if (t >= p.seq) return;
    device float *row = x + ulong(t) * p.hidden;
    float mean = 0.0f;
    for (uint i = 0; i < p.hidden; ++i) {
        mean += row[i];
    }
    mean /= float(p.hidden);
    float var = 0.0f;
    for (uint i = 0; i < p.hidden; ++i) {
        const float d = row[i] - mean;
        var += d * d;
    }
    var /= float(p.hidden);
    const float inv = rsqrt(var + p.eps);
    for (uint i = 0; i < p.hidden; ++i) {
        row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    }
}

// Exact CPU linear_nt: y[s,o] = bias[o] + dot(x[s], W[o]) over `seq` rows.
// Tiled: a 16x16 threadgroup stages 16 rows of x and 16 rows of W, 32
// columns at a time, in threadgroup memory (loaded along k, so the loads
// are contiguous), then each thread accumulates its dot product from that
// tile. The per-output accumulation order is still t = 0..k-1, so the
// result is the one the naive kernel produced. Dispatched by threadgroups
// (ceil(seq/16) x ceil(out/16)); the padding tiles are zero and add
// nothing.
#define LIN_TILE 16u
#define LIN_TILE_K 32u
kernel void linear_nt_kernel(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *bias [[buffer(2)]],
    device float *y [[buffer(3)]],
    constant LinearParams &p [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]]
) {
    threadgroup float xs[LIN_TILE][LIN_TILE_K];
    threadgroup float ws[LIN_TILE][LIN_TILE_K];
    const uint s0 = tg.x * LIN_TILE;
    const uint o0 = tg.y * LIN_TILE;
    const uint s = s0 + tid.x;
    const uint o = o0 + tid.y;
    const bool live = s < p.seq && o < p.out;
    float acc = (live && p.has_bias != 0) ? bias[o] : 0.0f;
    const uint lane = tid.y * LIN_TILE + tid.x; // 0..255
    for (uint t0 = 0; t0 < p.k; t0 += LIN_TILE_K) {
        for (uint idx = lane; idx < LIN_TILE * LIN_TILE_K; idx += LIN_TILE * LIN_TILE) {
            const uint r = idx / LIN_TILE_K;
            const uint c = idx % LIN_TILE_K;
            const uint t = t0 + c;
            const uint sr = s0 + r;
            const uint orow = o0 + r;
            xs[r][c] = (sr < p.seq && t < p.k) ? x[ulong(sr) * p.k + t] : 0.0f;
            ws[r][c] = (orow < p.out && t < p.k) ? w[ulong(orow) * p.k + t] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < LIN_TILE_K; ++c) {
            acc += xs[tid.x][c] * ws[tid.y][c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (live) {
        y[ulong(s) * p.out + o] = acc;
    }
}

// The same product on simdgroup matrix units (Apple GPU family 7 and
// later): a threadgroup of four simdgroups (128 threads) produces a 32x32
// tile of y, each simdgroup a 16x16 block as four 8x8 accumulators, with
// W read transposed straight from its [out, k] layout. Rows, outputs and k
// must be multiples of 32, 32 and 8; the host pads the row count and the
// bias is added by add_bias_rows_kernel. The accumulation order differs
// from the scalar kernels, so results agree to float rounding, not bit
// for bit.
kernel void linear_nt_simd_kernel(
    device const float *x [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device float *y [[buffer(2)]],
    constant LinearParams &p [[buffer(3)]],
    uint2 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]]
) {
    const uint row0 = tg.y * 32u + (sg / 2u) * 16u;
    const uint out0 = tg.x * 32u + (sg % 2u) * 16u;
    if (row0 >= p.seq || out0 >= p.out) return;
    simdgroup_float8x8 c00(0.0f), c01(0.0f), c10(0.0f), c11(0.0f);
    for (uint kk = 0; kk < p.k; kk += 8u) {
        simdgroup_float8x8 a0, a1, b0, b1;
        simdgroup_load(a0, x + ulong(row0) * p.k + kk, p.k);
        simdgroup_load(a1, x + ulong(row0 + 8u) * p.k + kk, p.k);
        simdgroup_load(b0, w + ulong(out0) * p.k + kk, p.k, ulong2(0, 0), true);
        simdgroup_load(b1, w + ulong(out0 + 8u) * p.k + kk, p.k, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    simdgroup_store(c00, y + ulong(row0) * p.out + out0, p.out);
    simdgroup_store(c01, y + ulong(row0) * p.out + out0 + 8u, p.out);
    simdgroup_store(c10, y + ulong(row0 + 8u) * p.out + out0, p.out);
    simdgroup_store(c11, y + ulong(row0 + 8u) * p.out + out0 + 8u, p.out);
}

// y[s, o] += bias[o] over `seq` rows of `out` columns.
kernel void add_bias_rows_kernel(
    device float *y [[buffer(0)]],
    device const float *bias [[buffer(1)]],
    constant LinearParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint o = gid.x;
    const uint s = gid.y;
    if (s >= p.seq || o >= p.out) return;
    y[ulong(s) * p.out + o] += bias[o];
}

// MSL metal_math has no erf/erfc (probed: make metal-erf-probe).
// Hart–Cheney piecewise + complementary exp tail. Keep in sync with
// erf_approx.hpp. HF gelu is the erf form, not gelu_new / tanh.
inline float erf_hart(float x) {
    const float ax = fabs(x);
    if (!(ax == ax)) {
        return x;
    }
    if (ax >= 4.0f) {
        return copysign(1.0f, x);
    }
    if (ax < 0.84375f) {
        const float z = x * x;
        const float p =
            fma(fma(-1.86261395e-03f, z, -3.36030394e-01f), z, 1.28379166e-01f);
        const float q = fma(
            fma(fma(-1.98859372e-03f, z, 2.16070414e-02f), z, 3.12324315e-01f),
            z,
            1.0f
        );
        return fma(x, p / q, x);
    }
    if (ax < 1.25f) {
        const float s = ax - 1.0f;
        const float P = fma(
            fma(fma(8.67677554e-02f, s, -2.09395722e-01f), s, 4.15109307e-01f),
            s,
            3.65041046e-06f
        );
        const float Q = fma(
            fma(fma(3.92478965e-02f, s, 3.71248513e-01f), s, 4.95560974e-01f),
            s,
            1.0f
        );
        return copysign(8.42697144e-01f + P / Q, x);
    }
    const float inv2 = 1.0f / (ax * ax);
    float R;
    float S;
    if (ax < 2.85715f) {
        R = fma(
            fma(fma(-6.91554189e-01f, inv2, -1.66828310f), inv2, -5.43658376e-01f),
            inv2,
            -9.88156721e-03f
        );
        S = fma(
            fma(fma(5.53855181e-01f, inv2, 4.10799170f), inv2, 4.48581553f),
            inv2,
            1.0f
        );
    } else {
        R = fma(fma(-1.84115684f, inv2, -5.48049808e-01f), inv2, -9.86496918e-03f);
        S = fma(
            fma(fma(-7.61900663e-01f, inv2, 3.04982710f), inv2, 4.87132740f),
            inv2,
            1.0f
        );
    }
    const float z = as_type<float>(as_type<uint>(ax) & 0xffffe000u);
    const float r =
        exp(-z * z - 0.5625f) * exp((z - ax) * (z + ax) + R / S);
    return copysign(1.0f - r / ax, x);
}

kernel void gelu_erf_kernel(
    device float *x [[buffer(0)]],
    constant ElemParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    const float v = x[i];
    x[i] = 0.5f * v * (1.0f + erf_hart(v * 0.7071067811865476f));
}

kernel void add_inplace_kernel(
    device float *x [[buffer(0)]],
    device const float *r [[buffer(1)]],
    constant ElemParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    x[i] += r[i];
}

kernel void residual_from_ctx_kernel(
    device float *x [[buffer(0)]],
    device const float *ctx [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    constant ElemParams &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    x[i] = ctx[i] + residual[i];
}

kernel void copy_f32_kernel(
    device float *dst [[buffer(0)]],
    device const float *src [[buffer(1)]],
    constant ElemParams &p [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    dst[i] = src[i];
}

kernel void zero_f32_kernel(
    device float *x [[buffer(0)]],
    constant ElemParams &p [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= p.n) return;
    x[i] = 0.0f;
}

kernel void attention_scores_kernel(
    device const float *q [[buffer(0)]],
    device const float *k [[buffer(1)]],
    device float *attn [[buffer(2)]],
    device const int *mask [[buffer(3)]],
    constant AttnParams &p [[buffer(4)]],
    uint3 gid [[thread_position_in_grid]]
) {
    // z indexes (row, head) over the whole batch; rows are seq tokens apart.
    const uint j = gid.x;
    const uint i = gid.y;
    const uint bh = gid.z;
    const uint b = bh / p.heads;
    const uint h = bh % p.heads;
    if (b >= p.batch || i >= p.seq || j >= p.seq) return;
    const ulong row0 = ulong(b) * p.seq;
    const device float *qi = q + (row0 + i) * p.hidden + h * p.dh;
    const device float *kj = k + (row0 + j) * p.hidden + h * p.dh;
    float dot = 0.0f;
    for (uint d = 0; d < p.dh; ++d) {
        dot += qi[d] * kj[d];
    }
    float s = dot * p.scale;
    if (mask[row0 + j] == 0) {
        s = -10000.0f;
    }
    attn[(ulong(bh) * p.seq + i) * p.seq + j] = s;
}

kernel void softmax_rows_kernel(
    device float *attn [[buffer(0)]],
    constant AttnParams &p [[buffer(1)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint bh = gid.y;
    if (bh >= p.batch * p.heads || i >= p.seq) return;
    device float *row = attn + (ulong(bh) * p.seq + i) * p.seq;
    float m = row[0];
    for (uint j = 1; j < p.seq; ++j) {
        if (row[j] > m) m = row[j];
    }
    float sum = 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        row[j] = exp(row[j] - m);
        sum += row[j];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        row[j] *= inv;
    }
}

// One thread per (token i, hidden column c, row b): ctx[i, c] =
// sum_j attn[h, i, j] * v[j, c] with h = c / dh. Adjacent threads read
// adjacent columns of v.
kernel void attention_ctx_kernel(
    device const float *attn [[buffer(0)]],
    device const float *v [[buffer(1)]],
    device float *ctx [[buffer(2)]],
    constant AttnParams &p [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {
    const uint i = gid.x;
    const uint c = gid.y;
    const uint b = gid.z;
    if (b >= p.batch || i >= p.seq || c >= p.hidden) return;
    const uint h = c / p.dh;
    const ulong row0 = ulong(b) * p.seq;
    const device float *srow = attn + ((ulong(b) * p.heads + h) * p.seq + i) * p.seq;
    float acc = 0.0f;
    for (uint j = 0; j < p.seq; ++j) {
        acc += srow[j] * v[(row0 + j) * p.hidden + c];
    }
    ctx[(row0 + i) * p.hidden + c] = acc;
}

// Pooler over the [CLS] row of one sequence (BERT NSP pooler, tanh).
kernel void pooler_kernel(
    device const float *cls [[buffer(0)]],
    device const float *pw [[buffer(1)]],
    device const float *pb [[buffer(2)]],
    device float *pooled [[buffer(3)]],
    constant PoolParams &p [[buffer(4)]],
    uint o [[thread_position_in_grid]]
) {
    if (o >= p.hidden) return;
    float acc = pb[o];
    const device float *wr = pw + ulong(o) * p.hidden;
    for (uint h = 0; h < p.hidden; ++h) {
        acc += cls[h] * wr[h];
    }
    pooled[o] = tanh(acc);
}

// Classifier head: logit[l] = cb[l] + dot(head_in, cw[l]); `out` labels.
struct HeadParams {
    uint hidden;
    uint out;
    uint activation; // 0 none, 1 sigmoid, 2 softmax (applied by a later pass)
};

kernel void classifier_kernel(
    device const float *head_in [[buffer(0)]],
    device const float *cw [[buffer(1)]],
    device const float *cb [[buffer(2)]],
    device float *logit [[buffer(3)]],
    constant HeadParams &p [[buffer(4)]],
    uint l [[thread_position_in_grid]]
) {
    if (l >= p.out) return;
    float acc = cb[l];
    const device float *wr = cw + ulong(l) * p.hidden;
    for (uint h = 0; h < p.hidden; ++h) {
        acc += head_in[h] * wr[h];
    }
    if (p.activation == 1u) {
        acc = 1.0f / (1.0f + exp(-acc));
    }
    logit[l] = acc;
}

// Softmax over one row of `n` values, one thread.
kernel void softmax_row_kernel(
    device float *x [[buffer(0)]],
    constant ElemParams &p [[buffer(1)]]
) {
    float m = x[0];
    for (uint i = 1; i < p.n; ++i) {
        if (x[i] > m) m = x[i];
    }
    float sum = 0.0f;
    for (uint i = 0; i < p.n; ++i) {
        x[i] = exp(x[i] - m);
        sum += x[i];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint i = 0; i < p.n; ++i) {
        x[i] *= inv;
    }
}

// Sentence pooling of `batch` sequences' hidden states into `out` (one
// row of `hidden` floats per sequence): mean over live tokens (mask != 0),
// the [CLS] row, or the last live row. One thread per (column, sequence).
struct SentencePoolParams {
    uint seq;
    uint hidden;
    uint mode; // 1 mean, 2 cls, 3 last
    uint batch;
};

kernel void sentence_pool_kernel(
    device const float *xb [[buffer(0)]],
    device const int *maskb [[buffer(1)]],
    device float *outb [[buffer(2)]],
    constant SentencePoolParams &p [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    const uint h = gid.x;
    const uint r = gid.y;
    if (h >= p.hidden || r >= p.batch) return;
    const device float *x = xb + ulong(r) * p.seq * p.hidden;
    const device int *mask = maskb + ulong(r) * p.seq;
    device float *out = outb + ulong(r) * p.hidden;
    if (p.mode == 2u) {
        out[h] = x[h];
        return;
    }
    if (p.mode == 3u) {
        uint last = 0;
        for (uint t = 0; t < p.seq; ++t) {
            if (mask[t] != 0) last = t;
        }
        out[h] = x[ulong(last) * p.hidden + h];
        return;
    }
    float sum = 0.0f;
    uint n = 0;
    for (uint t = 0; t < p.seq; ++t) {
        if (mask[t] != 0) {
            sum += x[ulong(t) * p.hidden + h];
            ++n;
        }
    }
    out[h] = n > 0 ? sum / float(n) : 0.0f;
}

// L2-normalize row `r` of a packed [rows, n] matrix in place, one thread
// per row (n is the model dimension, a few hundred).
kernel void l2_normalize_kernel(
    device float *vb [[buffer(0)]],
    constant ElemParams &p [[buffer(1)]],
    uint r [[thread_position_in_grid]]
) {
    device float *v = vb + ulong(r) * p.n;
    float sum = 0.0f;
    for (uint i = 0; i < p.n; ++i) {
        sum += v[i] * v[i];
    }
    if (sum > 1e-24f) {
        const float inv = rsqrt(sum);
        for (uint i = 0; i < p.n; ++i) {
            v[i] *= inv;
        }
    }
}
)METAL"
