/* SPDX-License-Identifier: Apache-2.0
 *
 * The BERT encoder's kernels for the levelzero backend, in OpenCL C,
 * compiled to SPIR-V by build.rs and built for the device by its driver:
 * the linear layers, the embedding lookup, LayerNorm, GELU, attention and
 * pooling.
 *
 * The rows of a batch are packed: row r's positions up to its last live
 * token sit one after another with the other rows', from rows[2r] for
 * rows[2r + 1] tokens, so padding past that token is never computed. No
 * output depends on it: attention skips masked keys, and no pooling reads
 * past the last live token. Positions are the row's own column indices.
 *
 * In F32, the arithmetic follows the CPU encoder where the order matters:
 * LayerNorm's mean and variance are summed in F64 and its scale and shift
 * are two F32 operations; mean pooling sums each dimension over the row's
 * positions in order, then scales by 1 / count; the L2 norm is summed in
 * F64 and floored at 1e-12. Softmax runs online over the keys, rescaling
 * as a larger score arrives, which equals subtracting the largest. At
 * FASTEST the linear layers and attention run on the matrix engines from
 * F16 operands with F32 sums; the rest is as in F32.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
/* Products and sums round separately, as on the CPU, unless fma says so. */
#pragma OPENCL FP_CONTRACT OFF

/* Work-items per group for the row kernels. */
#define BLOCK 128

#define POOLING_MEAN 1
#define POOLING_CLS 2
#define POOLING_LAST 3

/* The linear kernel's epilogue. */
#define LINEAR_BIAS 1
#define LINEAR_GELU 2

float gelu(float v) { return 0.5f * v * (1.0f + erf(v * 0.70710678118654752440f)); }

/* ---- Linear layers ------------------------------------------------------
 *
 * y[t, o] = sum_i x[t, i] w[o, i] (+ bias[o], then GELU, as flags say), x
 * [tokens, n_in] and w [n_out, n_in] row-major. A split of the terms
 * (group z of n_in / k_len) writes its partial sums to its own [tokens,
 * n_out] slice of y, for the next kernel to add; the epilogue runs only
 * unsplit. Each group of 16 x 16
 * work-items computes a 64 x 64 tile of y, 4 x 4 values each, stepping
 * through n_in 16 at a time with both operands' slices in local memory. */

#define TILE 64
#define STEP 16

__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void linear(__global const float *x,
                                                                    __global const float *w,
                                                                    __global const float *bias,
                                                                    __global float *y, int tokens,
                                                                    int n_out, int n_in, int flags,
                                                                    int k_len) {
    __local float xs[STEP][TILE + 1];
    __local float ws[STEP][TILE + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int t0 = get_group_id(1) * TILE, o0 = get_group_id(0) * TILE;
    const int k_start = get_group_id(2) * k_len, k_end = min(n_in, k_start + k_len);
    y += (size_t)get_group_id(2) * tokens * n_out;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = k_start; k0 < k_end; k0 += STEP) {
        for (int e = tid; e < TILE * STEP; e += 256) {
            const int r = e / STEP, c = e % STEP;
            const int t = t0 + r, o = o0 + r, k = k0 + c;
            xs[c][r] = t < tokens && k < k_end ? x[(size_t)t * n_in + k] : 0.0f;
            ws[c][r] = o < n_out && k < k_end ? w[(size_t)o * n_in + k] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int kk = 0; kk < STEP; kk++) {
            float a[4], b[4];
            for (int i = 0; i < 4; i++) a[i] = xs[kk][ty + 16 * i];
            for (int j = 0; j < 4; j++) b[j] = ws[kk][tx + 16 * j];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int i = 0; i < 4; i++) {
        const int t = t0 + ty + 16 * i;
        if (t >= tokens) continue;
        for (int j = 0; j < 4; j++) {
            const int o = o0 + tx + 16 * j;
            if (o >= n_out) continue;
            float v = acc[i][j];
            if (flags & LINEAR_BIAS) v = v + bias[o];
            if (flags & LINEAR_GELU) v = gelu(v);
            y[(size_t)t * n_out + o] = v;
        }
    }
}

/* The same in F32 by sub-group: 16 lanes compute 8 tokens by 64 outputs,
 * each lane four outputs 16 apart, 16 terms at a time: a block read gives
 * each lane one term of a token's row, which a shuffle hands to every
 * lane, and each lane reads its outputs' weights itself. n_in is a
 * multiple of 16; the split of the terms is as in linear. */

__attribute__((overloadable)) float intel_sub_group_shuffle(float x, uint c);
__attribute__((overloadable)) uint intel_sub_group_block_read(const __global uint *p);

#define SG_T 8
#define SG_N 64

__kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1))) void
linear_sg(__global const float *x, __global const float *w, __global const float *bias, __global float *y,
          int tokens, int n_out, int n_in, int flags, int k_len) {
    const int lane = get_sub_group_local_id();
    const int o0 = get_group_id(0) * SG_N, t0 = get_group_id(1) * SG_T;
    x += get_group_id(2) * k_len;
    w += get_group_id(2) * k_len;
    y += (size_t)get_group_id(2) * tokens * n_out;
    float acc[SG_T][4];
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < SG_T; m++)
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) acc[m][j] = 0.0f;
    __global const float *wr[4];
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++)
        wr[j] = w + (size_t)min(o0 + 16 * j + lane, n_out - 1) * n_in;
    for (int k0 = 0; k0 < k_len; k0 += 16) {
        float a[SG_T];
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < SG_T; m++) {
            const int t = t0 + m;
            a[m] = t < tokens ? as_float(intel_sub_group_block_read((__global const uint *)(x + (size_t)t * n_in + k0)))
                              : 0.0f;
        }
        float16 b[4];
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) b[j] = vload16(0, wr[j] + k0);
        __attribute__((opencl_unroll_hint)) for (int kk = 0; kk < 16; kk++) {
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < SG_T; m++) {
                const float am = intel_sub_group_shuffle(a[m], kk);
                __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) acc[m][j] = fma(am, b[j][kk], acc[m][j]);
            }
        }
    }
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) {
        const int o = o0 + 16 * j + lane;
        if (o >= n_out) continue;
        const float bo = flags & LINEAR_BIAS ? bias[o] : 0.0f;
        __attribute__((opencl_unroll_hint)) for (int m = 0; m < SG_T; m++) {
            const int t = t0 + m;
            if (t >= tokens) continue;
            float v = acc[m][j];
            if (flags & LINEAR_BIAS) v = v + bo;
            if (flags & LINEAR_GELU) v = gelu(v);
            y[(size_t)t * n_out + o] = v;
        }
    }
}

/* The linear layers at FASTEST, on the matrix engines (XMX,
 * cl_intel_subgroup_matrix_multiply_accumulate): the same y, from F16
 * weights and F16 activations, with F32 sums. A product takes 8 x 16 of A
 * (a lane per term, 8 tokens' values each) times 16 x 16 of B (a lane per
 * output, its 16 terms in pairs) into 8 x 16 sums (a lane per output, 8
 * tokens each). One kernel per operand type: X the activations', read as
 * F32 and rounded, or as F16; Y the output's. n_in is a multiple of 16;
 * the split of the terms is as in linear.
 *
 * A sub-group computes 8 tokens by 64 outputs, as linear_sg, one product
 * per 16 outputs. */

__attribute__((overloadable)) float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);
__attribute__((overloadable)) ushort intel_sub_group_block_read_us(const __global ushort *p);

#define LINEAR_XMX(NAME, X, Y, TO_A, FROM_F)                                                                    \
    __kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1))) void \
    NAME(__global const X *x, __global const half *w, __global const float *bias, __global Y *y, int tokens,     \
         int n_out, int n_in, int flags, int k_len, int k_sub) {                                                 \
        const int lane = get_sub_group_local_id();                                                               \
        const int o0 = get_group_id(0) * SG_N, t0 = get_group_id(1) * SG_T;                                           \
        x += get_group_id(2) * k_len;                                                                            \
        w += get_group_id(2) * k_len;                                                                            \
        y += (size_t)get_group_id(2) * tokens * n_out;                                                           \
        float8 acc[4];                                                                                           \
        __global const half *wr[4];                                                                              \
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) {                                        \
            acc[j] = (float8)(0.0f);                                                                             \
            wr[j] = w + (size_t)min(o0 + 16 * j + lane, n_out - 1) * n_in;                                       \
        }                                                                                                        \
        for (int k0 = 0; k0 < k_len; k0 += 16) {                                                                 \
            short8 a;                                                                                            \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                                    \
                const int t = t0 + m;                                                                            \
                a[m] = t < tokens ? TO_A(x + (size_t)t * n_in + k0) : 0;                                         \
            }                                                                                                    \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) acc[j] =                             \
                intel_sub_group_f16_f16_matrix_mad_k16(a, as_int8(vload8(0, (__global const uint *)(wr[j] + k0))), acc[j]); \
        }                                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) {                                        \
            const int o = o0 + 16 * j + lane;                                                                    \
            if (o >= n_out) continue;                                                                            \
            const float bo = flags & LINEAR_BIAS ? bias[o] : 0.0f;                                               \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                                    \
                const int t = t0 + m;                                                                            \
                if (t >= tokens) continue;                                                                       \
                float v = acc[j][m];                                                                             \
                if (flags & LINEAR_BIAS) v = v + bo;                                                             \
                if (flags & LINEAR_GELU) v = gelu(v);                                                            \
                y[(size_t)t * n_out + o] = FROM_F(v);                                                            \
            }                                                                                                    \
        }                                                                                                        \
    }

/* The same for layers with too few tiles to fill the device: a group of
 * KS sub-groups computes 32 tokens by 32 outputs, each sub-group summing
 * its own k_sub of the terms; the sums meet in local memory, and each
 * sub-group finishes 8 of the tokens. */

#define XM 32
#define XN 32
#define KS 4
#define LINEAR_XMX_SHARED(NAME, X, Y, TO_A, FROM_F)                                                                     \
    __kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * KS, 1, 1))) \
    void NAME(__global const X *x, __global const half *w, __global const float *bias, __global Y *y,            \
              int tokens, int n_out, int n_in, int flags, int k_len, int k_sub) {                                \
        __local float part[KS][4][2][8][16];                                                                     \
        const int lane = get_sub_group_local_id(), sg = get_sub_group_id();                                      \
        const int o0 = get_group_id(0) * XN, t0 = get_group_id(1) * XM;                                          \
        x += get_group_id(2) * k_len + sg * k_sub;                                                               \
        w += get_group_id(2) * k_len + sg * k_sub;                                                               \
        y += (size_t)get_group_id(2) * tokens * n_out;                                                           \
        float8 acc[4][2];                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < 4; i++)                                          \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) acc[i][j] = (float8)(0.0f);          \
        if (sg * k_sub < k_len) {                                                                                \
            for (int k0 = 0; k0 < k_sub; k0 += 16) {                                                             \
                short8 a[4];                                                                                     \
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 4; i++)                                  \
                    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                            \
                    const int t = t0 + i * 8 + m;                                                                \
                    a[i][m] = t < tokens ? TO_A(x + (size_t)t * n_in + k0) : 0;                                  \
                }                                                                                                \
                int8 b[2];                                                                                       \
                __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) {                                \
                    const int o = o0 + j * 16 + lane;                                                            \
                    b[j] = o < n_out ? as_int8(vload8(0, (__global const uint *)(w + (size_t)o * n_in + k0)))    \
                                     : (int8)(0);                                                                \
                }                                                                                                \
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < 4; i++)                                  \
                    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) acc[i][j] =                  \
                    intel_sub_group_f16_f16_matrix_mad_k16(a[i], b[j], acc[i][j]);                               \
            }                                                                                                    \
        }                                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < 4; i++)                                          \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++)                                      \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) part[sg][i][j][m][lane] =        \
                acc[i][j][m];                                                                                    \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                            \
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) {                                        \
            const int o = o0 + j * 16 + lane;                                                                    \
            if (o >= n_out) continue;                                                                            \
            const float bo = flags & LINEAR_BIAS ? bias[o] : 0.0f;                                               \
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                                    \
                const int t = t0 + sg * 8 + m;                                                                   \
                if (t >= tokens) continue;                                                                       \
                float v = part[0][sg][j][m][lane];                                                               \
                __attribute__((opencl_unroll_hint)) for (int s = 1; s < KS; s++) v = v + part[s][sg][j][m][lane]; \
                if (flags & LINEAR_BIAS) v = v + bo;                                                             \
                if (flags & LINEAR_GELU) v = gelu(v);                                                            \
                y[(size_t)t * n_out + o] = FROM_F(v);                                                            \
            }                                                                                                    \
        }                                                                                                        \
    }

#define F32_TO_A(p) as_short(convert_half(as_float(intel_sub_group_block_read((__global const uint *)(p)))))
#define F16_TO_A(p) as_short(intel_sub_group_block_read_us((__global const ushort *)(p)))
#define TO_F32(v) (v)
#define TO_F16(v) convert_half(v)

LINEAR_XMX(linear_xmx, float, float, F32_TO_A, TO_F32)
LINEAR_XMX(linear_xmx_to_half, float, half, F32_TO_A, TO_F16)
LINEAR_XMX(linear_xmx_from_half, short, float, F16_TO_A, TO_F32)
LINEAR_XMX_SHARED(linear_xmx_shared, float, float, F32_TO_A, TO_F32)
LINEAR_XMX_SHARED(linear_xmx_shared_to_half, float, half, F32_TO_A, TO_F16)
LINEAR_XMX_SHARED(linear_xmx_shared_from_half, short, float, F16_TO_A, TO_F32)

/* ---- Rows ---------------------------------------------------------------- */

/* row = (row - mean) / sqrt(var + eps) * w + b, with the mean and the
 * biased variance summed in F64. Each work-item touches only the columns
 * it wrote, so no barrier is needed before this. */
void layer_norm_row(__global float *row, int n, __global const float *w, __global const float *b, float eps) {
    const int lid = get_local_id(0);
    double s = 0;
    for (int d = lid; d < n; d += BLOCK) s += row[d];
    const double mean = work_group_reduce_add(s) / n;
    double v = 0;
    for (int d = lid; d < n; d += BLOCK) {
        const double c = row[d] - mean;
        v += c * c;
    }
    const double var = work_group_reduce_add(v) / n;
    const double inv = 1.0 / sqrt(var + (double)eps);
    for (int d = lid; d < n; d += BLOCK) {
        const float xn = (float)((row[d] - mean) * inv);
        row[d] = xn * w[d] + b[d];
    }
}

/* One group per packed token: word + position + type, then LayerNorm; and
 * the token's mask entry. ids, positions, types (when has_types is set;
 * else every type is 0) and mask are packed by the host into memory the
 * device reads directly, with the row table, which the first groups copy
 * to device memory for the kernels after this one. */
__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void embed_layer_norm(
    __global const int *ids, __global const int *positions, __global const int *types, int has_types,
    __global const int *mask, __global const int *rows, int batch, __global const float *word,
    __global const float *position, __global const float *type, __global const float *ln_w,
    __global const float *ln_b, float eps, int hidden, __global float *x, __global int *packed_mask,
    __global int *device_rows) {
    const int t = get_group_id(0);
    __global const float *wr = word + (size_t)ids[t] * hidden;
    __global const float *pr = position + (size_t)positions[t] * hidden;
    __global const float *tr = type + (size_t)(has_types ? types[t] : 0) * hidden;
    __global float *row = x + (size_t)t * hidden;
    for (int d = get_local_id(0); d < hidden; d += BLOCK) row[d] = wr[d] + pr[d] + tr[d];
    if (get_local_id(0) == 0) packed_mask[t] = mask[t];
    if (t < batch && get_local_id(0) < 2) device_rows[2 * t + get_local_id(0)] = rows[2 * t + get_local_id(0)];
    layer_norm_row(row, hidden, ln_w, ln_b, eps);
}

/* x = LayerNorm(x + (y + bias)), one group per packed token. y is the sum
 * of parts partial sums, each [tokens, hidden], one after another. */
__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void add_layer_norm(
    __global float *x, __global const float *y, __global const float *bias, __global const float *ln_w,
    __global const float *ln_b, float eps, int hidden, int parts) {
    const size_t t = get_group_id(0);
    const size_t part = (size_t)get_num_groups(0) * hidden;
    __global float *row = x + t * hidden;
    __global const float *yr = y + t * hidden;
    for (int d = get_local_id(0); d < hidden; d += BLOCK) {
        float sum = yr[d];
        for (int p = 1; p < parts; p++) sum = sum + yr[p * part + d];
        row[d] = row[d] + (sum + bias[d]);
    }
    layer_norm_row(row, hidden, ln_w, ln_b, eps);
}

/* ---- Attention ------------------------------------------------------------
 *
 * qkv is [tokens, 3 * hidden], each token's query, key and value with their
 * biases, heads side by side in each. ctx is [tokens, hidden].
 *
 * The tiled kernels: one work-item per query of a row and head, up to 256
 * queries a group, the query and its running context in registers. The
 * row's keys and values stream through local memory 64 at a time, and
 * each work-item keeps an online softmax over them, so every key and value
 * is read from global memory once per group. One kernel per head width, so
 * the registers are sized at compile time. */

#define KEYS 64
#define QUERIES 256

#define FLASH(HD)                                                                                                  \
    __kernel __attribute__((reqd_work_group_size(QUERIES, 1, 1))) void attention_##HD(                                \
        __global const float *qkv, __global const int *mask, __global const int *rows, int hidden, float scale,    \
        __global float *ctx) {                                                                                     \
        __local float4 ks[KEYS][HD / 4];                                                                           \
        __local float4 vs[KEYS][HD / 4];                                                                           \
        __local int live[KEYS];                                                                                    \
        const int head = get_group_id(1), r = get_group_id(2);                                                     \
        const int start = rows[2 * r], len = rows[2 * r + 1];                                                      \
        const int i = get_group_id(0) * QUERIES + get_local_id(0);                                                    \
        if (get_group_id(0) * QUERIES >= len) return;                                                                 \
        const int lid = get_local_id(0);                                                                           \
        const int stride = 3 * hidden, col = head * HD;                                                            \
        const bool mine = i < len;                                                                                 \
        float4 q[HD / 4], acc[HD / 4];                                                                             \
        __global const float *qrow = qkv + (size_t)(start + (mine ? i : 0)) * stride + col;                        \
        __attribute__((opencl_unroll_hint)) for (int d = 0; d < HD / 4; d++) {                                     \
            q[d] = vload4(d, qrow) * scale;                                                                        \
            acc[d] = (float4)(0.0f);                                                                               \
        }                                                                                                          \
        float m = -INFINITY, l = 0.0f;                                                                             \
        for (int j0 = 0; j0 < len; j0 += KEYS) {                                                                   \
            const int n = min(KEYS, len - j0);                                                                     \
            for (int e = lid; e < n * (HD / 4); e += QUERIES) {                                                       \
                const int j = e / (HD / 4), d = e % (HD / 4);                                                      \
                __global const float *kv = qkv + (size_t)(start + j0 + j) * stride + col;                          \
                ks[j][d] = vload4(d, kv + hidden);                                                                 \
                vs[j][d] = vload4(d, kv + 2 * hidden);                                                             \
            }                                                                                                      \
            for (int j = lid; j < n; j += QUERIES) live[j] = mask[start + j0 + j];                                                       \
            barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
            if (mine) {                                                                                            \
                for (int j = 0; j < n; j++) {                                                                      \
                    if (live[j] == 0) continue;                                                                    \
                    float4 s4 = q[0] * ks[j][0];                                                                   \
                    __attribute__((opencl_unroll_hint)) for (int d = 1; d < HD / 4; d++) s4 = fma(q[d], ks[j][d], s4); \
                    const float s = (s4.x + s4.y) + (s4.z + s4.w);                                                 \
                    if (s > m) {                                                                                   \
                        const float c = exp(m - s);                                                                \
                        l *= c;                                                                                    \
                        __attribute__((opencl_unroll_hint)) for (int d = 0; d < HD / 4; d++) acc[d] *= c;          \
                        m = s;                                                                                     \
                    }                                                                                              \
                    const float p = exp(s - m);                                                                    \
                    l += p;                                                                                        \
                    __attribute__((opencl_unroll_hint)) for (int d = 0; d < HD / 4; d++) acc[d] = fma(p, vs[j][d], acc[d]); \
                }                                                                                                  \
            }                                                                                                      \
            barrier(CLK_LOCAL_MEM_FENCE);                                                                          \
        }                                                                                                          \
        if (!mine) return;                                                                                         \
        const float inv = 1.0f / l;                                                                                \
        __global float *out = ctx + (size_t)(start + i) * hidden + col;                                            \
        __attribute__((opencl_unroll_hint)) for (int d = 0; d < HD / 4; d++) vstore4(acc[d] * inv, d, out);        \
    }

FLASH(32)
FLASH(64)
FLASH(128)

/* At FASTEST, on the matrix engines: a sub-group takes 16 queries of a row
 * and head, a lane each, and walks the row's keys 16 at a time. The scores
 * are K times Q transposed, so a lane holds its own query's 16 scores
 * (K as A, 8 keys a product and a lane per term; Q as B, a lane per
 * query), and its online softmax needs no other lane. The context is V
 * transposed times the weights (V as A, 8 of the width a product and a
 * lane per key; the weights as B, a lane per query). qkv and ctx are F16,
 * the sums and the softmax F32. */
#define FLASH_XMX(HD)                                                                                              \
    __kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16, 1, 1))) void   \
    attention_xmx_##HD(__global const half *qkv, __global const int *mask, __global const int *rows, int hidden,   \
                       float scale, __global half *ctx) {                                                          \
        const int lane = get_sub_group_local_id();                                                                 \
        const int q0 = get_group_id(0) * 16, head = get_group_id(1), r = get_group_id(2);                          \
        const int start = rows[2 * r], len = rows[2 * r + 1];                                                      \
        if (q0 >= len) return;                                                                                     \
        const int stride = 3 * hidden, col = head * HD;                                                            \
        __global const half *base = qkv + (size_t)start * stride + col;                                           \
        const int query = min(q0 + lane, len - 1);                                                                 \
        int8 qb[HD / 16];                                                                                          \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 16; b++) qb[b] =                              \
            as_int8(vload8(0, (__global const uint *)(base + (size_t)query * stride + b * 16)));                   \
        float8 acc[HD / 8];                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++) acc[b] = (float8)(0.0f);              \
        float mx = -INFINITY, l = 0.0f;                                                                            \
        for (int j0 = 0; j0 < len; j0 += 16) {                                                                     \
            float8 s[2];                                                                                           \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++) {                                      \
                s[h] = (float8)(0.0f);                                                                             \
                __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 16; b++) {                            \
                    short8 ka;                                                                                     \
                    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                              \
                        const int key = min(j0 + h * 8 + m, len - 1);                                              \
                        ka[m] = as_short(base[(size_t)key * stride + hidden + b * 16 + lane]);                     \
                    }                                                                                              \
                    s[h] = intel_sub_group_f16_f16_matrix_mad_k16(ka, qb[b], s[h]);                                \
                }                                                                                                  \
            }                                                                                                      \
            float cmax = -INFINITY;                                                                                \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++)                                        \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                                  \
                const int key = j0 + h * 8 + m;                                                                    \
                const bool live = key < len && mask[start + min(key, len - 1)] != 0;                               \
                s[h][m] = live ? s[h][m] * scale : -INFINITY;                                                      \
                cmax = fmax(cmax, s[h][m]);                                                                        \
            }                                                                                                      \
            const float newm = fmax(mx, cmax);                                                                     \
            const float corr = newm == -INFINITY ? 1.0f : exp(mx - newm);                                          \
            float8 p[2];                                                                                           \
            float psum = 0.0f;                                                                                     \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++)                                        \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                                  \
                p[h][m] = s[h][m] == -INFINITY ? 0.0f : exp(s[h][m] - newm);                                       \
                psum += p[h][m];                                                                                   \
            }                                                                                                      \
            l = l * corr + psum;                                                                                   \
            mx = newm;                                                                                             \
            int8 pb;                                                                                               \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) {                                      \
                pb[j] = as_int((ushort2)(as_ushort(convert_half(p[0][2 * j])), as_ushort(convert_half(p[0][2 * j + 1])))); \
                pb[j + 4] = as_int((ushort2)(as_ushort(convert_half(p[1][2 * j])), as_ushort(convert_half(p[1][2 * j + 1])))); \
            }                                                                                                      \
            __global const half *vrow = base + (size_t)min(j0 + lane, len - 1) * stride + 2 * hidden;              \
            __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++) {                                 \
                const short8 va = as_short8(vload8(b, (__global const ushort *)vrow));                             \
                acc[b] = intel_sub_group_f16_f16_matrix_mad_k16(va, pb, acc[b] * corr);                            \
            }                                                                                                      \
        }                                                                                                          \
        if (q0 + lane >= len) return;                                                                              \
        const float inv = 1.0f / l;                                                                                \
        __global half *out = ctx + (size_t)(start + q0 + lane) * hidden + col;                                     \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++)                                       \
            vstore_half8(acc[b] * inv, b, out);                                                                    \
    }

FLASH_XMX(32)

/* Any other head width: one group per (query, head, row), the row's
 * scores in local memory, sized by the host: the query's head, the row's
 * scores, and one partial context per group of work-items. */
__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void attention(
    __global const float *qkv, __global const int *mask, __global const int *rows, int hidden, int head_dim,
    float scale, __global float *ctx, __local float *sm) {
    const int i = get_group_id(0), head = get_group_id(1), r = get_group_id(2);
    const int start = rows[2 * r], len = rows[2 * r + 1];
    if (i >= len) return;
    __local float *qi = sm;
    __local float *score = qi + head_dim;
    __local float *part = score + len;
    const int lid = get_local_id(0);
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int lanes = get_max_sub_group_size(), groups_of_lanes = get_num_sub_groups();
    const int stride = 3 * hidden, col = head * head_dim;
    __global const int *m = mask + start;
    __global const float *base = qkv + (size_t)start * stride + col;
    for (int d = lid; d < head_dim; d += BLOCK) qi[d] = base[(size_t)i * stride + d];
    barrier(CLK_LOCAL_MEM_FENCE);
    // A sub-group per key: its lanes split the head's width.
    for (int j = sg; j < len; j += groups_of_lanes) {
        if (m[j] == 0) continue;
        __global const float *krow = base + (size_t)j * stride + hidden;
        float s = 0.0f;
        for (int d = lane; d < head_dim; d += lanes) s += qi[d] * krow[d];
        s = sub_group_reduce_add(s);
        if (lane == 0) score[j] = s * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float mx = -INFINITY;
    for (int j = lid; j < len; j += BLOCK)
        if (m[j] != 0) mx = fmax(mx, score[j]);
    mx = work_group_reduce_max(mx);
    float sum = 0.0f;
    for (int j = lid; j < len; j += BLOCK) {
        float e = 0.0f;
        if (m[j] != 0) {
            e = exp(score[j] - mx);
            sum += e;
        }
        score[j] = e;
    }
    sum = work_group_reduce_add(sum);
    const float inv = 1.0f / sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    __global float *out = ctx + (size_t)(start + i) * hidden + col;
    if (head_dim <= BLOCK) {
        const int groups = BLOCK / head_dim;
        const int d = lid % head_dim, g = lid / head_dim;
        if (g < groups) {
            float c = 0.0f;
            for (int j = g; j < len; j += groups) {
                if (m[j] == 0) continue;
                c += score[j] * inv * base[(size_t)j * stride + 2 * hidden + d];
            }
            part[g * head_dim + d] = c;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (lid < head_dim) {
            float c = 0.0f;
            for (int gg = 0; gg < groups; gg++) c += part[gg * head_dim + lid];
            out[lid] = c;
        }
    } else {
        for (int d = lid; d < head_dim; d += BLOCK) {
            float c = 0.0f;
            for (int j = 0; j < len; j++) {
                if (m[j] == 0) continue;
                c += score[j] * inv * base[(size_t)j * stride + 2 * hidden + d];
            }
            out[d] = c;
        }
    }
}

/* ---- Pooling ----------------------------------------------------------------
 *
 * One group per row: the mean over the tokens whose mask is 1, the first
 * token, or the last live one; cut to output_dim; then, when l2 is set,
 * divided by its L2 norm. */

__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void pool(__global const float *x,
                                                                    __global const int *mask,
                                                                    __global const int *rows, int hidden,
                                                                    int output_dim, int pooling, int l2,
                                                                    __global float *out) {
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    const int start = rows[2 * b], len = rows[2 * b + 1];
    __global const int *m = mask + start;
    __global const float *tokens = x + (size_t)start * hidden;
    __global float *dst = out + (size_t)b * output_dim;
    double ss = 0;
    for (int d = lid; d < output_dim; d += BLOCK) {
        float val;
        if (pooling == POOLING_CLS) {
            val = tokens[d];
        } else if (pooling == POOLING_LAST) {
            val = tokens[(size_t)(len - 1) * hidden + d];
        } else {
            float s = 0.0f;
            uint n = 0;
            for (int p = 0; p < len; p++) {
                if (m[p] == 0) continue;
                s += tokens[(size_t)p * hidden + d];
                n++;
            }
            val = s * (1.0f / (float)n);
        }
        dst[d] = val;
        ss += (double)val * (double)val;
    }
    const double total = work_group_reduce_add(ss);
    if (!l2) return;
    const double norm = fmax(sqrt(total), 1e-12);
    const float inv = (float)(1.0 / norm);
    for (int d = lid; d < output_dim; d += BLOCK) dst[d] *= inv;
}

/* ---- Widening ------------------------------------------------------------------
 *
 * An F16 or BF16 model's weights in F32, for the sessions that compute in
 * F32. */

__kernel void widen_f16(__global const half *src, ulong n, __global float *dst) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = vload_half(i, src);
}

__kernel void narrow_f16(__global const float *src, ulong n, __global half *dst) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = convert_half(src[i]);
}

__kernel void widen_bf16(__global const ushort *src, ulong n, __global float *dst) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = as_float((uint)src[i] << 16);
}
