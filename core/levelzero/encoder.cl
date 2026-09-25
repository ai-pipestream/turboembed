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
 * F16 operands with F32 sums, and the LayerNorms after the projections
 * sum in F32 in their epilogues; the rest is as in F32.
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

/* The same for a handful of tokens, 8 at most: a group of GV sub-groups
 * computes 8 tokens by 32 outputs, each summing its own k_sub of the
 * terms, and the sums meet in local memory. */
#define GV 8
__kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * GV, 1, 1))) void
linear_gemv(__global const float *x, __global const float *w, __global const float *bias, __global float *y,
            int tokens, int n_out, int n_in, int flags, int k_len, int k_sub) {
    __local float part[GV][2][8][16];
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int o0 = get_group_id(0) * 32;
    x += get_group_id(2) * k_len + sg * k_sub;
    w += get_group_id(2) * k_len + sg * k_sub;
    y += (size_t)get_group_id(2) * tokens * n_out;
    float acc[8][2];
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) acc[m][0] = acc[m][1] = 0.0f;
    __global const float *w0 = w + (size_t)min(o0 + lane, n_out - 1) * n_in;
    __global const float *w1 = w + (size_t)min(o0 + 16 + lane, n_out - 1) * n_in;
    if (sg * k_sub < k_len) {
        for (int k0 = 0; k0 < k_sub; k0 += 16) {
            float a[8];
            __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) a[m] =
                m < tokens ? as_float(intel_sub_group_block_read((__global const uint *)(x + (size_t)m * n_in + k0))) : 0.0f;
            const float16 b0 = vload16(0, w0 + k0), b1 = vload16(0, w1 + k0);
            __attribute__((opencl_unroll_hint)) for (int kk = 0; kk < 16; kk++) {
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {
                    const float am = intel_sub_group_shuffle(a[m], kk);
                    acc[m][0] = fma(am, b0[kk], acc[m][0]);
                    acc[m][1] = fma(am, b1[kk], acc[m][1]);
                }
            }
        }
    }
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {
        part[sg][0][m][lane] = acc[m][0];
        part[sg][1][m][lane] = acc[m][1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    /* Sub-group g finishes token g. */
    if (sg >= tokens) return;
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) {
        const int o = o0 + j * 16 + lane;
        if (o >= n_out) continue;
        float v = part[0][j][sg][lane];
        __attribute__((opencl_unroll_hint)) for (int s = 1; s < GV; s++) v = v + part[s][j][sg][lane];
        if (flags & LINEAR_BIAS) v = v + bias[o];
        if (flags & LINEAR_GELU) v = gelu(v);
        y[(size_t)sg * n_out + o] = v;
    }
}

/* The linear layers at FASTEST, on the matrix engines (XMX,
 * cl_intel_subgroup_matrix_multiply_accumulate): the same y, from F16
 * activations and F16 weights transposed to wt [n_in, n_out], with F32
 * sums, written as F32 or F16. A product takes 8 x 16 of A (a lane per
 * term, 8 tokens' values each) times 16 x 16 of B (a lane per output, its
 * 16 terms in pairs) into 8 x 16 sums (a lane per output, 8 tokens each).
 *
 * Both operands arrive by 2D block reads (cl_intel_subgroup_2d_block_io),
 * already in the product's layout: A as rows of x, B through the read's
 * VNNI transform, which pairs two rows of wt in each lane. A read past the
 * last token gives zeros and a write past it is dropped, so a tile needs
 * no bounds of its own. n_in, k_len and n_out are multiples of 32, and the
 * buffers 64-byte aligned.
 *
 * A sub-group computes TM tokens by TN outputs, 32 terms a step; a group
 * of WM x WN sub-groups computes a TM * WM by TN * WN tile, so the group's
 * sub-groups read the same rows of x and wt from cache. Group z sums the
 * z-th k_len of the terms into its own slice of y, as linear does. */

__attribute__((overloadable)) float8 intel_sub_group_f16_f16_matrix_mad_k16(short8 a, int8 b, float8 acc);
__attribute__((overloadable)) void intel_sub_group_2d_block_read_16b_8r16x2c(__global void *base, int width,
                                                                             int height, int pitch, int2 coord,
                                                                             __private ushort *dst);
__attribute__((overloadable)) void intel_sub_group_2d_block_read_16b_16r16x2c(__global void *base, int width,
                                                                              int height, int pitch, int2 coord,
                                                                              __private ushort *dst);
__attribute__((overloadable)) void intel_sub_group_2d_block_read_transform_16b_32r16x1c(__global void *base,
                                                                                        int width, int height,
                                                                                        int pitch, int2 coord,
                                                                                        __private uint *dst);
__attribute__((overloadable)) void intel_sub_group_2d_block_write_32b_8r16x1c(__global void *base, int width,
                                                                              int height, int pitch, int2 coord,
                                                                              __private uint *src);
__attribute__((overloadable)) void intel_sub_group_2d_block_write_16b_8r16x1c(__global void *base, int width,
                                                                              int height, int pitch, int2 coord,
                                                                              __private ushort *src);

/* A's two k-halves for 8 or 16 tokens: a 2D read of 8 or 16 rows by two
 * 16-term blocks, block by block, 8 rows a short8. A LINEAR_DPAS instance
 * names the one for its TM. */
#define DPAS_READ_A8(x, w, h, k, t, a)                                                                              \
    do {                                                                                                            \
        short8 r_[2];                                                                                               \
        intel_sub_group_2d_block_read_16b_8r16x2c((__global void *)(x), w, h, w, (int2)(k, t), (__private ushort *)r_); \
        a[0][0] = r_[0];                                                                                            \
        a[1][0] = r_[1];                                                                                            \
    } while (0)
#define DPAS_READ_A16(x, w, h, k, t, a, i)                                                                          \
    do {                                                                                                            \
        short8 r_[4];                                                                                               \
        intel_sub_group_2d_block_read_16b_16r16x2c((__global void *)(x), w, h, w, (int2)(k, t), (__private ushort *)r_); \
        a[0][i] = r_[0];                                                                                            \
        a[0][i + 1] = r_[1];                                                                                        \
        a[1][i] = r_[2];                                                                                            \
        a[1][i + 1] = r_[3];                                                                                        \
    } while (0)

#define DPAS_READ_A_16(x, w, h, k, t, a) DPAS_READ_A16(x, w, h, k, t, a, 0)
#define DPAS_READ_A_8(x, w, h, k, t, a) DPAS_READ_A8(x, w, h, k, t, a)

#define DPAS_STORE_F32(y, n_out, tokens, o, t, v)                                                                   \
    intel_sub_group_2d_block_write_32b_8r16x1c((__global void *)(y), (n_out) * 4, tokens, (n_out) * 4, (int2)(o, t), \
                                               (__private uint *)&(v))
#define DPAS_STORE_F16(y, n_out, tokens, o, t, v)                                                                   \
    do {                                                                                                            \
        ushort8 h_ = as_ushort8(convert_half8(v));                                                                  \
        intel_sub_group_2d_block_write_16b_8r16x1c((__global void *)(y), (n_out) * 2, tokens, (n_out) * 2,          \
                                                   (int2)(o, t), (__private ushort *)&h_);                          \
    } while (0)

#define LINEAR_DPAS(NAME, TM, TN, WM, WN, READ_A, Y, STORE)                                                                 \
    __kernel __attribute__((intel_reqd_sub_group_size(16)))                                                         \
    __attribute__((reqd_work_group_size(16 * (WM) * (WN), 1, 1))) void                                              \
    NAME(__global const half *x, __global const half *wt, __global const float *bias, __global Y *y, int tokens,    \
         int n_out, int n_in, int flags, int k_len) {                                                               \
        const int sg = get_sub_group_id(), lane = get_sub_group_local_id();                                         \
        const int o0 = (get_group_id(0) * (WN) + sg % (WN)) * (TN);                                                 \
        const int t0 = (get_group_id(1) * (WM) + sg / (WN)) * (TM);                                                 \
        if (o0 >= n_out || t0 >= tokens) return;                                                                    \
        const int k0 = get_group_id(2) * k_len;                                                                     \
        y += (size_t)get_group_id(2) * tokens * n_out;                                                              \
        float8 acc[(TM) / 8][(TN) / 16];                                                                            \
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++)                                      \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < (TN) / 16; j++) acc[i][j] = (float8)(0.0f);     \
        for (int k = k0; k < k0 + k_len; k += 32) {                                                                 \
            short8 a[2][(TM) / 8];                                                                                  \
            int8 b[2][(TN) / 16];                                                                                   \
            READ_A(x, n_in * 2, tokens, k, t0, a);                                                                  \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < (TN) / 16; j++) {                               \
                int8 r[2];                                                                                          \
                intel_sub_group_2d_block_read_transform_16b_32r16x1c((__global void *)wt, n_out * 2, n_in,          \
                                                                     n_out * 2, (int2)(o0 + 16 * j, k),             \
                                                                     (__private uint *)r);                          \
                b[0][j] = r[0];                                                                                     \
                b[1][j] = r[1];                                                                                     \
            }                                                                                                       \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++)                                         \
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++)                              \
                    __attribute__((opencl_unroll_hint)) for (int j = 0; j < (TN) / 16; j++) acc[i][j] =             \
                        intel_sub_group_f16_f16_matrix_mad_k16(a[h][i], b[h][j], acc[i][j]);                        \
        }                                                                                                           \
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < (TN) / 16; j++) {                                   \
            const float bo = flags & LINEAR_BIAS ? bias[o0 + 16 * j + lane] : 0.0f;                                 \
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++) {                                \
                float8 v = acc[i][j];                                                                               \
                __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                                   \
                    if (flags & LINEAR_BIAS) v[m] = v[m] + bo;                                                      \
                    if (flags & LINEAR_GELU) v[m] = gelu(v[m]);                                                     \
                }                                                                                                   \
                STORE(y, n_out, tokens, o0 + 16 * j, t0 + 8 * i, v);                                                \
            }                                                                                                       \
        }                                                                                                           \
    }

/* A group of 8 x 2 sub-groups of 16 tokens by 32 outputs; and for at most
 * 8 tokens, 1 x 4 sub-groups of 8 by 32. */
LINEAR_DPAS(linear_dpas, 16, 32, 8, 2, DPAS_READ_A_16, float, DPAS_STORE_F32)
LINEAR_DPAS(linear_dpas_to_half, 16, 32, 8, 2, DPAS_READ_A_16, half, DPAS_STORE_F16)
LINEAR_DPAS(linear_dpas_few, 8, 32, 1, 4, DPAS_READ_A_8, float, DPAS_STORE_F32)
LINEAR_DPAS(linear_dpas_few_to_half, 8, 32, 1, 4, DPAS_READ_A_8, half, DPAS_STORE_F16)

/* The attention output and the feed-forward output at FASTEST, with the
 * LayerNorm after them: x = LayerNorm(x + (y + bias)) as add_layer_norm,
 * and xh its F16 copy, y = act . wt as in LINEAR_DPAS with n_out = hidden.
 * A group computes TM tokens by the whole hidden width, a sub-group each
 * 32 outputs, so a group holds whole rows, and the rows' sums meet in
 * local memory. The sums are F32. hidden is at most 32 * DPAS_LN_SUBGROUPS. */

__attribute__((overloadable)) void intel_sub_group_2d_block_read_32b_8r16x1c(__global void *base, int width,
                                                                             int height, int pitch, int2 coord,
                                                                             __private uint *dst);

#define DPAS_LN_SUBGROUPS 64

/* The sum over a sub-group's lanes of each of v's 8 values. */
float8 sub_group_sum8(float8 v) {
    float8 s;
    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) s[m] = sub_group_reduce_add(v[m]);
    return s;
}

#define LINEAR_DPAS_LN(NAME, TM)                                                                                    \
    __kernel __attribute__((intel_reqd_sub_group_size(16))) void NAME(                                              \
        __global const half *act, __global const half *wt, __global const float *bias, __global float *x,           \
        __global half *xh, __global const float *ln_w, __global const float *ln_b, float eps, int tokens,          \
        int hidden, int n_in) {                                                                                     \
        __local float part[DPAS_LN_SUBGROUPS][TM];                                                                  \
        __local float mean[TM], inv[TM];                                                                            \
        const int sg = get_sub_group_id(), lane = get_sub_group_local_id(), lid = get_local_id(0);                 \
        const int subgroups = get_num_sub_groups();                                                                 \
        const int o0 = sg * 32, t0 = get_group_id(0) * (TM);                                                        \
        float8 acc[(TM) / 8][2];                                                                                    \
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++) acc[i][0] = acc[i][1] = (float8)(0.0f); \
        for (int k = 0; k < n_in; k += 32) {                                                                        \
            short8 a[2][(TM) / 8];                                                                                  \
            int8 b[2][2];                                                                                           \
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i += 2)                               \
                DPAS_READ_A16(act, n_in * 2, tokens, k, t0 + 8 * i, a, i);                                          \
            __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) {                                       \
                int8 r[2];                                                                                          \
                intel_sub_group_2d_block_read_transform_16b_32r16x1c((__global void *)wt, hidden * 2, n_in,         \
                                                                     hidden * 2, (int2)(o0 + 16 * j, k),            \
                                                                     (__private uint *)r);                          \
                b[0][j] = r[0];                                                                                     \
                b[1][j] = r[1];                                                                                     \
            }                                                                                                       \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 2; h++)                                         \
                __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++)                              \
                    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) acc[i][j] =                     \
                        intel_sub_group_f16_f16_matrix_mad_k16(a[h][i], b[h][j], acc[i][j]);                        \
        }                                                                                                           \
        /* The residual and the bias, then each row's mean. */                                                     \
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) {                                           \
            const float bo = bias[o0 + 16 * j + lane];                                                              \
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++) {                                \
                float8 r;                                                                                           \
                intel_sub_group_2d_block_read_32b_8r16x1c((__global void *)x, hidden * 4, tokens, hidden * 4,       \
                                                          (int2)(o0 + 16 * j, t0 + 8 * i), (__private uint *)&r);   \
                acc[i][j] = r + (acc[i][j] + bo);                                                                   \
            }                                                                                                       \
        }                                                                                                           \
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++) {                                    \
            const float8 s = sub_group_sum8(acc[i][0] + acc[i][1]);                                                 \
            if (lane == 0) vstore8(s, 0, &part[sg][8 * i]);                                                         \
        }                                                                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                               \
        if (lid < (TM)) {                                                                                           \
            float s = 0.0f;                                                                                         \
            for (int g = 0; g < subgroups; g++) s += part[g][lid];                                                  \
            mean[lid] = s / hidden;                                                                                 \
        }                                                                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                               \
        /* The biased variance about it. */                                                                         \
        __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++) {                                    \
            const float8 mu = vload8(0, &mean[8 * i]);                                                              \
            const float8 d0 = acc[i][0] - mu, d1 = acc[i][1] - mu;                                                  \
            const float8 s = sub_group_sum8(d0 * d0 + d1 * d1);                                                     \
            if (lane == 0) vstore8(s, 0, &part[sg][8 * i]);                                                         \
        }                                                                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                               \
        if (lid < (TM)) {                                                                                           \
            float s = 0.0f;                                                                                         \
            for (int g = 0; g < subgroups; g++) s += part[g][lid];                                                  \
            inv[lid] = 1.0f / sqrt(s / hidden + eps);                                                               \
        }                                                                                                           \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                               \
        __attribute__((opencl_unroll_hint)) for (int j = 0; j < 2; j++) {                                           \
            const int o = o0 + 16 * j;                                                                              \
            const float w = ln_w[o + lane], sh = ln_b[o + lane];                                                    \
            __attribute__((opencl_unroll_hint)) for (int i = 0; i < (TM) / 8; i++) {                                \
                float8 y = (acc[i][j] - vload8(0, &mean[8 * i])) * vload8(0, &inv[8 * i]) * w + sh;                 \
                DPAS_STORE_F32(x, hidden, tokens, o, t0 + 8 * i, y);                                                \
                DPAS_STORE_F16(xh, hidden, tokens, o, t0 + 8 * i, y);                                               \
            }                                                                                                       \
        }                                                                                                           \
    }

LINEAR_DPAS_LN(linear_dpas_layer_norm, 16)

/* ---- Rows ---------------------------------------------------------------- */

/* Rows run a sub-group a token, ROWS tokens a group, so the LayerNorm's
 * sums are sub-group reductions with no barrier. */
#define ROWS 8

/* row = (row - mean) / sqrt(var + eps) * w + b, with the mean and the
 * biased variance summed in F64, by one sub-group; and the same in F16 to
 * half_row where that is given. Each lane touches only the columns it
 * wrote, so nothing need be waited for before this. */
void layer_norm_row(__global float *row, int n, __global const float *w, __global const float *b, float eps,
                    __global half *half_row) {
    const int lane = get_sub_group_local_id();
    double s = 0;
    for (int d = lane; d < n; d += 16) s += row[d];
    const double mean = sub_group_reduce_add(s) / n;
    double v = 0;
    for (int d = lane; d < n; d += 16) {
        const double c = row[d] - mean;
        v += c * c;
    }
    const double var = sub_group_reduce_add(v) / n;
    const double inv = 1.0 / sqrt(var + (double)eps);
    for (int d = lane; d < n; d += 16) {
        const float xn = (float)((row[d] - mean) * inv);
        const float y = xn * w[d] + b[d];
        row[d] = y;
        if (half_row) half_row[d] = convert_half(y);
    }
}

/* One sub-group per packed token: word + position + type, then LayerNorm;
 * and the token's mask entry. ids, positions, types (when has_types is set;
 * else every type is 0) and mask are packed by the host into memory the
 * device reads directly, with the row table, which the first tokens copy
 * to device memory for the kernels after this one. At FASTEST xh takes
 * the rows in F16 as well, for the linear layers. */
__kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * ROWS, 1, 1))) void
embed_layer_norm(__global const int *ids, __global const int *positions, __global const int *types, int has_types,
                 __global const int *mask, __global const int *rows, int batch, __global const float *word,
                 __global const float *position, __global const float *type, __global const float *ln_w,
                 __global const float *ln_b, float eps, int hidden, __global float *x, __global int *packed_mask,
                 __global int *device_rows, int tokens, __global half *xh) {
    const int t = get_group_id(0) * ROWS + get_sub_group_id();
    if (t >= tokens) return;
    const int lane = get_sub_group_local_id();
    __global const float *wr = word + (size_t)ids[t] * hidden;
    __global const float *pr = position + (size_t)positions[t] * hidden;
    __global const float *tr = type + (size_t)(has_types ? types[t] : 0) * hidden;
    __global float *row = x + (size_t)t * hidden;
    for (int d = lane; d < hidden; d += 16) row[d] = wr[d] + pr[d] + tr[d];
    if (lane == 0) packed_mask[t] = mask[t];
    if (t < batch && lane < 2) device_rows[2 * t + lane] = rows[2 * t + lane];
    layer_norm_row(row, hidden, ln_w, ln_b, eps, xh ? xh + (size_t)t * hidden : 0);
}

/* x = LayerNorm(x + (y + bias)), one sub-group per packed token. y is the
 * sum of parts partial sums, each [tokens, hidden], one after another. */
__kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * ROWS, 1, 1))) void
add_layer_norm(__global float *x, __global const float *y, __global const float *bias, __global const float *ln_w,
               __global const float *ln_b, float eps, int hidden, int parts, int tokens, __global half *xh) {
    const int t = get_group_id(0) * ROWS + get_sub_group_id();
    if (t >= tokens) return;
    const int lane = get_sub_group_local_id();
    const size_t part = (size_t)tokens * hidden;
    __global float *row = x + (size_t)t * hidden;
    __global const float *yr = y + (size_t)t * hidden;
    for (int d = lane; d < hidden; d += 16) {
        float sum = yr[d];
        for (int p = 1; p < parts; p++) sum = sum + yr[p * part + d];
        row[d] = row[d] + (sum + bias[d]);
    }
    layer_norm_row(row, hidden, ln_w, ln_b, eps, xh ? xh + (size_t)t * hidden : 0);
}

/* The same two kernels for few tokens, where a sub-group's serial walk
 * of a row is the run's latency: a group of BLOCK work-items per token,
 * with group reductions. */

void layer_norm_row_group(__global float *row, int n, __global const float *w, __global const float *b, float eps,
                          __global half *half_row) {
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
        const float v = xn * w[d] + b[d];
        row[d] = v;
        if (half_row) half_row[d] = convert_half(v);
    }
}

__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void embed_layer_norm_group(
    __global const int *ids, __global const int *positions, __global const int *types, int has_types,
    __global const int *mask, __global const int *rows, int batch, __global const float *word,
    __global const float *position, __global const float *type, __global const float *ln_w,
    __global const float *ln_b, float eps, int hidden, __global float *x, __global int *packed_mask,
    __global int *device_rows, int tokens, __global half *xh) {
    const int t = get_group_id(0);
    __global const float *wr = word + (size_t)ids[t] * hidden;
    __global const float *pr = position + (size_t)positions[t] * hidden;
    __global const float *tr = type + (size_t)(has_types ? types[t] : 0) * hidden;
    __global float *row = x + (size_t)t * hidden;
    for (int d = get_local_id(0); d < hidden; d += BLOCK) row[d] = wr[d] + pr[d] + tr[d];
    if (get_local_id(0) == 0) packed_mask[t] = mask[t];
    if (t < batch && get_local_id(0) < 2) device_rows[2 * t + get_local_id(0)] = rows[2 * t + get_local_id(0)];
    layer_norm_row_group(row, hidden, ln_w, ln_b, eps, xh ? xh + (size_t)t * hidden : 0);
}

__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void add_layer_norm_group(
    __global float *x, __global const float *y, __global const float *bias, __global const float *ln_w,
    __global const float *ln_b, float eps, int hidden, int parts, int tokens, __global half *xh) {
    const size_t t = get_group_id(0);
    const size_t part = (size_t)tokens * hidden;
    __global float *row = x + t * hidden;
    __global const float *yr = y + t * hidden;
    for (int d = get_local_id(0); d < hidden; d += BLOCK) {
        float sum = yr[d];
        for (int p = 1; p < parts; p++) sum = sum + yr[p * part + d];
        row[d] = row[d] + (sum + bias[d]);
    }
    layer_norm_row_group(row, hidden, ln_w, ln_b, eps, xh ? xh + (size_t)t * hidden : 0);
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

#define FLASH(NAME, HD, CT, STORE4)                                                                                \
    __kernel __attribute__((reqd_work_group_size(QUERIES, 1, 1))) void NAME(                                          \
        __global const float *qkv, __global const int *mask, __global const int *rows, int hidden, float scale,    \
        __global CT *ctx) {                                                                                     \
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
        __global CT *out = ctx + (size_t)(start + i) * hidden + col;                                               \
        __attribute__((opencl_unroll_hint)) for (int d = 0; d < HD / 4; d++) STORE4(acc[d] * inv, d, out);         \
    }

FLASH(attention_32, 32, float, vstore4)
FLASH(attention_64, 64, float, vstore4)
FLASH(attention_128, 128, float, vstore4)
/* At FASTEST, for the next layer's F16 operand. */
FLASH(attention_128_to_half, 128, half, vstore_half4)

/* At FASTEST, on the matrix engines: a group of ATT_KS sub-groups takes 16
 * queries of a row and head, a lane each; sub-group g walks the row's keys
 * 32 at a time from the g-th 32, ATT_KS * 32 apart. The scores are K
 * times Q transposed, so a lane holds its own query's 32 scores (K as A,
 * 8 keys a product and a lane per term; Q as B, a lane per query), and its
 * online softmax needs no other lane. The context is V transposed times
 * the weights (V as A, 8 of the width a product and a lane per key; the
 * weights as B, a lane per query). K and V arrive by 2D block reads, V's
 * transposed; rows past the row's last token read as zeros. The softmax
 * runs in base 2 on scores scaled by log2(e). The sub-groups' maxima, sums
 * and contexts meet in local memory, rescaled to the largest maximum. qkv
 * and ctx are F16, the sums and the softmax F32. The head width is a
 * multiple of 32, and so is hidden, so a head's keys and values start
 * 64-byte aligned for the 2D reads. */
#define ATT_KS 4

__attribute__((overloadable)) void intel_sub_group_2d_block_read_transpose_32b_16r8x1c(__global void *base, int width,
                                                                                       int height, int pitch,
                                                                                       int2 coord,
                                                                                       __private uint *dst);

/* Two float8s of probabilities, rounded to F16 in pairs: the B operand for
 * 16 keys, a lane per query. */
int8 pack_probabilities(float8 lo, float8 hi) {
    int8 pb;
    __attribute__((opencl_unroll_hint)) for (int j = 0; j < 4; j++) {
        pb[j] = as_int((ushort2)(as_ushort(convert_half(lo[2 * j])), as_ushort(convert_half(lo[2 * j + 1]))));
        pb[j + 4] = as_int((ushort2)(as_ushort(convert_half(hi[2 * j])), as_ushort(convert_half(hi[2 * j + 1]))));
    }
    return pb;
}

#define FLASH_XMX(HD)                                                                                              \
    __kernel __attribute__((intel_reqd_sub_group_size(16)))                                                        \
    __attribute__((reqd_work_group_size(16 * ATT_KS, 1, 1))) void                                                  \
    attention_xmx_##HD(__global const half *qkv, __global const int *mask, __global const int *rows, int hidden,   \
                       float scale, __global half *ctx) {                                                          \
        __local float red_m[ATT_KS][16], red_l[ATT_KS][16];                                                        \
        __local float8 red_acc[ATT_KS][HD / 8][16];                                                                \
        const int lane = get_sub_group_local_id(), sg = get_sub_group_id();                                        \
        const int q0 = get_group_id(0) * 16, head = get_group_id(1), r = get_group_id(2);                          \
        const int start = rows[2 * r], len = rows[2 * r + 1];                                                      \
        if (q0 >= len) return;                                                                                     \
        const int stride = 3 * hidden, col = head * HD;                                                            \
        __global const half *base = qkv + (size_t)start * stride + col;                                           \
        const int query = min(q0 + lane, len - 1);                                                                 \
        int8 qb[HD / 16];                                                                                          \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 16; b++) qb[b] =                              \
            as_int8(vload8(0, (__global const uint *)(base + (size_t)query * stride + b * 16)));                   \
        const float scale2 = scale * 1.44269504088896340736f;                                                      \
        float8 acc[HD / 8];                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++) acc[b] = (float8)(0.0f);              \
        float mx = -INFINITY, l = 0.0f;                                                                            \
        for (int j0 = sg * 32; j0 < len; j0 += 32 * ATT_KS) {                                                      \
            /* s[2c + h]: keys j0 + 16c + 8h .. + 7. */                                                            \
            float8 s[4] = {(float8)(0.0f), (float8)(0.0f), (float8)(0.0f), (float8)(0.0f)};                       \
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < 2; c++)                                        \
                __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 32; b++) {                            \
                short8 ka[4];                                                                                      \
                intel_sub_group_2d_block_read_16b_16r16x2c((__global void *)(base + hidden), HD * 2, len,          \
                                                           stride * 2, (int2)(32 * b, j0 + 16 * c),                \
                                                           (__private ushort *)ka);                                \
                s[2 * c] = intel_sub_group_f16_f16_matrix_mad_k16(ka[0], qb[2 * b], s[2 * c]);                    \
                s[2 * c + 1] = intel_sub_group_f16_f16_matrix_mad_k16(ka[1], qb[2 * b], s[2 * c + 1]);            \
                s[2 * c] = intel_sub_group_f16_f16_matrix_mad_k16(ka[2], qb[2 * b + 1], s[2 * c]);                \
                s[2 * c + 1] = intel_sub_group_f16_f16_matrix_mad_k16(ka[3], qb[2 * b + 1], s[2 * c + 1]);        \
            }                                                                                                      \
            /* Lane i reads the mask of keys j0 + i and j0 + 16 + i. */                                            \
            const bool live0 = j0 + lane < len && mask[start + min(j0 + lane, len - 1)] != 0;                       \
            const bool live1 = j0 + 16 + lane < len && mask[start + min(j0 + 16 + lane, len - 1)] != 0;             \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 4; h++) s[h] *= scale2;                        \
            if (!sub_group_all(live0 && live1)) {                                                                  \
                __attribute__((opencl_unroll_hint)) for (int h = 0; h < 4; h++)                                    \
                    __attribute__((opencl_unroll_hint)) for (int m = 0; m < 8; m++) {                              \
                    const bool live = sub_group_broadcast((int)(h < 2 ? live0 : live1), (h & 1) * 8 + m) != 0;                 \
                    if (!live) s[h][m] = -INFINITY;                                                                \
                }                                                                                                  \
            }                                                                                                      \
            const float8 top = fmax(fmax(s[0], s[1]), fmax(s[2], s[3]));                                           \
            const float4 t4 = fmax(top.lo, top.hi);                                                                \
            const float cmax = fmax(fmax(t4.x, t4.y), fmax(t4.z, t4.w));                                           \
            const float newm = fmax(mx, cmax);                                                                     \
            if (newm == -INFINITY) continue;                                                                       \
            const float corr = mx == -INFINITY ? 0.0f : native_exp2(mx - newm);                                    \
            float8 p[4];                                                                                           \
            __attribute__((opencl_unroll_hint)) for (int h = 0; h < 4; h++) p[h] = native_exp2(s[h] - newm);        \
            const float8 p8 = (p[0] + p[1]) + (p[2] + p[3]);                                                       \
            const float4 p4 = p8.lo + p8.hi;                                                                       \
            l = l * corr + ((p4.x + p4.y) + (p4.z + p4.w));                                                        \
            mx = newm;                                                                                             \
            __attribute__((opencl_unroll_hint)) for (int c = 0; c < 2; c++) {                                      \
                const int8 pb = pack_probabilities(p[2 * c], p[2 * c + 1]);                                        \
                __attribute__((opencl_unroll_hint)) for (int v = 0; v < HD / 16; v++) {                            \
                    uint8 va;                                                                                      \
                    intel_sub_group_2d_block_read_transpose_32b_16r8x1c((__global void *)(base + 2 * hidden),      \
                                                                        HD * 2, len, stride * 2,                   \
                                                                        (int2)(8 * v, j0 + 16 * c),                \
                                                                        (__private uint *)&va);                    \
                    const float8 a0 = c == 0 ? acc[2 * v] * corr : acc[2 * v];                                     \
                    const float8 a1 = c == 0 ? acc[2 * v + 1] * corr : acc[2 * v + 1];                             \
                    acc[2 * v] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(va.lo), pb, a0);                 \
                    acc[2 * v + 1] = intel_sub_group_f16_f16_matrix_mad_k16(as_short8(va.hi), pb, a1);             \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
        red_m[sg][lane] = mx;                                                                                      \
        red_l[sg][lane] = l;                                                                                       \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++) red_acc[sg][b][lane] = acc[b];        \
        barrier(CLK_LOCAL_MEM_FENCE);                                                                              \
        if (sg != 0 || q0 + lane >= len) return;                                                                   \
        float m_all = -INFINITY;                                                                                   \
        for (int g = 0; g < ATT_KS; g++) m_all = fmax(m_all, red_m[g][lane]);                                      \
        float l_all = 0.0f;                                                                                        \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++) acc[b] = (float8)(0.0f);              \
        for (int g = 0; g < ATT_KS; g++) {                                                                         \
            const float mg = red_m[g][lane];                                                                       \
            const float c = mg == -INFINITY ? 0.0f : native_exp2(mg - m_all);                                      \
            l_all += red_l[g][lane] * c;                                                                           \
            __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++) acc[b] += red_acc[g][b][lane] * c; \
        }                                                                                                          \
        const float inv = 1.0f / l_all;                                                                            \
        __global half *out = ctx + (size_t)(start + q0 + lane) * hidden + col;                                     \
        __attribute__((opencl_unroll_hint)) for (int b = 0; b < HD / 8; b++)                                       \
            vstore_half8(acc[b] * inv, b, out);                                                                    \
    }

FLASH_XMX(32)
FLASH_XMX(64)

/* Any other head width: one group per (query, head, row), the row's
 * scores in local memory, sized by the host: the query's head, the row's
 * scores, and one partial context per group of work-items. The context
 * goes to ctx_half in F16 instead where that is given, at FASTEST. */
__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void attention(
    __global const float *qkv, __global const int *mask, __global const int *rows, int hidden, int head_dim,
    float scale, __global float *ctx, __global half *ctx_half, __local float *sm) {
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
    const size_t at = (size_t)(start + i) * hidden + col;
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
            if (ctx_half)
                ctx_half[at + lid] = convert_half(c);
            else
                ctx[at + lid] = c;
        }
    } else {
        for (int d = lid; d < head_dim; d += BLOCK) {
            float c = 0.0f;
            for (int j = 0; j < len; j++) {
                if (m[j] == 0) continue;
                c += score[j] * inv * base[(size_t)j * stride + 2 * hidden + d];
            }
            if (ctx_half)
                ctx_half[at + d] = convert_half(c);
            else
                ctx[at + d] = c;
        }
    }
}

/* ---- Pooling ----------------------------------------------------------------
 *
 * Each row's vector, cut to output_dim: the mean over the tokens whose mask
 * is 1, the first token, or the last live one. A group takes 16 of a row's
 * dimensions, a lane each, and its 8 sub-groups each sum every eighth of
 * the row's tokens; the eight sums are added in order. Then, when asked,
 * normalize divides each row by its L2 norm, summed in F64 and floored at
 * 1e-12. */

#define POOL_SLICES 8

__kernel __attribute__((intel_reqd_sub_group_size(16))) __attribute__((reqd_work_group_size(16 * POOL_SLICES, 1, 1)))
void pool(__global const float *x, __global const int *mask, __global const int *rows, int hidden, int output_dim,
          int pooling, __global float *out) {
    __local float part[POOL_SLICES][16];
    __local uint count[POOL_SLICES];
    const int b = get_group_id(0), lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int d = get_group_id(1) * 16 + lane;
    const int start = rows[2 * b], len = rows[2 * b + 1];
    __global const int *m = mask + start;
    __global const float *tokens = x + (size_t)start * hidden;
    const int dd = min(d, output_dim - 1);
    float s = 0.0f;
    uint n = 0;
    if (pooling == POOLING_CLS) {
        s = sg == 0 ? tokens[dd] : 0.0f;
    } else if (pooling == POOLING_LAST) {
        s = sg == 0 ? tokens[(size_t)(len - 1) * hidden + dd] : 0.0f;
    } else {
        for (int p = sg; p < len; p += POOL_SLICES) {
            if (m[p] == 0) continue;
            s += tokens[(size_t)p * hidden + dd];
            n++;
        }
    }
    part[sg][lane] = s;
    if (lane == 0) count[sg] = n;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (sg != 0 || d >= output_dim) return;
    float v = part[0][lane];
    uint total = count[0];
    for (int g = 1; g < POOL_SLICES; g++) {
        v += part[g][lane];
        total += count[g];
    }
    if (pooling != POOLING_CLS && pooling != POOLING_LAST) v = v * (1.0f / (float)total);
    out[(size_t)b * output_dim + d] = v;
}

__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void normalize(__global float *out, int output_dim) {
    __global float *dst = out + (size_t)get_group_id(0) * output_dim;
    const int lid = get_local_id(0);
    double ss = 0;
    for (int d = lid; d < output_dim; d += BLOCK) ss += (double)dst[d] * (double)dst[d];
    const double norm = fmax(sqrt(work_group_reduce_add(ss)), 1e-12);
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

/* w [n_out, n_in] in F32 to wt [n_in, n_out] in F16, the layout the
 * XMX linear kernels read. */
__kernel void narrow_f16_transposed(__global const float *w, int n_out, int n_in, __global half *wt) {
    const size_t n = (size_t)n_out * n_in;
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const size_t k = i / n_out, o = i % n_out;
        wt[i] = convert_half(w[o * n_in + k]);
    }
}

__kernel void widen_bf16(__global const ushort *src, ulong n, __global float *dst) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = as_float((uint)src[i] << 16);
}
