/* SPDX-License-Identifier: Apache-2.0
 *
 * The BERT encoder's kernels for the levelzero backend, in OpenCL C,
 * compiled to SPIR-V by build.rs and built for the device by its driver:
 * the linear layers, the embedding lookup, LayerNorm, GELU, attention and
 * pooling. The arithmetic follows the CPU encoder where the order matters:
 * LayerNorm's mean and variance are summed in F64 and its scale and shift
 * are two F32 operations; softmax subtracts the largest live score; mean
 * pooling sums each dimension over the row's positions in order, then
 * scales by 1 / count; the L2 norm is summed in F64 and floored at 1e-12.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable
/* Products and sums round separately, as on the CPU, unless fma says so. */
#pragma OPENCL FP_CONTRACT OFF

/* Work-items per group for the row kernels. */
#define BLOCK 128

#define POOLING_MEAN 1
#define POOLING_CLS 2
#define POOLING_LAST 3

/* ---- Linear layers ------------------------------------------------------
 *
 * y[t, o] = sum_i x[t, i] w[o, i], x [tokens, n_in] and w [n_out, n_in]
 * row-major. Each group of 16 x 16 work-items computes a 64 x 64 tile of
 * y, 4 x 4 values each, stepping through n_in 16 at a time with both
 * operands' slices in local memory. */

#define TILE 64
#define STEP 16

__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void linear(__global const float *x,
                                                                    __global const float *w,
                                                                    __global float *y, int tokens,
                                                                    int n_out, int n_in) {
    __local float xs[STEP][TILE + 1];
    __local float ws[STEP][TILE + 1];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int t0 = get_group_id(1) * TILE, o0 = get_group_id(0) * TILE;
    float acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;
    for (int k0 = 0; k0 < n_in; k0 += STEP) {
        for (int e = tid; e < TILE * STEP; e += 256) {
            const int r = e / STEP, c = e % STEP;
            const int t = t0 + r, o = o0 + r, k = k0 + c;
            xs[c][r] = t < tokens && k < n_in ? x[(size_t)t * n_in + k] : 0.0f;
            ws[c][r] = o < n_out && k < n_in ? w[(size_t)o * n_in + k] : 0.0f;
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
            if (o < n_out) y[(size_t)t * n_out + o] = acc[i][j];
        }
    }
}

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

/* One group per token: word + position + type, then LayerNorm. types is
 * read only when has_types is set; else every type is 0. */
__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void embed_layer_norm(
    __global const int *ids, __global const int *types, int has_types, __global const float *word,
    __global const float *position, __global const float *type, __global const float *ln_w,
    __global const float *ln_b, float eps, int seq, int hidden, __global float *x) {
    const size_t t = get_group_id(0);
    const int p = (int)(t % seq);
    __global const float *wr = word + (size_t)ids[t] * hidden;
    __global const float *pr = position + (size_t)p * hidden;
    __global const float *tr = type + (size_t)(has_types ? types[t] : 0) * hidden;
    __global float *row = x + t * hidden;
    for (int d = get_local_id(0); d < hidden; d += BLOCK) row[d] = wr[d] + pr[d] + tr[d];
    layer_norm_row(row, hidden, ln_w, ln_b, eps);
}

/* x = LayerNorm(x + (y + bias)), one group per token. */
__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void add_layer_norm(
    __global float *x, __global const float *y, __global const float *bias, __global const float *ln_w,
    __global const float *ln_b, float eps, int hidden) {
    const size_t t = get_group_id(0);
    __global float *row = x + t * hidden;
    __global const float *yr = y + t * hidden;
    for (int d = get_local_id(0); d < hidden; d += BLOCK) row[d] = row[d] + (yr[d] + bias[d]);
    layer_norm_row(row, hidden, ln_w, ln_b, eps);
}

/* y = GELU(y + bias), with erf, over n values of rows width wide. */
__kernel void bias_gelu(__global float *y, __global const float *bias, ulong n, int width) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) {
        const float v = y[i] + bias[i % width];
        y[i] = 0.5f * v * (1.0f + erf(v * 0.70710678118654752440f));
    }
}

/* ---- Attention ------------------------------------------------------------
 *
 * One group per (query, head, row). Local memory, sized by the host: the
 * query's head, the row's scores, and one partial context per group of
 * work-items. */

__kernel __attribute__((reqd_work_group_size(BLOCK, 1, 1))) void attention(
    __global const float *q, __global const float *k, __global const float *v, __global const float *bq,
    __global const float *bk, __global const float *bv, __global const int *mask, int seq, int hidden,
    int head_dim, float scale, __global float *ctx, __local float *sm) {
    __local float *qi = sm;
    __local float *score = qi + head_dim;
    __local float *part = score + seq;
    const int i = get_group_id(0), head = get_group_id(1), b = get_group_id(2);
    const int lid = get_local_id(0);
    const int lane = get_sub_group_local_id(), sg = get_sub_group_id();
    const int lanes = get_max_sub_group_size(), groups_of_lanes = get_num_sub_groups();
    const int col = head * head_dim;
    __global const int *m = mask + (size_t)b * seq;
    const size_t base = (size_t)b * seq;
    __global const float *qrow = q + (base + i) * hidden + col;
    for (int d = lid; d < head_dim; d += BLOCK) qi[d] = qrow[d] + bq[col + d];
    barrier(CLK_LOCAL_MEM_FENCE);
    // A sub-group per key: its lanes split the head's width.
    for (int j = sg; j < seq; j += groups_of_lanes) {
        if (m[j] == 0) continue;
        __global const float *krow = k + (base + j) * hidden + col;
        float s = 0.0f;
        for (int d = lane; d < head_dim; d += lanes) s += qi[d] * (krow[d] + bk[col + d]);
        s = sub_group_reduce_add(s);
        if (lane == 0) score[j] = s * scale;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float mx = -INFINITY;
    for (int j = lid; j < seq; j += BLOCK)
        if (m[j] != 0) mx = fmax(mx, score[j]);
    mx = work_group_reduce_max(mx);
    float sum = 0.0f;
    for (int j = lid; j < seq; j += BLOCK) {
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
    __global float *out = ctx + (base + i) * hidden + col;
    if (head_dim <= BLOCK) {
        // Groups of head_dim work-items each take every groups-th key; the
        // first group adds the partial sums in group order.
        const int groups = BLOCK / head_dim;
        const int d = lid % head_dim, g = lid / head_dim;
        if (g < groups) {
            float c = 0.0f;
            for (int j = g; j < seq; j += groups) {
                if (m[j] == 0) continue;
                const float p = score[j] * inv;
                c += p * (v[(base + j) * hidden + col + d] + bv[col + d]);
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
            for (int j = 0; j < seq; j++) {
                if (m[j] == 0) continue;
                const float p = score[j] * inv;
                c += p * (v[(base + j) * hidden + col + d] + bv[col + d]);
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
                                                                    __global const int *mask, int seq,
                                                                    int hidden, int output_dim, int pooling,
                                                                    int l2, __global float *out) {
    const int b = get_group_id(0);
    const int lid = get_local_id(0);
    __global const int *m = mask + (size_t)b * seq;
    __global const float *rows = x + (size_t)b * seq * hidden;
    __global float *dst = out + (size_t)b * output_dim;
    int l = 0;
    for (int p = lid; p < seq; p += BLOCK)
        if (m[p] != 0) l = p;
    const int last = work_group_reduce_max(l);
    double ss = 0;
    for (int d = lid; d < output_dim; d += BLOCK) {
        float val;
        if (pooling == POOLING_CLS) {
            val = rows[d];
        } else if (pooling == POOLING_LAST) {
            val = rows[(size_t)last * hidden + d];
        } else {
            float s = 0.0f;
            uint n = 0;
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

__kernel void widen_bf16(__global const ushort *src, ulong n, __global float *dst) {
    for (size_t i = get_global_id(0); i < n; i += get_global_size(0)) dst[i] = as_float((uint)src[i] << 16);
}
