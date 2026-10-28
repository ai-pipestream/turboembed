/* SPDX-License-Identifier: Apache-2.0
 *
 * The rotary families' device code (rotary.h), included by kernels.cu
 * inside its own namespace, after the row kernels and before the GEMMs,
 * whose epilogues call the two helpers at the top: the rotation of a
 * pair-adjacent Q or K pair (EPI_QKV_ROPE) and a gated MLP's activation
 * (EPI_GATED_*). Then the embedding lookup and the pre-norm's add and
 * LayerNorm, warp-per-token kernels as kernels.cu's own.
 */

/* Token t's position in its row: its column, the packed row's own. */
__device__ __forceinline__ int rope_position(const GemmArgs &g, int t) { return t - g.row_start[g.tok_row[t]]; }

/* The rotation of the pair at even column c of token t's QKV product, its
 * sums v0 (column c) and v1 (column c + 1) with their biases: Q's and K's
 * columns (c under 2 hidden) are each head's (i, i + d/2) side by side,
 * so the pair is i = (c mod d) / 2, rotated by the table's cos and sin of
 * (position, i), y1 = x1 cos - x2 sin and y2 = x2 cos + x1 sin, as the
 * CPU's Rope (core/src/cpu/kernels/pairs.rs); V's columns are left. */
__device__ __forceinline__ void rope_pair(const GemmArgs &g, int t, int c, float &v0, float &v1) {
    if (c >= 2 * g.hidden) return;
    const size_t at = (size_t)rope_position(g, t) * g.rope_pairs + (c % g.head_dim) / 2;
    const float cs = __ldg(g.rope_cos + at), sn = __ldg(g.rope_sin + at);
    const float x1 = v0, x2 = v1;
    v0 = __fsub_rn(__fmul_rn(x1, cs), __fmul_rn(x2, sn));
    v1 = __fadd_rn(__fmul_rn(x2, cs), __fmul_rn(x1, sn));
}

/* act(gate) * up for a gated epilogue EPI; GELU by erff at an F32 output
 * whichever GELU it names. */
template <int EPI, typename TOut> __device__ __forceinline__ float gated_of(float gate, float up) {
    float a;
    if constexpr (EPI == EPI_GATED_SILU)
        a = sizeof(TOut) == 2 ? gate / (1.0f + __expf(-gate)) : gate / (1.0f + expf(-gate));
    else if constexpr (EPI == EPI_GATED_TANH && sizeof(TOut) == 2)
        a = gelu_tanh(gate);
    else
        a = gelu(gate);
    return a * up;
}

/* Whether EPI writes the QKV product head-major. */
__host__ __device__ constexpr bool qkv_epilogue_of(int epi) { return epi == EPI_QKV || epi == EPI_QKV_ROPE; }

__host__ __device__ constexpr bool gated_epilogue_of(int epi) {
    return epi == EPI_GATED_ERF || epi == EPI_GATED_TANH || epi == EPI_GATED_SILU;
}

/* x[t] = LayerNorm(word[ids] + type[types]) for every packed token, the
 * type row only when type is not NULL (row 0 where the run has no types),
 * no position row: the positions are in the rotation. */
template <int V>
__global__ void __launch_bounds__(ROW_BLOCK)
    rotary_embed_kernel(const int32_t *rows, const float *word, const float *type, const float *ln_w,
                        const float *ln_b, float eps, Packing p, int hidden, float *x, __half *x16) {
    const Info in = *p.info;
    const int32_t *ids = ids_of(rows, in), *types = types_of(rows, in);
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < in.tokens; t += gridDim.x * ROW_WARPS) {
        const int b = p.tok_row[t], pos = t - p.start[b];
        const size_t at = (size_t)b * in.seq + pos;
        const float *w = word + (size_t)ids[at] * hidden;
        const float *tt = type ? type + (size_t)(in.has_types ? types[at] : 0) * hidden : nullptr;
        float v[V][4];
#pragma unroll
        for (int i = 0; i < V; i++) {
            const int d = 4 * lane + 128 * i;
            if (d < hidden) {
                float a[4], e[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                load4(w + d, a);
                if (tt) load4(tt + d, e);
#pragma unroll
                for (int j = 0; j < 4; j++) v[i][j] = a[j] + e[j];
            } else {
#pragma unroll
                for (int j = 0; j < 4; j++) v[i][j] = 0.0f;
            }
        }
        layer_norm_regs(v, hidden, ln_w, ln_b, eps, x ? x + (size_t)t * hidden : nullptr,
                        x16 ? x16 + (size_t)t * hidden : nullptr);
    }
}

/* The pre-norm's step: x[t] += y[t], kept in x, and LayerNorm of the sum
 * into xn (F32) or xn16 (F16), whichever is not NULL. PROD16: y is the F16
 * product with its (zero) bias in it, y16. */
template <int V, bool PROD16>
__global__ void __launch_bounds__(ROW_BLOCK)
    add_norm_into_kernel(float *x, const float *y, const __half *y16, const float *ln_w, const float *ln_b,
                         float eps, const Info *info, int hidden, float *xn, __half *xn16) {
    const int tokens = info->tokens;
    const int lane = threadIdx.x & 31;
    for (int t = blockIdx.x * ROW_WARPS + (threadIdx.x >> 5); t < tokens; t += gridDim.x * ROW_WARPS) {
        float *row = x + (size_t)t * hidden;
        float v[V][4];
#pragma unroll
        for (int i = 0; i < V; i++) {
            const int d = 4 * lane + 128 * i;
            if (d >= hidden) {
#pragma unroll
                for (int j = 0; j < 4; j++) v[i][j] = 0.0f;
                continue;
            }
            float pr[4];
            if constexpr (PROD16)
                load4_rw(y16 + (size_t)t * hidden + d, pr);
            else
                load4(y + (size_t)t * hidden + d, pr);
            float4 r = *reinterpret_cast<const float4 *>(row + d);
            r = make_float4(r.x + pr[0], r.y + pr[1], r.z + pr[2], r.w + pr[3]);
            *reinterpret_cast<float4 *>(row + d) = r;
            v[i][0] = r.x;
            v[i][1] = r.y;
            v[i][2] = r.z;
            v[i][3] = r.w;
        }
        layer_norm_regs(v, hidden, ln_w, ln_b, eps, xn ? xn + (size_t)t * hidden : nullptr,
                        xn16 ? xn16 + (size_t)t * hidden : nullptr);
    }
}
