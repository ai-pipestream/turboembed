/* SPDX-License-Identifier: Apache-2.0
 *
 * The CUDA backend's kernels, as the host side launches them. Each
 * launcher queues its kernel on the stream and returns what
 * cudaGetLastError says about the launch; none of them allocates.
 *
 * Rows are the written batch's full grid, [batch, seq], token t at row
 * t / seq and position t % seq. Hidden states are [tokens, hidden] F32,
 * row-major. Every linear layer's product comes from cuBLAS without its
 * bias; the kernel that reads the product next adds the bias first, so
 * each value is the dot product plus the bias, as on the CPU.
 */

#ifndef TURBO_CUDA_KERNELS_H
#define TURBO_CUDA_KERNELS_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace turbo_cuda {

/* x[t] = LayerNorm(word[ids[t]] + position[t % seq] + type[types[t]]).
 * types may be NULL for all zero. */
cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *ids, const int32_t *types, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             int tokens, int seq, int hidden, float *x);

/* x[t] = LayerNorm(x[t] + (y[t] + bias)), row by row. */
cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *y, const float *bias, const float *ln_w,
                           const float *ln_b, float eps, int tokens, int hidden);

/* y = GELU(y + bias), with the error function, over [tokens, width]. */
cudaError_t bias_gelu(cudaStream_t s, float *y, const float *bias, int tokens, int width);

/* Scaled dot-product attention within each row, head by head, over the
 * keys whose mask is 1: ctx[t, head] = softmax(q_t . k_j / sqrt(d)) v_j,
 * with q, k and v their projections plus their biases. */
cudaError_t attention(cudaStream_t s, const float *q, const float *k, const float *v, const float *bq, const float *bk,
                      const float *bv, const int32_t *mask, int batch, int seq, int hidden, int heads, float *ctx);

/* The dynamic shared memory attention needs for a row of seq tokens; the
 * most the current device lets it have, past its own static use; and the
 * setting that lets a launch have that most, made outside any run when a
 * session needs more than the default 48 KiB. */
size_t attention_shared_bytes(int seq, int hidden, int heads);
cudaError_t attention_max_shared(size_t *bytes);
cudaError_t attention_allow_shared();

/* out[b] = the row's pooled vector, cut to output_dim, L2-normalized when
 * l2 is not 0. pooling is TURBO_POOLING_MEAN, CLS or LAST. */
cudaError_t pool(cudaStream_t s, const float *x, const int32_t *mask, int batch, int seq, int hidden, int output_dim,
                 uint32_t pooling, int l2, float *out);

/* F16 or BF16 values widened to F32, exactly. */
cudaError_t widen_f16(cudaStream_t s, const uint16_t *src, size_t n, float *dst);
cudaError_t widen_bf16(cudaStream_t s, const uint16_t *src, size_t n, float *dst);

/* Whether this build carries code the current device runs: cudaSuccess,
 * or the error that says why not. */
cudaError_t kernels_run_here();

} // namespace turbo_cuda

#endif /* TURBO_CUDA_KERNELS_H */
