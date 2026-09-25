/* SPDX-License-Identifier: Apache-2.0
 *
 * The CUDA backend's kernels, as the host side launches them. Each
 * launcher queues its kernel on the stream and returns what
 * cudaGetLastError says about the launch; none of them allocates.
 *
 * The written rows are [batch, seq], row b's position p at b * seq + p.
 * The encoder computes them packed, as the CPU encoder does: row b's
 * positions up to its last live token, len[b] of them, sit one after
 * another from packed token start[b], and the rows follow each other, so
 * the padding past a row's last live token is never computed. Positions
 * are the row's own column indices. Hidden states are [tokens, hidden]
 * F32, row-major, over the packed tokens.
 *
 * Every linear layer's product comes from cuBLAS without its bias; the
 * kernel that reads the product next adds the bias first, so each value
 * is the dot product plus the bias, as on the CPU. Where a kernel takes a
 * uint16_t pointer that may be NULL, it also writes its output there in
 * F16, rounded to nearest, for the next GEMM of an F16 session.
 */

#ifndef TURBO_CUDA_KERNELS_H
#define TURBO_CUDA_KERNELS_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace turbo_cuda {

/* len[b] = 1 + the index of row b's last mask entry of 1, and start[b]
 * the sum of the lengths before it: the packing, from the mask on the
 * device. Every row has a live token, as the core checked. */
cudaError_t pack_rows(cudaStream_t s, const int32_t *mask, int batch, int seq, int32_t *start, int32_t *len);

/* x[start[b] + p] = LayerNorm(word[ids[b, p]] + position[p] + type[types[b, p]])
 * for p under len[b]; max_len is the largest len. types may be NULL for
 * all zero. */
cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *ids, const int32_t *types, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             const int32_t *start, const int32_t *len, int batch, int max_len, int seq, int hidden,
                             float *x, uint16_t *x16);

/* x[t] = LayerNorm(x[t] + (y[t] + bias)), token by token. */
cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *y, const float *bias, const float *ln_w,
                           const float *ln_b, float eps, int tokens, int hidden, uint16_t *x16);

/* y = GELU(y + bias), with the error function, over [tokens, width]: in
 * place, or into y16 alone when it is not NULL. */
cudaError_t bias_gelu(cudaStream_t s, float *y, const float *bias, int tokens, int width, uint16_t *y16);

/* Scaled dot-product attention within each row, head by head, over the
 * row's keys up to its length whose mask is 1: ctx[t, head] =
 * softmax(q_t . k_j / sqrt(d)) v_j, with q, k and v their projections
 * plus their biases, each ld floats apart from one token to the next.
 * The context goes to ctx, or to ctx16 alone when it is not NULL, hidden
 * values per token. shared_most is what attention_max_shared gave. */
cudaError_t attention(cudaStream_t s, const float *q, const float *k, const float *v, int ld, const float *bq,
                      const float *bk, const float *bv, const int32_t *mask, const int32_t *start, const int32_t *len,
                      int batch, int max_len, int seq, int hidden, int heads, size_t shared_most, float *ctx,
                      uint16_t *ctx16);

/* The widest head attention computes. */
constexpr int ATTENTION_MAX_HEAD_DIM = 128;

/* The least dynamic shared memory attention needs for rows of seq
 * tokens; the most the current device lets it have, past its own static
 * use; and the setting that lets a launch have that most, made outside
 * any run. */
size_t attention_shared_bytes(int seq, int hidden, int heads);
cudaError_t attention_max_shared(size_t *bytes);
cudaError_t attention_allow_shared();

/* out[b] = the row's pooled vector, cut to output_dim, L2-normalized when
 * l2 is not 0. pooling is TURBO_POOLING_MEAN, CLS or LAST. */
cudaError_t pool(cudaStream_t s, const float *x, const int32_t *mask, const int32_t *start, const int32_t *len,
                 int batch, int seq, int hidden, int output_dim, uint32_t pooling, int l2, float *out);

/* F16 or BF16 values widened to F32, exactly. */
cudaError_t widen_f16(cudaStream_t s, const uint16_t *src, size_t n, float *dst);
cudaError_t widen_bf16(cudaStream_t s, const uint16_t *src, size_t n, float *dst);

/* F32 values rounded to the nearest F16. *overflow is set to 1 when a
 * finite value is past F16's range, and left as it was otherwise. */
cudaError_t narrow_f16(cudaStream_t s, const float *src, size_t n, uint16_t *dst, int32_t *overflow);

/* Whether this build carries code the current device runs: cudaSuccess,
 * or the error that says why not. */
cudaError_t kernels_run_here();

} // namespace turbo_cuda

#endif /* TURBO_CUDA_KERNELS_H */
