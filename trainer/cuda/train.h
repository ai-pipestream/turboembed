/* SPDX-License-Identifier: Apache-2.0
 *
 * The encoder trainer's CUDA kernels (train.cu), as the Rust side
 * (trainer/src/encoder/cuda) launches them: the C API of one static
 * library, built by trainer/build.rs for the device (nvcc, feature
 * `cuda`) or as plain C++ on the host (shim.h, feature `cuda-shim`).
 *
 * Every function returns 0 or a CUDA error number (tt_error_string
 * names it). Every pointer but a host copy's is device memory; every
 * matrix is row-major with its natural leading dimension. The kernels
 * mirror the CPU ops of trainer/src/encoder one for one, so the host
 * side can run a layer as layer.rs does, and the numbers they give are
 * held to the CPU encoder's. Each output element is owned by one thread
 * and summed in a fixed order: no atomics, so a run gives the same
 * bytes on the same device every time. The launches are queued on the
 * default stream and return at once; tt_sync waits for them.
 */

#ifndef TURBO_TRAIN_CUDA_H
#define TURBO_TRAIN_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int tt_device_count(int *n);
int tt_set_device(int device);
const char *tt_error_string(int error);

int tt_malloc(size_t bytes, void **p);
int tt_free(void *p);
int tt_upload(void *dst, const void *src, size_t bytes);
int tt_download(void *dst, const void *src, size_t bytes);
int tt_copy(void *dst, const void *src, size_t bytes);
int tt_zero(void *dst, size_t bytes);
int tt_sync(void);

/* c [m, n] = alpha op(a) [m, k] op(b) [k, n] + beta c; a is stored [m, k],
 * or [k, m] with ta; b [k, n], or [n, k] with tb. beta 0 leaves c unread. */
int tt_gemm(const float *a, const float *b, float *c, int m, int n, int k, int ta, int tb, float alpha, float beta);

/* out [n, h] = the rows of table at ids; table [.., h] += rows at ids,
 * in token order, so a repeated id sums in order. */
int tt_gather(const float *table, const int32_t *ids, int n, int h, float *out);
int tt_scatter_add(const float *rows, const int32_t *ids, int n, int h, float *table);

/* LayerNorm without a bias over the rows of x [rows, width], the mean
 * and the biased variance summed in double, y = (x - mean) rstd w; mean
 * and rstd [rows] kept for backward. Backward adds to dx, and to dw when
 * it is not null. */
int tt_layernorm(const float *x, const float *w, int rows, int width, double eps, float *y, double *mean,
                 double *rstd);
int tt_layernorm_back(const float *x, const float *w, const double *mean, const double *rstd, const float *dy,
                      int rows, int width, float *dx, float *dw);

/* Rotates columns 0..cols of each row of x [rows, ld] in place, in heads
 * of 2 pairs columns, row t at position t, column i of a head with
 * column i + pairs by cs and sn [positions, pairs]; back applies the
 * transpose. */
int tt_rope(float *x, int rows, int ld, int cols, int pairs, const float *cs, const float *sn, int back);

/* Softmax attention over qkv [n, 3 hidden] (Q, K, V, each head-major, Q
 * and K rotated), query i reading keys |i - j| <= window (window -1: all),
 * scores scaled by 1 / sqrt(d): p [heads, n, n] the probabilities (0
 * outside the window; round rounds them to bfloat16), ctx [n, hidden]
 * their weighted values. Backward from dctx: ds [heads, n, n] the score
 * gradients p (dp - sum p dp) / sqrt(d), dp_j = dctx_i . v_j; then dq
 * adds the queries' gradients into dqkv [n, 3 hidden], dkv the keys'
 * and the values'. */
int tt_attention_probs(const float *qkv, int n, int hidden, int heads, int window, int round, float *p);
int tt_attention_ctx(const float *qkv, const float *p, int n, int hidden, int heads, int window, float *ctx);
int tt_attention_ds(const float *qkv, const float *p, const float *dctx, int n, int hidden, int heads, int window,
                    float *ds);
int tt_attention_dq(const float *qkv, const float *ds, int n, int hidden, int heads, int window, float *dqkv);
int tt_attention_dkv(const float *qkv, const float *p, const float *ds, const float *dctx, int n, int hidden,
                     int heads, int window, float *dqkv);

/* The gated GELU: act [rows, inter] = gelu(pre[.., ..inter]) pre[.., inter..],
 * from pre [rows, 2 inter], GELU by the error function in double; backward
 * adds to dpre. */
int tt_geglu(const float *pre, int rows, int inter, float *act);
int tt_geglu_back(const float *pre, const float *dact, int rows, int inter, float *dpre);

/* Inverted dropout in place: element i is dropped when the i-th draw of
 * SplitMix64 from key, as the trainer's Rng draws a unit, is below p, the
 * rest scaled by 1 / (1 - p): the mask the CPU draws from the same key. */
int tt_dropout(float *x, size_t count, double p, uint64_t key);

int tt_axpy(float *y, const float *x, size_t count, float alpha);
/* Rounds every element to the nearest bfloat16, ties to even. */
int tt_round_bf16(float *x, size_t count);

#ifdef __cplusplus
}
#endif

#endif
