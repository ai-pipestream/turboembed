/* SPDX-License-Identifier: Apache-2.0
 *
 * A DeBERTa-v2 model's kernels on the CUDA backend (TURBO_FAMILY_DEBERTA_V2):
 * the parts that differ from the BERT encoder, held to the CPU encoder
 * (core/src/deberta), which is the reference. The linear layers, the
 * LayerNorms after them and the pooling are the BERT encoder's, in F32.
 *
 * The relative table is LayerNormed once a model, then put through each
 * layer's fused Q|K|V projection (shared keys): rows of [3 * hidden], the
 * query part first, then the key part. A session holds the bucket of every
 * distance its rows can span, buckets[i - j + span] for span max_seq - 1.
 * A visible query's score of a visible key is
 * (q_i . k_j + q_i . pos_key[b] + k_j . pos_query[b]) / sqrt(3 head_dim),
 * b the bucket of i - j, over visible keys alone; a masked query attends
 * evenly to every position of its row, as upstream's mask leaves it.
 */
#ifndef TURBO_CUDA_DEBERTA_H
#define TURBO_CUDA_DEBERTA_H

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#include "kernels.h"

namespace turbo_cuda {

/* x[t] = LayerNorm(word[ids]) times the token's mask, for every packed
 * token: a masked position is zero. No position or type rows. */
cudaError_t deberta_embed(cudaStream_t s, const int32_t *rows, const float *word, const float *ln_w,
                          const float *ln_b, float eps, const Packing &p, int hidden, float *x, int grid);

/* Each of rows rows of x, hidden wide, LayerNormed in place. */
cudaError_t deberta_layer_norm_rows(cudaStream_t s, float *x, const float *w, const float *b, float eps, int rows,
                                    int hidden);

/* out[r, o] = bias[o] + sum_i x[r, i] w[o, i], x [rows, n_in], w [n_out,
 * n_in] row-major: the relative rows through a layer's projection. */
cudaError_t deberta_relative(cudaStream_t s, const float *x, const float *w, const float *bias, int rows, int n_out,
                             int n_in, float *out);

/* What attention reads: the QKV GEMM's head-major output, F32; the context
 * it writes, [tokens, hidden]; the packing; the layer's relative rows; the
 * session's buckets. */
struct DebertaArgs {
    const float *qkv;
    float *ctx;
    Packing p;
    int heads, head_dim, hidden, tcap, batch_cap, seq_cap;
    const float *pos;
    const int32_t *buckets;
    int span;
    float scale;
};

/* A session's attention launch: the tiled kernel for head widths 32, 64 and
 * 128 unless general is asked for, it does not fit the device's shared
 * memory, or a query tile and key block of rows of seq_cap tokens reach
 * more relative rows than it stages (deberta_tiled_rows); else the general
 * kernel, a block per query and head. buckets is the session's table, of
 * 2 * seq_cap - 1 entries. */
struct DebertaPlan {
    bool tiled = false;
    int threads = 0, grid = 0;
    size_t smem = 0;
};
cudaError_t deberta_plan(int head_dim, int batch_cap, int seq_cap, int heads, int sms, size_t smem_optin,
                         bool general, const int32_t *buckets, DebertaPlan *out);

/* The most relative rows the tiled kernel at head_dim reaches at any query
 * tile and key block of rows of seq_cap tokens, from the session's buckets
 * (the bucket of a pair's largest distance less that of its smallest, plus
 * one: past the log buckets' midpoint a distance may skip a row), and the
 * rows it stages; false for a width with no tiled kernel. */
bool deberta_tiled_rows(int head_dim, int seq_cap, const int32_t *buckets, int *reach, int *room);
cudaError_t deberta_attention(cudaStream_t s, const DebertaArgs &a, const DebertaPlan &plan);

/* HF make_log_bucket_position plus the table's offset, as
 * core/src/deberta/math.rs computes it: the row of distance d. */
uint32_t deberta_bucket(long long d, uint32_t buckets, uint32_t maximum);

} // namespace turbo_cuda

#endif /* TURBO_CUDA_DEBERTA_H */
