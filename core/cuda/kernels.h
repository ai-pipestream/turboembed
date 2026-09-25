/* SPDX-License-Identifier: Apache-2.0
 *
 * The CUDA backend's kernels, as the host side launches them. Each
 * launcher queues its kernel on the stream and returns what
 * cudaGetLastError says about the launch; none of them allocates, and
 * none of them waits.
 *
 * The written rows are [batch, seq] int32, row b's position p at
 * b * pitch + p, pitch being the session's max_seq. The encoder computes
 * them packed, as the CPU encoder does: row b's positions up to its last
 * live token, len[b] of them, sit one after another from packed token
 * start[b], and the rows follow each other, so the padding past a row's
 * last live token is never computed. Positions are the row's own column
 * indices. Hidden states are [tokens, hidden] F32, row-major, over the
 * packed tokens.
 *
 * Nothing about a run's shape is a launch argument. pack_rows writes the
 * packed token count and the run's options into an Info in device memory,
 * every later kernel reads them there, and each is launched for the
 * session's largest batch: a block loops over its share of the work that
 * exists and leaves when there is none. So one CUDA graph captured when
 * the session is made serves every run, whatever its rows.
 */

#ifndef TURBO_CUDA_KERNELS_H
#define TURBO_CUDA_KERNELS_H

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace turbo_cuda {

/* What a run's write sets: the rows' shape and the embed options. */
struct RunArgs {
    int32_t batch, seq, pooling, l2, output_dim, has_types;
};

/* What pack_rows finds, in device memory, for every later kernel. */
struct Info {
    int32_t tokens; /* packed tokens: the GEMMs' M */
    int32_t batch, seq, pooling, l2, output_dim, has_types;
    int32_t items; /* attention's work items, (row, head, query tile) */
};

/* The packing, in device memory. start, len and holes (whether a masked
 * token sits before the last live one) are by row; order is the rows
 * longest first (by query tiles), which only attention's schedule reads,
 * and item_start[i] the first work item of order[i], batch + 1 entries;
 * tok_row and key_bias are by packed token: the row a token is in, and 0
 * for a live token or -1e30 for a masked one. */
struct Packing {
    Info *info;
    int32_t *start, *len, *holes, *order, *item_start, *tok_row;
    float *key_bias;
};

/* Queries per attention block. */
constexpr int ATTENTION_QUERIES = 64;

/* The widest head attention computes. */
constexpr int ATTENTION_MAX_HEAD_DIM = 64;

/* The widest hidden state the row kernels hold in registers. */
constexpr int MAX_HIDDEN = 1024;

/* A session's fixed shape, from which make_plan sizes every launch. */
struct Shape {
    int batch_cap = 0, seq_cap = 0; /* max_batch, max_seq */
    int tcap = 0;                   /* batch_cap * seq_cap */
    int hidden = 0, heads = 0, inter = 0;
    bool half = false;         /* FASTEST: F16 GEMM operands and attention */
    bool tensor_cores = false; /* sm_80 or newer: mma.sync */
    int sms = 0;
    size_t smem_optin = 0; /* cudaDevAttrMaxSharedMemoryPerBlockOptin */
};

/* Grids and shared memory for every launch of a session. */
struct Plan {
    size_t pack_smem = 0;
    int rows_grid = 0; /* the warp-per-token kernels */
    int pool_grid = 0;
    int epi_grid = 0; /* the cuBLAS epilogues */
    int qkv_grid = 0, out_grid = 0, ffn1_grid = 0, ffn2_grid = 0;
    int ffn2_splits = 1, ffn2_ksplit = 0;
    int attn_grid = 0, attn_chunk = 0;
    size_t attn_smem = 0;
};

cudaError_t make_plan(const Shape &shape, Plan *plan);

// ---- The packing -------------------------------------------------------------

struct PackArgs {
    const int32_t *mask;
    int32_t pitch; /* the session's max_seq */
    int32_t heads;
    Packing p;
    RunArgs run;
};

/* The run's lengths, starts, token count, schedule and options, from the
 * mask on the device. Every row has a live token, as the core checked. */
cudaError_t pack_rows(cudaStream_t s, const PackArgs &a, const Plan &plan);

/* pack_rows's launch as a graph kernel node's parameters, its one
 * argument read from *a (args is where the pointer to it is kept; both
 * must live until the parameters are set), and the kernel's address, to
 * find its node in a captured graph. */
void pack_rows_node(const PackArgs *a, void **args, const Plan &plan, cudaKernelNodeParams *out);
const void *pack_rows_function();

// ---- Row kernels --------------------------------------------------------------

/* x[t] = LayerNorm(word[ids] + position[p] + type[types]) for every packed
 * token t, its row and position from the packing; types are read only
 * when the run has them. */
cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *ids, const int32_t *types, int pitch, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             const Packing &p, int hidden, float *x, uint16_t *x16, const Plan &plan);

/* x[t] = LayerNorm(x[t] + ((part[0][t] + ... + part[splits - 1][t]) + bias)),
 * part being splits partial products of tcap rows each, summed in order;
 * the result into x16 as F16 too when it is not NULL. */
cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *part, int splits, int tcap, const float *bias,
                           const float *ln_w, const float *ln_b, float eps, const Info *info, int hidden,
                           uint16_t *x16, const Plan &plan);

/* out[b] = the row's pooled vector, cut to output_dim, L2-normalized when
 * the run asks: the pooling, output_dim and normalization are the Info's. */
cudaError_t pool(cudaStream_t s, const float *x, const int32_t *mask, int pitch, const Packing &p, int hidden,
                 float *out, const Plan &plan);

// ---- GEMMs --------------------------------------------------------------------
//
// out = a w^T over the packed tokens, a [tokens, k] and w [n, k] row-major
// (a linear layer's weight), F32 operands with F32 FMAs, or F16 operands
// with F32 accumulation (mma.sync on the tensor cores, FMAs before
// sm_80), and what follows the product done before it is stored:
//
//   QKV: + bias, written head-major, [3][heads][tcap][head_dim], so
//        attention reads each (row, head)'s keys contiguously;
//   GELU: + bias, then GELU with the error function, [tokens, n];
//   PARTIAL: the bare product of split s's share of k, [splits][tcap][n],
//        which add_layer_norm sums in order: split-K without atomics.
//
// QKV and GELU store F16 when the operands are F16, F32 otherwise.
// Tiles past the packed token count do nothing, so the launch is sized
// for the session's largest batch. n and k are multiples of 8.

enum Epilogue : int { EPI_QKV = 0, EPI_GELU = 1, EPI_PARTIAL = 2 };

struct GemmArgs {
    const void *a, *w;
    const float *bias;
    void *out;
    const Info *info; /* info->tokens is M */
    int n, k;
    int splits, ksplit; /* k per split, a multiple of 32 */
    int heads, head_dim, hidden, tcap;
};

/* The grid for a GEMM: enough blocks for the largest M's tiles, capped at
 * what the device holds at once. */
cudaError_t gemm_grid(Epilogue e, bool half, bool tensor_cores, int n, int tcap, int splits, int sms, int *grid);
cudaError_t gemm(cudaStream_t s, Epilogue e, bool half, bool tensor_cores, const GemmArgs &g, int grid);

/* The same epilogues over a product cuBLAS made, raw [tokens, n] F32, for
 * sessions told to compute a GEMM with cuBLAS (TURBO_CUDA_CUBLAS). */
cudaError_t qkv_epilogue(cudaStream_t s, const float *raw, const GemmArgs &g, bool half, const Plan &plan);
cudaError_t gelu_epilogue(cudaStream_t s, const float *raw, const GemmArgs &g, bool half, const Plan &plan);

// ---- Attention ----------------------------------------------------------------

/* Scaled dot-product attention within each row, head by head, over the
 * row's live keys: ctx[t, head] = softmax(q_t . k_j / sqrt(d)) v_j. q, k and
 * v are the QKV GEMM's head-major output with their biases, F32 or F16;
 * the context goes to ctx in the same dtype, [tokens, hidden]. One block
 * per 64 queries of one (row, head), rows longest first; keys go through
 * shared memory plan.attn_chunk at a time, the softmax carried from chunk
 * to chunk. */
struct AttnArgs {
    const void *qkv;
    void *ctx;
    Packing p;
    int heads, head_dim, hidden, tcap, chunk;
    float scale;
};

cudaError_t attention(cudaStream_t s, const AttnArgs &a, const Shape &shape, const Plan &plan);

// ---- Weights ------------------------------------------------------------------

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
