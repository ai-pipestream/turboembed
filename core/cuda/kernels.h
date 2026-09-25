/* SPDX-License-Identifier: Apache-2.0
 *
 * The CUDA backend's kernels, as the host side launches them. Each
 * launcher queues its kernel on the stream and returns what
 * cudaGetLastError says about the launch; none of them allocates, and
 * none of them waits.
 *
 * The written rows are one int32 array of [k][batch][seq], k 2 or 3:
 * the ids, the mask and, when the run has them, the token types, each
 * [batch, seq] with row b's position p at b * seq + p. The encoder computes
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
 * for a live token or -1e30 for a masked one. The attention of 128
 * queries tests key_bias for nonzero rather than adding it, so it must
 * stay 0 or -1e30, not become a general additive bias. */
struct Packing {
    Info *info;
    int32_t *start, *len, *holes, *order, *item_start, *tok_row;
    float *key_bias;
};

/* The widest hidden state TILE_SWIZZLED_ROWS normalizes in the attention
 * output and second feed-forward GEMMs' epilogue: its tile's width. */
constexpr int ROW_LN_WIDTH = 384;

/* The widest head attention computes. */
constexpr int ATTENTION_MAX_HEAD_DIM = 64;

/* The widest hidden state the row kernels hold in registers. */
constexpr int MAX_HIDDEN = 2048;

/* The GEMMs' tile, rows by columns: TILE_DEFAULT is 128 x 64 for the
 * FMA GEMM and TF32, and for F16 on the tensor cores TILE_EIGHT_WARPS;
 * TURBO_CUDA_TILE names one tile for all of them. The FMA GEMM gives a
 * thread 8 x 8 outputs but at TILE_128x128_16x8, 16 x 8 over 128
 * threads, and takes 128 x 64 for the tensor cores' own tiles. */
enum Tile : int {
    TILE_DEFAULT = 0,
    TILE_64x64 = 1,
    TILE_128x64 = 2,
    TILE_128x128 = 3,
    TILE_128x128_16x8 = 4,
    TILE_128x128_4W = 5, /* the tensor cores' 128 x 128 over four warps of 64 x 64 */
    TILE_256x128 = 6,    /* the tensor cores' 256 x 128 over eight warps of 64 x 64 */
    TILE_EIGHT_WARPS = 7, /* F16 on the tensor cores: 128 x 128 for QKV and GELU, 128 x 64 for the others */
    /* TILE_EIGHT_WARPS with F16 accumulators over each 64 terms of k,
     * added into F32 ones (TURBO_CUDA_F16_ACCUMULATE=1); F16 operands
     * only, other GEMMs take TILE_DEFAULT. */
    TILE_EIGHT_WARPS_F16_ACCUMULATE = 8,
    /* F16 operands on the swizzled kernel: stage rows of 64 bytes, the
     * loads running ahead across tiles, the epilogue from registers.
     * 128 x 128 over four warps of 64 x 64, two blocks to an SM. */
    TILE_SWIZZLED = 9,
    /* The swizzled kernel at the eight-warp mix's shapes, three stages. */
    TILE_SWIZZLED_8W = 10,
    /* TILE_SWIZZLED, but 256 x 128 over eight warps for GELU. */
    TILE_SWIZZLED_256x128 = 11,
    /* TILE_SWIZZLED_8W with F16 accumulators over each 64 terms of k, as
     * TILE_EIGHT_WARPS_F16_ACCUMULATE. (Warps of 64 x 64 have no
     * registers for both kinds of accumulator.) */
    TILE_SWIZZLED_8W_F16_ACCUMULATE = 12,
    /* TILE_SWIZZLED_8W, but the attention output and second feed-forward
     * GEMMs 64 x 384, whole rows (hidden widths up to ROW_LN_WIDTH), over
     * eight warps of 32 x 96, one block to an SM, with the residual and
     * the LayerNorm in their epilogue. */
    TILE_SWIZZLED_ROWS = 13,
    /* F16 operands summed in F16 accumulators over the whole of k, with
     * no F32 sums (TensorRT's F16 GEMMs): the swizzled kernel at 128 x 128
     * over four warps of 64 x 64, four stages, one block to an SM; and at
     * three stages, two. An experiment, for its accuracy first. */
    TILE_F16_WHOLE_K = 14,
    TILE_F16_WHOLE_K_3 = 15,
    /* TILE_F16_WHOLE_K_3, but the attention output and second
     * feed-forward GEMMs as TILE_SWIZZLED_ROWS takes them: 64 x 384,
     * whole rows, over eight warps of 32 x 96, one block to an SM, with
     * the residual and the LayerNorm in their epilogue. */
    TILE_F16_WHOLE_K_ROWS = 16,
    /* TILE_F16_WHOLE_K_3, but QKV and GELU 256 x 128 over eight warps of
     * 64 x 64, one block to an SM. */
    TILE_F16_WHOLE_K_256 = 17,
    /* TILE_F16_WHOLE_K at two stages, three blocks to an SM. */
    TILE_F16_WHOLE_K_2 = 18
};

/* The four GEMMs of a layer, in the order a layer runs them. */
enum Gemm : int { GEMM_QKV = 0, GEMM_OUT = 1, GEMM_FFN1 = 2, GEMM_FFN2 = 3, GEMM_COUNT = 4 };

/* How a GEMM's launch shares out its work: stream-K, each block an equal
 * run of k steps (a tile may be split between blocks, finished by the one
 * holding its last step), or whole tiles to a block, with no partial
 * products and no waits. */
enum SkMode : int { SK_STREAM = 0, SK_TILES = 1 };

/* Fewer k steps than this per block and a stream-K launch uses fewer
 * blocks; GemmArgs.min_steps, when set, in its place. */
constexpr int SK_MIN_STEPS = 4;

/* GemmArgs.min_steps for SK_TILES. */
constexpr int SK_WHOLE_TILES = -1;

/* One GEMM's kernel: its tile, how its launch shares the work, and, for
 * F32 operands on a device with tensor cores, whether it computes in TF32
 * on them (FASTEST's F16 operands take them whenever the device has
 * them). */
struct GemmChoice {
    Tile tile = TILE_DEFAULT;
    SkMode sk = SK_STREAM;
    int sk_steps = 0; /* SK_STREAM's fewest k steps per block; 0 for SK_MIN_STEPS */
    bool tf32 = false;
};

/* A session's fixed shape, from which make_plan sizes every launch. */
struct Shape {
    int batch_cap = 0, seq_cap = 0; /* max_batch, max_seq */
    int tcap = 0;                   /* batch_cap * seq_cap */
    int hidden = 0, heads = 0, inter = 0;
    bool half = false;         /* FASTEST: F16 GEMM operands and attention */
    /* sm_80 or newer: mma.sync for F16 GEMMs and attention, and for the
     * F32 GEMMs whose choice asks for TF32. */
    bool tensor_cores = false;
    int sms = 0;
    size_t smem_optin = 0; /* cudaDevAttrMaxSharedMemoryPerBlockOptin */
    /* Each GEMM's kernel, by Gemm. */
    GemmChoice gemm[GEMM_COUNT];
    /* The FMA attention with each query's keys split among four warps
     * (TURBO_CUDA_ATTENTION=split), for measuring against the default,
     * which computes Q K^T and P V as register tiles. */
    bool split_attention = false;
    /* FASTEST's attention on the tensor cores at 128 queries to a block,
     * keys and values through cp.async (TURBO_CUDA_ATTENTION=128), for
     * measuring against the default of 64. */
    bool wide_attention = false;
    /* That attention's softmax with exp2f and the scores scaled before
     * the largest is taken (TURBO_CUDA_ATTENTION=exact), the arithmetic
     * before ex2.approx and the scale in the exponent's multiply-add, for
     * comparing bits and times with the default. */
    bool exact_exp2 = false;
    /* LayerNorm in the attention output and second feed-forward GEMMs
     * (EPI_ADD_LN) when the hidden width allows and
     * TURBO_CUDA_LAYER_NORM=fused asks for it; otherwise the GEMM's
     * product and add_layer_norm after it. The session sets it. */
    bool fused_ln = false;
    /* The pooling kernel of a thread per column
     * (TURBO_CUDA_POOL=columns), for measuring against the default. */
    bool column_pool = false;
};

/* Whether a GEMM of the shape runs on the tensor cores. */
inline bool gemm_mma(const Shape &s, Gemm g) { return s.tensor_cores && (s.half || s.gemm[g].tf32); }

/* Grids and shared memory for every launch of a session. */
struct Plan {
    size_t pack_smem = 0;
    int rows_grid = 0; /* the warp-per-token kernels */
    int pool_grid = 0;
    int epi_grid = 0; /* the cuBLAS epilogues */
    int fetch_grid = 0;
    int gemm_grid[GEMM_COUNT] = {0, 0, 0, 0}; /* by Gemm */
    /* The GEMMs' stream-K workspace: a slot of partial products per block
     * of the largest launch, floats in all, and a flag per block. */
    size_t sk_floats = 0;
    int sk_flags = 0;
    /* A GEMM whose kernel was built for more blocks to an SM than fit. */
    bool gemm_crowded = false;
    /* The attention output and second feed-forward GEMMs normalize their
     * rows (EPI_ADD_LN; out_grid and ffn2_grid are its launch), with
     * ln_counts counters after the stream-K flags. */
    bool fused_ln = false;
    int ln_counts = 0;
    bool column_pool = false;
    int attn_grid = 0, attn_chunk = 0, attn_queries = 0;
    size_t attn_smem = 0;
};

cudaError_t make_plan(const Shape &shape, Plan *plan);

// ---- The packing -------------------------------------------------------------

struct PackArgs {
    const int32_t *rows; /* the written rows, [k][batch][seq] */
    int32_t heads, queries; /* attention's heads and queries per work item */
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

/* n int32 of the written rows from src, the page-locked staging as the
 * device addresses it, to dst in device memory: the run's first kernel,
 * so the rows reach the device inside the run's graph, with no copy
 * queued ahead of it. n is 0 when the rows were sent another way. */
struct FetchArgs {
    const int32_t *src;
    int32_t *dst;
    int32_t n;
};

cudaError_t fetch_rows(cudaStream_t s, const FetchArgs &a, const Plan &plan);
void fetch_rows_node(const FetchArgs *a, void **args, const Plan &plan, cudaKernelNodeParams *out);
const void *fetch_rows_function();

// ---- Row kernels --------------------------------------------------------------

/* x[t] = LayerNorm(word[ids] + position[p] + type[types]) for every packed
 * token t, its row and position from the packing, rows being the written
 * rows; types are read only when the run has them. */
cudaError_t embed_layer_norm(cudaStream_t s, const int32_t *rows, const float *word,
                             const float *position, const float *type, const float *ln_w, const float *ln_b, float eps,
                             const Packing &p, int hidden, float *x, uint16_t *x16, const Plan &plan);

/* x[t] = LayerNorm(x[t] + (y[t] + bias)), y a GEMM's product; the result
 * into x16 as F16 too when it is not NULL. */
cudaError_t add_layer_norm(cudaStream_t s, float *x, const float *y, const float *bias,
                           const float *ln_w, const float *ln_b, float eps, const Info *info, int hidden,
                           uint16_t *x16, const Plan &plan);

/* out[b] = the row's pooled vector, cut to output_dim, L2-normalized when
 * the run asks: the pooling, output_dim and normalization are the Info's. */
cudaError_t pool(cudaStream_t s, const float *x, const int32_t *rows, const Packing &p, int hidden, float *out,
                 const Plan &plan);

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
//   PLAIN: the bare product, F32, [tokens, n], which add_layer_norm adds;
//   ADD_LN: out is the hidden states, F32 [tokens, n], n = hidden; each
//        output becomes out + (product + bias), and once every tile of
//        a block of rows has, the block finishing the last of them
//        normalizes those rows as add_layer_norm does, with the same
//        sums in the same order, so the same bits: LayerNorm inside the
//        GEMM, for hidden widths up to LN_FUSED_MAX_HIDDEN.
//
// QKV and GELU store F16 when the operands are F16, F32 otherwise. The
// launch is the blocks the device holds at once, sharing the tiles' k
// steps evenly among them (stream-K): a tile split between blocks is
// finished by the block holding its last k step, which adds the others'
// partial products, from the workspace, nearest block first. So the sums
// depend on the token count and the launch, never on which rows run
// beside a row in a batch of the same token count. n and k are multiples
// of 8.

enum Epilogue : int { EPI_QKV = 0, EPI_GELU = 1, EPI_PLAIN = 2, EPI_ADD_LN = 3 };

/* The widest hidden state ADD_LN normalizes, 16 values to a lane. */
constexpr int LN_FUSED_MAX_HIDDEN = 512;

/* ADD_LN's counters of finished tiles, one per block of rows: enough for
 * the smallest tile's rows. */
inline int ln_counters(int tcap) { return tcap / 64 + 1; }

/* The epilogue a GEMM of a layer runs: the attention output and second
 * feed-forward GEMMs normalize their rows when the LayerNorm is fused. */
inline Epilogue gemm_epilogue(Gemm g, bool fused_ln) {
    if (g == GEMM_QKV) return EPI_QKV;
    if (g == GEMM_FFN1) return EPI_GELU;
    return fused_ln ? EPI_ADD_LN : EPI_PLAIN;
}

struct GemmArgs {
    const void *a, *w;
    const float *bias;
    void *out;
    const Info *info; /* info->tokens is M */
    int n, k;
    int heads, head_dim, hidden, tcap;
    /* The stream-K workspace: a slot of BM x BN floats per block, and a
     * flag per block, 0 between launches. */
    float *ws;
    int *flags;
    /* Set to 1 by a wait that gave up; the host reads it after the run.
     * Host memory the device writes, so reading it copies nothing. */
    int *fault;
    /* The fewest k steps a block takes before the kernel runs on fewer
     * blocks; 0 for the kernels' own, SK_MIN_STEPS; SK_WHOLE_TILES for
     * whole tiles to a block. */
    int min_steps;
    /* ADD_LN only: the LayerNorm's weight and bias and epsilon, the F16
     * copy of the hidden states (NULL at F32), and a counter per block of
     * rows, 0 between launches, which the block finishing the rows'
     * last tile clears. */
    const float *ln_w, *ln_b;
    float eps;
    uint16_t *x16;
    int *rows_done;
};

/* A GEMM's launch: the blocks the device holds at once, and the
 * workspace floats it needs; and the shared memory setting its kernel
 * needs, made outside any run. The launch never depends on the token
 * count, which only the device knows (the graph is made once for every
 * run): the kernel shares out the tiles of the run's M, and the launch
 * is sized for the device, not for the largest M's tiles. *crowded,
 * when given, is set when fewer blocks fit on an SM than the kernel was
 * built for. */
cudaError_t gemm_grid(Epilogue e, bool half, bool tensor_cores, Tile tile, int sms, int *grid, size_t *ws_floats,
                      bool *crowded = nullptr);
cudaError_t gemm_prepare(Epilogue e, bool half, bool tensor_cores, Tile tile);
cudaError_t gemm(cudaStream_t s, Epilogue e, bool half, bool tensor_cores, Tile tile, const GemmArgs &g, int grid);

/* The same epilogues over a product cuBLAS made, raw [tokens, n] F32, for
 * sessions told to compute a GEMM with cuBLAS (TURBO_CUDA_CUBLAS). */
cudaError_t qkv_epilogue(cudaStream_t s, const float *raw, const GemmArgs &g, bool half, const Plan &plan);
cudaError_t gelu_epilogue(cudaStream_t s, const float *raw, const GemmArgs &g, bool half, const Plan &plan);

// ---- Attention ----------------------------------------------------------------

/* Scaled dot-product attention within each row, head by head, over the
 * row's live keys: ctx[t, head] = softmax(q_t . k_j / sqrt(d)) v_j. q, k and
 * v are the QKV GEMM's head-major output with their biases, F32 or F16;
 * the context goes to ctx in the same dtype, [tokens, hidden]. One block
 * per plan.attn_queries queries of one (row, head), rows longest first;
 * keys go through shared memory plan.attn_chunk at a time, the softmax
 * carried from chunk to chunk. */
struct AttnArgs {
    const void *qkv;
    void *ctx;
    Packing p;
    int heads, head_dim, hidden, tcap, chunk, queries;
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
