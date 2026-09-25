/* SPDX-License-Identifier: Apache-2.0
 *
 * A CUDA session's kernel choices: which kernel each GEMM, attention, the
 * LayerNorms and the pooling take, per token bin. A session holds one
 * graph per bin of packed tokens, each captured from its own choices, and
 * a run launches the graph of its rows' bin. The choices come from the
 * backend's defaults, the environment and the tests, which force them.
 */

#ifndef TURBO_CUDA_AUTOTUNE_H
#define TURBO_CUDA_AUTOTUNE_H

#include <cstdint>

#include "kernels.h"

namespace turbo_cuda {

/* The attention kernels: the FMA kernel of register tiles or of keys
 * split among warps (F32, and F16 without the tensor cores' attention),
 * and the tensor cores' kernel of 64 or 128 queries to a block (F16 on
 * heads of 32 or 64). */
enum AttnVariant : int { ATT_FMA_TILED = 0, ATT_FMA_SPLIT = 1, ATT_MMA_64 = 2, ATT_MMA_128 = 3 };

/* The attention output and second feed-forward GEMMs' LayerNorm: a kernel
 * of its own after the product, or in the GEMM's epilogue. */
enum LnVariant : int { LN_SEPARATE = 0, LN_FUSED = 1 };

/* The pooling kernel: groups of tokens summed apart, or a thread per
 * column. */
enum PoolVariant : int { POOL_GROUPS = 0, POOL_COLUMNS = 1 };

/* The bins of packed tokens: bin i holds the counts above BIN_EDGE[i - 1]
 * up to BIN_EDGE[i]. A bin whose lowest count is above the session's
 * max_batch x max_seq does not exist for it. */
constexpr int BIN_COUNT = 5;
constexpr int BIN_EDGE[BIN_COUNT] = {256, 1024, 4096, 16384, INT32_MAX};
constexpr const char *BIN_NAME[BIN_COUNT] = {"le256", "le1k", "le4k", "le16k", "gt16k"};

/* The bins that exist for tcap packed tokens at most: at least one. */
inline int bins_for(int tcap) {
    int n = 1;
    while (n < BIN_COUNT && BIN_EDGE[n - 1] < tcap) n++;
    return n;
}

/* The bin of a run of tokens packed tokens, of bins bins. */
inline int bin_of(uint32_t tokens, int bins) {
    int b = 0;
    while (b + 1 < bins && tokens > (uint32_t)BIN_EDGE[b]) b++;
    return b;
}

/* The tiles by name, as TURBO_CUDA_TILE spells them; the F16
 * accumulators' two, which TURBO_CUDA_F16_ACCUMULATE picks, last. */
struct TileName {
    const char *name;
    Tile tile;
};
constexpr TileName TILE_NAMES[] = {
    {"64x64", TILE_64x64},
    {"128x64", TILE_128x64},
    {"128x128", TILE_128x128},
    {"128x128-16x8", TILE_128x128_16x8},
    {"128x128-4w", TILE_128x128_4W},
    {"256x128", TILE_256x128},
    {"8w", TILE_EIGHT_WARPS},
    {"sw", TILE_SWIZZLED},
    {"sw8w", TILE_SWIZZLED_8W},
    {"sw256", TILE_SWIZZLED_256x128},
    {"swrow", TILE_SWIZZLED_ROWS},
    {"acc16-8w", TILE_EIGHT_WARPS_F16_ACCUMULATE},
    {"acc16-sw8w", TILE_SWIZZLED_8W_F16_ACCUMULATE},
};
constexpr int TILE_NAMES_ACCUMULATE = 11; /* the first of the F16 accumulators' */

/* Whether a tile sums F16 products in F16 accumulators. */
inline bool f16_accumulates(Tile t) {
    return t == TILE_EIGHT_WARPS_F16_ACCUMULATE || t == TILE_SWIZZLED_8W_F16_ACCUMULATE;
}

/* What a choice was fixed by, rather than left to the defaults: a bit per
 * knob. */
enum Knob : uint32_t {
    KNOB_TILE = 1,
    KNOB_SK = 2,
    KNOB_TF32 = 4,
    KNOB_ATTN = 8,
    KNOB_LN = 16,
    KNOB_POOL = 32,
};

/* What one bin's graph is captured from, with the knobs forced in it. */
struct BinChoices {
    GemmChoice gemm[GEMM_COUNT];
    uint32_t gemm_forced[GEMM_COUNT] = {0, 0, 0, 0}; /* KNOB_TILE, KNOB_SK, KNOB_TF32 */
    AttnVariant attention = ATT_FMA_TILED;
    LnVariant ln = LN_SEPARATE;
    uint32_t forced = 0; /* KNOB_ATTN, KNOB_LN */
};

/* Everything a session bakes into its graphs. */
struct Choices {
    int bins = 1; /* the bins that exist for the session */
    BinChoices bin[BIN_COUNT];
    PoolVariant pool = POOL_GROUPS;
    uint32_t pool_forced = 0; /* KNOB_POOL */
};

/* The knobs forced anywhere in c. */
inline uint32_t forced_knobs(const Choices &c) {
    uint32_t k = c.pool_forced;
    for (int b = 0; b < c.bins; b++) {
        k |= c.bin[b].forced;
        for (int g = 0; g < GEMM_COUNT; g++) k |= c.bin[b].gemm_forced[g];
    }
    return k;
}

} // namespace turbo_cuda

#endif /* TURBO_CUDA_AUTOTUNE_H */
