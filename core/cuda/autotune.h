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

#include <cstddef>
#include <cstdint>
#include <string>

#include "kernels.h"

namespace turbo_cuda {

/* The attention kernels: the FMA kernel of register tiles or of keys
 * split among warps (F32, and F16 without the tensor cores' attention),
 * and the tensor cores' kernel of 64 or 128 queries to a block (F16 on
 * heads of 32 or 64), the latter also with its earlier softmax of
 * exp2f, or at heads of 32 with 32 queries to each of four warps. */
enum AttnVariant : int {
    ATT_FMA_TILED = 0,
    ATT_FMA_SPLIT = 1,
    ATT_MMA_64 = 2,
    ATT_MMA_128 = 3,
    ATT_MMA_128_EXACT = 4,
    ATT_MMA_128_FA32 = 5, /* heads of 32 only: 32 queries to a warp, four warps */
};

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

/* The GEMMs by name, as a choices string's items spell them, by Gemm. */
constexpr const char *GEMM_NAMES[GEMM_COUNT] = {"qkv", "out", "ffn1", "ffn2"};

/* The tiles by name, as a choices string spells them, and whether
 * TURBO_CUDA_TILE takes the name: every one but the F16 accumulators'
 * eight-warp tiles, which TURBO_CUDA_F16_ACCUMULATE picks. */
struct TileName {
    const char *name;
    Tile tile;
    bool switch_names;
};
constexpr TileName TILE_NAMES[] = {
    {"64x64", TILE_64x64, true},
    {"128x64", TILE_128x64, true},
    {"128x128", TILE_128x128, true},
    {"128x128-16x8", TILE_128x128_16x8, true},
    {"128x128-4w", TILE_128x128_4W, true},
    {"256x128", TILE_256x128, true},
    {"8w", TILE_EIGHT_WARPS, true},
    {"sw", TILE_SWIZZLED, true},
    {"sw8w", TILE_SWIZZLED_8W, true},
    {"sw256", TILE_SWIZZLED_256x128, true},
    {"swrow", TILE_SWIZZLED_ROWS, true},
    {"acc16-8w", TILE_EIGHT_WARPS_F16_ACCUMULATE, false},
    {"acc16-sw8w", TILE_SWIZZLED_8W_F16_ACCUMULATE, false},
    {"f16k", TILE_F16_WHOLE_K, true},
    {"f16k3", TILE_F16_WHOLE_K_3, true},
    {"f16krow", TILE_F16_WHOLE_K_ROWS, true},
    {"f16k256", TILE_F16_WHOLE_K_256, true},
};

/* Whether a tile sums F16 products in F16 accumulators: over each 64
 * terms of k, or over the whole of a block's k. */
inline bool f16_accumulates(Tile t) {
    return t == TILE_EIGHT_WARPS_F16_ACCUMULATE || t == TILE_SWIZZLED_8W_F16_ACCUMULATE || t == TILE_F16_WHOLE_K ||
           t == TILE_F16_WHOLE_K_3 || t == TILE_F16_WHOLE_K_ROWS || t == TILE_F16_WHOLE_K_256;
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

/* The knobs' names, as a choices string's forced= item lists them. */
constexpr const char *KNOB_NAMES[] = {"tile", "sk", "tf32", "attn", "ln", "pool"};

/* Whether every knob of c was forced: each GEMM's tile and stream-K,
 * each bin's attention and LayerNorm, and the pooling. */
bool all_forced(const Choices &c);

/* The kernels of c as they run in a session of the base shape: each
 * choice a name that runs no kernel of its own (a tile the operands or the
 * device do not take, the default tile, stream-K's default steps,
 * attention the session does not have) becomes the name of the kernel it
 * runs, so the string reports what ran and forcing it back runs the same.
 * The pooling is set from the plan, after. */
void canonicalize(const Shape &base, Choices *c);

/* g as it runs in a GEMM of the base shape: the tile the kernel family
 * takes in its place, and TF32 only for F32 operands on tensor cores. */
GemmChoice canonical_gemm(Gemm which, const Shape &base, GemmChoice g);

/* A GEMM's kernel as the variants name it: its tile, then "/tf32" when
 * it computes in TF32. */
std::string variant_name(const GemmChoice &g);

/* The TURBO_NUMERIC_* class a GEMM of the choice computes in. */
uint32_t gemm_numeric(const Shape &base, const GemmChoice &g);

/* The TURBO_NUMERIC_* classes the GEMMs of c compute in, in every bin
 * that exists. */
uint32_t numerics_of(const Shape &base, const Choices &c);

/* The classes of n by name, joined by " and ". */
std::string numerics_named(uint32_t n);

/* NULL when every kernel c chooses computes in a class of allowed, else
 * the name of the first that does not, as the string spells it, and its
 * class's name in *numeric. */
const char *outside(const Shape &base, const Choices &c, uint32_t allowed, char *name, size_t len,
                    const char **numeric);

/* c as one line: per bin that exists, "<bin>:qkv=<tile>/<sk>[/tf32],out=
 * ...,ffn1=...,ffn2=...,attn=<attention>,ln=<layer norm>", then
 * "pool=<pooling>" and "forced=<knob>,...", separated by ";". Writes what
 * fits in len bytes with the NUL; returns the length it needs without. */
size_t format_choices(const Choices &c, char *out, size_t len);

/* Fills only the items s names, as format_choices writes them, and sets
 * their forced bits. "all:" names every bin; a GEMM's item may leave out
 * the stream-K, which is then not forced; forced= is ignored. A bin the
 * session does not have (into->bins) is left out, its bit set in
 * *absent: one line forces sessions of any size. An unknown item or
 * value fails with why naming it. */
bool parse_choices(const char *s, Choices *into, uint32_t *absent, char *why, size_t why_len);

/* One kernel variant of a GEMM that a session of the shape can force:
 * its name as the string spells a GEMM's tile ("8w", "128x64/tf32"), its
 * tile and TF32, its TURBO_NUMERIC_* class, and whether the tuner times
 * it. */
struct Variant {
    char name[24];
    Tile tile;
    bool tf32;
    uint32_t numeric;
    bool candidate;
};

/* The GEMM variants of a session of the shape, candidates first, into
 * out (cap entries at most): how many. */
int gemm_variants(const Shape &base, Variant *out, int cap);

} // namespace turbo_cuda

#endif /* TURBO_CUDA_AUTOTUNE_H */
