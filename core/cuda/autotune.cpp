/* SPDX-License-Identifier: Apache-2.0
 *
 * A CUDA session's kernel choices as one line, the variants each knob
 * may take, and which numeric class each computes in (autotune.h).
 */

#include "autotune.h"

#include <turbo/turbo_backend.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <strings.h>

namespace turbo_cuda {

namespace {

const char *tile_name(Tile t) {
    for (const TileName &n : TILE_NAMES)
        if (n.tile == t) return n.name;
    return "?";
}

constexpr const char *ATTN_NAME[] = {"fma-tiled", "fma-split", "mma64", "mma128", "mma128-exact", "mma128-fa32"};
constexpr const char *LN_NAME[] = {"separate", "fused"};
constexpr const char *POOL_NAME[] = {"groups", "columns"};

bool mma_of(const Shape &base, const GemmChoice &g) { return base.tensor_cores && (base.half || g.tf32); }

/* The tile a GEMM of the kernel family runs, by the name it would take. */
Tile canonical_tile(Gemm which, const Shape &base, bool mma, Tile t) {
    const bool narrow = which == GEMM_OUT || which == GEMM_FFN2;
    if (!mma) {
        // The FMA kernel: F32 operands, or F16 without the tensor cores.
        switch (t) {
        case TILE_64x64:
        case TILE_128x128:
        case TILE_128x128_16x8: return t;
        default: return TILE_128x64;
        }
    }
    if (t == TILE_128x128_16x8) return TILE_128x128;
    if (!base.half) {
        // TF32: F32 operands and outputs on the tensor cores.
        switch (t) {
        case TILE_64x64:
        case TILE_128x128:
        case TILE_128x128_4W: return t;
        case TILE_256x128: return TILE_128x128_4W;
        default: return TILE_128x64;
        }
    }
    switch (t) {
    case TILE_DEFAULT: return TILE_EIGHT_WARPS;
    // An F32 output tile of 256 x 128 does not fit: the kernel takes
    // 128 x 128 over four warps.
    case TILE_256x128: return narrow ? TILE_128x128_4W : t;
    // 256 x 128 only for GELU; the others the swizzled 128 x 128.
    case TILE_SWIZZLED_256x128: return which == GEMM_FFN1 ? t : TILE_SWIZZLED;
    // Whole rows only for the GEMMs that normalize them, when they fit.
    case TILE_SWIZZLED_ROWS: return narrow && base.hidden <= ROW_LN_WIDTH ? t : TILE_SWIZZLED_8W;
    case TILE_F16_WHOLE_K_ROWS: return narrow && base.hidden <= ROW_LN_WIDTH ? t : TILE_F16_WHOLE_K_3;
    // Whole-k 256 x 128 only for QKV and GELU, their F16 outputs.
    case TILE_F16_WHOLE_K_256: return narrow ? TILE_F16_WHOLE_K_3 : t;
    default: return t;
    }
}

const char *numeric_name(uint32_t n) {
    switch (n) {
    case TURBO_NUMERIC_F32_FMA: return "F32 FMAs";
    case TURBO_NUMERIC_TF32: return "TF32";
    case TURBO_NUMERIC_F16_F32ACC: return "F16 with F32 sums";
    default: return "F16 with F16 sums within a chunk";
    }
}

void append(std::string &s, const char *fmt, const char *a, const char *b = "") {
    char buf[96];
    snprintf(buf, sizeof buf, fmt, a, b);
    s += buf;
}

/* A GEMM's item after its "=": tile, then the stream-K and TF32. */
std::string gemm_item(const GemmChoice &g) {
    std::string s = tile_name(g.tile);
    if (g.sk == SK_TILES) {
        s += "/tiles";
    } else {
        char buf[16];
        snprintf(buf, sizeof buf, "/sk%d", g.sk_steps > 0 ? g.sk_steps : SK_MIN_STEPS);
        s += buf;
    }
    if (g.tf32) s += "/tf32";
    return s;
}

/* The index of word (n bytes) in names, or -1. */
template <size_t N> int index_of(const char *const (&names)[N], const char *w, size_t n) {
    for (size_t i = 0; i < N; i++)
        if (strlen(names[i]) == n && strncmp(names[i], w, n) == 0) return (int)i;
    return -1;
}

bool fail(char *why, size_t len, const char *what, const char *w, size_t n) {
    snprintf(why, len, "%s \"%.*s\"", what, (int)n, w);
    return false;
}

/* One GEMM's value: "<tile>[/sk<n>|/tiles][/tf32]". */
bool parse_gemm(const char *w, size_t n, GemmChoice *g, uint32_t *forced, char *why, size_t len) {
    const char *end = w + n, *p = w;
    while (p < end && *p != '/') p++;
    bool found = false;
    for (const TileName &t : TILE_NAMES)
        if (strlen(t.name) == (size_t)(p - w) && strncmp(t.name, w, (size_t)(p - w)) == 0) {
            g->tile = t.tile;
            found = true;
        }
    if (!found) return fail(why, len, "TURBO_CUDA_CHOICES: no GEMM tile is named", w, (size_t)(p - w));
    *forced |= KNOB_TILE | KNOB_TF32;
    g->tf32 = false;
    while (p < end) {
        const char *q = ++p;
        while (p < end && *p != '/') p++;
        const size_t m = (size_t)(p - q);
        if (m == 4 && !strncmp(q, "tf32", 4)) {
            g->tf32 = true;
        } else if (m == 5 && !strncmp(q, "tiles", 5)) {
            g->sk = SK_TILES;
            g->sk_steps = 0;
            *forced |= KNOB_SK;
        } else if (m > 2 && !strncmp(q, "sk", 2)) {
            char digits[8] = {0};
            if (m - 2 >= sizeof digits) return fail(why, len, "TURBO_CUDA_CHOICES: no stream-K is named", q, m);
            memcpy(digits, q + 2, m - 2);
            char *e = nullptr;
            const long steps = strtol(digits, &e, 10);
            if (*e != 0 || steps < 1 || steps > 64)
                return fail(why, len, "TURBO_CUDA_CHOICES: stream-K takes 1 to 64 steps, not", q, m);
            g->sk = SK_STREAM;
            g->sk_steps = (int)steps;
            *forced |= KNOB_SK;
        } else {
            return fail(why, len, "TURBO_CUDA_CHOICES: a GEMM's kernel has no part named", q, m);
        }
    }
    return true;
}

} // namespace

bool all_forced(const Choices &c) {
    if (!c.pool_forced) return false;
    for (int b = 0; b < c.bins; b++) {
        const BinChoices &bc = c.bin[b];
        if ((bc.forced & (KNOB_ATTN | KNOB_LN)) != (KNOB_ATTN | KNOB_LN)) return false;
        for (uint32_t f : bc.gemm_forced)
            if ((f & (KNOB_TILE | KNOB_SK)) != (KNOB_TILE | KNOB_SK)) return false;
    }
    return true;
}

void canonicalize(const Shape &base, Choices *c) {
    const int d = base.hidden / base.heads;
    const bool mma_attention = base.half && base.tensor_cores && (d == 32 || d == 64);
    for (BinChoices &bc : c->bin) {
        for (int g = 0; g < GEMM_COUNT; g++) {
            GemmChoice &gc = bc.gemm[g];
            gc = canonical_gemm((Gemm)g, base, gc);
            if (gc.sk == SK_TILES)
                gc.sk_steps = 0;
            else if (gc.sk_steps <= 0)
                gc.sk_steps = SK_MIN_STEPS;
        }
        // attention_kernel_for: the tensor cores' kernel wherever it
        // applies, 128 queries only when asked; else the FMA kernel,
        // split only when asked.
        if (mma_attention && bc.attention == ATT_MMA_128_FA32 && d != 32)
            bc.attention = ATT_MMA_128;
        else if (mma_attention)
            bc.attention = bc.attention == ATT_MMA_128 || bc.attention == ATT_MMA_128_EXACT ||
                                   bc.attention == ATT_MMA_128_FA32
                               ? bc.attention
                               : ATT_MMA_64;
        else
            bc.attention = bc.attention == ATT_FMA_SPLIT ? ATT_FMA_SPLIT : ATT_FMA_TILED;
        if (base.hidden > LN_FUSED_MAX_HIDDEN) bc.ln = LN_SEPARATE;
    }
}

GemmChoice canonical_gemm(Gemm which, const Shape &base, GemmChoice g) {
    g.tf32 = g.tf32 && !base.half && base.tensor_cores;
    g.tile = canonical_tile(which, base, mma_of(base, g), g.tile);
    return g;
}

std::string variant_name(const GemmChoice &g) {
    std::string s = tile_name(g.tile);
    if (g.tf32) s += "/tf32";
    return s;
}

uint32_t gemm_numeric(const Shape &base, const GemmChoice &g) {
    if (!base.half) return mma_of(base, g) ? TURBO_NUMERIC_TF32 : TURBO_NUMERIC_F32_FMA;
    return mma_of(base, g) && f16_accumulates(g.tile) ? TURBO_NUMERIC_F16_CHUNKACC : TURBO_NUMERIC_F16_F32ACC;
}

uint32_t numerics_of(const Shape &base, const Choices &c) {
    uint32_t n = 0;
    for (int b = 0; b < c.bins; b++)
        for (const GemmChoice &g : c.bin[b].gemm) n |= gemm_numeric(base, g);
    return n;
}

std::string numerics_named(uint32_t n) {
    std::string s;
    for (uint32_t bit = 1; bit <= TURBO_NUMERIC_F16_CHUNKACC; bit <<= 1)
        if (n & bit) {
            if (!s.empty()) s += " and ";
            s += numeric_name(bit);
        }
    return s;
}

const char *outside(const Shape &base, const Choices &c, uint32_t allowed, char *name, size_t len,
                    const char **numeric) {
    for (int b = 0; b < c.bins; b++)
        for (int g = 0; g < GEMM_COUNT; g++) {
            const uint32_t n = gemm_numeric(base, c.bin[b].gemm[g]);
            if (n & allowed) continue;
            snprintf(name, len, "%s:%s=%s", BIN_NAME[b], GEMM_NAMES[g], gemm_item(c.bin[b].gemm[g]).c_str());
            *numeric = numeric_name(n);
            return name;
        }
    return nullptr;
}

size_t format_choices(const Choices &c, char *out, size_t len) {
    std::string s;
    for (int b = 0; b < c.bins; b++) {
        const BinChoices &bc = c.bin[b];
        s += BIN_NAME[b];
        s += ':';
        for (int g = 0; g < GEMM_COUNT; g++) {
            s += GEMM_NAMES[g];
            s += '=';
            s += gemm_item(bc.gemm[g]);
            s += ',';
        }
        append(s, "attn=%s,ln=%s;", ATTN_NAME[bc.attention], LN_NAME[bc.ln]);
    }
    append(s, "pool=%s;forced=", POOL_NAME[c.pool]);
    const uint32_t k = forced_knobs(c);
    bool first = true;
    for (int i = 0; i < 6; i++)
        if (k & (1u << i)) {
            if (!first) s += ',';
            s += KNOB_NAMES[i];
            first = false;
        }
    if (len) {
        const size_t n = s.size() < len - 1 ? s.size() : len - 1;
        memcpy(out, s.data(), n);
        out[n] = 0;
    }
    return s.size();
}

bool parse_choices(const char *s, Choices *into, uint32_t *absent, char *why, size_t why_len) {
    *absent = 0;
    const char *p = s;
    while (*p) {
        const char *item = p;
        while (*p && *p != ';') p++;
        const size_t n = (size_t)(p - item);
        if (*p) p++;
        if (n == 0) continue;
        if (n >= 7 && !strncmp(item, "forced=", 7)) continue;
        if (n >= 5 && !strncmp(item, "pool=", 5)) {
            const int v = index_of(POOL_NAME, item + 5, n - 5);
            if (v < 0) return fail(why, why_len, "TURBO_CUDA_CHOICES: no pooling is named", item + 5, n - 5);
            into->pool = (PoolVariant)v;
            into->pool_forced = KNOB_POOL;
            continue;
        }
        const char *colon = (const char *)memchr(item, ':', n);
        if (!colon) return fail(why, why_len, "TURBO_CUDA_CHOICES: no item is", item, n);
        const size_t bn = (size_t)(colon - item);
        const bool all = bn == 3 && !strncmp(item, "all", 3);
        const int bin = all ? -1 : index_of(BIN_NAME, item, bn);
        if (!all && bin < 0) return fail(why, why_len, "TURBO_CUDA_CHOICES: no token bin is named", item, bn);
        // Each of the bin's parts, into every bin it names.
        const char *q = colon + 1, *end = item + n;
        while (q < end) {
            const char *part = q;
            while (q < end && *q != ',') q++;
            const size_t m = (size_t)(q - part);
            if (q < end) q++;
            const char *eq = (const char *)memchr(part, '=', m);
            if (!eq) return fail(why, why_len, "TURBO_CUDA_CHOICES: no part is", part, m);
            const size_t kn = (size_t)(eq - part), vn = m - kn - 1;
            const char *v = eq + 1;
            if (!all && bin >= into->bins) *absent |= 1u << bin;
            for (int b = all ? 0 : bin; b < (all ? into->bins : bin + 1) && b < into->bins; b++) {
                BinChoices &bc = into->bin[b];
                const int g = index_of(GEMM_NAMES, part, kn);
                if (g >= 0) {
                    if (!parse_gemm(v, vn, &bc.gemm[g], &bc.gemm_forced[g], why, why_len)) return false;
                } else if (kn == 4 && !strncmp(part, "attn", 4)) {
                    const int a = index_of(ATTN_NAME, v, vn);
                    if (a < 0) return fail(why, why_len, "TURBO_CUDA_CHOICES: no attention is named", v, vn);
                    bc.attention = (AttnVariant)a;
                    bc.forced |= KNOB_ATTN;
                } else if (kn == 2 && !strncmp(part, "ln", 2)) {
                    const int l = index_of(LN_NAME, v, vn);
                    if (l < 0) return fail(why, why_len, "TURBO_CUDA_CHOICES: no LayerNorm is named", v, vn);
                    bc.ln = (LnVariant)l;
                    bc.forced |= KNOB_LN;
                } else {
                    return fail(why, why_len, "TURBO_CUDA_CHOICES: a token bin has no part named", part, kn);
                }
            }
        }
    }
    return true;
}

int gemm_variants(const Shape &base, Variant *out, int cap) {
    int n = 0;
    auto add = [&](Tile t, bool tf32, bool candidate) {
        if (n == cap) return;
        Variant &v = out[n++];
        snprintf(v.name, sizeof v.name, "%s%s", tile_name(t), tf32 ? "/tf32" : "");
        v.tile = t;
        v.tf32 = tf32;
        GemmChoice g;
        g.tile = t;
        g.tf32 = tf32;
        v.numeric = gemm_numeric(base, g);
        v.candidate = candidate;
    };
    if (base.half && base.tensor_cores) {
        // F16 on the tensor cores: the eight-warp shapes, plain and
        // swizzled, each with F32 sums and with F16 sums within a chunk.
        for (Tile t : {TILE_EIGHT_WARPS, TILE_SWIZZLED_8W, TILE_EIGHT_WARPS_F16_ACCUMULATE,
                       TILE_SWIZZLED_8W_F16_ACCUMULATE})
            add(t, false, true);
        // F16 sums over the whole of a block's k are F16 sums within a
        // chunk, the block's segment of k, and are never timed.
        for (Tile t : {TILE_64x64, TILE_128x64, TILE_128x128, TILE_128x128_4W, TILE_256x128, TILE_SWIZZLED,
                       TILE_SWIZZLED_256x128, TILE_SWIZZLED_ROWS, TILE_F16_WHOLE_K, TILE_F16_WHOLE_K_3,
                       TILE_F16_WHOLE_K_ROWS, TILE_F16_WHOLE_K_256, TILE_F16_WHOLE_K_2})
            add(t, false, false);
        return n;
    }
    // The FMA kernel's tiles, F32 or F16 operands.
    for (Tile t : {TILE_128x64, TILE_128x128_16x8, TILE_128x128, TILE_64x64}) add(t, false, true);
    if (!base.half && base.tensor_cores) {
        for (Tile t : {TILE_128x64, TILE_128x128}) add(t, true, true);
        for (Tile t : {TILE_64x64, TILE_128x128_4W}) add(t, true, false);
    }
    return n;
}

} // namespace turbo_cuda
