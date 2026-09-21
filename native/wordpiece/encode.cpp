// SPDX-License-Identifier: Apache-2.0
//
// BERT uncased BasicTokenizer + WordPiece. Writes ids into caller
// memory. Heap containers and hash maps are forbidden on this path.

#include "vocab.hpp"

#include <atomic>
#include <algorithm>
#include <limits>
#include "utf8proc/utf8proc.h"
#include "bert_unicode_categories.hpp"
#include <cstring>

namespace {

constexpr size_t kMaxWordChars = 100;
constexpr uint32_t kSpecials = 3;

std::atomic<uint64_t> g_hot_allocs{0};

bool is_control(uint32_t cp) {
    return cp != '\t' && cp != '\n' && cp != '\r' && wordpiece_unicode::is_other(cp);
}

bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x0085 ||
           cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_punctuation(uint32_t cp) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
           (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126) ||
           wordpiece_unicode::is_punctuation(cp);
}

bool is_cjk(uint32_t cp) {
    // Match the model reference's BertNormalizer, including its extension-E boundary.
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B920 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

size_t utf8_next(const char *s, size_t n, size_t i, uint32_t *cp) {
    int32_t decoded = 0;
    const auto len = utf8proc_iterate(reinterpret_cast<const uint8_t *>(s + i),
        static_cast<utf8proc_ssize_t>(std::min(n - i, size_t{4})), &decoded);
    if (len < 1) { return 0; }
    *cp = static_cast<uint32_t>(decoded);
    return static_cast<size_t>(len);
}

constexpr uint64_t kFnvOff = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

uint64_t fnv1a(const char *a, size_t an, const char *b, size_t bn) {
    uint64_t h = kFnvOff;
    for (size_t i = 0; i < an; ++i) {
        h ^= static_cast<uint8_t>(a[i]);
        h *= kFnvPrime;
    }
    for (size_t i = 0; i < bn; ++i) {
        h ^= static_cast<uint8_t>(b[i]);
        h *= kFnvPrime;
    }
    return h;
}

bool lookup(
    const wordpiece_vocab *v,
    const char *a,
    size_t an,
    const char *b,
    size_t bn,
    int32_t *id
) {
    const uint64_t h = fnv1a(a, an, b, bn);
    const size_t total = an + bn;
    for (uint32_t i = 0; i < v->n_slots; ++i) {
        const uint32_t idx = static_cast<uint32_t>((h + i) & v->mask);
        const wordpiece::impl::Slot &s = v->slots[idx];
        if (!s.occupied) {
            return false;
        }
        if (s.len != total) {
            continue;
        }
        const char *tok = v->blob + s.off;
        if (an > 0 && std::memcmp(tok, a, an) != 0) {
            continue;
        }
        if (bn > 0 && std::memcmp(tok + an, b, bn) != 0) {
            continue;
        }
        *id = s.id;
        return true;
    }
    return false;
}

void store(void *base, size_t i, int32_t val, uint32_t width) {
    if (base == nullptr) { return; }
    auto *dest = static_cast<char *>(base) + i * width;
    if (width == 8) {
        const int64_t wide = val;
        std::memcpy(dest, &wide, sizeof(wide));
    } else {
        std::memcpy(dest, &val, sizeof(val));
    }
}

void zero_n(void *base, uint32_t n, uint32_t width) {
    if (base == nullptr || n == 0) {
        return;
    }
    std::memset(base, 0, static_cast<size_t>(n) * width);
}

bool width_ok(uint32_t w) {
    return w == 4 || w == 8;
}

void emit_id(void *ids, size_t cap, uint32_t width, size_t *count, int32_t id) {
    if (*count < cap) {
        store(ids, *count, id, width);
        ++*count;
    }
}

void wordpiece_word(
    const wordpiece_vocab *v, const uint32_t *word, size_t nch,
    void *ids, size_t cap, uint32_t width, size_t *count
) {
    if (nch == 0 || *count == cap) { return; }
    if (nch > kMaxWordChars) {
        emit_id(ids, cap, width, count, v->unk_id);
        return;
    }
    char bytes[kMaxWordChars * 4];
    uint16_t offsets[kMaxWordChars + 1];
    offsets[0] = 0;
    for (size_t i = 0; i < nch; ++i) {
        const auto len = utf8proc_encode_char(static_cast<int32_t>(word[i]),
            reinterpret_cast<uint8_t *>(bytes + offsets[i]));
        offsets[i + 1] = static_cast<uint16_t>(offsets[i] + len);
    }
    int32_t sub[kMaxWordChars];
    size_t nsub = 0;
    size_t start = 0;
    while (start < nch) {
        size_t end = nch;
        int32_t found = -1;
        while (start < end) {
            const char *span = bytes + offsets[start];
            const size_t len = offsets[end] - offsets[start];
            const bool hit = start > 0 ? lookup(v, "##", 2, span, len, &found)
                                       : lookup(v, span, len, nullptr, 0, &found);
            if (hit) { break; }
            --end;
        }
        if (found < 0) {
            emit_id(ids, cap, width, count, v->unk_id);
            return;
        }
        sub[nsub++] = found;
        start = end;
    }
    for (size_t i = 0; i < nsub; ++i) { emit_id(ids, cap, width, count, sub[i]); }
}

int tokenize_into(
    const wordpiece_vocab *v, const char *utf8, size_t n,
    void *ids, size_t cap, uint32_t width, size_t *count
) {
    if (count == nullptr) { return WORDPIECE_ERR_INVALID_ARGUMENT; }
    *count = 0;
    if (v == nullptr || !v->loaded || !width_ok(width) ||
        (utf8 == nullptr && n > 0) || n > static_cast<size_t>(PTRDIFF_MAX) ||
        (ids != nullptr && cap > static_cast<size_t>(PTRDIFF_MAX) / width)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (ids == nullptr) { cap = SIZE_MAX; }
    uint32_t word[kMaxWordChars];
    uint8_t classes[kMaxWordChars];
    size_t nch = 0;
    auto flush = [&]() {
        wordpiece_word(v, word, nch, ids, cap, width, count);
        nch = 0;
    };
    auto append = [&](uint32_t cp) {
        if (nch > kMaxWordChars) { return; }
        if (nch == kMaxWordChars) { ++nch; return; }
        const auto cc = utf8proc_get_property(static_cast<int32_t>(cp))->combining_class;
        size_t at = nch;
        // Canonical ordering of marks that remain after accent removal.
        while (at > 0 && cc != 0 && classes[at - 1] > cc) {
            word[at] = word[at - 1]; classes[at] = classes[at - 1]; --at;
        }
        word[at] = cp; classes[at] = static_cast<uint8_t>(cc); ++nch;
    };
    const char *specials[] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
    const int32_t special_ids[] = {v->pad_id, v->unk_id, v->cls_id, v->sep_id, v->mask_id};
    for (size_t i = 0; i < n;) {
        bool matched = false;
        if (utf8[i] == '[') {
            for (unsigned k = 0; k < 5; ++k) {
                const size_t len = k == 4 ? 6 : 5;
                if ((v->added_specials & (1u << k)) && len <= n - i &&
                    std::memcmp(utf8 + i, specials[k], len) == 0) {
                    flush(); emit_id(ids, cap, width, count, special_ids[k]);
                    i += len; matched = true; break;
                }
            }
        }
        if (matched) { continue; }
        uint32_t cp = 0;
        const size_t adv = utf8_next(utf8, n, i, &cp);
        if (adv == 0) { return WORDPIECE_ERR_INVALID_ARGUMENT; }
        i += adv;
        if (cp == 0 || cp == 0xFFFD || is_control(cp)) { continue; }
        if (is_whitespace(cp)) { flush(); continue; }
        const bool chinese = is_cjk(cp);
        if (chinese) { flush(); }
        int32_t decomposed[32];
        utf8proc_ssize_t len = 1;
        if (v->strip_accents) {
            len = utf8proc_decompose_char(static_cast<int32_t>(cp), decomposed, 32, UTF8PROC_DECOMPOSE, nullptr);
            if (len < 0 || len > 32) { return WORDPIECE_ERR_INTERNAL; }
        } else {
            decomposed[0] = static_cast<int32_t>(cp);
        }
        for (utf8proc_ssize_t k = 0; k < len; ++k) {
            if (v->strip_accents && wordpiece_unicode::is_mark_nonspacing(static_cast<uint32_t>(decomposed[k]))) { continue; }
            const uint32_t lower = v->lowercase ? static_cast<uint32_t>(utf8proc_tolower(decomposed[k]))
                                                : static_cast<uint32_t>(decomposed[k]);
            if (is_whitespace(lower)) { flush(); }
            else if (is_punctuation(lower)) {
                flush(); wordpiece_word(v, &lower, 1, ids, cap, width, count);
            } else { append(lower); }
        }
        if (chinese) { flush(); }
    }
    flush();
    return WORDPIECE_OK;
}

void write_special(
    void *ids,
    void *mask,
    void *types,
    void *pos,
    uint32_t i,
    int32_t id,
    int32_t type,
    uint32_t width
) {
    store(ids, i, id, width);
    store(mask, i, 1, width);
    store(types, i, type, width);
    store(pos, i, static_cast<int32_t>(i), width);
}

void fill_pad_pos(
    void *ids,
    void *mask,
    void *types,
    void *pos,
    uint32_t from,
    uint32_t seq,
    int32_t pad,
    uint32_t width
) {
    for (uint32_t p = from; p < seq; ++p) {
        store(ids, p, pad, width);
        store(mask, p, 0, width);
        store(types, p, 0, width);
        store(pos, p, static_cast<int32_t>(p), width);
    }
}

} // namespace

extern "C" {

uint64_t wordpiece_hot_alloc_counter(void) {
    return g_hot_allocs.load(std::memory_order_relaxed);
}

void wordpiece_hot_alloc_counter_reset(void) {
    g_hot_allocs.store(0, std::memory_order_relaxed);
}

void wordpiece_note_hot_alloc(void) {
    g_hot_allocs.fetch_add(1, std::memory_order_relaxed);
}

int wordpiece_tokenize(
    const wordpiece_vocab *v,
    const char *utf8,
    size_t utf8_len,
    void *ids,
    size_t ids_cap,
    uint32_t elem_width,
    size_t *n_out
) {
    if (n_out == nullptr) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    return tokenize_into(v, utf8, utf8_len, ids, ids_cap, elem_width, n_out);
}

int wordpiece_encode_sentence(
    const wordpiece_vocab *v,
    const char *utf8,
    size_t utf8_len,
    void *ids,
    void *mask,
    void *types,
    void *pos,
    uint32_t seq,
    uint32_t stride,
    uint32_t elem_width
) {
    if (v == nullptr || !v->loaded || ids == nullptr || mask == nullptr ||
        seq < 2 || !width_ok(elem_width)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (stride == 0) {
        stride = seq;
    }
    if (stride < seq || stride > static_cast<size_t>(PTRDIFF_MAX) / elem_width ||
        seq > static_cast<uint32_t>(INT32_MAX)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    zero_n(ids, stride, elem_width);
    zero_n(mask, stride, elem_width);
    zero_n(types, stride, elem_width);
    if (pos) {
        zero_n(pos, stride, elem_width);
    }
    write_special(ids, mask, types, pos, 0, v->cls_id, 0, elem_width);
    size_t ntok = 0;
    const size_t cap = static_cast<size_t>(seq - 2);
    char *id_row = static_cast<char *>(ids) + elem_width; /* ids+1 */
    const int st = tokenize_into(
        v, utf8, utf8_len, id_row, cap, elem_width, &ntok
    );
    if (st != WORDPIECE_OK) {
        return st;
    }
    if (ntok > cap) {
        ntok = cap;
    }
    for (size_t t = 0; t < ntok; ++t) {
        const uint32_t i = static_cast<uint32_t>(t + 1);
        store(mask, i, 1, elem_width);
        store(types, i, 0, elem_width);
        store(pos, i, static_cast<int32_t>(i), elem_width);
    }
    const uint32_t sep_i = static_cast<uint32_t>(1 + ntok);
    write_special(ids, mask, types, pos, sep_i, v->sep_id, 0, elem_width);
    fill_pad_pos(ids, mask, types, pos, sep_i + 1, seq, v->pad_id, elem_width);
    return WORDPIECE_OK;
}

int wordpiece_pack_pair(
    const wordpiece_vocab *v,
    const char *query,
    size_t query_len,
    const char *doc,
    size_t doc_len,
    void *ids,
    void *mask,
    void *types,
    void *pos,
    uint32_t seq,
    uint32_t stride,
    uint32_t elem_width,
    uint32_t truncation,
    uint32_t max_length
) {
    if (v == nullptr || !v->loaded || ids == nullptr || mask == nullptr ||
        types == nullptr || pos == nullptr || seq < kSpecials ||
        !width_ok(elem_width) || truncation > WORDPIECE_TRUNC_ERROR) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if ((query == nullptr && query_len > 0) || (doc == nullptr && doc_len > 0)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (stride == 0) {
        stride = seq;
    }
    if (stride < seq || stride > static_cast<size_t>(PTRDIFF_MAX) / elem_width ||
        seq > static_cast<uint32_t>(INT32_MAX)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }

    uint32_t cap = max_length == 0 ? seq : max_length;
    if (cap > seq) {
        cap = seq;
    }
    if (cap < kSpecials) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    const uint32_t budget = cap - kSpecials;

    size_t nq = 0;
    size_t nd = 0;
    int st = tokenize_into(v, query, query_len, nullptr, 0, elem_width, &nq);
    if (st != WORDPIECE_OK) {
        return st;
    }
    st = tokenize_into(v, doc, doc_len, nullptr, 0, elem_width, &nd);
    if (st != WORDPIECE_OK) {
        return st;
    }

    if (nq > budget || nd > budget - nq) {
        switch (truncation) {
        case WORDPIECE_TRUNC_ERROR:
            return WORDPIECE_ERR_INVALID_ARGUMENT;
        case WORDPIECE_TRUNC_QUERY_PRIORITY:
            if (nq > budget) {
                nq = budget;
                nd = 0;
            } else {
                nd = budget - nq;
            }
            break;
        case WORDPIECE_TRUNC_LONGEST_FIRST: {
            const bool query_longer = nq > nd;
            size_t shortest = query_longer ? nd : nq;
            size_t longest = query_longer ? nq : nd;
            if (shortest > budget / 2) {
                shortest = budget / 2;
                longest = budget - shortest;
            } else {
                longest = budget - shortest;
            }
            nq = query_longer ? longest : shortest;
            nd = query_longer ? shortest : longest;
            break;
        }
        }
    }

    zero_n(ids, stride, elem_width);
    zero_n(mask, stride, elem_width);
    zero_n(types, stride, elem_width);
    zero_n(pos, stride, elem_width);

    write_special(ids, mask, types, pos, 0, v->cls_id, 0, elem_width);

    size_t got_q = 0;
    char *qdest = static_cast<char *>(ids) + elem_width;
    st = tokenize_into(v, query, query_len, qdest, nq, elem_width, &got_q);
    if (st != WORDPIECE_OK) {
        return st;
    }
    if (got_q > nq) {
        got_q = nq;
    }
    for (size_t t = 0; t < got_q; ++t) {
        const uint32_t i = static_cast<uint32_t>(t + 1);
        store(mask, i, 1, elem_width);
        store(types, i, 0, elem_width);
        store(pos, i, static_cast<int32_t>(i), elem_width);
    }

    const uint32_t sep_q = static_cast<uint32_t>(1 + got_q);
    write_special(ids, mask, types, pos, sep_q, v->sep_id, 0, elem_width);

    size_t got_d = 0;
    char *ddest = static_cast<char *>(ids) + static_cast<size_t>(sep_q + 1) * elem_width;
    st = tokenize_into(v, doc, doc_len, ddest, nd, elem_width, &got_d);
    if (st != WORDPIECE_OK) {
        return st;
    }
    if (got_d > nd) {
        got_d = nd;
    }
    for (size_t t = 0; t < got_d; ++t) {
        const uint32_t i = sep_q + 1 + static_cast<uint32_t>(t);
        store(mask, i, 1, elem_width);
        store(types, i, 1, elem_width);
        store(pos, i, static_cast<int32_t>(i), elem_width);
    }

    const uint32_t sep_d = sep_q + 1 + static_cast<uint32_t>(got_d);
    write_special(ids, mask, types, pos, sep_d, v->sep_id, 1, elem_width);
    fill_pad_pos(ids, mask, types, pos, sep_d + 1, seq, v->pad_id, elem_width);
    return WORDPIECE_OK;
}

} // extern "C"
