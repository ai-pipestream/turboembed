// SPDX-License-Identifier: Apache-2.0
//
// BERT uncased BasicTokenizer + WordPiece. Writes ids into caller
// memory. Heap containers and hash maps are forbidden on this path.

#include "vocab.hpp"

#include <atomic>
#include <cctype>
#include <cstring>

namespace {

constexpr size_t kMaxWordBytes = 1024;
constexpr size_t kMaxWordChars = 256;
constexpr size_t kMaxSub = 64;
constexpr uint32_t kSpecials = 3;

std::atomic<uint64_t> g_hot_allocs{0};

bool is_control(uint32_t cp) {
    if (cp == '\t' || cp == '\n' || cp == '\r') {
        return false;
    }
    return cp < 32 || (cp >= 0x7F && cp <= 0x9F);
}

bool is_whitespace(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == 0x00A0 ||
           cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 ||
           cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

bool is_punct_ascii(uint32_t cp) {
    return (cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
           (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126);
}

bool is_punctuation(uint32_t cp) {
    if (is_punct_ascii(cp)) {
        return true;
    }
    return (cp >= 0x00A1 && cp <= 0x00BF && cp != 0x00A0) ||
           (cp >= 0x2010 && cp <= 0x2027) || (cp >= 0x2030 && cp <= 0x205E) ||
           (cp >= 0x3001 && cp <= 0x303F) || cp == 0x00AD || cp == 0x058A ||
           cp == 0x05BE || cp == 0x055A || cp == 0x055B;
}

bool is_cjk(uint32_t cp) {
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF) ||
           (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F) ||
           (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF) ||
           (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F);
}

bool is_combining_mark(uint32_t cp) {
    return (cp >= 0x0300 && cp <= 0x036F) || (cp >= 0x1AB0 && cp <= 0x1AFF) ||
           (cp >= 0x1DC0 && cp <= 0x1DFF) || (cp >= 0x20D0 && cp <= 0x20FF) ||
           (cp >= 0xFE20 && cp <= 0xFE2F);
}

uint32_t strip_latin_accent(uint32_t cp) {
    if (cp >= 'A' && cp <= 'Z') {
        return cp - 'A' + 'a';
    }
    if (cp >= 'a' && cp <= 'z') {
        return cp;
    }
    switch (cp) {
    case 0x00C0: case 0x00C1: case 0x00C2: case 0x00C3: case 0x00C4:
    case 0x00C5: case 0x0100: case 0x0102: case 0x0104:
    case 0x00E0: case 0x00E1: case 0x00E2: case 0x00E3: case 0x00E4:
    case 0x00E5: case 0x0101: case 0x0103: case 0x0105:
        return 'a';
    case 0x00C7: case 0x0106: case 0x0108: case 0x010A: case 0x010C:
    case 0x00E7: case 0x0107: case 0x0109: case 0x010B: case 0x010D:
        return 'c';
    case 0x00D0: case 0x010E: case 0x0110:
    case 0x00F0: case 0x010F: case 0x0111:
        return 'd';
    case 0x00C8: case 0x00C9: case 0x00CA: case 0x00CB: case 0x0112:
    case 0x0114: case 0x0116: case 0x0118: case 0x011A:
    case 0x00E8: case 0x00E9: case 0x00EA: case 0x00EB: case 0x0113:
    case 0x0115: case 0x0117: case 0x0119: case 0x011B:
        return 'e';
    case 0x011C: case 0x011E: case 0x0120: case 0x0122:
    case 0x011D: case 0x011F: case 0x0121: case 0x0123:
        return 'g';
    case 0x0124: case 0x0126: case 0x0125: case 0x0127:
        return 'h';
    case 0x00CC: case 0x00CD: case 0x00CE: case 0x00CF: case 0x0128:
    case 0x012A: case 0x012C: case 0x012E: case 0x0130:
    case 0x00EC: case 0x00ED: case 0x00EE: case 0x00EF: case 0x0129:
    case 0x012B: case 0x012D: case 0x012F: case 0x0131:
        return 'i';
    case 0x0134: case 0x0135:
        return 'j';
    case 0x0136: case 0x0137:
        return 'k';
    case 0x0139: case 0x013B: case 0x013D: case 0x013F: case 0x0141:
    case 0x013A: case 0x013C: case 0x013E: case 0x0140: case 0x0142:
        return 'l';
    case 0x0143: case 0x0145: case 0x0147: case 0x00D1:
    case 0x0144: case 0x0146: case 0x0148: case 0x00F1:
        return 'n';
    case 0x00D2: case 0x00D3: case 0x00D4: case 0x00D5: case 0x00D6:
    case 0x00D8: case 0x014C: case 0x014E: case 0x0150:
    case 0x00F2: case 0x00F3: case 0x00F4: case 0x00F5: case 0x00F6:
    case 0x00F8: case 0x014D: case 0x014F: case 0x0151:
        return 'o';
    case 0x0154: case 0x0156: case 0x0158:
    case 0x0155: case 0x0157: case 0x0159:
        return 'r';
    case 0x015A: case 0x015C: case 0x015E: case 0x0160:
    case 0x015B: case 0x015D: case 0x015F: case 0x0161:
    case 0x00DF:
        return 's';
    case 0x0162: case 0x0164: case 0x0166:
    case 0x0163: case 0x0165: case 0x0167:
        return 't';
    case 0x00D9: case 0x00DA: case 0x00DB: case 0x00DC: case 0x0168:
    case 0x016A: case 0x016C: case 0x016E: case 0x0170: case 0x0172:
    case 0x00F9: case 0x00FA: case 0x00FB: case 0x00FC: case 0x0169:
    case 0x016B: case 0x016D: case 0x016F: case 0x0171: case 0x0173:
        return 'u';
    case 0x0174: case 0x0175:
        return 'w';
    case 0x00DD: case 0x0176: case 0x0178:
    case 0x00FD: case 0x00FF: case 0x0177:
        return 'y';
    case 0x0179: case 0x017B: case 0x017D:
    case 0x017A: case 0x017C: case 0x017E:
        return 'z';
    case 0x00C6: case 0x00E6:
        return 0;
    default:
        if (cp < 128) {
            return static_cast<uint32_t>(std::tolower(static_cast<int>(cp)));
        }
        return cp;
    }
}

size_t utf8_next(const char *s, size_t n, size_t i, uint32_t *cp) {
    if (i >= n) {
        *cp = 0;
        return 0;
    }
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
        *cp = c;
        return 1;
    }
    if ((c >> 5) == 0x6 && i + 1 < n) {
        *cp = (static_cast<uint32_t>(c & 0x1F) << 6) |
              (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        return 2;
    }
    if ((c >> 4) == 0xE && i + 2 < n) {
        *cp = (static_cast<uint32_t>(c & 0x0F) << 12) |
              ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
              (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        return 3;
    }
    if ((c >> 3) == 0x1E && i + 3 < n) {
        *cp = (static_cast<uint32_t>(c & 0x07) << 18) |
              ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
              ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
              ((static_cast<unsigned char>(s[i + 3]) & 0x3F));
        return 4;
    }
    *cp = 0xFFFD;
    return 1;
}

size_t append_utf8_buf(char *out, size_t cap, size_t used, uint32_t cp) {
    char tmp[4];
    size_t n = 0;
    if (cp < 0x80) {
        tmp[0] = static_cast<char>(cp);
        n = 1;
    } else if (cp < 0x800) {
        tmp[0] = static_cast<char>(0xC0 | (cp >> 6));
        tmp[1] = static_cast<char>(0x80 | (cp & 0x3F));
        n = 2;
    } else if (cp < 0x10000) {
        tmp[0] = static_cast<char>(0xE0 | (cp >> 12));
        tmp[1] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        tmp[2] = static_cast<char>(0x80 | (cp & 0x3F));
        n = 3;
    } else {
        tmp[0] = static_cast<char>(0xF0 | (cp >> 18));
        tmp[1] = static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
        tmp[2] = static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
        tmp[3] = static_cast<char>(0x80 | (cp & 0x3F));
        n = 4;
    }
    if (used + n > cap) {
        return used;
    }
    for (size_t k = 0; k < n; ++k) {
        out[used + k] = tmp[k];
    }
    return used + n;
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

void store(void *base, uint32_t i, int32_t val, uint32_t width) {
    if (base == nullptr) {
        return;
    }
    if (width == 8) {
        reinterpret_cast<int64_t *>(base)[i] = val;
    } else {
        reinterpret_cast<int32_t *>(base)[i] = val;
    }
}

int32_t load_id(const void *base, uint32_t i, uint32_t width) {
    if (base == nullptr) {
        return 0;
    }
    if (width == 8) {
        return static_cast<int32_t>(reinterpret_cast<const int64_t *>(base)[i]);
    }
    return reinterpret_cast<const int32_t *>(base)[i];
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

bool wordpiece_word(
    const wordpiece_vocab *v,
    const char *word,
    size_t word_n,
    void *ids,
    size_t ids_cap,
    uint32_t width,
    size_t *n_out
) {
    if (word_n == 0) {
        return true;
    }
    uint16_t coff[kMaxWordChars];
    uint16_t clen[kMaxWordChars];
    size_t nch = 0;
    size_t i = 0;
    while (i < word_n && nch < kMaxWordChars) {
        uint32_t cp = 0;
        const size_t adv = utf8_next(word, word_n, i, &cp);
        coff[nch] = static_cast<uint16_t>(i);
        clen[nch] = static_cast<uint16_t>(adv);
        ++nch;
        i += adv;
    }
    if (i < word_n) {
        if (*n_out >= ids_cap) {
            return false;
        }
        store(ids, static_cast<uint32_t>(*n_out), v->unk_id, width);
        ++*n_out;
        return true;
    }

    int32_t sub[kMaxSub];
    size_t nsub = 0;
    bool is_bad = false;
    size_t start = 0;
    while (start < nch) {
        size_t end = nch;
        int32_t found = -1;
        while (start < end) {
            const size_t boff = coff[start];
            const size_t bend = static_cast<size_t>(coff[end - 1]) + clen[end - 1];
            const char *span = word + boff;
            const size_t slen = bend - boff;
            int32_t id = -1;
            const bool hit = start > 0
                                 ? lookup(v, "##", 2, span, slen, &id)
                                 : lookup(v, span, slen, nullptr, 0, &id);
            if (hit) {
                found = id;
                break;
            }
            --end;
        }
        if (found < 0) {
            is_bad = true;
            break;
        }
        if (nsub >= kMaxSub) {
            is_bad = true;
            break;
        }
        sub[nsub++] = found;
        start = end;
    }
    if (is_bad) {
        if (*n_out >= ids_cap) {
            return false;
        }
        store(ids, static_cast<uint32_t>(*n_out), v->unk_id, width);
        ++*n_out;
        return true;
    }
    if (*n_out + nsub > ids_cap) {
        return false;
    }
    for (size_t k = 0; k < nsub; ++k) {
        store(ids, static_cast<uint32_t>(*n_out), sub[k], width);
        ++*n_out;
    }
    return true;
}

bool emit_word(
    const wordpiece_vocab *v,
    const char *word,
    size_t word_n,
    void *ids,
    size_t ids_cap,
    uint32_t width,
    size_t *n_out
) {
    if (word_n == 0) {
        return true;
    }
    char piece[kMaxWordBytes];
    size_t piece_n = 0;
    size_t j = 0;
    while (j < word_n) {
        uint32_t cp = 0;
        const size_t adv = utf8_next(word, word_n, j, &cp);
        if (is_punctuation(cp)) {
            if (piece_n > 0) {
                if (!wordpiece_word(v, piece, piece_n, ids, ids_cap, width, n_out)) {
                    return false;
                }
                piece_n = 0;
            }
            char pbuf[8];
            const size_t pn = append_utf8_buf(pbuf, sizeof(pbuf), 0, cp);
            if (!wordpiece_word(v, pbuf, pn, ids, ids_cap, width, n_out)) {
                return false;
            }
        } else {
            if (piece_n + adv > sizeof(piece)) {
                if (!wordpiece_word(v, piece, piece_n, ids, ids_cap, width, n_out)) {
                    return false;
                }
                piece_n = 0;
                if (*n_out >= ids_cap) {
                    return false;
                }
                store(ids, static_cast<uint32_t>(*n_out), v->unk_id, width);
                ++*n_out;
                j += adv;
                continue;
            }
            for (size_t k = 0; k < adv; ++k) {
                piece[piece_n++] = word[j + k];
            }
        }
        j += adv;
    }
    if (piece_n > 0) {
        return wordpiece_word(v, piece, piece_n, ids, ids_cap, width, n_out);
    }
    return true;
}

int tokenize_into(
    const wordpiece_vocab *v,
    const char *utf8,
    size_t n,
    void *ids,
    size_t ids_cap,
    uint32_t width,
    size_t *n_out
) {
    if (v == nullptr || !v->loaded || n_out == nullptr) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (!width_ok(width)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (ids == nullptr && ids_cap == 0) {
        ids_cap = static_cast<size_t>(-1);
    }
    if (utf8 == nullptr && n > 0) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    *n_out = 0;
    if (utf8 == nullptr) {
        return WORDPIECE_OK;
    }

    char word[kMaxWordBytes];
    size_t word_n = 0;
    auto flush = [&]() -> bool {
        const bool ok = emit_word(v, word, word_n, ids, ids_cap, width, n_out);
        word_n = 0;
        return ok;
    };

    size_t i = 0;
    while (i < n) {
        uint32_t cp = 0;
        const size_t adv = utf8_next(utf8, n, i, &cp);
        i += adv;
        if (cp == 0 || is_control(cp)) {
            continue;
        }
        if (is_whitespace(cp)) {
            if (!flush()) {
                return WORDPIECE_ERR_INTERNAL;
            }
            continue;
        }
        if (is_cjk(cp)) {
            if (!flush()) {
                return WORDPIECE_ERR_INTERNAL;
            }
            char cjk[8];
            const size_t cn = append_utf8_buf(cjk, sizeof(cjk), 0, cp);
            if (!emit_word(v, cjk, cn, ids, ids_cap, width, n_out)) {
                return WORDPIECE_ERR_INTERNAL;
            }
            continue;
        }
        if (is_combining_mark(cp)) {
            continue;
        }
        if (cp == 0x00C6 || cp == 0x00E6) {
            if (word_n + 2 > sizeof(word)) {
                if (!flush()) {
                    return WORDPIECE_ERR_INTERNAL;
                }
            }
            if (word_n + 2 <= sizeof(word)) {
                word[word_n++] = 'a';
                word[word_n++] = 'e';
            }
            continue;
        }
        const uint32_t stripped = strip_latin_accent(cp);
        if (stripped == 0) {
            continue;
        }
        const size_t next = append_utf8_buf(word, sizeof(word), word_n, stripped);
        if (next == word_n) {
            if (!flush()) {
                return WORDPIECE_ERR_INTERNAL;
            }
            if (*n_out >= ids_cap) {
                return WORDPIECE_ERR_INTERNAL;
            }
            store(ids, static_cast<uint32_t>(*n_out), v->unk_id, width);
            ++*n_out;
            continue;
        }
        word_n = next;
    }
    if (!flush()) {
        return WORDPIECE_ERR_INTERNAL;
    }
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
    if (ids == nullptr && ids_cap != 0 && ids_cap != static_cast<size_t>(-1)) {
        /* count-only callers pass ids=NULL and ids_cap=0 or SIZE_MAX */
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
    if (stride < seq) {
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
        !width_ok(elem_width)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if ((query == nullptr && query_len > 0) || (doc == nullptr && doc_len > 0)) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (query_len == 0 && doc_len == 0) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    if (stride == 0) {
        stride = seq;
    }
    if (stride < seq) {
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

    if (nq + nd > budget) {
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
        case WORDPIECE_TRUNC_LONGEST_FIRST:
        default:
            while (nq + nd > budget) {
                if (nq >= nd && nq > 0) {
                    --nq;
                } else if (nd > 0) {
                    --nd;
                } else {
                    break;
                }
            }
            break;
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
        (void)load_id;
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
