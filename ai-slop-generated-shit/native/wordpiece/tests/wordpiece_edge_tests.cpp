// SPDX-License-Identifier: Apache-2.0
//
// Edge-case suite: empty/NUL/unicode inputs, count queries, cap canaries,
// elem widths, exact encode/pack layout, truncation modes, hot-alloc
// tripwire across a burst, and load/destroy error semantics.

#include "wordpiece.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

static int g_fails = 0;
static int g_passes = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_fails;                                                           \
        } else {                                                                 \
            ++g_passes;                                                           \
        }                                                                        \
    } while (0)

#define CHECK_EQ(a, b) CHECK((a) == (b))

static std::string workspace_root() {
    if (const char *e = std::getenv("INFERSTREAM_ROOT")) {
        return e;
    }
    return ".";
}

static bool file_exists(const std::string &p) {
    std::ifstream in(p);
    return static_cast<bool>(in);
}

static void check_row_i32(const char *label, const int32_t *got, const int32_t *want, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (got[i] != want[i]) {
            std::fprintf(
                stderr, "FAIL %s [%zu]: got %d want %d\n", label, i, got[i], want[i]
            );
            ++g_fails;
        } else {
            ++g_passes;
        }
    }
}

static wordpiece_vocab *load_tiny() {
    const std::string path = workspace_root() + "/testdata/turborerank/tiny-vocab/vocab.txt";
    if (!file_exists(path)) {
        std::fprintf(stderr, "SKIP tiny vocab missing\n");
        return nullptr;
    }
    wordpiece_vocab *v = nullptr;
    CHECK_EQ(wordpiece_vocab_load(path.c_str(), &v), WORDPIECE_OK);
    CHECK(v != nullptr);
    return v;
}

// Tiny vocab ids: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 hello=4 world=5 ##ing=6 cafe=7.

static void test_empty_and_whitespace() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    int32_t ids[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    size_t n = 999;

    // Empty string tokenizes to zero ids.
    CHECK_EQ(wordpiece_tokenize(v, "", 0, ids, 8, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 0u);

    // NULL pointer with zero length is a valid empty input (pointer+length contract).
    CHECK_EQ(wordpiece_tokenize(v, nullptr, 0, ids, 8, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 0u);

    // Whitespace-only text (space, tab, CR, LF, NBSP) yields zero ids.
    const char ws[] = " \t\r\n\xC2\xA0";
    CHECK_EQ(wordpiece_tokenize(v, ws, sizeof(ws) - 1, ids, 8, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 0u);

    // Punctuation-only text: each punctuation char is its own word; the tiny
    // vocab has no punctuation tokens, so each resolves to [UNK].
    const char punct[] = "!?";
    CHECK_EQ(wordpiece_tokenize(v, punct, sizeof(punct) - 1, ids, 8, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 2u);
    CHECK_EQ(ids[0], 1);
    CHECK_EQ(ids[1], 1);

    // Mixed word + punctuation + word: punctuation flushes the word and emits
    // its own id between them.
    const char mixed[] = "hello, world";
    CHECK_EQ(wordpiece_tokenize(v, mixed, sizeof(mixed) - 1, ids, 8, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 3u);
    CHECK_EQ(ids[0], 4);
    CHECK_EQ(ids[1], 1); // "," -> [UNK] on the tiny vocab
    CHECK_EQ(ids[2], 5);

    wordpiece_vocab_destroy(v);
}

static void test_embedded_nul_and_controls() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    int32_t a[8];
    int32_t b[8];
    size_t na = 0;
    size_t nb = 0;

    // BERT clean_text removes NUL (it is not turned into whitespace), so a NUL
    // inside a word joins the surrounding pieces into one unknown word.
    const char with_nul[] = {'h', 'e', 'l', 'l', 'o', '\0', 'w', 'o', 'r', 'l', 'd'};
    const char joined[] = "helloworld";
    CHECK_EQ(wordpiece_tokenize(v, with_nul, sizeof(with_nul), a, 8, 4, &na), WORDPIECE_OK);
    CHECK_EQ(wordpiece_tokenize(v, joined, sizeof(joined) - 1, b, 8, 4, &nb), WORDPIECE_OK);
    CHECK_EQ(na, nb);
    CHECK_EQ(na, 1u);
    CHECK_EQ(a[0], 1); // "helloworld" is not in the tiny vocab -> [UNK]
    check_row_i32("nul-join", a, b, na);

    // A NUL adjacent to whitespace cleans away and matches the no-NUL text.
    const char nul_before_ws[] = {'h', 'e', 'l', 'l', 'o', '\0', ' ', 'w', 'o', 'r', 'l', 'd'};
    const char plain[] = "hello world";
    CHECK_EQ(wordpiece_tokenize(v, nul_before_ws, sizeof(nul_before_ws), a, 8, 4, &na), WORDPIECE_OK);
    CHECK_EQ(wordpiece_tokenize(v, plain, sizeof(plain) - 1, b, 8, 4, &nb), WORDPIECE_OK);
    CHECK_EQ(na, nb);
    CHECK_EQ(na, 2u);
    check_row_i32("nul-clean", a, b, na);

    // Other C0 controls are removed the same way (BERT clean_text).
    const char ctl_before_ws[] = {'h', 'e', 'l', 'l', 'o', '\x01', ' ', 'w', 'o', 'r', 'l', 'd'};
    CHECK_EQ(wordpiece_tokenize(v, ctl_before_ws, sizeof(ctl_before_ws), a, 8, 4, &na), WORDPIECE_OK);
    CHECK_EQ(na, 2u);
    check_row_i32("ctl-clean", a, b, na);

    wordpiece_vocab_destroy(v);
}

static void test_unicode_tiny_vocab() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    int32_t ids[16];
    size_t n = 0;

    // Uncased preset: accents are stripped and text lowercased, so "café"
    // (and even "CAFÉ") collapses onto the tiny vocab's "cafe" entry.
    const char cafe[] = "caf\xC3\xA9";
    CHECK_EQ(wordpiece_tokenize(v, cafe, sizeof(cafe) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 1u);
    CHECK_EQ(ids[0], 7);

    const char cafe_upper[] = "CAF\xC3\x89";
    CHECK_EQ(wordpiece_tokenize(v, cafe_upper, sizeof(cafe_upper) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 1u);
    CHECK_EQ(ids[0], 7);

    // CJK chars are split into single-char words; each misses the tiny vocab
    // and resolves to [UNK].
    const char cjk[] = "\xE4\xB8\x96\xE7\x95\x8C"; // 世界
    CHECK_EQ(wordpiece_tokenize(v, cjk, sizeof(cjk) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 2u);
    CHECK_EQ(ids[0], 1);
    CHECK_EQ(ids[1], 1);

    // Emoji is a single word that misses the vocab -> [UNK].
    const char emoji[] = "\xF0\x9F\x98\x80"; // U+1F600
    CHECK_EQ(wordpiece_tokenize(v, emoji, sizeof(emoji) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 1u);
    CHECK_EQ(ids[0], 1);

    // Combined sequence mixes known words, [UNK] punctuation, [UNK] CJK, the
    // accent-stripped hit, and an [UNK] emoji. Used again for width parity.
    const char combo[] = "hello, \xE4\xB8\x96\xE7\x95\x8C caf\xC3\xA9 \xF0\x9F\x98\x80";
    CHECK_EQ(wordpiece_tokenize(v, combo, sizeof(combo) - 1, ids, 16, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 6u);
    const int32_t want_combo[6] = {4, 1, 1, 1, 7, 1};
    check_row_i32("unicode-combo", ids, want_combo, 6);

    wordpiece_vocab_destroy(v);
}

static void test_count_query_and_cap_canaries() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    // "hello world helloing" -> hello(4) world(5) hello(4) ##ing(6): 4 tokens.
    const char text[] = "hello world helloing";
    size_t count = 0;
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, nullptr, 0, 4, &count), WORDPIECE_OK);
    CHECK_EQ(count, 4u);

    // ids == NULL ignores ids_cap entirely (documented).
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, nullptr, 0, 4, &count), WORDPIECE_OK);
    CHECK_EQ(count, 4u);

    // A real buffer with cap 2 reports the written count (2), not the total.
    int32_t ids[4] = {-1, -1, -1, -1};
    size_t n = 999;
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, ids, 2, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 2u);
    CHECK_EQ(ids[0], 4);
    CHECK_EQ(ids[1], 5);

    // ids_cap one too small: status stays OK, count is clamped to the cap,
    // and nothing is written past the caller's buffer (canaries survive).
    constexpr int32_t kSentinel4 = 0x55555555;
    int32_t buf4[6];
    for (int32_t &x : buf4) { x = kSentinel4; }
    n = 999;
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, buf4 + 2, 1, 4, &n), WORDPIECE_OK);
    CHECK_EQ(n, 1u);
    CHECK_EQ(buf4[2], 4);
    for (size_t i = 0; i < 6; ++i) {
        if (i != 2) { CHECK_EQ(buf4[i], kSentinel4); }
    }

    // Same no-overflow guarantee for the i64 element width.
    constexpr int64_t kSentinel8 = 0x5555555555555555ll;
    int64_t buf8[4];
    for (int64_t &x : buf8) { x = kSentinel8; }
    n = 999;
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, buf8 + 1, 1, 8, &n), WORDPIECE_OK);
    CHECK_EQ(n, 1u);
    CHECK_EQ(buf8[1], 4);
    CHECK_EQ(buf8[0], kSentinel8);
    CHECK_EQ(buf8[2], kSentinel8);
    CHECK_EQ(buf8[3], kSentinel8);

    // Invalid widths and a missing n_out are rejected.
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, ids, 4, 2, &n),
             WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, ids, 4, 0, &n),
             WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, ids, 4, 4, nullptr),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // Non-UTF-8 input is rejected and reports a zero count.
    const char bad_utf8[] = {'h', 'i', static_cast<char>(0xFF)};
    n = 999;
    CHECK_EQ(wordpiece_tokenize(v, bad_utf8, sizeof(bad_utf8), ids, 4, 4, &n),
             WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(n, 0u);

    wordpiece_vocab_destroy(v);
}

static void test_elem_width_parity() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    const char text[] = "hello, world caf\xC3\xA9";
    int32_t ids4[16];
    int64_t ids8[16];
    size_t n4 = 0;
    size_t n8 = 0;
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, ids4, 16, 4, &n4), WORDPIECE_OK);
    CHECK_EQ(wordpiece_tokenize(v, text, sizeof(text) - 1, ids8, 16, 8, &n8), WORDPIECE_OK);
    CHECK_EQ(n4, n8);
    for (size_t i = 0; i < n4; ++i) { CHECK_EQ(ids4[i], static_cast<int32_t>(ids8[i])); }

    // encode_sentence produces identical ids/mask/types/pos at width 4 and 8.
    int32_t i4[8], m4[8], t4[8], p4[8];
    int64_t i8[8], m8[8], t8[8], p8[8];
    const char hello[] = "hello world";
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       i4, m4, t4, p4, 8, 8, 4), WORDPIECE_OK);
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       i8, m8, t8, p8, 8, 8, 8), WORDPIECE_OK);
    for (int i = 0; i < 8; ++i) {
        CHECK_EQ(i4[i], static_cast<int32_t>(i8[i]));
        CHECK_EQ(m4[i], static_cast<int32_t>(m8[i]));
        CHECK_EQ(t4[i], static_cast<int32_t>(t8[i]));
        CHECK_EQ(p4[i], static_cast<int32_t>(p8[i]));
    }

    wordpiece_vocab_destroy(v);
}

static void test_encode_sentence_layout() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    const char hello[] = "hello world";

    // Exact layout with PAD tail: [CLS] hello world [SEP] [PAD x4].
    int32_t ids[8], mask[8], types[8], pos[8];
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       ids, mask, types, pos, 8, 8, 4), WORDPIECE_OK);
    const int32_t want_ids[8] = {2, 4, 5, 3, 0, 0, 0, 0};
    const int32_t want_mask[8] = {1, 1, 1, 1, 0, 0, 0, 0};
    const int32_t want_types[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int32_t want_pos[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    check_row_i32("enc ids", ids, want_ids, 8);
    check_row_i32("enc mask", mask, want_mask, 8);
    check_row_i32("enc types", types, want_types, 8);
    check_row_i32("enc pos", pos, want_pos, 8);

    // Exact fit: seq == tokens + 2 leaves no PAD and an all-ones mask.
    int32_t ids4buf[4], mask4buf[4];
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       ids4buf, mask4buf, nullptr, nullptr, 4, 4, 4),
             WORDPIECE_OK);
    const int32_t want_fit[4] = {2, 4, 5, 3};
    const int32_t want_fit_mask[4] = {1, 1, 1, 1};
    check_row_i32("enc-fit ids", ids4buf, want_fit, 4);
    check_row_i32("enc-fit mask", mask4buf, want_fit_mask, 4);

    // One slot short silently right-truncates the tokens to the seq-2 budget.
    int32_t ids3buf[3], mask3buf[3];
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       ids3buf, mask3buf, nullptr, nullptr, 3, 3, 4),
             WORDPIECE_OK);
    const int32_t want_trunc[3] = {2, 4, 3};
    const int32_t want_trunc_mask[3] = {1, 1, 1};
    check_row_i32("enc-trunc ids", ids3buf, want_trunc, 3);
    check_row_i32("enc-trunc mask", mask3buf, want_trunc_mask, 3);

    // Empty input still emits [CLS] [SEP].
    int32_t idse[8], maske[8];
    CHECK_EQ(wordpiece_encode_sentence(v, "", 0, idse, maske, nullptr, nullptr, 8, 8, 4),
             WORDPIECE_OK);
    CHECK_EQ(idse[0], 2);
    CHECK_EQ(idse[1], 3);
    CHECK_EQ(idse[2], 0);
    CHECK_EQ(maske[0], 1);
    CHECK_EQ(maske[1], 1);
    CHECK_EQ(maske[2], 0);

    // pos == NULL is documented as allowed; ids == NULL is not.
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       ids, mask, types, nullptr, 8, 8, 4), WORDPIECE_OK);
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       nullptr, mask, types, pos, 8, 8, 4),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // seq < 2 and stride < seq are rejected.
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       ids, mask, types, pos, 1, 1, 4),
             WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       ids, mask, types, pos, 8, 4, 4),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // stride > seq additionally zeroes the tail of every row.
    int32_t idsw[10], maskw[10], typesw[10], posw[10];
    for (int i = 0; i < 10; ++i) {
        idsw[i] = maskw[i] = typesw[i] = posw[i] = -7;
    }
    CHECK_EQ(wordpiece_encode_sentence(v, hello, sizeof(hello) - 1,
                                       idsw, maskw, typesw, posw, 8, 10, 4), WORDPIECE_OK);
    CHECK_EQ(idsw[8], 0);
    CHECK_EQ(idsw[9], 0);
    CHECK_EQ(maskw[9], 0);
    CHECK_EQ(typesw[9], 0);
    CHECK_EQ(posw[9], 0);

    wordpiece_vocab_destroy(v);
}

static void check_pack_row(const char *label, const int32_t *ids, const int32_t *mask,
                           const int32_t *types, const int32_t *pos,
                           const int32_t *want_ids, const int32_t *want_mask,
                           const int32_t *want_types, size_t n) {
    check_row_i32(label, ids, want_ids, n);
    char sub[64];
    std::snprintf(sub, sizeof(sub), "%s mask", label);
    check_row_i32(sub, mask, want_mask, n);
    std::snprintf(sub, sizeof(sub), "%s types", label);
    check_row_i32(sub, types, want_types, n);
    for (size_t i = 0; i < n; ++i) { CHECK_EQ(pos[i], static_cast<int32_t>(i)); }
}

static void test_pack_pair_truncation() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    const char q[] = "hello world";   // 2 tokens: 4 5
    const char d[] = "helloing";      // 2 tokens: 4 6
    int32_t ids[8], mask[8], types[8], pos[8];

    // Exact max_length boundary (2+2 tokens + 3 specials = 7): no truncation,
    // full type-id split at the document boundary.
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 7), WORDPIECE_OK);
    const int32_t want_a_ids[8] = {2, 4, 5, 3, 4, 6, 3, 0};
    const int32_t want_a_mask[8] = {1, 1, 1, 1, 1, 1, 1, 0};
    const int32_t want_a_types[8] = {0, 0, 0, 0, 1, 1, 1, 0};
    check_pack_row("pack-exact", ids, mask, types, pos,
                   want_a_ids, want_a_mask, want_a_types, 8);

    // Same boundary with TRUNC_ERROR also fits and succeeds.
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_ERROR, 7), WORDPIECE_OK);

    // max_length one short (budget 3, tie): longest-first halves the tie and
    // the query keeps 1 token; query-priority keeps the whole query (2) and
    // truncates the doc to 1; error mode fails loudly.
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 6), WORDPIECE_OK);
    const int32_t want_b0_ids[8] = {2, 4, 3, 4, 6, 3, 0, 0};
    const int32_t want_b0_mask[8] = {1, 1, 1, 1, 1, 1, 0, 0};
    const int32_t want_b0_types[8] = {0, 0, 0, 1, 1, 1, 0, 0};
    check_pack_row("pack-lf", ids, mask, types, pos,
                   want_b0_ids, want_b0_mask, want_b0_types, 8);

    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_QUERY_PRIORITY, 6), WORDPIECE_OK);
    const int32_t want_b1_ids[8] = {2, 4, 5, 3, 4, 3, 0, 0};
    const int32_t want_b1_mask[8] = {1, 1, 1, 1, 1, 1, 0, 0};
    const int32_t want_b1_types[8] = {0, 0, 0, 0, 1, 1, 0, 0};
    check_pack_row("pack-qp", ids, mask, types, pos,
                   want_b1_ids, want_b1_mask, want_b1_types, 8);

    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_ERROR, 6),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // Query longer than doc, budget 2: longest-first gives each side 1;
    // query-priority starves the doc entirely (0 tokens) but both SEPs stay.
    const char ql[] = "hello world helloing"; // 4 tokens: 4 5 4 6
    const char ds[] = "hello";                // 1 token: 4
    CHECK_EQ(wordpiece_pack_pair(v, ql, sizeof(ql) - 1, ds, sizeof(ds) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 5), WORDPIECE_OK);
    const int32_t want_c0_ids[8] = {2, 4, 3, 4, 3, 0, 0, 0};
    const int32_t want_c0_mask[8] = {1, 1, 1, 1, 1, 0, 0, 0};
    const int32_t want_c0_types[8] = {0, 0, 0, 1, 1, 0, 0, 0};
    check_pack_row("pack-lf-split", ids, mask, types, pos,
                   want_c0_ids, want_c0_mask, want_c0_types, 8);

    CHECK_EQ(wordpiece_pack_pair(v, ql, sizeof(ql) - 1, ds, sizeof(ds) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_QUERY_PRIORITY, 5), WORDPIECE_OK);
    const int32_t want_c1_ids[8] = {2, 4, 5, 3, 3, 0, 0, 0};
    const int32_t want_c1_mask[8] = {1, 1, 1, 1, 1, 0, 0, 0};
    const int32_t want_c1_types[8] = {0, 0, 0, 0, 1, 0, 0, 0};
    check_pack_row("pack-qp-starve", ids, mask, types, pos,
                   want_c1_ids, want_c1_mask, want_c1_types, 8);

    CHECK_EQ(wordpiece_pack_pair(v, ql, sizeof(ql) - 1, ds, sizeof(ds) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_ERROR, 5),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // max_length below the 3-special floor is an invalid argument, as is an
    // unknown truncation mode.
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 2),
             WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4, 3u, 8),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // max_length 0 means "use seq": seq 7 fits the pair exactly with no PAD.
    int32_t ids7[7], mask7[7], types7[7], pos7[7];
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids7, mask7, types7, pos7, 7, 7, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 0), WORDPIECE_OK);
    const int32_t want_e_ids[7] = {2, 4, 5, 3, 4, 6, 3};
    const int32_t want_e_mask[7] = {1, 1, 1, 1, 1, 1, 1};
    const int32_t want_e_types[7] = {0, 0, 0, 0, 1, 1, 1};
    check_pack_row("pack-maxlen0", ids7, mask7, types7, pos7,
                   want_e_ids, want_e_mask, want_e_types, 7);

    // max_length above seq clamps to seq: seq 6 reproduces the budget-3 split.
    int32_t ids6[6], mask6[6], types6[6], pos6[6];
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids6, mask6, types6, pos6, 6, 6, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 16), WORDPIECE_OK);
    const int32_t want_f_ids[6] = {2, 4, 3, 4, 6, 3};
    const int32_t want_f_mask[6] = {1, 1, 1, 1, 1, 1};
    const int32_t want_f_types[6] = {0, 0, 0, 1, 1, 1};
    check_pack_row("pack-clamp", ids6, mask6, types6, pos6,
                   want_f_ids, want_f_mask, want_f_types, 6);

    // Empty query packs as [CLS] [SEP] doc [SEP] with doc types from slot 2.
    int32_t idsg[8], maskg[8], typesg[8], posg[8];
    CHECK_EQ(wordpiece_pack_pair(v, "", 0, ds, sizeof(ds) - 1,
                                 idsg, maskg, typesg, posg, 8, 8, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 0), WORDPIECE_OK);
    const int32_t want_g_ids[8] = {2, 3, 4, 3, 0, 0, 0, 0};
    const int32_t want_g_mask[8] = {1, 1, 1, 1, 0, 0, 0, 0};
    const int32_t want_g_types[8] = {0, 0, 1, 1, 0, 0, 0, 0};
    check_pack_row("pack-empty-q", idsg, maskg, typesg, posg,
                   want_g_ids, want_g_mask, want_g_types, 8);

    // seq below the 3-special floor and NULL text with nonzero length fail.
    CHECK_EQ(wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 2, 2, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 0),
             WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(wordpiece_pack_pair(v, nullptr, 3, d, sizeof(d) - 1,
                                 ids, mask, types, pos, 8, 8, 4,
                                 WORDPIECE_TRUNC_LONGEST_FIRST, 0),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    wordpiece_vocab_destroy(v);
}

static void test_hot_alloc_burst() {
    wordpiece_vocab *v = load_tiny();
    if (v == nullptr) { return; }

    int32_t ids[16], mask[16], types[16], pos[16];
    int64_t ids8[16];
    size_t n = 0;
    const char q[] = "hello world";
    const char d[] = "helloing";

    // Warmup, then the counter must stay at zero across a mixed burst that
    // exercises tokenize (both widths), encode, and all pack truncation modes
    // over empty, ASCII, punctuation, CJK, accent, and emoji inputs.
    CHECK_EQ(wordpiece_encode_sentence(v, q, sizeof(q) - 1, ids, mask, types, pos, 16, 16, 4),
             WORDPIECE_OK);
    wordpiece_hot_alloc_counter_reset();
    const char *inputs[] = {
        "",
        "hello world",
        "!? ,.",
        "\xE4\xB8\x96\xE7\x95\x8C",
        "caf\xC3\xA9",
        "\xF0\x9F\x98\x80",
        "hello, world caf\xC3\xA9 \xE4\xB8\x96",
    };
    for (int round = 0; round < 32; ++round) {
        for (const char *text : inputs) {
            const size_t len = std::strlen(text);
            CHECK_EQ(wordpiece_tokenize(v, text, len, ids, 16, 4, &n), WORDPIECE_OK);
            CHECK_EQ(wordpiece_tokenize(v, text, len, ids8, 16, 8, &n), WORDPIECE_OK);
            CHECK_EQ(wordpiece_encode_sentence(v, text, len, ids, mask, types, pos, 16, 16, 4),
                     WORDPIECE_OK);
        }
        for (uint32_t mode = 0; mode <= WORDPIECE_TRUNC_ERROR; ++mode) {
            const int st = wordpiece_pack_pair(v, q, sizeof(q) - 1, d, sizeof(d) - 1,
                                               ids, mask, types, pos, 16, 16, 4, mode,
                                               6 + static_cast<uint32_t>(round % 4));
            CHECK(st == WORDPIECE_OK || st == WORDPIECE_ERR_INVALID_ARGUMENT);
        }
    }
    CHECK_EQ(wordpiece_hot_alloc_counter(), 0u);

    wordpiece_vocab_destroy(v);
}

static void test_vocab_load_errors_and_destroy() {
    const std::string root = workspace_root();
    wordpiece_vocab *v = reinterpret_cast<wordpiece_vocab *>(0x1); // poison; must be overwritten

    // Missing file and missing directory both report NOT_FOUND with *out NULL.
    CHECK_EQ(wordpiece_vocab_load((root + "/testdata/turborerank/no-such-vocab.txt").c_str(), &v),
             WORDPIECE_ERR_NOT_FOUND);
    CHECK(v == nullptr);
    CHECK_EQ(wordpiece_vocab_load_dir((root + "/testdata/turborerank/no-such-dir").c_str(), &v),
             WORDPIECE_ERR_NOT_FOUND);
    CHECK(v == nullptr);

    // Empty path and NULL path are invalid arguments.
    CHECK_EQ(wordpiece_vocab_load("", &v), WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK(v == nullptr);
    CHECK_EQ(wordpiece_vocab_load(nullptr, &v), WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK_EQ(wordpiece_vocab_load_dir(nullptr, &v), WORDPIECE_ERR_INVALID_ARGUMENT);

    // NULL out-param is rejected.
    CHECK_EQ(wordpiece_vocab_load((root + "/testdata/turborerank/tiny-vocab/vocab.txt").c_str(),
                                  nullptr),
             WORDPIECE_ERR_INVALID_ARGUMENT);

    // An empty vocab file is malformed (no tokens, no specials).
    const std::string empty_path = "/tmp/wordpiece_edge_empty_vocab.txt";
    { std::ofstream out(empty_path, std::ios::binary | std::ios::trunc); }
    CHECK_EQ(wordpiece_vocab_load(empty_path.c_str(), &v), WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK(v == nullptr);
    std::remove(empty_path.c_str());

    // A vocab containing invalid UTF-8 is malformed.
    const std::string badutf_path = "/tmp/wordpiece_edge_badutf_vocab.txt";
    {
        std::ofstream out(badutf_path, std::ios::binary | std::ios::trunc);
        out << "[PAD]\n[UNK]\n[CLS]\n[SEP]\n" << static_cast<char>(0xFF) << "\n";
    }
    CHECK_EQ(wordpiece_vocab_load(badutf_path.c_str(), &v), WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK(v == nullptr);
    std::remove(badutf_path.c_str());

    // A well-formed UTF-8 vocab missing the special tokens is unsupported.
    const std::string nospecial_path = "/tmp/wordpiece_edge_nospecial_vocab.txt";
    {
        std::ofstream out(nospecial_path, std::ios::binary | std::ios::trunc);
        out << "hello\nworld\n";
    }
    CHECK_EQ(wordpiece_vocab_load(nospecial_path.c_str(), &v), WORDPIECE_ERR_INVALID_ARGUMENT);
    CHECK(v == nullptr);
    std::remove(nospecial_path.c_str());

    // Destroy tolerates NULL; a live handle reports loaded until destroyed.
    wordpiece_vocab_destroy(nullptr);
    CHECK(wordpiece_vocab_is_loaded(nullptr) == 0);
    v = nullptr;
    CHECK_EQ(wordpiece_vocab_load(
                 (root + "/testdata/turborerank/tiny-vocab/vocab.txt").c_str(), &v),
             WORDPIECE_OK);
    CHECK(wordpiece_vocab_is_loaded(v) == 1);
    CHECK_EQ(wordpiece_unk_id(v), 1);
    CHECK_EQ(wordpiece_cls_id(v), 2);
    CHECK_EQ(wordpiece_sep_id(v), 3);
    CHECK_EQ(wordpiece_pad_id(v), 0);
    wordpiece_vocab_destroy(v);
}

static void test_minilm_parity() {
    const std::string path =
        workspace_root() + "/models/rerank/ms-marco-minilm-l6/vocab.txt";
    if (!file_exists(path)) {
        std::fprintf(stderr, "SKIP MiniLM vocab (not fetched)\n");
        return;
    }
    wordpiece_vocab *v = nullptr;
    CHECK_EQ(wordpiece_vocab_load(path.c_str(), &v), WORDPIECE_OK);
    CHECK(v != nullptr);

    // BERT-cleaning on the real vocab: a NUL before the space matches the
    // plain sentence exactly.
    const char plain[] = "hello world";
    const char with_nul[] = {'h', 'e', 'l', 'l', 'o', '\0', ' ', 'w', 'o', 'r', 'l', 'd'};
    int32_t a[16], b[16];
    size_t na = 0, nb = 0;
    CHECK_EQ(wordpiece_tokenize(v, plain, sizeof(plain) - 1, a, 16, 4, &na), WORDPIECE_OK);
    CHECK_EQ(wordpiece_tokenize(v, with_nul, sizeof(with_nul), b, 16, 4, &nb), WORDPIECE_OK);
    CHECK_EQ(na, nb);
    CHECK(na > 0);
    check_row_i32("minilm-nul", a, b, na);

    // Count query equals fill count on a longer sentence.
    const char sentence[] = "the quick brown fox jumps over the lazy dog";
    size_t count = 0;
    CHECK_EQ(wordpiece_tokenize(v, sentence, sizeof(sentence) - 1, nullptr, 0, 4, &count),
             WORDPIECE_OK);
    CHECK_EQ(wordpiece_tokenize(v, sentence, sizeof(sentence) - 1, a, 16, 4, &na), WORDPIECE_OK);
    CHECK_EQ(count, na);
    CHECK(count > 4); // genuinely multi-token on the real vocab

    // Width 4 vs width 8 produce identical sequences on the real vocab.
    int64_t a8[16];
    size_t n8 = 0;
    CHECK_EQ(wordpiece_tokenize(v, sentence, sizeof(sentence) - 1, a8, 16, 8, &n8), WORDPIECE_OK);
    CHECK_EQ(na, n8);
    for (size_t i = 0; i < na; ++i) { CHECK_EQ(a[i], static_cast<int32_t>(a8[i])); }

    wordpiece_vocab_destroy(v);
}

int main() {
    test_empty_and_whitespace();
    test_embedded_nul_and_controls();
    test_unicode_tiny_vocab();
    test_count_query_and_cap_canaries();
    test_elem_width_parity();
    test_encode_sentence_layout();
    test_pack_pair_truncation();
    test_hot_alloc_burst();
    test_vocab_load_errors_and_destroy();
    test_minilm_parity();
    std::fprintf(stderr, "wordpiece_edge_tests: %d pass, %d fail\n", g_passes, g_fails);
    return g_fails == 0 ? 0 : 1;
}
