/* SPDX-License-Identifier: Apache-2.0
 *
 * Frozen BERT WordPiece — write-through into caller memory.
 *
 * Vocab is a mmap / one-shot image built at load. Encode / pack write
 * i32 or i64 ids (and mask / type / pos) directly into caller buffers
 * (turbo_buffer rented rows). The hot path after load does not allocate
 * a token vector, a std::string scratch list, or an unordered_map.
 *
 * Canonical path: include/wordpiece.h
 * Apple copy (keep identical): swift/Sources/WordPieceC/include/wordpiece.h
 */

#ifndef WORDPIECE_H
#define WORDPIECE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define WORDPIECE_OK 0
#define WORDPIECE_ERR_INVALID_ARGUMENT 1
#define WORDPIECE_ERR_NOT_FOUND 2
#define WORDPIECE_ERR_INTERNAL 3

/** 0 = longest-first, 1 = query-priority, 2 = error if pair exceeds budget. */
#define WORDPIECE_TRUNC_LONGEST_FIRST 0u
#define WORDPIECE_TRUNC_QUERY_PRIORITY 1u
#define WORDPIECE_TRUNC_ERROR 2u

typedef struct wordpiece_vocab wordpiece_vocab;

/** Load vocab.txt (mmap) or tokenizer.json (WordPiece vocab object). */
int wordpiece_vocab_load(const char *path, wordpiece_vocab **out);

/** First existing of `dir/vocab.txt` then `dir/tokenizer.json`. */
int wordpiece_vocab_load_dir(const char *dir, wordpiece_vocab **out);

void wordpiece_vocab_destroy(wordpiece_vocab *v);

int wordpiece_vocab_is_loaded(const wordpiece_vocab *v);
int32_t wordpiece_unk_id(const wordpiece_vocab *v);
int32_t wordpiece_cls_id(const wordpiece_vocab *v);
int32_t wordpiece_sep_id(const wordpiece_vocab *v);
int32_t wordpiece_pad_id(const wordpiece_vocab *v);

/**
 * WordPiece without specials. Writes up to `ids_cap` ids.
 * `elem_width` is 4 (i32) or 8 (i64). `ids` may be NULL to count only.
 */
int wordpiece_tokenize(
    const wordpiece_vocab *v,
    const char *utf8,
    size_t utf8_len,
    void *ids,
    size_t ids_cap,
    uint32_t elem_width,
    size_t *n_out
);

/**
 * Single-sequence BERT encode into one row:
 * [CLS] tokens [SEP] [PAD…]. Mask 1 on non-pad, types 0, pos = arange.
 * `stride` is elements (often == seq). `pos` may be NULL.
 */
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
);

/**
 * Cross-encoder pair into one row:
 * [CLS] query [SEP] doc [SEP] [PAD…].
 * Query type=0, doc type=1. Writes directly into the destination row.
 */
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
);

/** Tests: encode/pack must leave this at 0 after warmup. */
uint64_t wordpiece_hot_alloc_counter(void);
void wordpiece_hot_alloc_counter_reset(void);
/** Call only if a hot-path heap token staging is reintroduced. */
void wordpiece_note_hot_alloc(void);

#ifdef __cplusplus
}
#endif

#endif /* WORDPIECE_H */
