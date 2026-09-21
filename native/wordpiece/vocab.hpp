// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "wordpiece.h"

#include <cstddef>
#include <cstdint>

namespace wordpiece {
namespace impl {

struct Slot {
    uint32_t off = 0;
    uint16_t len = 0;
    uint16_t occupied = 0;
    int32_t id = 0;
};

} // namespace impl
} // namespace wordpiece

struct wordpiece_vocab {
    const char *blob = nullptr;
    size_t blob_size = 0;
    wordpiece::impl::Slot *slots = nullptr;
    uint32_t n_slots = 0;
    uint32_t mask = 0;
    int32_t unk_id = 100;
    int32_t cls_id = 101;
    int32_t sep_id = 102;
    int32_t pad_id = 0;
    int32_t mask_id = -1;
    // Bit order: PAD, UNK, CLS, SEP, MASK; matched before normalization.
    uint8_t added_specials = 0;
    // BertNormalizer settings from tokenizer.json: uncased models lowercase
    // and strip accents; cased models do neither.
    uint8_t lowercase = 1;
    uint8_t strip_accents = 1;
    int loaded = 0;
    int fd = -1;
    int blob_mmap = 0;
    void *map = nullptr;
    size_t map_size = 0;
};

// Internal C++ entry for a bundle already captured and hash-verified in memory.
int wordpiece_vocab_load_json_bytes(const char *, size_t, wordpiece_vocab **);
