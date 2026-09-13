// SPDX-License-Identifier: Apache-2.0
//
// TurboRerank wrappers over the frozen WordPiece image. Pack/tokenize
// write directly into caller / arena rows — no heap token vector.

#include "internal.hpp"
#include "wordpiece.h"

namespace turborerank {
namespace impl {

void free_vocab(Vocab *vocab) {
    if (vocab == nullptr) {
        return;
    }
    if (vocab->img != nullptr) {
        wordpiece_vocab_destroy(vocab->img);
        vocab->img = nullptr;
    }
    vocab->loaded = false;
}

bool load_vocab_txt(const char *path, Vocab *vocab, std::string *err) {
    if (path == nullptr || vocab == nullptr) {
        if (err) {
            *err = "load_vocab_txt: null argument";
        }
        return false;
    }
    free_vocab(vocab);
    wordpiece_vocab *img = nullptr;
    const int st = wordpiece_vocab_load(path, &img);
    if (st != WORDPIECE_OK || img == nullptr) {
        if (err) {
            *err = std::string("vocab not loaded: ") + path;
        }
        return false;
    }
    vocab->img = img;
    vocab->unk_id = wordpiece_unk_id(img);
    vocab->cls_id = wordpiece_cls_id(img);
    vocab->sep_id = wordpiece_sep_id(img);
    vocab->pad_id = wordpiece_pad_id(img);
    vocab->loaded = true;
    return true;
}

size_t tokenize_wordpiece(
    const Vocab &vocab,
    const char *utf8,
    size_t utf8_len,
    int32_t *ids,
    size_t ids_cap,
    std::string *err
) {
    if (!vocab.loaded || vocab.img == nullptr) {
        if (err) {
            *err = "tokenize: vocab not loaded";
        }
        return 0;
    }
    if (ids == nullptr || ids_cap == 0) {
        if (err) {
            *err = "tokenize: null id buffer";
        }
        return 0;
    }
    size_t n = 0;
    const int st = wordpiece_tokenize(
        vocab.img, utf8, utf8_len, ids, ids_cap, 4, &n
    );
    if (st != WORDPIECE_OK) {
        if (err) {
            *err = "tokenize: id scratch overflow";
        }
        return 0;
    }
    return n;
}

} // namespace impl
} // namespace turborerank
