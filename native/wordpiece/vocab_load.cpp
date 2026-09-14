// SPDX-License-Identifier: Apache-2.0
//
// Create-time vocab image. Heap / mmap here is allowed. The hot path
// in encode.cpp only reads the frozen slots + blob.

#include "vocab.hpp"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <new>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <memory>
#include <set>
#include "nlohmann/json.hpp"
#include "utf8proc/utf8proc.h"

#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace {

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

uint32_t next_pow2(uint32_t n) {
    uint32_t p = 8;
    while (p < n) {
        p <<= 1;
        if (p == 0) {
            return 0;
        }
    }
    return p;
}

bool insert_slot(
    wordpiece_vocab *v,
    uint32_t off,
    uint16_t len,
    int32_t id
) {
    if (v->slots == nullptr || v->n_slots == 0 || !v->blob) {
        return false;
    }
    const uint64_t h = fnv1a(v->blob + off, len, nullptr, 0);
    for (uint32_t i = 0; i < v->n_slots; ++i) {
        const uint32_t idx = static_cast<uint32_t>((h + i) & v->mask);
        wordpiece::impl::Slot &s = v->slots[idx];
        if (!s.occupied) {
            s.off = off;
            s.len = len;
            s.occupied = 1;
            s.id = id;
            return true;
        }
        if (s.len == len && std::memcmp(v->blob + s.off, v->blob + off, len) == 0) {
            return false; // Duplicate vocabulary token.
        }
    }
    return false;
}

int32_t lookup_special(const wordpiece_vocab *v, const char *tok) {
    const size_t n = std::strlen(tok);
    const uint64_t h = fnv1a(tok, n, nullptr, 0);
    for (uint32_t i = 0; i < v->n_slots; ++i) {
        const uint32_t idx = static_cast<uint32_t>((h + i) & v->mask);
        const wordpiece::impl::Slot &s = v->slots[idx];
        if (!s.occupied) {
            return -1;
        }
        if (s.len == n && std::memcmp(v->blob + s.off, tok, n) == 0) {
            return s.id;
        }
    }
    return -1;
}

bool set_specials(wordpiece_vocab *v) {
    v->unk_id = lookup_special(v, "[UNK]");
    v->cls_id = lookup_special(v, "[CLS]");
    v->sep_id = lookup_special(v, "[SEP]");
    v->pad_id = lookup_special(v, "[PAD]");
    v->mask_id = lookup_special(v, "[MASK]");
    return v->unk_id >= 0 && v->cls_id >= 0 && v->sep_id >= 0 && v->pad_id >= 0;
}

bool alloc_slots(wordpiece_vocab *v, uint32_t n_tokens) {
    if (n_tokens > (UINT32_MAX - 8u) / 2u) {
        return false;
    }
    uint32_t n = next_pow2(n_tokens * 2u + 8u);
    if (n == 0) {
        return false;
    }
    v->slots = new (std::nothrow) wordpiece::impl::Slot[n];
    if (v->slots == nullptr) {
        return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        v->slots[i] = wordpiece::impl::Slot{};
    }
    v->n_slots = n;
    v->mask = n - 1;
    return true;
}

void destroy_image(wordpiece_vocab *v) {
    if (v == nullptr) {
        return;
    }
    delete[] v->slots;
    v->slots = nullptr;
#ifndef _WIN32
    if (v->blob_mmap && v->map != nullptr && v->map != MAP_FAILED) {
        munmap(v->map, v->map_size);
    }
    if (v->fd >= 0) {
        close(v->fd);
    }
#endif
    if (!v->blob_mmap && v->blob != nullptr) {
        delete[] const_cast<char *>(v->blob);
    }
    delete v;
}

using Image = std::unique_ptr<wordpiece_vocab, decltype(&destroy_image)>;

bool valid_utf8(const char *text, size_t size) {
    size_t i = 0;
    while (i < size) {
        int32_t cp = 0;
        const auto n = utf8proc_iterate(
            reinterpret_cast<const uint8_t *>(text + i),
            static_cast<utf8proc_ssize_t>(size - i), &cp);
        if (n < 1) { return false; }
        i += static_cast<size_t>(n);
    }
    return true;
}

bool load_vocab_txt(const char *path, wordpiece_vocab **out) {
#ifdef _WIN32
    (void)path;
    (void)out;
    return false;
#else
    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size <= 0 || static_cast<uint64_t>(st.st_size) > UINT32_MAX) {
        close(fd);
        return false;
    }
    void *map = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        close(fd);
        return false;
    }

    auto *v = new (std::nothrow) wordpiece_vocab{};
    if (v == nullptr) {
        munmap(map, static_cast<size_t>(st.st_size));
        close(fd);
        return false;
    }
    v->blob = static_cast<const char *>(map);
    v->blob_size = static_cast<size_t>(st.st_size);
    v->map = map;
    v->map_size = v->blob_size;
    v->fd = fd;
    v->blob_mmap = 1;

    Image image(v, &destroy_image);
    uint32_t n_tok = 0;
    for (size_t i = 0; i < v->blob_size; ++i) {
        if (v->blob[i] == '\n') {
            ++n_tok;
        }
    }
    if (v->blob_size > 0 && v->blob[v->blob_size - 1] != '\n') {
        ++n_tok;
    }
    if (n_tok == 0 || n_tok > INT32_MAX || !alloc_slots(v, n_tok)) {
        return false;
    }

    size_t start = 0;
    int32_t id = 0;
    for (size_t i = 0; i <= v->blob_size; ++i) {
        if ((i == v->blob_size && start < i) || (i < v->blob_size && v->blob[i] == '\n')) {
            size_t end = i;
            if (end > start && v->blob[end - 1] == '\r') {
                --end;
            }
            const size_t len = end - start;
            if (len > 0xffffu || !valid_utf8(v->blob + start, len)) {
                return false;
            }
            if (!insert_slot(v, static_cast<uint32_t>(start), static_cast<uint16_t>(len), id)) {
                return false;
            }
            ++id;
            start = i + 1;
        }
    }
    if (!set_specials(v)) {
        return false;
    }
    v->added_specials = v->mask_id >= 0 ? 31 : 15;
    v->loaded = 1;
    *out = image.release();
    return true;
#endif
}

using Json = nlohmann::json;

bool valid_id(const Json &id) {
    return id.is_number_integer() && id >= 0 && id <= INT32_MAX;
}

bool supported_processor(const Json &p, const wordpiece_vocab *v) {
    const Json bert = {{"type", "BertProcessing"},
        {"cls", {"[CLS]", v->cls_id}}, {"sep", {"[SEP]", v->sep_id}}};
    if (p == bert) { return true; }
    auto special = [](const char *id, int type) {
        return Json{{"SpecialToken", {{"id", id}, {"type_id", type}}}};
    };
    auto sequence = [](const char *id, int type) {
        return Json{{"Sequence", {{"id", id}, {"type_id", type}}}};
    };
    auto record = [](const char *token, int id) {
        return Json{{"id", token}, {"ids", Json::array({id})},
                    {"tokens", Json::array({token})}};
    };
    const Json templ = {
        {"type", "TemplateProcessing"},
        {"single", Json::array({special("[CLS]", 0), sequence("A", 0), special("[SEP]", 0)})},
        {"pair", Json::array({special("[CLS]", 0), sequence("A", 0), special("[SEP]", 0),
                              sequence("B", 1), special("[SEP]", 1)})},
        {"special_tokens", {{"[CLS]", record("[CLS]", v->cls_id)},
                            {"[SEP]", record("[SEP]", v->sep_id)}}}
    };
    return p == templ;
}

bool supported_config(const Json &j, wordpiece_vocab *v) {
    if (j.at("version") != "1.0") { return false; }
    const auto &m = j.at("model");
    if (m.at("type") != "WordPiece" || m.at("unk_token") != "[UNK]" ||
        m.at("continuing_subword_prefix") != "##" ||
        m.at("max_input_chars_per_word") != 100) { return false; }
    auto normal = j.at("normalizer");
    if (normal.at("strip_accents").is_null()) { normal["strip_accents"] = true; }
    if (normal != Json{{"type", "BertNormalizer"}, {"clean_text", true},
            {"handle_chinese_chars", true}, {"strip_accents", true}, {"lowercase", true}} ||
        j.at("pre_tokenizer") != Json{{"type", "BertPreTokenizer"}} ||
        !supported_processor(j.at("post_processor"), v)) { return false; }

    const auto &added = j.at("added_tokens");
    if (!added.is_array()) { return false; }
    const char *names[] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"};
    for (const auto &token : added) {
        bool matched = false;
        for (unsigned i = 0; i < 5; ++i) {
            const int32_t id = lookup_special(v, names[i]);
            if (id >= 0 && token == Json{{"id", id}, {"content", names[i]},
                    {"special", true}, {"single_word", false}, {"lstrip", false},
                    {"rstrip", false}, {"normalized", false}}) {
                if (v->added_specials & (1u << i)) { return false; }
                v->added_specials |= static_cast<uint8_t>(1u << i);
                matched = true;
                break;
            }
        }
        if (!matched) { return false; }
    }
    // Sequence lengths are supplied by the caller; other metadata must match
    // the native right-padding/right-truncation policy.
    const auto &trunc = j.at("truncation");
    if (!trunc.is_null() && (trunc.value("direction", "Right") != "Right" ||
        trunc.at("strategy") != "LongestFirst" || trunc.at("stride") != 0 ||
        !valid_id(trunc.at("max_length")))) { return false; }
    const auto &pad = j.at("padding");
    if (!pad.is_null()) {
        if (pad.at("direction") != "Right" || !pad.at("pad_to_multiple_of").is_null() ||
            pad.at("pad_id") != v->pad_id || pad.at("pad_type_id") != 0 ||
            pad.at("pad_token") != "[PAD]") { return false; }
        const auto &strategy = pad.at("strategy");
        if (strategy != "BatchLongest" &&
            !(strategy.is_object() && strategy.size() == 1 &&
              strategy.contains("Fixed") && valid_id(strategy["Fixed"]))) { return false; }
    }
    return true;
}

bool load_tokenizer_json(const char *path, wordpiece_vocab **out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { return false; }
    // Reject duplicate keys rather than allowing a later value to shadow
    // tokenizer configuration inspected by other consumers of the bundle.
    std::vector<std::set<std::string>> keys;
    const auto callback = [&keys](int, Json::parse_event_t event, Json &value) {
        if (event == Json::parse_event_t::object_start) { keys.emplace_back(); }
        else if (event == Json::parse_event_t::object_end) { keys.pop_back(); }
        else if (event == Json::parse_event_t::key &&
                 !keys.back().insert(value.get<std::string>()).second) {
            throw std::invalid_argument("duplicate tokenizer key");
        }
        return true;
    };
    const auto j = Json::parse(in, callback);
    const auto &vocab = j.at("model").at("vocab");
    if (!vocab.is_object() || vocab.empty() || vocab.size() > INT32_MAX) { return false; }
    size_t blob_n = 0;
    std::set<int32_t> ids;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        if (it.key().empty() || it.key().size() > UINT16_MAX ||
            !valid_utf8(it.key().data(), it.key().size()) || !valid_id(it.value()) ||
            !ids.insert(it.value().get<int32_t>()).second ||
            it.key().size() > UINT32_MAX - blob_n) { return false; }
        blob_n += it.key().size();
    }
    Image image(new wordpiece_vocab{}, &destroy_image);
    auto *v = image.get();
    v->blob = new char[blob_n];
    v->blob_size = blob_n;
    if (!alloc_slots(v, static_cast<uint32_t>(vocab.size()))) { return false; }
    size_t off = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        std::memcpy(const_cast<char *>(v->blob) + off, it.key().data(), it.key().size());
        if (!insert_slot(v, static_cast<uint32_t>(off),
                static_cast<uint16_t>(it.key().size()), it.value().get<int32_t>())) { return false; }
        off += it.key().size();
    }
    if (!set_specials(v) || !supported_config(j, v)) { return false; }
    v->loaded = 1;
    *out = image.release();
    return true;
}

bool ends_with(const char *path, const char *suf) {
    const size_t n = std::strlen(path);
    const size_t m = std::strlen(suf);
    return n >= m && std::memcmp(path + n - m, suf, m) == 0;
}

bool file_exists(const std::string &p) {
    std::ifstream in(p);
    return static_cast<bool>(in);
}

} // namespace

extern "C" {

int wordpiece_vocab_load(const char *path, wordpiece_vocab **out) {
    if (out == nullptr) { return WORDPIECE_ERR_INVALID_ARGUMENT; }
    *out = nullptr;
    if (path == nullptr || path[0] == '\0') { return WORDPIECE_ERR_INVALID_ARGUMENT; }
    try {
        if (!file_exists(path)) { return WORDPIECE_ERR_NOT_FOUND; }
        const bool json = ends_with(path, ".json");
        const bool ok = json ? load_tokenizer_json(path, out) : load_vocab_txt(path, out);
        return ok && *out != nullptr ? WORDPIECE_OK : WORDPIECE_ERR_INVALID_ARGUMENT;
    } catch (const std::bad_alloc &) {
        return WORDPIECE_ERR_INTERNAL;
    } catch (...) {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
}

int wordpiece_vocab_load_dir(const char *dir, wordpiece_vocab **out) {
    if (out == nullptr) { return WORDPIECE_ERR_INVALID_ARGUMENT; }
    *out = nullptr;
    if (dir == nullptr || dir[0] == '\0') { return WORDPIECE_ERR_INVALID_ARGUMENT; }
    try {
        const std::string base(dir);
        const std::string txt = base + "/vocab.txt";
        if (file_exists(txt)) { return wordpiece_vocab_load(txt.c_str(), out); }
        const std::string js = base + "/tokenizer.json";
        return wordpiece_vocab_load(js.c_str(), out);
    } catch (...) {
        return WORDPIECE_ERR_INTERNAL;
    }
}

void wordpiece_vocab_destroy(wordpiece_vocab *v) {
    destroy_image(v);
}

int wordpiece_vocab_is_loaded(const wordpiece_vocab *v) {
    return v != nullptr && v->loaded;
}

int32_t wordpiece_unk_id(const wordpiece_vocab *v) {
    return v ? v->unk_id : 100;
}
int32_t wordpiece_cls_id(const wordpiece_vocab *v) {
    return v ? v->cls_id : 101;
}
int32_t wordpiece_sep_id(const wordpiece_vocab *v) {
    return v ? v->sep_id : 102;
}
int32_t wordpiece_pad_id(const wordpiece_vocab *v) {
    return v ? v->pad_id : 0;
}

} // extern "C"
