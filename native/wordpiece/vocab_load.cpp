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
            s.id = id;
            return true;
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

void set_specials(wordpiece_vocab *v) {
    const int32_t unk = lookup_special(v, "[UNK]");
    const int32_t cls = lookup_special(v, "[CLS]");
    const int32_t sep = lookup_special(v, "[SEP]");
    const int32_t pad = lookup_special(v, "[PAD]");
    if (unk >= 0) {
        v->unk_id = unk;
    }
    if (cls >= 0) {
        v->cls_id = cls;
    }
    if (sep >= 0) {
        v->sep_id = sep;
    }
    if (pad >= 0) {
        v->pad_id = pad;
    }
}

bool alloc_slots(wordpiece_vocab *v, uint32_t n_tokens) {
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
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
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

    uint32_t n_tok = 0;
    for (size_t i = 0; i < v->blob_size; ++i) {
        if (v->blob[i] == '\n') {
            ++n_tok;
        }
    }
    if (v->blob_size > 0 && v->blob[v->blob_size - 1] != '\n') {
        ++n_tok;
    }
    if (n_tok == 0 || !alloc_slots(v, n_tok)) {
        destroy_image(v);
        return false;
    }

    size_t start = 0;
    int32_t id = 0;
    for (size_t i = 0; i <= v->blob_size; ++i) {
        if (i == v->blob_size || v->blob[i] == '\n') {
            size_t end = i;
            if (end > start && v->blob[end - 1] == '\r') {
                --end;
            }
            const size_t len = end - start;
            if (len > 0xffffu) {
                destroy_image(v);
                return false;
            }
            if (!insert_slot(v, static_cast<uint32_t>(start), static_cast<uint16_t>(len), id)) {
                destroy_image(v);
                return false;
            }
            ++id;
            start = i + 1;
        }
    }
    set_specials(v);
    v->loaded = 1;
    *out = v;
    return true;
#endif
}

int hex_val(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }
    return -1;
}

bool append_utf8(std::string *out, uint32_t cp) {
    if (cp < 0x80) {
        out->push_back(static_cast<char>(cp));
    } else if (cp < 0x800) {
        out->push_back(static_cast<char>(0xC0 | (cp >> 6)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp < 0x10000) {
        out->push_back(static_cast<char>(0xE0 | (cp >> 12)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out->push_back(static_cast<char>(0xF0 | (cp >> 18)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out->push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return true;
}

bool parse_json_string(const std::string &text, size_t *i, std::string *out) {
    if (*i >= text.size() || text[*i] != '"') {
        return false;
    }
    ++*i;
    out->clear();
    while (*i < text.size()) {
        const char c = text[*i];
        ++*i;
        if (c == '"') {
            return true;
        }
        if (c != '\\') {
            out->push_back(c);
            continue;
        }
        if (*i >= text.size()) {
            return false;
        }
        const char e = text[*i];
        ++*i;
        switch (e) {
        case '"':
        case '\\':
        case '/':
            out->push_back(e);
            break;
        case 'b':
            out->push_back('\b');
            break;
        case 'f':
            out->push_back('\f');
            break;
        case 'n':
            out->push_back('\n');
            break;
        case 'r':
            out->push_back('\r');
            break;
        case 't':
            out->push_back('\t');
            break;
        case 'u': {
            if (*i + 4 > text.size()) {
                return false;
            }
            uint32_t cp = 0;
            for (int k = 0; k < 4; ++k) {
                const int h = hex_val(text[*i + k]);
                if (h < 0) {
                    return false;
                }
                cp = (cp << 4) | static_cast<uint32_t>(h);
            }
            *i += 4;
            append_utf8(out, cp);
            break;
        }
        default:
            return false;
        }
    }
    return false;
}

bool skip_ws(const std::string &text, size_t *i) {
    while (*i < text.size()) {
        const char c = text[*i];
        if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
            return true;
        }
        ++*i;
    }
    return false;
}

bool parse_int(const std::string &text, size_t *i, int32_t *out) {
    if (!skip_ws(text, i)) {
        return false;
    }
    bool neg = false;
    if (text[*i] == '-') {
        neg = true;
        ++*i;
    }
    if (*i >= text.size() || text[*i] < '0' || text[*i] > '9') {
        return false;
    }
    int64_t v = 0;
    while (*i < text.size() && text[*i] >= '0' && text[*i] <= '9') {
        v = v * 10 + (text[*i] - '0');
        ++*i;
    }
    *out = static_cast<int32_t>(neg ? -v : v);
    return true;
}

size_t find_wordpiece_vocab_object(const std::string &text) {
    const char *needles[] = {"\"type\":\"WordPiece\"", "\"type\": \"WordPiece\""};
    size_t from = 0;
    for (const char *n : needles) {
        const size_t p = text.find(n);
        if (p != std::string::npos) {
            from = p;
            break;
        }
    }
    size_t pos = text.find("\"vocab\"", from);
    if (pos == std::string::npos) {
        pos = text.find("\"vocab\"");
    }
    if (pos == std::string::npos) {
        return std::string::npos;
    }
    size_t i = pos + 7;
    if (!skip_ws(text, &i) || i >= text.size() || text[i] != ':') {
        return std::string::npos;
    }
    ++i;
    if (!skip_ws(text, &i) || i >= text.size() || text[i] != '{') {
        return std::string::npos;
    }
    return i;
}

bool load_tokenizer_json(const char *path, wordpiece_vocab **out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return false;
    }
    std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    const size_t obj = find_wordpiece_vocab_object(text);
    if (obj == std::string::npos) {
        return false;
    }
    size_t i = obj + 1;
    struct Pair {
        std::string tok;
        int32_t id;
    };
    std::vector<Pair> pairs;
    pairs.reserve(30522);
    while (i < text.size()) {
        if (!skip_ws(text, &i)) {
            return false;
        }
        if (text[i] == '}') {
            break;
        }
        if (text[i] == ',') {
            ++i;
            continue;
        }
        Pair p;
        if (!parse_json_string(text, &i, &p.tok)) {
            return false;
        }
        if (!skip_ws(text, &i) || text[i] != ':') {
            return false;
        }
        ++i;
        if (!parse_int(text, &i, &p.id)) {
            return false;
        }
        pairs.push_back(std::move(p));
    }
    if (pairs.empty()) {
        return false;
    }

    size_t blob_n = 0;
    for (const auto &p : pairs) {
        blob_n += p.tok.size();
    }
    auto *v = new (std::nothrow) wordpiece_vocab{};
    if (v == nullptr) {
        return false;
    }
    char *blob = new (std::nothrow) char[blob_n ? blob_n : 1];
    if (blob == nullptr || !alloc_slots(v, static_cast<uint32_t>(pairs.size()))) {
        delete[] blob;
        destroy_image(v);
        return false;
    }
    v->blob = blob;
    v->blob_size = blob_n;
    v->blob_mmap = 0;
    size_t off = 0;
    for (const auto &p : pairs) {
        if (p.tok.size() > 0xffffu) {
            destroy_image(v);
            return false;
        }
        if (!p.tok.empty()) {
            std::memcpy(blob + off, p.tok.data(), p.tok.size());
        }
        if (!insert_slot(
                v, static_cast<uint32_t>(off), static_cast<uint16_t>(p.tok.size()), p.id
            )) {
            destroy_image(v);
            return false;
        }
        off += p.tok.size();
    }
    set_specials(v);
    v->loaded = 1;
    *out = v;
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
    if (path == nullptr || out == nullptr || path[0] == '\0') {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    const bool json = ends_with(path, ".json");
    const bool ok = json ? load_tokenizer_json(path, out) : load_vocab_txt(path, out);
    if (!ok || *out == nullptr) {
        return WORDPIECE_ERR_NOT_FOUND;
    }
    return WORDPIECE_OK;
}

int wordpiece_vocab_load_dir(const char *dir, wordpiece_vocab **out) {
    if (dir == nullptr || out == nullptr || dir[0] == '\0') {
        return WORDPIECE_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    const std::string base(dir);
    const std::string txt = base + "/vocab.txt";
    if (file_exists(txt)) {
        return wordpiece_vocab_load(txt.c_str(), out);
    }
    const std::string js = base + "/tokenizer.json";
    if (file_exists(js)) {
        return wordpiece_vocab_load(js.c_str(), out);
    }
    return WORDPIECE_ERR_NOT_FOUND;
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
