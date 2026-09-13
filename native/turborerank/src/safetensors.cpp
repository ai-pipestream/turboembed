// SPDX-License-Identifier: Apache-2.0
//
// Minimal safetensors reader. Header JSON is scanned for named F32
// tensors; the data payload is mmap'd. Misaligned tensors are copied
// once at load into 64-byte storage.

#include "internal.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace turborerank {
namespace impl {
namespace {

bool parse_u64_le(const uint8_t *p, uint64_t *out) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) {
        v |= static_cast<uint64_t>(p[i]) << (8 * i);
    }
    *out = v;
    return true;
}

bool find_key_object(const std::string &header, const std::string &name, std::string *obj) {
    const std::string key = std::string("\"") + name + "\":";
    size_t pos = header.find(key);
    if (pos == std::string::npos) {
        return false;
    }
    pos += key.size();
    while (pos < header.size() && (header[pos] == ' ' || header[pos] == '\n')) {
        ++pos;
    }
    if (pos >= header.size() || header[pos] != '{') {
        return false;
    }
    int depth = 0;
    size_t start = pos;
    for (; pos < header.size(); ++pos) {
        if (header[pos] == '{') {
            ++depth;
        } else if (header[pos] == '}') {
            --depth;
            if (depth == 0) {
                *obj = header.substr(start, pos - start + 1);
                return true;
            }
        }
    }
    return false;
}

bool parse_offsets(const std::string &obj, uint64_t *a, uint64_t *b) {
    const size_t p = obj.find("\"data_offsets\"");
    if (p == std::string::npos) {
        return false;
    }
    const size_t lb = obj.find('[', p);
    const size_t rb = obj.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) {
        return false;
    }
    const std::string inner = obj.substr(lb + 1, rb - lb - 1);
    size_t comma = inner.find(',');
    if (comma == std::string::npos) {
        return false;
    }
    *a = std::strtoull(inner.c_str(), nullptr, 10);
    *b = std::strtoull(inner.c_str() + comma + 1, nullptr, 10);
    return *b >= *a;
}

bool parse_shape2(const std::string &obj, uint32_t *r, uint32_t *c) {
    const size_t p = obj.find("\"shape\"");
    if (p == std::string::npos) {
        return false;
    }
    const size_t lb = obj.find('[', p);
    const size_t rb = obj.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) {
        return false;
    }
    const std::string inner = obj.substr(lb + 1, rb - lb - 1);
    std::vector<uint32_t> dims;
    size_t i = 0;
    while (i < inner.size()) {
        while (i < inner.size() && (inner[i] == ' ' || inner[i] == ',')) {
            ++i;
        }
        if (i >= inner.size()) {
            break;
        }
        char *end = nullptr;
        const unsigned long v = std::strtoul(inner.c_str() + i, &end, 10);
        dims.push_back(static_cast<uint32_t>(v));
        i = static_cast<size_t>(end - inner.c_str());
    }
    if (dims.empty()) {
        return false;
    }
    if (dims.size() == 1) {
        *r = 1;
        *c = dims[0];
        return true;
    }
    *r = dims[0];
    *c = dims[1];
    return true;
}

bool parse_dtype_f32(const std::string &obj) {
    return obj.find("\"F32\"") != std::string::npos ||
           obj.find("\"f32\"") != std::string::npos;
}

bool load_tensor(
    const std::string &header,
    const uint8_t *data,
    size_t data_len,
    const std::string &name,
    TensorView *view,
    std::vector<OwnedFloat> *owned,
    std::string *err
) {
    std::string obj;
    if (!find_key_object(header, name, &obj)) {
        if (err) {
            *err = "safetensors missing tensor: " + name;
        }
        return false;
    }
    if (!parse_dtype_f32(obj)) {
        if (err) {
            *err = "safetensors tensor is not F32: " + name;
        }
        return false;
    }
    uint64_t off0 = 0, off1 = 0;
    uint32_t rows = 0, cols = 0;
    if (!parse_offsets(obj, &off0, &off1) || !parse_shape2(obj, &rows, &cols)) {
        if (err) {
            *err = "safetensors bad shape/offsets: " + name;
        }
        return false;
    }
    if (off1 > data_len || off1 < off0) {
        if (err) {
            *err = "safetensors offsets out of range: " + name;
        }
        return false;
    }
    const size_t nbytes = static_cast<size_t>(off1 - off0);
    const size_t expect = static_cast<size_t>(rows) * static_cast<size_t>(cols) * sizeof(float);
    if (nbytes < expect) {
        if (err) {
            *err = "safetensors size mismatch: " + name;
        }
        return false;
    }
    const uint8_t *src = data + off0;
    const uintptr_t addr = reinterpret_cast<uintptr_t>(src);
    if ((addr % 64) == 0) {
        view->data = reinterpret_cast<const float *>(src);
        view->owned = false;
    } else {
        Status st = Status::Ok;
        float *copy = static_cast<float *>(
            aligned_alloc_bytes(expect, kCpuAlignment, &st)
        );
        if (copy == nullptr) {
            if (err) {
                *err = "safetensors OOM copying " + name;
            }
            return false;
        }
        std::memcpy(copy, src, expect);
        view->data = copy;
        view->owned = true;
        owned->push_back(OwnedFloat{copy, expect});
    }
    view->rows = rows;
    view->cols = cols;
    return true;
}

bool load_named(
    const std::string &header,
    const uint8_t *data,
    size_t data_len,
    const std::vector<std::string> &names,
    TensorView *view,
    std::vector<OwnedFloat> *owned,
    std::string *err
) {
    std::string last;
    for (const auto &n : names) {
        last.clear();
        if (load_tensor(header, data, data_len, n, view, owned, &last)) {
            return true;
        }
    }
    if (err) {
        *err = last.empty() ? "safetensors missing tensor" : last;
    }
    return false;
}

} // namespace

void free_mapped(MappedFile *map) {
    if (map == nullptr) {
        return;
    }
    if (map->map != nullptr && map->map != MAP_FAILED) {
        munmap(map->map, map->size);
    }
    if (map->fd >= 0) {
        close(map->fd);
    }
    map->map = nullptr;
    map->fd = -1;
    map->size = 0;
}

void free_owned(std::vector<OwnedFloat> *owned) {
    if (owned == nullptr) {
        return;
    }
    for (auto &o : *owned) {
        aligned_free_bytes(o.ptr);
        o.ptr = nullptr;
    }
    owned->clear();
}

void free_scratch(Scratch *s) {
    if (s == nullptr) {
        return;
    }
    if (s->arena != nullptr) {
        turbo_buffer_view *views[] = {
            &s->v_x,
            &s->v_residual,
            &s->v_q,
            &s->v_k,
            &s->v_v,
            &s->v_attn,
            &s->v_ctx,
            &s->v_inter,
            &s->v_tmp,
            &s->v_tok_q,
            &s->v_tok_d,
        };
        for (turbo_buffer_view *v : views) {
            if (v->ptr != nullptr) {
                (void)turbo_buffer_arena_return(s->arena, v);
            }
        }
    }
    *s = Scratch{};
}

bool alloc_scratch(
    Scratch *s,
    turbo_buffer_arena *arena,
    turbo_buffer_placement host_place,
    const BertConfig &cfg,
    std::string *err
) {
    free_scratch(s);
    if (arena == nullptr) {
        if (err) {
            *err = "scratch requires a turbo_buffer arena; refusing private malloc";
        }
        return false;
    }
    const uint32_t B = cfg.max_batch == 0 ? 1 : cfg.max_batch;
    const uint32_t S = cfg.max_position;
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate;
    const uint32_t heads = cfg.heads;
    auto rent_f32 = [&](turbo_buffer_view *v, uint32_t rows, uint32_t cols) -> float * {
        if (turbo_buffer_arena_rent(
                arena, TURBO_BUFFER_DTYPE_F32, host_place, rows, cols, cols, v
            ) != TURBO_BUFFER_OK) {
            return nullptr;
        }
        return turbo_buffer_view_f32(v);
    };
    auto rent_i32 = [&](turbo_buffer_view *v, uint32_t n) -> int32_t * {
        if (turbo_buffer_arena_rent(
                arena, TURBO_BUFFER_DTYPE_I32, host_place, 1, n, n, v
            ) != TURBO_BUFFER_OK) {
            return nullptr;
        }
        return turbo_buffer_view_i32(v);
    };
    s->arena = arena;
    s->x = rent_f32(&s->v_x, S, H);
    s->residual = rent_f32(&s->v_residual, S, H);
    s->q = rent_f32(&s->v_q, S, H);
    s->k = rent_f32(&s->v_k, S, H);
    s->v = rent_f32(&s->v_v, S, H);
    s->attn = rent_f32(&s->v_attn, heads, S * S);
    s->ctx = rent_f32(&s->v_ctx, S, H);
    s->inter = rent_f32(&s->v_inter, S, I);
    s->tmp = rent_f32(&s->v_tmp, S, H);
    const uint32_t tok_cap = 8192;
    s->tok_q = rent_i32(&s->v_tok_q, tok_cap);
    s->tok_d = rent_i32(&s->v_tok_d, tok_cap);
    if (s->x == nullptr || s->residual == nullptr || s->q == nullptr ||
        s->k == nullptr || s->v == nullptr || s->attn == nullptr ||
        s->ctx == nullptr || s->inter == nullptr || s->tmp == nullptr ||
        s->tok_q == nullptr || s->tok_d == nullptr) {
        if (err) {
            *err = std::string("scratch arena rent failed: ") +
                   turbo_buffer_last_error(arena);
        }
        free_scratch(s);
        return false;
    }
    s->tok_cap = tok_cap;
    s->max_batch = B;
    s->max_seq = S;
    s->hidden = H;
    (void)heads;
    return true;
}

bool load_safetensors(
    const char *path,
    const BertConfig &cfg,
    BertWeights *weights,
    MappedFile *map,
    std::vector<OwnedFloat> *owned,
    std::string *err
) {
    *weights = BertWeights{};
    free_mapped(map);
    free_owned(owned);

    const int fd = open(path, O_RDONLY);
    if (fd < 0) {
        if (err) {
            *err = std::string("cannot open safetensors: ") + path + " (" +
                   std::strerror(errno) + ")";
        }
        return false;
    }
    struct stat st {};
    if (fstat(fd, &st) != 0 || st.st_size < 16) {
        close(fd);
        if (err) {
            *err = std::string("bad safetensors size: ") + path;
        }
        return false;
    }
    void *m = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) {
        close(fd);
        if (err) {
            *err = std::string("mmap failed: ") + path;
        }
        return false;
    }
    map->map = m;
    map->size = static_cast<size_t>(st.st_size);
    map->fd = fd;

    const auto *base = static_cast<const uint8_t *>(m);
    uint64_t header_len = 0;
    parse_u64_le(base, &header_len);
    if (header_len == 0 || header_len + 8 >= map->size) {
        if (err) {
            *err = "safetensors header length invalid";
        }
        return false;
    }
    const std::string header(reinterpret_cast<const char *>(base + 8), static_cast<size_t>(header_len));
    const uint8_t *data = base + 8 + header_len;
    const size_t data_len = map->size - 8 - static_cast<size_t>(header_len);

    auto need = [&](const std::vector<std::string> &names, TensorView *v) -> bool {
        return load_named(header, data, data_len, names, v, owned, err);
    };

    if (!need({"bert.embeddings.word_embeddings.weight",
               "embeddings.word_embeddings.weight"},
              &weights->word)) {
        return false;
    }
    if (!need({"bert.embeddings.position_embeddings.weight",
               "embeddings.position_embeddings.weight"},
              &weights->pos)) {
        return false;
    }
    if (!need({"bert.embeddings.token_type_embeddings.weight",
               "embeddings.token_type_embeddings.weight"},
              &weights->type)) {
        return false;
    }
    if (!need({"bert.embeddings.LayerNorm.weight", "embeddings.LayerNorm.weight"},
              &weights->emb_ln_w)) {
        return false;
    }
    if (!need({"bert.embeddings.LayerNorm.bias", "embeddings.LayerNorm.bias"},
              &weights->emb_ln_b)) {
        return false;
    }

    const uint32_t L = cfg.layers;
    if (L > 12) {
        if (err) {
            *err = "Phase 1 CPU kernel supports at most 12 layers";
        }
        return false;
    }
    for (uint32_t i = 0; i < L; ++i) {
        const std::string p = "bert.encoder.layer." + std::to_string(i) + ".";
        const std::string q = "encoder.layer." + std::to_string(i) + ".";
        auto layer = [&](const char *suffix, TensorView *v) -> bool {
            return need({p + suffix, q + suffix}, v);
        };
        if (!layer("attention.self.query.weight", &weights->q_w[i]) ||
            !layer("attention.self.query.bias", &weights->q_b[i]) ||
            !layer("attention.self.key.weight", &weights->k_w[i]) ||
            !layer("attention.self.key.bias", &weights->k_b[i]) ||
            !layer("attention.self.value.weight", &weights->v_w[i]) ||
            !layer("attention.self.value.bias", &weights->v_b[i]) ||
            !layer("attention.output.dense.weight", &weights->attn_o_w[i]) ||
            !layer("attention.output.dense.bias", &weights->attn_o_b[i]) ||
            !layer("attention.output.LayerNorm.weight", &weights->attn_ln_w[i]) ||
            !layer("attention.output.LayerNorm.bias", &weights->attn_ln_b[i]) ||
            !layer("intermediate.dense.weight", &weights->ff_i_w[i]) ||
            !layer("intermediate.dense.bias", &weights->ff_i_b[i]) ||
            !layer("output.dense.weight", &weights->ff_o_w[i]) ||
            !layer("output.dense.bias", &weights->ff_o_b[i]) ||
            !layer("output.LayerNorm.weight", &weights->ff_ln_w[i]) ||
            !layer("output.LayerNorm.bias", &weights->ff_ln_b[i])) {
            return false;
        }
    }
    std::string pool_err;
    weights->has_pooler = need(
        {"bert.pooler.dense.weight", "pooler.dense.weight"},
        &weights->pool_w
    ) && need(
        {"bert.pooler.dense.bias", "pooler.dense.bias"},
        &weights->pool_b
    );
    if (!weights->has_pooler) {
        // Optional: some CE exports drop the unused NSP pooler.
        if (err) {
            err->clear();
        }
        (void)pool_err;
    }
    if (!need({"classifier.weight"}, &weights->cls_w) ||
        !need({"classifier.bias"}, &weights->cls_b)) {
        return false;
    }
    weights->n_layers = L;
    (void)cfg;
    return true;
}

} // namespace impl
} // namespace turborerank
