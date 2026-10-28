/* SPDX-License-Identifier: Apache-2.0
 *
 * The rotary families' load on the CUDA backend (rotary.h): the checks,
 * and the family's tensors laid out in BERT's slots.
 */
#include "rotary.h"

#include <cstddef>
#include <cstdio>
#include <cstring>

namespace turbo_cuda {

const char *rotary_check(const turbo_backend_model &d, int32_t *code, char *why, size_t len) {
    why[0] = '\0';
    const size_t need = offsetof(turbo_backend_model, global_every) + sizeof(uint32_t);
    if (d.struct_size < need) {
        *code = TURBO_E_INVALID_STRUCT_SIZE;
        snprintf(why, len, "turbo_backend_model of %u bytes: a rotary model needs %zu", d.struct_size, need);
        return why;
    }
    const bool modern = d.family == TURBO_FAMILY_MODERNBERT;
    const uint64_t emb = modern ? TURBO_MODERNBERT_EMBEDDING_TENSORS : TURBO_NOMIC_EMBEDDING_TENSORS;
    const uint64_t per = modern ? TURBO_MODERNBERT_LAYER_TENSORS : TURBO_NOMIC_LAYER_TENSORS;
    const uint64_t count = emb + (uint64_t)d.layers * per;
    *code = TURBO_E_INVALID_ARGUMENT;
    if (d.tensor_count != count || !d.tensors) {
        snprintf(why, len, "tensor_count %u: this family's encoder of %u layers has %llu", d.tensor_count, d.layers,
                 (unsigned long long)count);
        return why;
    }
    *code = TURBO_E_UNSUPPORTED;
    if (d.dtype != TURBO_DTYPE_F32) {
        snprintf(why, len, "a rotary model's weights in dtype %u: the cuda backend loads them in F32", d.dtype);
        return why;
    }
    const uint32_t act = modern ? TURBO_ACTIVATION_GELU_ERF : TURBO_ACTIVATION_SILU;
    const uint32_t norm = modern ? TURBO_NORM_LAYER_NO_BIAS : TURBO_NORM_LAYER;
    if (d.mlp != TURBO_MLP_GATED || d.activation != act || d.norm != norm) {
        snprintf(why, len, "mlp %u, activation %u, norm %u: the cuda backend runs this family with %u, %u, %u", d.mlp,
                 d.activation, d.norm, TURBO_MLP_GATED, act, norm);
        return why;
    }
    *code = TURBO_E_INVALID_ARGUMENT;
    const uint32_t head = d.heads ? d.hidden / d.heads : 0;
    if (head == 0 || d.hidden % d.heads != 0 || head % 2 != 0 || d.rope_pairs * 2 != head || !d.rope_table) {
        snprintf(why, len,
                 "hidden %u, heads %u, rope_pairs %u: a rotary head is an even width, twice rope_pairs, with a table",
                 d.hidden, d.heads, d.rope_pairs);
        return why;
    }
    if ((d.attention_window == 0) != (d.global_every == 0) || (modern && d.attention_window == 0)) {
        snprintf(why, len, "attention_window %u, global_every %u", d.attention_window, d.global_every);
        return why;
    }
    return why;
}

namespace {

/* A view of n floats. */
turbo_backend_tensor view(const float *p, size_t n, const char *name) {
    turbo_backend_tensor t{};
    t.name = name;
    t.data = p;
    t.dtype = TURBO_DTYPE_F32;
    t.bytes = (uint64_t)n * 4;
    return t;
}

const float *floats(const turbo_backend_tensor &t) { return static_cast<const float *>(t.data); }

} // namespace

void rotary_slots(const turbo_backend_model &d, std::vector<turbo_backend_tensor> &slots,
                  std::vector<std::vector<float>> &owned) {
    const bool modern = d.family == TURBO_FAMILY_MODERNBERT;
    const size_t h = d.hidden, inter = d.intermediate, hd = h / d.heads, half = hd / 2;
    const size_t emb = modern ? TURBO_MODERNBERT_EMBEDDING_TENSORS : TURBO_NOMIC_EMBEDDING_TENSORS;
    const size_t per = modern ? TURBO_MODERNBERT_LAYER_TENSORS : TURBO_NOMIC_LAYER_TENSORS;
    const turbo_backend_tensor *t = d.tensors;
    owned.clear();
    owned.reserve(1 + 2 * (size_t)d.layers);
    // One zero vector serves every absent bias.
    const size_t widest = 3 * h > 2 * inter ? 3 * h : 2 * inter;
    owned.emplace_back(widest, 0.0f);
    const float *zero = owned[0].data();
    slots.assign(TURBO_BERT_EMBEDDING_TENSORS + (size_t)d.layers * TURBO_BERT_LAYER_TENSORS, view(zero, 0, "empty"));
    if (modern) {
        slots[TURBO_BERT_WORD_EMBEDDINGS] = t[TURBO_MODERNBERT_WORD_EMBEDDINGS];
        slots[ROTARY_FINAL_NORM] = t[TURBO_MODERNBERT_FINAL_NORM_WEIGHT];
        slots[TURBO_BERT_EMBEDDINGS_LN_WEIGHT] = t[TURBO_MODERNBERT_EMBEDDINGS_LN_WEIGHT];
        slots[TURBO_BERT_EMBEDDINGS_LN_BIAS] = view(zero, h, "zero");
    } else {
        slots[TURBO_BERT_WORD_EMBEDDINGS] = t[TURBO_NOMIC_WORD_EMBEDDINGS];
        slots[TURBO_BERT_TOKEN_TYPE_EMBEDDINGS] = t[TURBO_NOMIC_TOKEN_TYPE_EMBEDDINGS];
        slots[TURBO_BERT_EMBEDDINGS_LN_WEIGHT] = t[TURBO_NOMIC_EMBEDDINGS_LN_WEIGHT];
        slots[TURBO_BERT_EMBEDDINGS_LN_BIAS] = t[TURBO_NOMIC_EMBEDDINGS_LN_BIAS];
    }
    for (size_t l = 0; l < d.layers; l++) {
        const turbo_backend_tensor *lt = t + emb + l * per;
        turbo_backend_tensor *s = slots.data() + TURBO_BERT_EMBEDDING_TENSORS + l * TURBO_BERT_LAYER_TENSORS;
        const float *wqkv = floats(lt[modern ? TURBO_MODERNBERT_QKV_WEIGHT : TURBO_NOMIC_QKV_WEIGHT]);
        // Q and K pair-adjacent within each head, V as stored.
        std::vector<float> qkv(3 * h * h);
        size_t row = 0;
        for (size_t third = 0; third < 2; third++)
            for (size_t head = 0; head < h; head += hd)
                for (size_t c = 0; c < half; c++)
                    for (size_t o : {third * h + head + c, third * h + head + c + half})
                        memcpy(&qkv[(row++) * h], wqkv + o * h, h * 4);
        memcpy(&qkv[2 * h * h], wqkv + 2 * h * h, h * h * 4);
        // Gate row j, then up row j.
        const float *gate, *up;
        if (modern) {
            gate = floats(lt[TURBO_MODERNBERT_FFN_IN_WEIGHT]);
            up = gate + inter * h;
        } else {
            gate = floats(lt[TURBO_NOMIC_FFN_GATE_WEIGHT]);
            up = floats(lt[TURBO_NOMIC_FFN_UP_WEIGHT]);
        }
        std::vector<float> ffn(2 * inter * h);
        for (size_t j = 0; j < inter; j++) {
            memcpy(&ffn[(2 * j) * h], gate + j * h, h * 4);
            memcpy(&ffn[(2 * j + 1) * h], up + j * h, h * 4);
        }
        owned.push_back(std::move(qkv));
        const float *q = owned.back().data();
        owned.push_back(std::move(ffn));
        const float *f = owned.back().data();
        s[TURBO_BERT_Q_WEIGHT] = view(q, h * h, "Wqkv");
        s[TURBO_BERT_K_WEIGHT] = view(q + h * h, h * h, "Wqkv");
        s[TURBO_BERT_V_WEIGHT] = view(q + 2 * h * h, h * h, "Wqkv");
        s[TURBO_BERT_Q_BIAS] = s[TURBO_BERT_K_BIAS] = s[TURBO_BERT_V_BIAS] = view(zero, h, "zero");
        s[TURBO_BERT_ATTN_OUT_BIAS] = s[TURBO_BERT_FFN_OUT_BIAS] = view(zero, h, "zero");
        s[TURBO_BERT_FFN_IN_WEIGHT] = view(f, 2 * inter * h, "gated mlp in");
        s[TURBO_BERT_FFN_IN_BIAS] = view(zero, 2 * inter, "zero");
        if (modern) {
            s[TURBO_BERT_ATTN_OUT_WEIGHT] = lt[TURBO_MODERNBERT_ATTN_OUT_WEIGHT];
            s[TURBO_BERT_FFN_OUT_WEIGHT] = lt[TURBO_MODERNBERT_FFN_OUT_WEIGHT];
            // Layer 0's is absent (data NULL, bytes 0): its slot stays empty.
            if (lt[TURBO_MODERNBERT_ATTN_LN_WEIGHT].data)
                s[TURBO_BERT_ATTN_LN_WEIGHT] = lt[TURBO_MODERNBERT_ATTN_LN_WEIGHT];
            s[TURBO_BERT_ATTN_LN_BIAS] = view(zero, h, "zero");
            s[TURBO_BERT_FFN_LN_WEIGHT] = lt[TURBO_MODERNBERT_FFN_LN_WEIGHT];
            s[TURBO_BERT_FFN_LN_BIAS] = view(zero, h, "zero");
        } else {
            s[TURBO_BERT_ATTN_OUT_WEIGHT] = lt[TURBO_NOMIC_ATTN_OUT_WEIGHT];
            s[TURBO_BERT_FFN_OUT_WEIGHT] = lt[TURBO_NOMIC_FFN_OUT_WEIGHT];
            s[TURBO_BERT_ATTN_LN_WEIGHT] = lt[TURBO_NOMIC_ATTN_LN_WEIGHT];
            s[TURBO_BERT_ATTN_LN_BIAS] = lt[TURBO_NOMIC_ATTN_LN_BIAS];
            s[TURBO_BERT_FFN_LN_WEIGHT] = lt[TURBO_NOMIC_FFN_LN_WEIGHT];
            s[TURBO_BERT_FFN_LN_BIAS] = lt[TURBO_NOMIC_FFN_LN_BIAS];
        }
    }
}

} // namespace turbo_cuda
