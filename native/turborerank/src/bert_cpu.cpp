// SPDX-License-Identifier: Apache-2.0
//
// First-party FP32 BertForSequenceClassification forward.
// Reads caller-owned int32 token buffers; writes one CLS logit.
// No malloc. Scratch is reserved at load.

#include "internal.hpp"

#include <cmath>
#include <cstring>

namespace turborerank {
namespace impl {
namespace {

inline float gelu_erf(float x) {
    // HF `gelu` (not gelu_new): 0.5 * x * (1 + erf(x / sqrt(2)))
    return 0.5f * x * (1.0f + std::erff(x * 0.7071067811865476f));
}

void layer_norm(
    float *x,
    uint32_t seq,
    uint32_t hidden,
    const float *gamma,
    const float *beta,
    float eps
) {
    for (uint32_t t = 0; t < seq; ++t) {
        float *row = x + static_cast<size_t>(t) * hidden;
        float mean = 0.0f;
        for (uint32_t i = 0; i < hidden; ++i) {
            mean += row[i];
        }
        mean /= static_cast<float>(hidden);
        float var = 0.0f;
        for (uint32_t i = 0; i < hidden; ++i) {
            const float d = row[i] - mean;
            var += d * d;
        }
        var /= static_cast<float>(hidden);
        const float inv = 1.0f / std::sqrt(var + eps);
        for (uint32_t i = 0; i < hidden; ++i) {
            row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
        }
    }
}

// y[S, O] = x[S, K] @ W[O, K]^T + b[O]
void linear_nt(
    const float *x,
    const float *w,
    const float *bias,
    float *y,
    uint32_t seq,
    uint32_t k,
    uint32_t out
) {
    constexpr uint32_t BM = 4;
    constexpr uint32_t BN = 16;
    for (uint32_t i0 = 0; i0 < seq; i0 += BM) {
        const uint32_t im = i0 + BM < seq ? i0 + BM : seq;
        for (uint32_t o0 = 0; o0 < out; o0 += BN) {
            const uint32_t om = o0 + BN < out ? o0 + BN : out;
            for (uint32_t i = i0; i < im; ++i) {
                const float *xr = x + static_cast<size_t>(i) * k;
                float *yr = y + static_cast<size_t>(i) * out;
                for (uint32_t o = o0; o < om; ++o) {
                    const float *wr = w + static_cast<size_t>(o) * k;
                    float acc = bias != nullptr ? bias[o] : 0.0f;
                    uint32_t t = 0;
                    for (; t + 8 <= k; t += 8) {
                        acc += xr[t] * wr[t] + xr[t + 1] * wr[t + 1] +
                               xr[t + 2] * wr[t + 2] + xr[t + 3] * wr[t + 3] +
                               xr[t + 4] * wr[t + 4] + xr[t + 5] * wr[t + 5] +
                               xr[t + 6] * wr[t + 6] + xr[t + 7] * wr[t + 7];
                    }
                    for (; t < k; ++t) {
                        acc += xr[t] * wr[t];
                    }
                    yr[o] = acc;
                }
            }
        }
    }
}

void softmax_row(float *row, uint32_t n) {
    float m = row[0];
    for (uint32_t i = 1; i < n; ++i) {
        if (row[i] > m) {
            m = row[i];
        }
    }
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        row[i] = std::exp(row[i] - m);
        sum += row[i];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t i = 0; i < n; ++i) {
        row[i] *= inv;
    }
}

void self_attention(
    const BertConfig &cfg,
    const float *x,
    float *q,
    float *k,
    float *v,
    float *attn,
    float *ctx,
    const float *qw,
    const float *qb,
    const float *kw,
    const float *kb,
    const float *vw,
    const float *vb,
    const float *ow,
    const float *ob,
    const int32_t *mask,
    uint32_t seq
) {
    const uint32_t H = cfg.hidden;
    const uint32_t heads = cfg.heads;
    const uint32_t dh = H / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    linear_nt(x, qw, qb, q, seq, H, H);
    linear_nt(x, kw, kb, k, seq, H, H);
    linear_nt(x, vw, vb, v, seq, H, H);

    // ctx[t, h*dh + d]
    std::memset(ctx, 0, static_cast<size_t>(seq) * H * sizeof(float));

    for (uint32_t h = 0; h < heads; ++h) {
        float *scores = attn + static_cast<size_t>(h) * seq * seq;
        for (uint32_t i = 0; i < seq; ++i) {
            const float *qi = q + static_cast<size_t>(i) * H + h * dh;
            float *srow = scores + static_cast<size_t>(i) * seq;
            for (uint32_t j = 0; j < seq; ++j) {
                const float *kj = k + static_cast<size_t>(j) * H + h * dh;
                float dot = 0.0f;
                for (uint32_t d = 0; d < dh; ++d) {
                    dot += qi[d] * kj[d];
                }
                float s = dot * scale;
                if (mask[j] == 0) {
                    s = -10000.0f;
                }
                srow[j] = s;
            }
            softmax_row(srow, seq);
        }
        for (uint32_t i = 0; i < seq; ++i) {
            const float *srow = scores + static_cast<size_t>(i) * seq;
            float *out = ctx + static_cast<size_t>(i) * H + h * dh;
            for (uint32_t d = 0; d < dh; ++d) {
                float acc = 0.0f;
                for (uint32_t j = 0; j < seq; ++j) {
                    const float *vj = v + static_cast<size_t>(j) * H + h * dh;
                    acc += srow[j] * vj[d];
                }
                out[d] = acc;
            }
        }
    }

    // Project concatenated heads: ctx is already [S, H] in head-major layout
    // matching HF (head 0 features, head 1, ...).
    linear_nt(ctx, ow, ob, q, seq, H, H); // reuse q as attn output
    std::memcpy(ctx, q, static_cast<size_t>(seq) * H * sizeof(float));
}

} // namespace

float bert_forward_row(
    const BertConfig &cfg,
    const BertWeights &w,
    Scratch *s,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    const int32_t *position_ids,
    uint32_t seq
) {
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate;
    float *x = s->x;
    float *residual = s->residual;

    // Embeddings: word + position + token_type
    for (uint32_t t = 0; t < seq; ++t) {
        const int32_t id = input_ids[t];
        const int32_t pos = position_ids != nullptr ? position_ids[t] : static_cast<int32_t>(t);
        const int32_t typ = token_type_ids != nullptr ? token_type_ids[t] : 0;
        const uint32_t uid = static_cast<uint32_t>(id);
        const uint32_t upos = static_cast<uint32_t>(pos);
        const uint32_t utyp = static_cast<uint32_t>(typ);
        float *row = x + static_cast<size_t>(t) * H;
        if (uid < w.word.rows && upos < w.pos.rows && utyp < w.type.rows) {
            const float *we = w.word.data + static_cast<size_t>(uid) * H;
            const float *pe = w.pos.data + static_cast<size_t>(upos) * H;
            const float *te = w.type.data + static_cast<size_t>(utyp) * H;
            for (uint32_t h = 0; h < H; ++h) {
                row[h] = we[h] + pe[h] + te[h];
            }
        } else {
            std::memset(row, 0, H * sizeof(float));
        }
    }
    layer_norm(x, seq, H, w.emb_ln_w.data, w.emb_ln_b.data, cfg.ln_eps);

    for (uint32_t layer = 0; layer < w.n_layers; ++layer) {
        std::memcpy(residual, x, static_cast<size_t>(seq) * H * sizeof(float));
        self_attention(
            cfg,
            x,
            s->q,
            s->k,
            s->v,
            s->attn,
            s->ctx,
            w.q_w[layer].data,
            w.q_b[layer].data,
            w.k_w[layer].data,
            w.k_b[layer].data,
            w.v_w[layer].data,
            w.v_b[layer].data,
            w.attn_o_w[layer].data,
            w.attn_o_b[layer].data,
            attention_mask,
            seq
        );
        // residual + LN
        for (uint32_t t = 0; t < seq; ++t) {
            float *xr = x + static_cast<size_t>(t) * H;
            const float *cr = s->ctx + static_cast<size_t>(t) * H;
            const float *rr = residual + static_cast<size_t>(t) * H;
            for (uint32_t h = 0; h < H; ++h) {
                xr[h] = cr[h] + rr[h];
            }
        }
        layer_norm(x, seq, H, w.attn_ln_w[layer].data, w.attn_ln_b[layer].data, cfg.ln_eps);

        std::memcpy(residual, x, static_cast<size_t>(seq) * H * sizeof(float));
        linear_nt(x, w.ff_i_w[layer].data, w.ff_i_b[layer].data, s->inter, seq, H, I);
        {
            const size_t n = static_cast<size_t>(seq) * I;
            for (size_t i = 0; i < n; ++i) {
                s->inter[i] = gelu_erf(s->inter[i]);
            }
        }
        linear_nt(s->inter, w.ff_o_w[layer].data, w.ff_o_b[layer].data, x, seq, I, H);
        for (uint32_t t = 0; t < seq; ++t) {
            float *xr = x + static_cast<size_t>(t) * H;
            const float *rr = residual + static_cast<size_t>(t) * H;
            for (uint32_t h = 0; h < H; ++h) {
                xr[h] += rr[h];
            }
        }
        layer_norm(x, seq, H, w.ff_ln_w[layer].data, w.ff_ln_b[layer].data, cfg.ln_eps);
    }

    // HF BertForSequenceClassification: pooler(tanh(dense(CLS))) then
    // linear classifier. "CLS logit" is this scalar, not raw hidden[0].
    const float *cls = x;
    float pooled_storage[1024];
    const float *head_in = cls;
    if (w.has_pooler && w.pool_w.data != nullptr && H <= 1024) {
        const float *pw = w.pool_w.data;
        const float *pb = w.pool_b.data;
        for (uint32_t o = 0; o < H; ++o) {
            float acc = pb != nullptr ? pb[o] : 0.0f;
            const float *wr = pw + static_cast<size_t>(o) * H;
            for (uint32_t h = 0; h < H; ++h) {
                acc += cls[h] * wr[h];
            }
            pooled_storage[o] = std::tanh(acc);
        }
        head_in = pooled_storage;
    }
    const float *cw = w.cls_w.data;
    float logit = w.cls_b.data != nullptr ? w.cls_b.data[0] : 0.0f;
    const uint32_t cls_k = w.cls_w.cols > 0 ? w.cls_w.cols : H;
    for (uint32_t h = 0; h < cls_k && h < H; ++h) {
        logit += head_in[h] * cw[h];
    }
    return logit;
}

} // namespace impl
} // namespace turborerank
