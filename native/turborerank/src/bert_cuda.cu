// SPDX-License-Identifier: Apache-2.0
//
// Device MiniLM-L6 BertForSequenceClassification.
// Linear layers: cuBLAS Sgemm. Embeddings / LayerNorm / GELU / attention /
// residual / pooler / classifier: first-party CUDA kernels.
//
// Token workspace stays in caller cudaHostAlloc memory. Each forward
// copies one packed int32 row H2D into device scratch (pinned → device
// is the fast path cudaHostAlloc exists for). No host heap growth.

#include "cuda_api.hpp"

#include <cuda_runtime.h>
#define TURBO_BUFFER_CUDA_INTERCEPT 1
#include "cuda_runtime_hooks.hpp"

#include <cmath>
#include <cstring>
#include <string>

namespace turborerank {
namespace impl {
namespace {

#define TR_CUDA(call, err)                                                     \
    do {                                                                       \
        const cudaError_t _e = (call);                                         \
        if (_e != cudaSuccess) {                                               \
            if (err) {                                                         \
                *(err) = std::string(#call) + ": " + cudaGetErrorString(_e);   \
            }                                                                  \
            return false;                                                      \
        }                                                                      \
    } while (0)

bool upload_f32(const TensorView &src, float **dst, std::string *err) {
    *dst = nullptr;
    if (src.data == nullptr || src.rows == 0 || src.cols == 0) {
        return true;
    }
    const size_t bytes =
        static_cast<size_t>(src.rows) * static_cast<size_t>(src.cols) * sizeof(float);
    TR_CUDA(cudaMalloc(reinterpret_cast<void **>(dst), bytes), err);
    TR_CUDA(cudaMemcpy(*dst, src.data, bytes, cudaMemcpyHostToDevice), err);
    return true;
}

void dfree(float **p) {
    if (p != nullptr && *p != nullptr) {
        (void)cudaFree(*p);
        *p = nullptr;
    }
}

void dfree_i(int32_t **p) {
    if (p != nullptr && *p != nullptr) {
        (void)cudaFree(*p);
        *p = nullptr;
    }
}

__global__ void embed_kernel(
    float *x,
    const int32_t *ids,
    const int32_t *pos,
    const int32_t *types,
    const float *word,
    const float *pos_w,
    const float *type_w,
    uint32_t seq,
    uint32_t hidden,
    uint32_t word_rows,
    uint32_t pos_rows,
    uint32_t type_rows
) {
    const uint32_t t = blockIdx.x;
    if (t >= seq) {
        return;
    }
    const uint32_t uid = static_cast<uint32_t>(ids[t]);
    const uint32_t upos =
        pos != nullptr ? static_cast<uint32_t>(pos[t]) : t;
    const uint32_t utyp =
        types != nullptr ? static_cast<uint32_t>(types[t]) : 0;
    float *row = x + static_cast<size_t>(t) * hidden;
    if (uid < word_rows && upos < pos_rows && utyp < type_rows) {
        const float *we = word + static_cast<size_t>(uid) * hidden;
        const float *pe = pos_w + static_cast<size_t>(upos) * hidden;
        const float *te = type_w + static_cast<size_t>(utyp) * hidden;
        for (uint32_t h = threadIdx.x; h < hidden; h += blockDim.x) {
            row[h] = we[h] + pe[h] + te[h];
        }
    } else {
        for (uint32_t h = threadIdx.x; h < hidden; h += blockDim.x) {
            row[h] = 0.0f;
        }
    }
}

// One thread per token: same reduction order as the CPU kernel.
__global__ void layer_norm_kernel(
    float *x,
    uint32_t seq,
    uint32_t hidden,
    const float *gamma,
    const float *beta,
    float eps
) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= seq) {
        return;
    }
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
    const float inv = rsqrtf(var + eps);
    for (uint32_t i = 0; i < hidden; ++i) {
        row[i] = (row[i] - mean) * inv * gamma[i] + beta[i];
    }
}

// Exact CPU linear_nt: y[s,o] = bias[o] + dot(x[s], W[o]).
__global__ void linear_nt_kernel(
    const float *x,
    const float *w,
    const float *bias,
    float *y,
    uint32_t seq,
    uint32_t k,
    uint32_t out
) {
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t s = blockIdx.y;
    if (s >= seq || o >= out) {
        return;
    }
    const float *xr = x + static_cast<size_t>(s) * k;
    const float *wr = w + static_cast<size_t>(o) * k;
    float acc = bias != nullptr ? bias[o] : 0.0f;
    for (uint32_t t = 0; t < k; ++t) {
        acc += xr[t] * wr[t];
    }
    y[static_cast<size_t>(s) * out + o] = acc;
}

__global__ void gelu_erf_kernel(float *x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float v = x[i];
        x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
    }
}

__global__ void add_inplace_kernel(float *x, const float *r, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += r[i];
    }
}

__global__ void residual_from_ctx_kernel(
    float *x,
    const float *ctx,
    const float *residual,
    size_t n
) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = ctx[i] + residual[i];
    }
}

__global__ void attention_scores_kernel(
    const float *q,
    const float *k,
    float *attn,
    const int32_t *mask,
    uint32_t seq,
    uint32_t hidden,
    uint32_t heads,
    uint32_t dh,
    float scale
) {
    const uint32_t h = blockIdx.z;
    const uint32_t i = blockIdx.y;
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= heads || i >= seq || j >= seq) {
        return;
    }
    const float *qi = q + static_cast<size_t>(i) * hidden + h * dh;
    const float *kj = k + static_cast<size_t>(j) * hidden + h * dh;
    float dot = 0.0f;
    for (uint32_t d = 0; d < dh; ++d) {
        dot += qi[d] * kj[d];
    }
    float s = dot * scale;
    if (mask[j] == 0) {
        s = -10000.0f;
    }
    attn[(static_cast<size_t>(h) * seq + i) * seq + j] = s;
}

__global__ void softmax_rows_kernel(float *attn, uint32_t seq, uint32_t heads) {
    const uint32_t h = blockIdx.y;
    const uint32_t i = blockIdx.x;
    if (h >= heads || i >= seq) {
        return;
    }
    float *row = attn + (static_cast<size_t>(h) * seq + i) * seq;
    float m = row[0];
    for (uint32_t j = 1; j < seq; ++j) {
        if (row[j] > m) {
            m = row[j];
        }
    }
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq; ++j) {
        row[j] = expf(row[j] - m);
        sum += row[j];
    }
    const float inv = sum > 0.0f ? 1.0f / sum : 0.0f;
    for (uint32_t j = 0; j < seq; ++j) {
        row[j] *= inv;
    }
}

__global__ void attention_ctx_kernel(
    const float *attn,
    const float *v,
    float *ctx,
    uint32_t seq,
    uint32_t hidden,
    uint32_t heads,
    uint32_t dh
) {
    const uint32_t h = blockIdx.y;
    const uint32_t i = blockIdx.x;
    if (h >= heads || i >= seq) {
        return;
    }
    const float *srow = attn + (static_cast<size_t>(h) * seq + i) * seq;
    float *out = ctx + static_cast<size_t>(i) * hidden + h * dh;
    for (uint32_t d = threadIdx.x; d < dh; d += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < seq; ++j) {
            const float *vj = v + static_cast<size_t>(j) * hidden + h * dh;
            acc += srow[j] * vj[d];
        }
        out[d] = acc;
    }
}

__global__ void pooler_kernel(
    const float *cls,
    const float *pw,
    const float *pb,
    float *pooled,
    uint32_t hidden
) {
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= hidden) {
        return;
    }
    float acc = pb != nullptr ? pb[o] : 0.0f;
    const float *wr = pw + static_cast<size_t>(o) * hidden;
    for (uint32_t h = 0; h < hidden; ++h) {
        acc += cls[h] * wr[h];
    }
    pooled[o] = tanhf(acc);
}

__global__ void classifier_kernel(
    const float *head_in,
    const float *cw,
    const float *cb,
    float *logit,
    uint32_t hidden
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    float acc = cb != nullptr ? cb[0] : 0.0f;
    for (uint32_t h = 0; h < hidden; ++h) {
        acc += head_in[h] * cw[h];
    }
    *logit = acc;
}

uint32_t grid1(size_t n, uint32_t threads = 256) {
    return static_cast<uint32_t>((n + threads - 1) / threads);
}

struct CudaForwardGuard {
    CudaForwardGuard() { turbo_buffer_cuda_forward_enter(); }
    ~CudaForwardGuard() { turbo_buffer_cuda_forward_leave(); }
    CudaForwardGuard(const CudaForwardGuard &) = delete;
    CudaForwardGuard &operator=(const CudaForwardGuard &) = delete;
};

bool linear_nt_cuda(
    const float *x,
    const float *w,
    const float *bias,
    float *y,
    uint32_t seq,
    uint32_t k,
    uint32_t out,
    std::string *err
) {
    if (seq == 0 || k == 0 || out == 0) {
        if (err) {
            *err = "linear_nt_cuda: empty gemm";
        }
        return false;
    }
    dim3 block(128);
    dim3 grid((out + 127) / 128, seq);
    linear_nt_kernel<<<grid, block>>>(x, w, bias, y, seq, k, out);
    TR_CUDA(cudaGetLastError(), err);
    return true;
}

bool copy_tensor(
    const TensorView &src,
    float **dst,
    std::string *err,
    const char *what
) {
    if (!upload_f32(src, dst, err)) {
        if (err && err->find(what) == std::string::npos) {
            *err = std::string(what) + ": " + *err;
        }
        return false;
    }
    return true;
}

} // namespace

bool cuda_resources_init(
    CudaResources *r,
    const BertConfig &cfg,
    const BertWeights &w,
    turbo_buffer_arena *arena,
    std::string *err
) {
    if (r == nullptr) {
        if (err) {
            *err = "cuda_resources_init: null";
        }
        return false;
    }
    cuda_resources_free(r);

    TR_CUDA(cudaSetDevice(0), err);

    auto up = [&](const TensorView &tv, float **dst, const char *name) -> bool {
        return copy_tensor(tv, dst, err, name);
    };

    if (!up(w.word, &r->word, "word") || !up(w.pos, &r->pos, "pos") ||
        !up(w.type, &r->type, "type") || !up(w.emb_ln_w, &r->emb_ln_w, "emb_ln_w") ||
        !up(w.emb_ln_b, &r->emb_ln_b, "emb_ln_b")) {
        cuda_resources_free(r);
        return false;
    }
    r->word_rows = w.word.rows;
    r->pos_rows = w.pos.rows;
    r->type_rows = w.type.rows;
    r->n_layers = w.n_layers;
    r->has_pooler = w.has_pooler;
    r->cls_cols = w.cls_w.cols > 0 ? w.cls_w.cols : cfg.hidden;

    for (uint32_t i = 0; i < w.n_layers && i < 12; ++i) {
        if (!up(w.q_w[i], &r->q_w[i], "q_w") || !up(w.q_b[i], &r->q_b[i], "q_b") ||
            !up(w.k_w[i], &r->k_w[i], "k_w") || !up(w.k_b[i], &r->k_b[i], "k_b") ||
            !up(w.v_w[i], &r->v_w[i], "v_w") || !up(w.v_b[i], &r->v_b[i], "v_b") ||
            !up(w.attn_o_w[i], &r->attn_o_w[i], "attn_o_w") ||
            !up(w.attn_o_b[i], &r->attn_o_b[i], "attn_o_b") ||
            !up(w.attn_ln_w[i], &r->attn_ln_w[i], "attn_ln_w") ||
            !up(w.attn_ln_b[i], &r->attn_ln_b[i], "attn_ln_b") ||
            !up(w.ff_i_w[i], &r->ff_i_w[i], "ff_i_w") ||
            !up(w.ff_i_b[i], &r->ff_i_b[i], "ff_i_b") ||
            !up(w.ff_o_w[i], &r->ff_o_w[i], "ff_o_w") ||
            !up(w.ff_o_b[i], &r->ff_o_b[i], "ff_o_b") ||
            !up(w.ff_ln_w[i], &r->ff_ln_w[i], "ff_ln_w") ||
            !up(w.ff_ln_b[i], &r->ff_ln_b[i], "ff_ln_b")) {
            cuda_resources_free(r);
            return false;
        }
    }
    if (w.has_pooler) {
        if (!up(w.pool_w, &r->pool_w, "pool_w") || !up(w.pool_b, &r->pool_b, "pool_b")) {
            cuda_resources_free(r);
            return false;
        }
    }
    if (!up(w.cls_w, &r->cls_w, "cls_w") || !up(w.cls_b, &r->cls_b, "cls_b")) {
        cuda_resources_free(r);
        return false;
    }

    const uint32_t S = cfg.max_position;
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate;
    const uint32_t heads = cfg.heads;
    if (arena == nullptr) {
        if (err) {
            *err = "CUDA activation scratch requires a turbo_buffer arena; "
                   "refusing private cudaMalloc";
        }
        cuda_resources_free(r);
        return false;
    }
    r->arena = arena;
    auto rent_slot = [&](turbo_buffer_dtype dt, size_t n, void **out) -> bool {
        if (r->n_rented >= 16) {
            if (err) {
                *err = "CUDA scratch view table full";
            }
            return false;
        }
        turbo_buffer_view *v = &r->rented[r->n_rented];
        const uint32_t cols = static_cast<uint32_t>(n);
        if (turbo_buffer_arena_rent(
                arena, dt, TURBO_BUFFER_PLACE_DEVICE, 1, cols, cols, v
            ) != TURBO_BUFFER_OK) {
            if (err) {
                *err = std::string("CUDA DEVICE rent failed: ") +
                       turbo_buffer_last_error(arena);
            }
            return false;
        }
        *out = v->ptr;
        r->n_rented += 1;
        return true;
    };
    auto dalloc = [&](float **p, size_t n) -> bool {
        return rent_slot(TURBO_BUFFER_DTYPE_F32, n, reinterpret_cast<void **>(p));
    };
    auto ialloc = [&](int32_t **p, size_t n) -> bool {
        return rent_slot(TURBO_BUFFER_DTYPE_I32, n, reinterpret_cast<void **>(p));
    };
    if (!dalloc(&r->x, static_cast<size_t>(S) * H) ||
        !dalloc(&r->residual, static_cast<size_t>(S) * H) ||
        !dalloc(&r->q, static_cast<size_t>(S) * H) ||
        !dalloc(&r->k, static_cast<size_t>(S) * H) ||
        !dalloc(&r->v, static_cast<size_t>(S) * H) ||
        !dalloc(&r->attn, static_cast<size_t>(heads) * S * S) ||
        !dalloc(&r->ctx, static_cast<size_t>(S) * H) ||
        !dalloc(&r->inter, static_cast<size_t>(S) * I) ||
        !dalloc(&r->tmp, static_cast<size_t>(S) * H) ||
        !dalloc(&r->pooled, H) || !dalloc(&r->logit, 1) ||
        !ialloc(&r->ids, S) || !ialloc(&r->mask, S) ||
        !ialloc(&r->types, S) || !ialloc(&r->pos_ids, S)) {
        cuda_resources_free(r);
        return false;
    }
    r->enabled = true;
    return true;
}

void cuda_resources_free(CudaResources *r) {
    if (r == nullptr) {
        return;
    }
    r->cublas = nullptr;
    dfree(&r->word);
    dfree(&r->pos);
    dfree(&r->type);
    dfree(&r->emb_ln_w);
    dfree(&r->emb_ln_b);
    for (int i = 0; i < 12; ++i) {
        dfree(&r->q_w[i]);
        dfree(&r->q_b[i]);
        dfree(&r->k_w[i]);
        dfree(&r->k_b[i]);
        dfree(&r->v_w[i]);
        dfree(&r->v_b[i]);
        dfree(&r->attn_o_w[i]);
        dfree(&r->attn_o_b[i]);
        dfree(&r->attn_ln_w[i]);
        dfree(&r->attn_ln_b[i]);
        dfree(&r->ff_i_w[i]);
        dfree(&r->ff_i_b[i]);
        dfree(&r->ff_o_w[i]);
        dfree(&r->ff_o_b[i]);
        dfree(&r->ff_ln_w[i]);
        dfree(&r->ff_ln_b[i]);
    }
    dfree(&r->pool_w);
    dfree(&r->pool_b);
    dfree(&r->cls_w);
    dfree(&r->cls_b);
    if (r->arena != nullptr) {
        for (uint32_t i = 0; i < r->n_rented; ++i) {
            if (r->rented[i].ptr != nullptr) {
                (void)turbo_buffer_arena_return(r->arena, &r->rented[i]);
            }
        }
    }
    r->x = nullptr;
    r->residual = nullptr;
    r->q = nullptr;
    r->k = nullptr;
    r->v = nullptr;
    r->attn = nullptr;
    r->ctx = nullptr;
    r->inter = nullptr;
    r->tmp = nullptr;
    r->pooled = nullptr;
    r->logit = nullptr;
    r->ids = nullptr;
    r->mask = nullptr;
    r->types = nullptr;
    r->pos_ids = nullptr;
    r->n_rented = 0;
    r->arena = nullptr;
    r->enabled = false;
}

bool bert_forward_row_cuda(
    CudaResources *r,
    const BertConfig &cfg,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    const int32_t *position_ids,
    uint32_t seq,
    float *logit_out,
    std::string *err
) {
    CudaForwardGuard fwd;
    if (r == nullptr || !r->enabled) {
        if (err) {
            *err = "CUDA MiniLM CE is not initialized";
        }
        return false;
    }
    if (input_ids == nullptr || attention_mask == nullptr || logit_out == nullptr ||
        seq == 0 || seq > cfg.max_position) {
        if (err) {
            *err = "bert_forward_row_cuda: bad arguments";
        }
        return false;
    }
    const uint32_t H = cfg.hidden;
    const uint32_t I = cfg.intermediate;
    const uint32_t heads = cfg.heads;
    const uint32_t dh = H / heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    const size_t nbytes_i = static_cast<size_t>(seq) * sizeof(int32_t);

    TR_CUDA(cudaMemcpy(r->ids, input_ids, nbytes_i, cudaMemcpyHostToDevice), err);
    TR_CUDA(cudaMemcpy(r->mask, attention_mask, nbytes_i, cudaMemcpyHostToDevice), err);
    if (token_type_ids != nullptr) {
        TR_CUDA(cudaMemcpy(r->types, token_type_ids, nbytes_i, cudaMemcpyHostToDevice), err);
    } else {
        TR_CUDA(cudaMemset(r->types, 0, nbytes_i), err);
    }
    if (position_ids != nullptr) {
        TR_CUDA(cudaMemcpy(r->pos_ids, position_ids, nbytes_i, cudaMemcpyHostToDevice), err);
    } else {
        // arange on host would allocate; write a tiny kernel-free loop into
        // the already-reserved device buffer via a one-shot host stack copy.
        int32_t tmp_pos[512];
        if (seq > 512) {
            if (err) {
                *err = "bert_forward_row_cuda: seq exceeds stack arange";
            }
            return false;
        }
        for (uint32_t t = 0; t < seq; ++t) {
            tmp_pos[t] = static_cast<int32_t>(t);
        }
        TR_CUDA(cudaMemcpy(r->pos_ids, tmp_pos, nbytes_i, cudaMemcpyHostToDevice), err);
    }

    embed_kernel<<<seq, 128>>>(
        r->x,
        r->ids,
        r->pos_ids,
        r->types,
        r->word,
        r->pos,
        r->type,
        seq,
        H,
        r->word_rows,
        r->pos_rows,
        r->type_rows
    );
    TR_CUDA(cudaGetLastError(), err);
    layer_norm_kernel<<<seq, 1>>>(
        r->x, seq, H, r->emb_ln_w, r->emb_ln_b, cfg.ln_eps
    );
    TR_CUDA(cudaGetLastError(), err);

    const size_t hidden_n = static_cast<size_t>(seq) * H;
    const size_t inter_n = static_cast<size_t>(seq) * I;

    for (uint32_t layer = 0; layer < r->n_layers; ++layer) {
        TR_CUDA(
            cudaMemcpy(r->residual, r->x, hidden_n * sizeof(float), cudaMemcpyDeviceToDevice),
            err
        );
        if (!linear_nt_cuda(r->x, r->q_w[layer], r->q_b[layer], r->q, seq, H, H, err) ||
            !linear_nt_cuda(r->x, r->k_w[layer], r->k_b[layer], r->k, seq, H, H, err) ||
            !linear_nt_cuda(r->x, r->v_w[layer], r->v_b[layer], r->v, seq, H, H, err)) {
            return false;
        }
        TR_CUDA(cudaMemset(r->ctx, 0, hidden_n * sizeof(float)), err);
        dim3 score_grid((seq + 31) / 32, seq, heads);
        attention_scores_kernel<<<score_grid, 32>>>(
            r->q, r->k, r->attn, r->mask, seq, H, heads, dh, scale
        );
        TR_CUDA(cudaGetLastError(), err);
        dim3 sm_grid(seq, heads);
        softmax_rows_kernel<<<sm_grid, 1>>>(r->attn, seq, heads);
        TR_CUDA(cudaGetLastError(), err);
        attention_ctx_kernel<<<sm_grid, 32>>>(r->attn, r->v, r->ctx, seq, H, heads, dh);
        TR_CUDA(cudaGetLastError(), err);
        if (!linear_nt_cuda(
                r->ctx, r->attn_o_w[layer], r->attn_o_b[layer], r->q, seq, H, H, err
            )) {
            return false;
        }
        TR_CUDA(cudaMemcpy(r->ctx, r->q, hidden_n * sizeof(float), cudaMemcpyDeviceToDevice), err);
        residual_from_ctx_kernel<<<grid1(hidden_n), 256>>>(r->x, r->ctx, r->residual, hidden_n);
        TR_CUDA(cudaGetLastError(), err);
        layer_norm_kernel<<<seq, 1>>>(
            r->x, seq, H, r->attn_ln_w[layer], r->attn_ln_b[layer], cfg.ln_eps
        );
        TR_CUDA(cudaGetLastError(), err);

        TR_CUDA(
            cudaMemcpy(r->residual, r->x, hidden_n * sizeof(float), cudaMemcpyDeviceToDevice),
            err
        );
        if (!linear_nt_cuda(
                r->x, r->ff_i_w[layer], r->ff_i_b[layer], r->inter, seq, H, I, err
            )) {
            return false;
        }
        gelu_erf_kernel<<<grid1(inter_n), 256>>>(r->inter, inter_n);
        TR_CUDA(cudaGetLastError(), err);
        if (!linear_nt_cuda(
                r->inter, r->ff_o_w[layer], r->ff_o_b[layer], r->x, seq, I, H, err
            )) {
            return false;
        }
        add_inplace_kernel<<<grid1(hidden_n), 256>>>(r->x, r->residual, hidden_n);
        TR_CUDA(cudaGetLastError(), err);
        layer_norm_kernel<<<seq, 1>>>(
            r->x, seq, H, r->ff_ln_w[layer], r->ff_ln_b[layer], cfg.ln_eps
        );
        TR_CUDA(cudaGetLastError(), err);
    }

    const float *head_in = r->x;
    if (r->has_pooler && r->pool_w != nullptr) {
        pooler_kernel<<<grid1(H, 128), 128>>>(r->x, r->pool_w, r->pool_b, r->pooled, H);
        TR_CUDA(cudaGetLastError(), err);
        head_in = r->pooled;
    }
    const uint32_t cls_k = r->cls_cols > 0 ? r->cls_cols : H;
    classifier_kernel<<<1, 1>>>(head_in, r->cls_w, r->cls_b, r->logit, cls_k);
    TR_CUDA(cudaGetLastError(), err);
    TR_CUDA(cudaMemcpy(logit_out, r->logit, sizeof(float), cudaMemcpyDeviceToHost), err);
    TR_CUDA(cudaDeviceSynchronize(), err);
    return true;
}

} // namespace impl
} // namespace turborerank
