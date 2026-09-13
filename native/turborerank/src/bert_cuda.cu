// SPDX-License-Identifier: Apache-2.0
//
// Device MiniLM-L6 BertForSequenceClassification.
// Linear layers: cuBLASLt (`cublasLtMatmul`) on turbo_buffer DEVICE
// activations + an arena-rented Lt workspace. Missing cuBLASLt fails
// load loud — there is no hand-rolled GEMM fallback.
// Embeddings / LayerNorm / GELU / attention / residual / tanh: first-party
// CUDA kernels.
//
// Token workspace is caller cudaHostAllocMapped memory. Host writes
// ids/mask/types/pos into those pages; kernels read the mapped device
// pointer. No per-forward cudaMemcpy H2D of the token row. No host
// heap growth.

#include "cuda_api.hpp"

#include <cuda_runtime.h>
#define TURBO_BUFFER_CUDA_INTERCEPT 1
#include "cuda_runtime_hooks.hpp"

#include <cublasLt.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#if !defined(CUBLAS_VERSION)
#error "TURBORERANK_CUDA requires cuBLASLt (cublasLt.h / cublas_api.h). Refusing hand-rolled GEMM."
#endif

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

__global__ void bias_row_kernel(
    float *y,
    const float *bias,
    uint32_t seq,
    uint32_t out
) {
    const uint32_t o = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t s = blockIdx.y;
    if (s >= seq || o >= out) {
        return;
    }
    y[static_cast<size_t>(s) * out + o] += bias[o];
}

__global__ void tanh_kernel(float *x, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] = tanhf(x[i]);
    }
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

uint32_t grid1(size_t n, uint32_t threads = 256) {
    return static_cast<uint32_t>((n + threads - 1) / threads);
}

struct CudaForwardGuard {
    CudaForwardGuard() { turbo_buffer_cuda_forward_enter(); }
    ~CudaForwardGuard() { turbo_buffer_cuda_forward_leave(); }
    CudaForwardGuard(const CudaForwardGuard &) = delete;
    CudaForwardGuard &operator=(const CudaForwardGuard &) = delete;
};

#define TR_LT(call, err)                                                       \
    do {                                                                       \
        const cublasStatus_t _s = (call);                                      \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                     \
            if (err) {                                                         \
                *(err) = std::string(#call) + ": cublasLt status " +           \
                         std::to_string(static_cast<int>(_s)) +                \
                         " (refusing hand-rolled GEMM)";                       \
            }                                                                  \
            return false;                                                      \
        }                                                                      \
    } while (0)

cublasLtHandle_t lt_handle(CudaResources *r) {
    return static_cast<cublasLtHandle_t>(r->cublaslt);
}

cublasLtMatmulDesc_t lt_desc(CudaResources *r, bool bias) {
    return static_cast<cublasLtMatmulDesc_t>(bias ? r->lt_desc_bias : r->lt_desc);
}

cublasLtMatrixLayout_t lt_layout_a(CudaResources *r) {
    return static_cast<cublasLtMatrixLayout_t>(r->lt_layout_a);
}

cublasLtMatrixLayout_t lt_layout_b(CudaResources *r) {
    return static_cast<cublasLtMatrixLayout_t>(r->lt_layout_b);
}

cublasLtMatrixLayout_t lt_layout_c(CudaResources *r) {
    return static_cast<cublasLtMatrixLayout_t>(r->lt_layout_c);
}

cublasLtMatmulPreference_t lt_pref(CudaResources *r) {
    return static_cast<cublasLtMatmulPreference_t>(r->lt_pref);
}

bool lt_set_layout(
    cublasLtMatrixLayout_t layout,
    uint64_t rows,
    uint64_t cols,
    int64_t ld,
    std::string *err
) {
    TR_LT(
        cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_ROWS, &rows, sizeof(rows)
        ),
        err
    );
    TR_LT(
        cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_COLS, &cols, sizeof(cols)
        ),
        err
    );
    TR_LT(
        cublasLtMatrixLayoutSetAttribute(
            layout, CUBLASLT_MATRIX_LAYOUT_LD, &ld, sizeof(ld)
        ),
        err
    );
    return true;
}

// Row-major Y[seq, out] = X[seq, k] @ W[out, k]^T  via column-major
// cublasLtMatmul: C[out, seq] = W^T_cm @ X_cm  (transa=T, transb=N).
bool linear_nt_cuda(
    CudaResources *r,
    const float *x,
    const float *w,
    const float *bias,
    float *y,
    uint32_t seq,
    uint32_t k,
    uint32_t out,
    std::string *err
) {
    if (r == nullptr || r->cublaslt == nullptr) {
        if (err) {
            *err = "cuBLASLt handle is null; refusing hand-rolled GEMM";
        }
        return false;
    }
    if (seq == 0 || k == 0 || out == 0) {
        if (err) {
            *err = "linear_nt_cuda: empty gemm";
        }
        return false;
    }

    // Column-major dims matching cublasSgemm(T, N, out, seq, k, W, k, X, k, Y, out).
    if (!lt_set_layout(lt_layout_a(r), k, out, static_cast<int64_t>(k), err) ||
        !lt_set_layout(lt_layout_b(r), k, seq, static_cast<int64_t>(k), err) ||
        !lt_set_layout(lt_layout_c(r), out, seq, static_cast<int64_t>(out), err)) {
        return false;
    }

    const bool use_bias = bias != nullptr;
    cublasLtMatmulDesc_t desc = lt_desc(r, use_bias);
    if (use_bias) {
        TR_LT(
            cublasLtMatmulDescSetAttribute(
                desc,
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &bias,
                sizeof(bias)
            ),
            err
        );
    }

    cublasLtMatmulHeuristicResult_t heur {};
    int returned = 0;
    TR_LT(
        cublasLtMatmulAlgoGetHeuristic(
            lt_handle(r),
            desc,
            lt_layout_a(r),
            lt_layout_b(r),
            lt_layout_c(r),
            lt_layout_c(r),
            lt_pref(r),
            1,
            &heur,
            &returned
        ),
        err
    );
    if (returned <= 0) {
        if (use_bias) {
            // Still cuBLASLt — bias epilogue unavailable for this shape.
            desc = lt_desc(r, false);
            returned = 0;
            TR_LT(
                cublasLtMatmulAlgoGetHeuristic(
                    lt_handle(r),
                    desc,
                    lt_layout_a(r),
                    lt_layout_b(r),
                    lt_layout_c(r),
                    lt_layout_c(r),
                    lt_pref(r),
                    1,
                    &heur,
                    &returned
                ),
                err
            );
        }
        if (returned <= 0) {
            if (err) {
                *err = "cuBLASLt has no algo for MiniLM GEMM; refusing "
                       "hand-rolled kernel";
            }
            return false;
        }
    }
    if (heur.workspaceSize > r->lt_workspace_bytes) {
        if (err) {
            *err = "cuBLASLt workspace exceeds arena slab; refusing "
                   "per-forward cudaMalloc";
        }
        return false;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    TR_LT(
        cublasLtMatmul(
            lt_handle(r),
            desc,
            &alpha,
            w,
            lt_layout_a(r),
            x,
            lt_layout_b(r),
            &beta,
            y,
            lt_layout_c(r),
            y,
            lt_layout_c(r),
            &heur.algo,
            r->lt_workspace,
            r->lt_workspace_bytes,
            0
        ),
        err
    );
    if (use_bias && desc == lt_desc(r, false)) {
        dim3 block(128);
        dim3 grid((out + 127) / 128, seq);
        bias_row_kernel<<<grid, block>>>(y, bias, seq, out);
        TR_CUDA(cudaGetLastError(), err);
    }
    return true;
}

void lt_destroy(CudaResources *r) {
    if (r == nullptr) {
        return;
    }
    if (r->lt_pref != nullptr) {
        (void)cublasLtMatmulPreferenceDestroy(
            static_cast<cublasLtMatmulPreference_t>(r->lt_pref)
        );
        r->lt_pref = nullptr;
    }
    if (r->lt_layout_a != nullptr) {
        (void)cublasLtMatrixLayoutDestroy(
            static_cast<cublasLtMatrixLayout_t>(r->lt_layout_a)
        );
        r->lt_layout_a = nullptr;
    }
    if (r->lt_layout_b != nullptr) {
        (void)cublasLtMatrixLayoutDestroy(
            static_cast<cublasLtMatrixLayout_t>(r->lt_layout_b)
        );
        r->lt_layout_b = nullptr;
    }
    if (r->lt_layout_c != nullptr) {
        (void)cublasLtMatrixLayoutDestroy(
            static_cast<cublasLtMatrixLayout_t>(r->lt_layout_c)
        );
        r->lt_layout_c = nullptr;
    }
    if (r->lt_desc != nullptr) {
        (void)cublasLtMatmulDescDestroy(
            static_cast<cublasLtMatmulDesc_t>(r->lt_desc)
        );
        r->lt_desc = nullptr;
    }
    if (r->lt_desc_bias != nullptr) {
        (void)cublasLtMatmulDescDestroy(
            static_cast<cublasLtMatmulDesc_t>(r->lt_desc_bias)
        );
        r->lt_desc_bias = nullptr;
    }
    if (r->cublaslt != nullptr) {
        (void)cublasLtDestroy(static_cast<cublasLtHandle_t>(r->cublaslt));
        r->cublaslt = nullptr;
    }
    r->lt_workspace = nullptr;
    r->lt_workspace_bytes = 0;
}

bool lt_resources_init(CudaResources *r, std::string *err) {
    cublasLtHandle_t handle = nullptr;
    const cublasStatus_t created = cublasLtCreate(&handle);
    if (created != CUBLAS_STATUS_SUCCESS || handle == nullptr) {
        if (err) {
            *err = "cuBLASLt is unavailable (cublasLtCreate status " +
                   std::to_string(static_cast<int>(created)) +
                   "); refusing hand-rolled GEMM fallback";
        }
        return false;
    }
    r->cublaslt = handle;

    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatmulDesc_t desc_bias = nullptr;
    TR_LT(
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), err
    );
    TR_LT(
        cublasLtMatmulDescCreate(&desc_bias, CUBLAS_COMPUTE_32F, CUDA_R_32F), err
    );
    r->lt_desc = desc;
    r->lt_desc_bias = desc_bias;

    const cublasOperation_t transa = CUBLAS_OP_T;
    const cublasOperation_t transb = CUBLAS_OP_N;
    TR_LT(
        cublasLtMatmulDescSetAttribute(
            desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)
        ),
        err
    );
    TR_LT(
        cublasLtMatmulDescSetAttribute(
            desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)
        ),
        err
    );
    TR_LT(
        cublasLtMatmulDescSetAttribute(
            desc_bias, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)
        ),
        err
    );
    TR_LT(
        cublasLtMatmulDescSetAttribute(
            desc_bias, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)
        ),
        err
    );
    const cublasLtEpilogue_t epi_bias = CUBLASLT_EPILOGUE_BIAS;
    TR_LT(
        cublasLtMatmulDescSetAttribute(
            desc_bias,
            CUBLASLT_MATMUL_DESC_EPILOGUE,
            &epi_bias,
            sizeof(epi_bias)
        ),
        err
    );

    // Dummy 1x1 layouts; rows/cols/ld are overwritten per GEMM.
    cublasLtMatrixLayout_t a = nullptr;
    cublasLtMatrixLayout_t b = nullptr;
    cublasLtMatrixLayout_t c = nullptr;
    TR_LT(cublasLtMatrixLayoutCreate(&a, CUDA_R_32F, 1, 1, 1), err);
    TR_LT(cublasLtMatrixLayoutCreate(&b, CUDA_R_32F, 1, 1, 1), err);
    TR_LT(cublasLtMatrixLayoutCreate(&c, CUDA_R_32F, 1, 1, 1), err);
    r->lt_layout_a = a;
    r->lt_layout_b = b;
    r->lt_layout_c = c;

    cublasLtMatmulPreference_t pref = nullptr;
    TR_LT(cublasLtMatmulPreferenceCreate(&pref), err);
    r->lt_pref = pref;
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

    if (!lt_resources_init(r, err)) {
        cuda_resources_free(r);
        return false;
    }

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
    if (!dalloc(&r->x, static_cast<size_t>(S) * H) ||
        !dalloc(&r->residual, static_cast<size_t>(S) * H) ||
        !dalloc(&r->q, static_cast<size_t>(S) * H) ||
        !dalloc(&r->k, static_cast<size_t>(S) * H) ||
        !dalloc(&r->v, static_cast<size_t>(S) * H) ||
        !dalloc(&r->attn, static_cast<size_t>(heads) * S * S) ||
        !dalloc(&r->ctx, static_cast<size_t>(S) * H) ||
        !dalloc(&r->inter, static_cast<size_t>(S) * I) ||
        !dalloc(&r->tmp, static_cast<size_t>(S) * H) ||
        !dalloc(&r->pooled, H) || !dalloc(&r->logit, 1)) {
        cuda_resources_free(r);
        return false;
    }

    // Size the Lt workspace from heuristics on the live MiniLM shapes.
    // 32 MiB cap is the arena slab we are willing to rent — never cudaMalloc
    // a larger workspace on forward.
    const uint64_t k_lt_workspace_cap = 32ull * 1024ull * 1024ull;
    if (cublasLtMatmulPreferenceSetAttribute(
            lt_pref(r),
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &k_lt_workspace_cap,
            sizeof(k_lt_workspace_cap)
        ) != CUBLAS_STATUS_SUCCESS) {
        if (err) {
            *err = "cublasLtMatmulPreferenceSetAttribute workspace cap failed";
        }
        cuda_resources_free(r);
        return false;
    }
    auto probe_ws = [&](uint32_t seq, uint32_t kk, uint32_t oo, bool with_bias) -> bool {
        if (!lt_set_layout(lt_layout_a(r), kk, oo, static_cast<int64_t>(kk), err) ||
            !lt_set_layout(lt_layout_b(r), kk, seq, static_cast<int64_t>(kk), err) ||
            !lt_set_layout(lt_layout_c(r), oo, seq, static_cast<int64_t>(oo), err)) {
            return false;
        }
        if (with_bias) {
            const float *dummy_bias =
                r->q_b[0] != nullptr ? r->q_b[0] : r->cls_b;
            if (dummy_bias == nullptr) {
                if (err) {
                    *err = "cuBLASLt bias probe has no device bias pointer";
                }
                return false;
            }
            const cublasStatus_t bs = cublasLtMatmulDescSetAttribute(
                lt_desc(r, true),
                CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                &dummy_bias,
                sizeof(dummy_bias)
            );
            if (bs != CUBLAS_STATUS_SUCCESS) {
                if (err) {
                    *err = "cublasLtMatmulDescSetAttribute bias probe failed";
                }
                return false;
            }
        }
        cublasLtMatmulHeuristicResult_t heur {};
        int returned = 0;
        const cublasStatus_t hs = cublasLtMatmulAlgoGetHeuristic(
            lt_handle(r),
            lt_desc(r, with_bias),
            lt_layout_a(r),
            lt_layout_b(r),
            lt_layout_c(r),
            lt_layout_c(r),
            lt_pref(r),
            1,
            &heur,
            &returned
        );
        if (hs != CUBLAS_STATUS_SUCCESS || returned <= 0) {
            if (err) {
                *err = "cuBLASLt has no algo for MiniLM GEMM shape; refusing "
                       "hand-rolled kernel";
            }
            return false;
        }
        if (heur.workspaceSize > r->lt_workspace_bytes) {
            r->lt_workspace_bytes = heur.workspaceSize;
        }
        return true;
    };
    const uint32_t Smax = S;
    if (!probe_ws(Smax, H, H, true) || !probe_ws(Smax, H, I, true) ||
        !probe_ws(Smax, I, H, true) || !probe_ws(1, H, H, true) ||
        !probe_ws(1, r->cls_cols > 0 ? r->cls_cols : H, 1, true) ||
        !probe_ws(Smax, H, H, false) || !probe_ws(1, H, 1, false)) {
        cuda_resources_free(r);
        return false;
    }
    if (r->lt_workspace_bytes > k_lt_workspace_cap) {
        if (err) {
            *err = "cuBLASLt workspace exceeds 32 MiB arena cap; refusing "
                   "hand-rolled GEMM";
        }
        cuda_resources_free(r);
        return false;
    }
    if (r->lt_workspace_bytes > 0) {
        const size_t n_f32 =
            (r->lt_workspace_bytes + sizeof(float) - 1) / sizeof(float);
        float *ws = nullptr;
        if (!dalloc(&ws, n_f32)) {
            cuda_resources_free(r);
            return false;
        }
        r->lt_workspace = ws;
    }
    const uint64_t rented_ws = static_cast<uint64_t>(r->lt_workspace_bytes);
    if (cublasLtMatmulPreferenceSetAttribute(
            lt_pref(r),
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &rented_ws,
            sizeof(rented_ws)
        ) != CUBLAS_STATUS_SUCCESS) {
        if (err) {
            *err = "cublasLtMatmulPreferenceSetAttribute rented workspace failed";
        }
        cuda_resources_free(r);
        return false;
    }

    // Warm the algo cache / any one-time Lt state at load, not forward.
    if (!linear_nt_cuda(r, r->x, r->q_w[0], r->q_b[0], r->q, Smax, H, H, err) ||
        !linear_nt_cuda(r, r->x, r->ff_i_w[0], r->ff_i_b[0], r->inter, Smax, H, I, err) ||
        !linear_nt_cuda(
            r, r->inter, r->ff_o_w[0], r->ff_o_b[0], r->x, Smax, I, H, err
        ) ||
        !linear_nt_cuda(
            r,
            r->x,
            r->cls_w,
            r->cls_b,
            r->logit,
            1,
            r->cls_cols > 0 ? r->cls_cols : H,
            1,
            err
        )) {
        cuda_resources_free(r);
        return false;
    }
    if (r->has_pooler && r->pool_w != nullptr) {
        if (!linear_nt_cuda(r, r->x, r->pool_w, r->pool_b, r->pooled, 1, H, H, err)) {
            cuda_resources_free(r);
            return false;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        if (err) {
            *err = "cuBLASLt warmup cudaDeviceSynchronize failed";
        }
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
    lt_destroy(r);
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

    auto map_i32 = [&](const int32_t *host, const int32_t **dev, const char *what)
        -> bool {
        if (host == nullptr) {
            *dev = nullptr;
            return true;
        }
        void *d = nullptr;
        if (turbo_buffer_cuda_mapped_device_ptr(host, &d) == 0 || d == nullptr) {
            if (err) {
                *err = std::string(what) +
                       ": tokens are not CUDA PINNED mapped; "
                       "refusing per-forward H2D";
            }
            return false;
        }
        *dev = static_cast<const int32_t *>(d);
        return true;
    };
    const int32_t *d_ids = nullptr;
    const int32_t *d_mask = nullptr;
    const int32_t *d_types = nullptr;
    const int32_t *d_pos = nullptr;
    if (!map_i32(input_ids, &d_ids, "input_ids") ||
        !map_i32(attention_mask, &d_mask, "attention_mask") ||
        !map_i32(token_type_ids, &d_types, "token_type_ids") ||
        !map_i32(position_ids, &d_pos, "position_ids")) {
        return false;
    }

    embed_kernel<<<seq, 128>>>(
        r->x,
        d_ids,
        d_pos,
        d_types,
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
        if (!linear_nt_cuda(r, r->x, r->q_w[layer], r->q_b[layer], r->q, seq, H, H, err) ||
            !linear_nt_cuda(r, r->x, r->k_w[layer], r->k_b[layer], r->k, seq, H, H, err) ||
            !linear_nt_cuda(r, r->x, r->v_w[layer], r->v_b[layer], r->v, seq, H, H, err)) {
            return false;
        }
        TR_CUDA(cudaMemset(r->ctx, 0, hidden_n * sizeof(float)), err);
        dim3 score_grid((seq + 31) / 32, seq, heads);
        attention_scores_kernel<<<score_grid, 32>>>(
            r->q, r->k, r->attn, d_mask, seq, H, heads, dh, scale
        );
        TR_CUDA(cudaGetLastError(), err);
        dim3 sm_grid(seq, heads);
        softmax_rows_kernel<<<sm_grid, 1>>>(r->attn, seq, heads);
        TR_CUDA(cudaGetLastError(), err);
        attention_ctx_kernel<<<sm_grid, 32>>>(r->attn, r->v, r->ctx, seq, H, heads, dh);
        TR_CUDA(cudaGetLastError(), err);
        if (!linear_nt_cuda(
                r, r->ctx, r->attn_o_w[layer], r->attn_o_b[layer], r->q, seq, H, H, err
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
                r, r->x, r->ff_i_w[layer], r->ff_i_b[layer], r->inter, seq, H, I, err
            )) {
            return false;
        }
        gelu_erf_kernel<<<grid1(inter_n), 256>>>(r->inter, inter_n);
        TR_CUDA(cudaGetLastError(), err);
        if (!linear_nt_cuda(
                r, r->inter, r->ff_o_w[layer], r->ff_o_b[layer], r->x, seq, I, H, err
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
        if (!linear_nt_cuda(r, r->x, r->pool_w, r->pool_b, r->pooled, 1, H, H, err)) {
            return false;
        }
        tanh_kernel<<<grid1(H, 128), 128>>>(r->pooled, H);
        TR_CUDA(cudaGetLastError(), err);
        head_in = r->pooled;
    }
    const uint32_t cls_k = r->cls_cols > 0 ? r->cls_cols : H;
    if (!linear_nt_cuda(r, head_in, r->cls_w, r->cls_b, r->logit, 1, cls_k, 1, err)) {
        return false;
    }
    TR_CUDA(cudaMemcpy(logit_out, r->logit, sizeof(float), cudaMemcpyDeviceToHost), err);
    TR_CUDA(cudaDeviceSynchronize(), err);
    return true;
}

const char *cuda_gemm_backend() { return "cublasLtMatmul"; }

} // namespace impl
} // namespace turborerank
