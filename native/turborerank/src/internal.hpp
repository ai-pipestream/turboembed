// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "reranker.hpp"
#include "turbo_buffer.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace turborerank {
namespace impl {

void set_create_error(const std::string &msg);
void set_engine_error(turborerank_engine *engine, const std::string &msg);
const char *create_error();

bool device_is_accelerator(turborerank_device d);
bool accelerator_unavailable(turborerank_device d, std::string *why);
/** AUTO resolves to CUDA, else OpenVINO GPU, else Metal, else stays AUTO (fail loud). */
turborerank_device resolve_create_device(turborerank_device requested);

struct Vocab {
    std::unordered_map<std::string, int32_t> token_to_id;
    int32_t unk_id = kUnkId;
    int32_t cls_id = kClsId;
    int32_t sep_id = kSepId;
    int32_t pad_id = kPadId;
    bool loaded = false;
};

bool load_vocab_txt(const char *path, Vocab *vocab, std::string *err);

/** WordPiece + BERT basic tokenize. Writes ids, returns count. No heap if
 *  `ids_cap` is sufficient; otherwise returns 0 and sets err. */
size_t tokenize_wordpiece(
    const Vocab &vocab,
    const char *utf8,
    size_t utf8_len,
    int32_t *ids,
    size_t ids_cap,
    std::string *err
);

struct BertConfig {
    uint32_t hidden = 384;
    uint32_t intermediate = 1536;
    uint32_t layers = 6;
    uint32_t heads = 12;
    uint32_t max_position = 512;
    uint32_t vocab_size = 30522;
    uint32_t type_vocab = 2;
    float ln_eps = 1e-12f;
    uint32_t max_batch = 32;
};

struct TensorView {
    const float *data = nullptr;
    uint32_t rows = 0;
    uint32_t cols = 0;
    bool owned = false;
};

struct BertWeights {
    TensorView word;     // [vocab, H]
    TensorView pos;      // [max_pos, H]
    TensorView type;     // [2, H]
    TensorView emb_ln_w; // [H]
    TensorView emb_ln_b; // [H]
    TensorView q_w[12];
    TensorView q_b[12];
    TensorView k_w[12];
    TensorView k_b[12];
    TensorView v_w[12];
    TensorView v_b[12];
    TensorView attn_o_w[12];
    TensorView attn_o_b[12];
    TensorView attn_ln_w[12];
    TensorView attn_ln_b[12];
    TensorView ff_i_w[12];
    TensorView ff_i_b[12];
    TensorView ff_o_w[12];
    TensorView ff_o_b[12];
    TensorView ff_ln_w[12];
    TensorView ff_ln_b[12];
    TensorView pool_w; // [H, H] BertPooler
    TensorView pool_b; // [H]
    TensorView cls_w;  // [1, H]
    TensorView cls_b;  // [1]
    bool has_pooler = false;
    uint32_t n_layers = 0;
};

struct MappedFile {
    void *map = nullptr;
    size_t size = 0;
    int fd = -1;
};

struct OwnedFloat {
    float *ptr = nullptr;
    size_t bytes = 0;
};

struct Scratch {
    float *x = nullptr;
    float *residual = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *attn = nullptr;
    float *ctx = nullptr;
    float *inter = nullptr;
    float *tmp = nullptr;
    int32_t *tok_q = nullptr;
    int32_t *tok_d = nullptr;
    size_t tok_cap = 0;
    uint32_t max_batch = 0;
    uint32_t max_seq = 0;
    uint32_t hidden = 0;
    turbo_buffer_arena *arena = nullptr;
    turbo_buffer_view v_x {};
    turbo_buffer_view v_residual {};
    turbo_buffer_view v_q {};
    turbo_buffer_view v_k {};
    turbo_buffer_view v_v {};
    turbo_buffer_view v_attn {};
    turbo_buffer_view v_ctx {};
    turbo_buffer_view v_inter {};
    turbo_buffer_view v_tmp {};
    turbo_buffer_view v_tok_q {};
    turbo_buffer_view v_tok_d {};
};

/** Device weights + activation arena. Opaque to the CPU path. */
struct CudaResources {
    bool enabled = false;
    void *cublas = nullptr; // cublasHandle_t
    float *word = nullptr;
    float *pos = nullptr;
    float *type = nullptr;
    float *emb_ln_w = nullptr;
    float *emb_ln_b = nullptr;
    float *q_w[12]{};
    float *q_b[12]{};
    float *k_w[12]{};
    float *k_b[12]{};
    float *v_w[12]{};
    float *v_b[12]{};
    float *attn_o_w[12]{};
    float *attn_o_b[12]{};
    float *attn_ln_w[12]{};
    float *attn_ln_b[12]{};
    float *ff_i_w[12]{};
    float *ff_i_b[12]{};
    float *ff_o_w[12]{};
    float *ff_o_b[12]{};
    float *ff_ln_w[12]{};
    float *ff_ln_b[12]{};
    float *pool_w = nullptr;
    float *pool_b = nullptr;
    float *cls_w = nullptr;
    float *cls_b = nullptr;
    bool has_pooler = false;
    uint32_t n_layers = 0;
    uint32_t word_rows = 0;
    uint32_t pos_rows = 0;
    uint32_t type_rows = 0;
    uint32_t cls_cols = 0;
    float *x = nullptr;
    float *residual = nullptr;
    float *q = nullptr;
    float *k = nullptr;
    float *v = nullptr;
    float *attn = nullptr;
    float *ctx = nullptr;
    float *inter = nullptr;
    float *tmp = nullptr;
    float *pooled = nullptr;
    float *logit = nullptr;
    turbo_buffer_arena *arena = nullptr;
    turbo_buffer_view rented[16] {};
    uint32_t n_rented = 0;
};

/** OpenVINO CompiledModel + infer request. Opaque hold lives in ov_api.cpp. */
struct OvResources {
    bool enabled = false;
    bool gpu = false;
    bool token_usm = false;
    bool remote_wrap = false;
    std::string ov_device;
    void *hold = nullptr;
    uint32_t max_batch = 0;
    uint32_t max_seq = 0;
};

/** Metal device BERT. Opaque hold lives in metal_api.mm. */
struct MetalResources {
    bool enabled = false;
    void *hold = nullptr;
    uint32_t max_seq = 0;
    uint32_t hidden = 0;
};

bool load_safetensors(
    const char *path,
    const BertConfig &cfg,
    BertWeights *weights,
    MappedFile *map,
    std::vector<OwnedFloat> *owned,
    std::string *err
);

void free_mapped(MappedFile *map);
void free_owned(std::vector<OwnedFloat> *owned);
void free_scratch(Scratch *s);

/** Rent scratch from `arena`. Null arena is a hard fail — no private malloc. */
bool alloc_scratch(
    Scratch *s,
    turbo_buffer_arena *arena,
    turbo_buffer_placement host_place,
    const BertConfig &cfg,
    std::string *err
);

turbo_buffer_device buffer_device_for(turborerank_device d);
turbo_buffer_placement host_visible_placement(turborerank_device d);

/** BERT CE forward. One packed row. No allocation. */
float bert_forward_row(
    const BertConfig &cfg,
    const BertWeights &w,
    Scratch *s,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    const int32_t *position_ids,
    uint32_t seq
);

Status pack_ids_into(
    turborerank_buffer *buffer,
    uint32_t row,
    const int32_t *query_ids,
    size_t n_query,
    const int32_t *doc_ids,
    size_t n_doc,
    turborerank_truncation truncation,
    uint32_t max_length,
    std::string *err
);

std::string resolve_model_dir(
    const char *alias,
    size_t alias_len,
    const char *config_path,
    const char *workspace_root
);

bool read_bert_config_json(const char *path, BertConfig *cfg, std::string *err);

} // namespace impl
} // namespace turborerank

struct turborerank_engine {
    turborerank_device device = TURBORERANK_DEVICE_CPU;
    std::string last_error;
    std::string config_path;
    std::string workspace_root;
    std::string loaded_alias;
    std::string model_dir;
    bool ready = false;
    turborerank::impl::BertConfig cfg;
    turborerank::impl::Vocab vocab;
    turborerank::impl::BertWeights weights;
    turborerank::impl::MappedFile mapped;
    std::vector<turborerank::impl::OwnedFloat> owned_weights;
    turborerank::impl::Scratch scratch;
    turborerank::impl::CudaResources cuda;
    turborerank::impl::OvResources ov;
    turborerank::impl::MetalResources metal;
    turbo_buffer_arena *arena = nullptr;
    turborerank_buffer *work = nullptr;
};
