// SPDX-License-Identifier: Apache-2.0
//
// Turbo Metal provider: BERT-family embeddings and cross-encoder reranking
// on Apple GPUs through Metal directly. No MLX, no Core ML, no offline
// shader toolchain: the kernels (kernels.inc, from the PoC) are compiled at
// runtime with newLibraryWithSource, weights come from an F32 safetensors
// checkpoint in the bundle, and every buffer is MTLResourceStorageModeShared
// on unified memory, so tokens are written straight into the GPU's memory
// and results are read straight out of it. The provider reports
// TURBO_CAP_DEVICE_RESULT and TURBO_CAP_UNIFIED_MEMORY, results are
// TURBO_PLACE_SHARED, and h2d/d2h byte counters stay at zero because
// nothing is copied across a bus.
//
// What runs where: WordPiece on the host (native/wordpiece); the encoder,
// pooling, L2 normalization, the NSP pooler and the classifier head on the
// GPU. fully_accelerated is 0 because of the tokenizer.
//
// One command buffer per run: the whole batch goes through the encoder as
// one set of dispatches in one compute encoder (Metal orders dependent
// dispatches on the same buffers), then the run waits for completion
// before it returns.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "turbo/turbo_provider.h"
#include "turbo/turbo_types.h"
#include "turbo_provider_common.hpp"
#include "safetensors.hpp"
#include "wordpiece.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

namespace turbo_metal {
using namespace turbo_pc; // NOLINT(google-build-using-namespace)

constexpr const char *kProviderId = "metal";
constexpr const char *kProviderVersion = "2.0.0-alpha.0";

static const char *kMetalSrc =
#include "kernels.inc"
    ;

// ---------------------------------------------------------------------------
// Provider state: the default Metal device and the compiled kernels
// ---------------------------------------------------------------------------

struct Pipelines {
    id<MTLComputePipelineState> embed, ln, linear, gelu, add, residual, copyf, zerof, scores, softmax, ctx, pooler,
        classifier, softmax_row, sentence_pool, l2;
};

struct State {
    id<MTLDevice> device = nil;
    id<MTLLibrary> library = nil;
    Pipelines p{};
    std::string why; // non-empty when the device is unusable
    std::string os;

    State() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                why = "no Metal device";
                return;
            }
            if (!device.hasUnifiedMemory) {
                why = "Metal device " + std::string(device.name.UTF8String) +
                      " has no unified memory; the metal provider keeps tokens and results in shared memory";
                return;
            }
            os = [[NSProcessInfo processInfo] operatingSystemVersionString].UTF8String;
            NSError *err = nil;
            MTLCompileOptions *opts = [MTLCompileOptions new];
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeSafe;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
            }
            library = [device newLibraryWithSource:[NSString stringWithUTF8String:kMetalSrc] options:opts error:&err];
            if (library == nil) {
                why = std::string("Metal shader compile failed: ") + (err ? err.localizedDescription.UTF8String : "unknown");
                return;
            }
            auto pipe = [&](const char *name) -> id<MTLComputePipelineState> {
                id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:name]];
                if (fn == nil) {
                    why = std::string("kernel `") + name + "` is missing from the compiled library";
                    return nil;
                }
                NSError *pe = nil;
                id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&pe];
                if (pso == nil) {
                    why = std::string("pipeline for `") + name + "` failed: " + (pe ? pe.localizedDescription.UTF8String : "unknown");
                }
                return pso;
            };
            p.embed = pipe("embed_kernel");
            p.ln = pipe("layer_norm_kernel");
            p.linear = pipe("linear_nt_kernel");
            p.gelu = pipe("gelu_erf_kernel");
            p.add = pipe("add_inplace_kernel");
            p.residual = pipe("residual_from_ctx_kernel");
            p.copyf = pipe("copy_f32_kernel");
            p.zerof = pipe("zero_f32_kernel");
            p.scores = pipe("attention_scores_kernel");
            p.softmax = pipe("softmax_rows_kernel");
            p.ctx = pipe("attention_ctx_kernel");
            p.pooler = pipe("pooler_kernel");
            p.classifier = pipe("classifier_kernel");
            p.softmax_row = pipe("softmax_row_kernel");
            p.sentence_pool = pipe("sentence_pool_kernel");
            p.l2 = pipe("l2_normalize_kernel");
        }
    }

    bool ready() const { return why.empty(); }
};

State &state() {
    static State s;
    return s;
}

void require_device(uint32_t ordinal) {
    const State &s = state();
    require(s.device != nil, TURBO_E_DEVICE_NOT_FOUND, "metal provider found no Metal device on this machine");
    require(ordinal == 0, TURBO_E_DEVICE_NOT_FOUND,
            "metal provider has one device (ordinal 0); ordinal " + std::to_string(ordinal) + " does not exist");
}

constexpr uint64_t kCaps = TURBO_CAP_DEVICE_RESULT | TURBO_CAP_UNIFIED_MEMORY | TURBO_CAP_HOST_PTR_IMPORT |
                           TURBO_CAP_DETERMINISTIC | TURBO_CAP_OPT_TRUNCATE | TURBO_CAP_OPT_MAX_TOKENS |
                           TURBO_CAP_OPT_PROMPT_ROLE | TURBO_CAP_OPT_NORMALIZE | TURBO_CAP_OPT_POOLING_OVERRIDE |
                           TURBO_CAP_OPT_OUTPUT_DIM | TURBO_CAP_OPT_TOP_N | TURBO_CAP_OPT_RAW_SCORES;

bool offers(uint32_t task, uint32_t modality) {
    return state().ready() && modality == TURBO_MODALITY_TEXT && (task == TURBO_TASK_EMBED || task == TURBO_TASK_RERANK);
}

template <typename T>
void release(void *p) noexcept {
    try {
        delete static_cast<T *>(p);
    } catch (...) {
    }
}

uint32_t checked_truncate(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_TRUNCATE_MODEL:
    case TURBO_TRUNCATE_NONE:
    case TURBO_TRUNCATE_RIGHT:
    case TURBO_TRUNCATE_LEFT:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "truncate " + std::to_string(v) + " is not a TURBO_TRUNCATE_* value", field);
    }
}

uint32_t checked_prompt_role(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_PROMPT_NONE:
    case TURBO_PROMPT_QUERY:
    case TURBO_PROMPT_DOCUMENT:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "prompt_role " + std::to_string(v) + " is not a TURBO_PROMPT_* value", field);
    }
}

uint32_t checked_normalize(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_NORMALIZE_MODEL:
    case TURBO_NORMALIZE_NONE:
    case TURBO_NORMALIZE_L2:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "normalize " + std::to_string(v) + " is not a TURBO_NORMALIZE_* value", field);
    }
}

uint32_t checked_pooling(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_POOLING_MODEL:
    case TURBO_POOLING_MEAN:
    case TURBO_POOLING_CLS:
    case TURBO_POOLING_LAST:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "pooling " + std::to_string(v) + " is not a TURBO_POOLING_* value", field);
    }
}

// ---------------------------------------------------------------------------
// Context and buffers
// ---------------------------------------------------------------------------

struct Context {
    id<MTLCommandQueue> queue = nil;

    Context() {
        require(state().ready(), TURBO_E_DEVICE_UNAVAILABLE, state().why);
        queue = [state().device newCommandQueue];
        require(queue != nil, TURBO_E_DEVICE_UNAVAILABLE, "Metal command queue creation failed");
    }
};

id<MTLBuffer> shared_buffer(uint64_t bytes, const void *src) {
    const size_t aligned = (static_cast<size_t>(bytes) + 255) & ~static_cast<size_t>(255);
    id<MTLBuffer> b = [state().device newBufferWithLength:aligned options:MTLResourceStorageModeShared];
    require(b != nil, TURBO_E_OUT_OF_MEMORY, "Metal shared buffer of " + std::to_string(bytes) + " bytes failed");
    std::memset(b.contents, 0, aligned);
    if (src != nullptr && bytes > 0) {
        std::memcpy(b.contents, src, static_cast<size_t>(bytes));
    }
    return b;
}

struct Buffer {
    turbo_buffer_desc desc{};
    id<MTLBuffer> mtl = nil; // SHARED placement
    void *host = nullptr;    // HOST placement or an imported pointer
    uint64_t bytes = 0;
    bool owns_host = true;

    ~Buffer() {
        if (host != nullptr && owns_host) {
            std::free(host);
        }
    }

    void *contents() const { return mtl != nil ? mtl.contents : host; }
};

turbo_buffer_desc packed_desc(uint32_t placement, uint32_t dtype, const std::vector<uint64_t> &shape, uint64_t elem) {
    turbo_buffer_desc d{};
    d.struct_size = sizeof(turbo_buffer_desc);
    d.placement = placement;
    d.dtype = dtype;
    d.ndim = static_cast<uint32_t>(shape.size());
    uint64_t acc = elem;
    for (size_t i = shape.size(); i-- > 0;) {
        d.shape[i] = shape[i];
        d.strides[i] = acc;
        acc *= shape[i];
    }
    d.bytes = acc;
    return d;
}

uint64_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case TURBO_DTYPE_BOOL:
    case TURBO_DTYPE_U8:
    case TURBO_DTYPE_I8:
        return 1;
    case TURBO_DTYPE_U16:
    case TURBO_DTYPE_I16:
    case TURBO_DTYPE_F16:
    case TURBO_DTYPE_BF16:
        return 2;
    case TURBO_DTYPE_U32:
    case TURBO_DTYPE_I32:
    case TURBO_DTYPE_F32:
        return 4;
    case TURBO_DTYPE_U64:
    case TURBO_DTYPE_I64:
    case TURBO_DTYPE_F64:
        return 8;
    default:
        fail(TURBO_E_UNSUPPORTED_DTYPE, "dtype " + std::to_string(dtype) + " is not supported by the metal provider");
    }
}

uint64_t describe_into(Buffer *b, const turbo_buffer_desc &in) {
    b->desc = in;
    b->desc.next = nullptr;
    const uint64_t elem = dtype_size(in.dtype);
    check_buffer_desc(in, elem);
    uint64_t bytes = in.bytes;
    if (bytes == 0) {
        bytes = packed_bytes(elem, in.ndim, in.shape);
        b->desc = packed_desc(in.placement, in.dtype, std::vector<uint64_t>(in.shape, in.shape + in.ndim), elem);
    }
    b->bytes = bytes;
    b->desc.bytes = bytes;
    return bytes;
}

std::unique_ptr<Buffer> make_buffer(const turbo_buffer_desc &in) {
    auto b = std::make_unique<Buffer>();
    const uint64_t bytes = describe_into(b.get(), in);
    require(bytes > 0, TURBO_E_INVALID_SHAPE, "zero-byte buffers are not allocated");
    switch (in.placement) {
    case TURBO_PLACE_SHARED:
        b->mtl = shared_buffer(bytes, nullptr);
        break;
    case TURBO_PLACE_HOST: {
        const size_t sz = (static_cast<size_t>(bytes) + 63) & ~static_cast<size_t>(63);
        b->host = std::aligned_alloc(64, sz);
        require(b->host != nullptr, TURBO_E_OUT_OF_MEMORY, "host allocation of " + std::to_string(bytes) + " bytes failed");
        std::memset(b->host, 0, sz);
        break;
    }
    default:
        fail(TURBO_E_UNSUPPORTED_PLACEMENT,
             "metal provider allocates TURBO_PLACE_SHARED (unified memory) and TURBO_PLACE_HOST; DEVICE and PINNED are not "
             "distinct placements on this hardware");
    }
    return b;
}

std::unique_ptr<Buffer> import_buffer(const turbo_buffer_desc &in, const turbo_native_handle &h) {
    require(h.kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED,
            "metal provider imports TURBO_HANDLE_HOST_PTR only; handle kind " + std::to_string(h.kind) + " is not offered");
    require(in.placement == TURBO_PLACE_HOST, TURBO_E_UNSUPPORTED_PLACEMENT,
            "an imported host pointer is TURBO_PLACE_HOST; placement " + std::to_string(in.placement) + " cannot describe caller memory");
    require(h.offset == 0, TURBO_E_UNSUPPORTED, "metal provider imports handles with offset 0 only");
    require(h.handle != 0, TURBO_E_INVALID_ARGUMENT, "imported host pointer is NULL");
    auto b = std::make_unique<Buffer>();
    const uint64_t bytes = describe_into(b.get(), in);
    require(bytes > 0, TURBO_E_INVALID_SHAPE, "an imported buffer must describe at least one byte");
    b->host = reinterpret_cast<void *>(static_cast<uintptr_t>(h.handle));
    b->owns_host = false;
    return b;
}

turbo_provider_buffer describe(Buffer *b) {
    turbo_provider_buffer out{};
    out.struct_size = sizeof(turbo_provider_buffer);
    out.handle = b;
    out.host_ptr = b->contents(); // unified memory: readable from the host either way
    out.desc = b->desc;
    return out;
}

// ---------------------------------------------------------------------------
// Models: an F32 BERT checkpoint uploaded to shared buffers
// ---------------------------------------------------------------------------

enum class Kind { Embedding, Reranker };
enum class Pool { Mean, Cls, Last };

struct Layer {
    id<MTLBuffer> q_w, q_b, k_w, k_b, v_w, v_b, o_w, o_b, ln1_w, ln1_b, ff_i_w, ff_i_b, ff_o_w, ff_o_b, ln2_w, ln2_b;
};

struct Model {
    Bundle bundle;
    Kind kind = Kind::Embedding;
    Pool pool = Pool::Mean;
    bool normalize = false;
    std::string activation; // reranker: sigmoid | none
    uint32_t hidden = 0, intermediate = 0, layers = 0, heads = 0, max_seq = 0, max_batch = 0;
    uint32_t word_rows = 0, pos_rows = 0, type_rows = 0;
    float ln_eps = 1e-12f;
    wordpiece_vocab *vocab = nullptr;
    Context *ctx = nullptr; // the context the model was loaded on (outlives it)
    id<MTLBuffer> word, pos, type, emb_ln_w, emb_ln_b, pool_w, pool_b, cls_w, cls_b, dummy_bias;
    bool has_pooler = false;
    std::vector<Layer> layer;

    ~Model() {
        if (vocab != nullptr) {
            wordpiece_vocab_destroy(vocab);
        }
    }
};

id<MTLBuffer> upload(const Tensor &t) { return shared_buffer(t.elements() * 4, t.data); }

std::unique_ptr<Model> load_model(Context *ctx, const std::string &dir, const std::map<std::string, std::string> &opts) {
    reject_unknown(opts, {}, "metal model");
    require(state().ready(), TURBO_E_DEVICE_UNAVAILABLE, state().why);
    auto m = std::make_unique<Model>();
    m->ctx = ctx;
    m->bundle = Bundle::read(dir);
    const Bundle &b = m->bundle;
    require(b.modality == "text", TURBO_E_UNSUPPORTED_MODALITY, "metal provider serves text bundles only");
    if (b.kind == "embedding") {
        m->kind = Kind::Embedding;
        require(b.dim > 0, TURBO_E_BUNDLE_INVALID, "embedding bundle must declare contract.dim");
        if (b.pooling == "mean") {
            m->pool = Pool::Mean;
        } else if (b.pooling == "cls") {
            m->pool = Pool::Cls;
        } else if (b.pooling == "last") {
            m->pool = Pool::Last;
        } else {
            fail(TURBO_E_BUNDLE_INVALID, "embedding bundle pooling `" + b.pooling + "` must be mean, cls, or last");
        }
        require(b.normalize == "l2" || b.normalize == "none", TURBO_E_BUNDLE_INVALID, "contract.normalize must be l2 or none");
        m->normalize = b.normalize == "l2";
    } else if (b.kind == "reranker") {
        m->kind = Kind::Reranker;
        m->activation = b.activation.empty() ? "sigmoid" : b.activation;
        require(m->activation == "sigmoid" || m->activation == "none", TURBO_E_UNSUPPORTED,
                "reranker activation `" + m->activation + "` is not supported by the metal provider (sigmoid, none)");
    } else {
        fail(TURBO_E_UNSUPPORTED_TASK, "metal provider does not serve `" + b.kind + "` bundles (embedding, reranker)");
    }

    const auto st_path = b.artifact("safetensors");
    const auto cfg_path = b.artifact("hf_config");
    if (!st_path || !cfg_path) {
        std::string have;
        for (const auto &[k, v] : b.artifacts) {
            have += (have.empty() ? "" : ", ") + k;
        }
        fail(TURBO_E_BUNDLE_NO_ARTIFACT, "bundle `" + b.model_id + "` needs a `safetensors` and an `hf_config` artifact; it has: " +
                                             (have.empty() ? "(none)" : have));
    }
    // The architecture facts the checkpoint does not carry come from the
    // model's own config.json, hashed into the bundle like every other file.
    nlohmann::json cfg;
    {
        std::ifstream in(*cfg_path);
        require(in.good(), TURBO_E_BUNDLE_NOT_FOUND, "cannot open hf_config artifact " + *cfg_path);
        try {
            in >> cfg;
        } catch (const std::exception &e) {
            fail(TURBO_E_BUNDLE_INVALID, *cfg_path + ": " + e.what());
        }
    }
    const std::string model_type = cfg.value("model_type", "");
    require(model_type == "bert", TURBO_E_UNSUPPORTED,
            "metal provider runs BERT-family checkpoints; hf_config model_type is `" + model_type + "`");
    const std::string act = cfg.value("hidden_act", "gelu");
    require(act == "gelu", TURBO_E_UNSUPPORTED, "hf_config hidden_act `" + act + "` is not the erf GELU the kernels implement");
    m->hidden = cfg.value("hidden_size", 0u);
    m->intermediate = cfg.value("intermediate_size", 0u);
    m->layers = cfg.value("num_hidden_layers", 0u);
    m->heads = cfg.value("num_attention_heads", 0u);
    m->ln_eps = cfg.value("layer_norm_eps", 1e-12);
    const uint32_t max_position = cfg.value("max_position_embeddings", 512u);
    require(m->hidden > 0 && m->intermediate > 0 && m->layers > 0 && m->heads > 0 && m->hidden % m->heads == 0,
            TURBO_E_BUNDLE_INVALID, "hf_config must give hidden_size, intermediate_size, num_hidden_layers, and num_attention_heads");
    require(m->layers <= 24, TURBO_E_UNSUPPORTED, "metal provider handles at most 24 encoder layers");
    if (m->kind == Kind::Embedding) {
        require(b.dim == m->hidden, TURBO_E_BUNDLE_INVALID,
                "contract.dim " + std::to_string(b.dim) + " does not match hidden_size " + std::to_string(m->hidden));
    }
    require(b.max_seq <= max_position, TURBO_E_BUNDLE_INVALID,
            "contract.max_seq " + std::to_string(b.max_seq) + " exceeds the checkpoint's max_position_embeddings " +
                std::to_string(max_position));
    m->max_seq = b.max_seq == 0 ? max_position : b.max_seq;
    // limits.max_batch 0 means "the provider's default" by the bundle
    // contract (docs/bundles.md); this provider's is 32 rows.
    m->max_batch = b.max_batch == 0 ? 32 : b.max_batch;

    SafeTensors st;
    st.open(*st_path);
    const uint32_t ckpt_layers = st.encoder_layers();
    require(ckpt_layers == m->layers, TURBO_E_BUNDLE_INVALID,
            "hf_config says " + std::to_string(m->layers) + " layers but the checkpoint holds " + std::to_string(ckpt_layers));
    const Tensor word = st.get({"embeddings.word_embeddings.weight", "bert.embeddings.word_embeddings.weight"}, 2);
    const Tensor pos = st.get({"embeddings.position_embeddings.weight", "bert.embeddings.position_embeddings.weight"}, 2);
    const Tensor type = st.get({"embeddings.token_type_embeddings.weight", "bert.embeddings.token_type_embeddings.weight"}, 2);
    require(word.cols() == m->hidden && pos.cols() == m->hidden && type.cols() == m->hidden, TURBO_E_BUNDLE_INVALID,
            "embedding tables do not have hidden_size columns");
    require(b.vocab_size == 0 || word.rows() == b.vocab_size, TURBO_E_BUNDLE_INVALID,
            "checkpoint vocabulary of " + std::to_string(word.rows()) + " rows does not match contract.vocab_size " + std::to_string(b.vocab_size));
    m->word_rows = static_cast<uint32_t>(word.rows());
    m->pos_rows = static_cast<uint32_t>(pos.rows());
    m->type_rows = static_cast<uint32_t>(type.rows());
    require(m->max_seq <= m->pos_rows, TURBO_E_BUNDLE_INVALID,
            "contract.max_seq " + std::to_string(m->max_seq) + " exceeds the checkpoint's " + std::to_string(m->pos_rows) + " positions");
    require(m->pos_rows == max_position, TURBO_E_BUNDLE_INVALID,
            "hf_config max_position_embeddings " + std::to_string(max_position) + " does not match the checkpoint's " +
                std::to_string(m->pos_rows) + " position rows");
    m->word = upload(word);
    m->pos = upload(pos);
    m->type = upload(type);
    m->emb_ln_w = upload(st.get({"embeddings.LayerNorm.weight", "bert.embeddings.LayerNorm.weight"}, 1));
    m->emb_ln_b = upload(st.get({"embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.bias"}, 1));
    for (uint32_t i = 0; i < m->layers; ++i) {
        const std::string p = "encoder.layer." + std::to_string(i) + ".";
        const std::string q = "bert." + p;
        auto get = [&](const char *suffix, size_t rank, uint64_t rows, uint64_t cols) {
            const std::string a = p + suffix, bname = q + suffix;
            const Tensor t = st.get({a.c_str(), bname.c_str()}, rank);
            require(t.rows() == rows && (rank == 1 || t.cols() == cols), TURBO_E_BUNDLE_INVALID,
                    "tensor " + a + " has shape [" + std::to_string(t.rows()) + (rank == 2 ? ", " + std::to_string(t.cols()) : "") +
                        "], expected [" + std::to_string(rows) + (rank == 2 ? ", " + std::to_string(cols) : "") + "]");
            return upload(t);
        };
        const uint64_t H = m->hidden, I = m->intermediate;
        Layer L;
        L.q_w = get("attention.self.query.weight", 2, H, H);
        L.q_b = get("attention.self.query.bias", 1, H, 0);
        L.k_w = get("attention.self.key.weight", 2, H, H);
        L.k_b = get("attention.self.key.bias", 1, H, 0);
        L.v_w = get("attention.self.value.weight", 2, H, H);
        L.v_b = get("attention.self.value.bias", 1, H, 0);
        L.o_w = get("attention.output.dense.weight", 2, H, H);
        L.o_b = get("attention.output.dense.bias", 1, H, 0);
        L.ln1_w = get("attention.output.LayerNorm.weight", 1, H, 0);
        L.ln1_b = get("attention.output.LayerNorm.bias", 1, H, 0);
        L.ff_i_w = get("intermediate.dense.weight", 2, I, H);
        L.ff_i_b = get("intermediate.dense.bias", 1, I, 0);
        L.ff_o_w = get("output.dense.weight", 2, H, I);
        L.ff_o_b = get("output.dense.bias", 1, H, 0);
        L.ln2_w = get("output.LayerNorm.weight", 1, H, 0);
        L.ln2_b = get("output.LayerNorm.bias", 1, H, 0);
        m->layer.push_back(L);
    }
    if (m->kind == Kind::Reranker) {
        // The NSP pooler is optional in cross-encoder exports; the head is not.
        if (st.has("bert.pooler.dense.weight") || st.has("pooler.dense.weight")) {
            const Tensor pw = st.get({"bert.pooler.dense.weight", "pooler.dense.weight"}, 2);
            require(pw.rows() == m->hidden && pw.cols() == m->hidden, TURBO_E_BUNDLE_INVALID, "pooler weight is not [hidden, hidden]");
            m->pool_w = upload(pw);
            m->pool_b = upload(st.get({"bert.pooler.dense.bias", "pooler.dense.bias"}, 1));
            m->has_pooler = true;
        }
        const Tensor cw = st.get({"classifier.weight"}, 2);
        require(cw.rows() == 1 && cw.cols() == m->hidden, TURBO_E_UNSUPPORTED,
                "reranker classifier is [" + std::to_string(cw.rows()) + ", " + std::to_string(cw.cols()) + "]; the metal provider expects one logit over hidden_size");
        m->cls_w = upload(cw);
        m->cls_b = upload(st.get({"classifier.bias"}, 1));
    }
    m->dummy_bias = shared_buffer(4, nullptr);

    require(!b.tokenizer_path.empty(), TURBO_E_BUNDLE_INVALID, "bundle has no tokenizer.json; the metal provider tokenizes natively");
    require(b.tokenizer_kind == "wordpiece", TURBO_E_UNSUPPORTED,
            "metal provider tokenizes WordPiece (BERT) natively; tokenizer kind `" + b.tokenizer_kind + "` is not supported");
    const int rc = wordpiece_vocab_load(b.tokenizer_path.c_str(), &m->vocab);
    require(rc == WORDPIECE_OK && m->vocab != nullptr, TURBO_E_BUNDLE_INVALID,
            "tokenizer.json is not a supported uncased BERT WordPiece configuration (wordpiece status " + std::to_string(rc) + ")");
    // Every id the tokenizer can produce must have a row; a vocabulary
    // larger than the table would gather nothing for its tail.
    require(static_cast<uint32_t>(wordpiece_vocab_size(m->vocab)) == m->word_rows, TURBO_E_BUNDLE_INVALID,
            "tokenizer.json has " + std::to_string(wordpiece_vocab_size(m->vocab)) + " ids but the checkpoint's word table has " +
                std::to_string(m->word_rows) + " rows");
    return m;
}

void fill_model_info(const Model &m, turbo_model_info &out) {
    out.task = m.kind == Kind::Embedding ? TURBO_TASK_EMBED : TURBO_TASK_RERANK;
    out.kind = m.kind == Kind::Embedding ? TURBO_MODEL_EMBEDDING : TURBO_MODEL_RERANKER;
    out.modality = TURBO_MODALITY_TEXT;
    out.dim = m.kind == Kind::Embedding ? m.hidden : 0;
    out.n_labels = 0;
    out.pooling = m.kind == Kind::Embedding ? (m.pool == Pool::Mean ? TURBO_POOLING_MEAN : m.pool == Pool::Cls ? TURBO_POOLING_CLS : TURBO_POOLING_LAST) : 0;
    out.normalize = m.kind == Kind::Embedding ? (m.normalize ? TURBO_NORMALIZE_L2 : TURBO_NORMALIZE_NONE) : 0;
    out.max_seq = m.max_seq;
    out.max_batch = m.max_batch;
    out.dtype_used = TURBO_DTYPE_F32;
    out.stage_placement[TURBO_STAGE_TOKENIZE] = TURBO_STAGE_HOST;
    out.stage_placement[TURBO_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
    out.stage_placement[TURBO_STAGE_POOL] = m.kind == Kind::Embedding ? TURBO_STAGE_DEVICE : TURBO_STAGE_UNUSED;
    out.stage_placement[TURBO_STAGE_NORMALIZE] = (m.kind == Kind::Embedding && m.normalize) ? TURBO_STAGE_DEVICE : TURBO_STAGE_UNUSED;
    out.stage_placement[TURBO_STAGE_POSTPROCESS] = m.kind == Kind::Reranker ? TURBO_STAGE_DEVICE : TURBO_STAGE_UNUSED;
    out.fully_accelerated = 0; // the tokenizer is host work
    out.n_inputs = 0;
    out.n_outputs = 0;
    out.vocab_size = m.bundle.vocab_size;
    put_str(out.model_id, m.bundle.model_id);
    put_str(out.revision, m.bundle.revision);
    put_str(out.tokenizer_sha256, m.bundle.tokenizer_sha256);
    put_str(out.provider_id, std::string(kProviderId));
    put_str(out.prefix_query, m.bundle.prefix_query);
    put_str(out.prefix_document, m.bundle.prefix_document);
}

// ---------------------------------------------------------------------------
// Sessions: shared token rows, scratch for one sequence, shared results
// ---------------------------------------------------------------------------

struct Session {
    Model *model = nullptr;
    Context *ctx = nullptr;
    uint32_t batch = 0, seq = 0;
    // Token rows [batch * seq] i32 in shared memory: the host writes them,
    // the kernels read them, nothing is copied.
    id<MTLBuffer> ids, mask, types, posids;
    // Scratch for the whole batch.
    id<MTLBuffer> x, residual, q, k, v, attn, ctxb, inter, pooled;
    // Results: [batch, width] f32 shared; width = dim (embed) or 1 (rerank).
    id<MTLBuffer> out;
    uint32_t width = 0;
    std::vector<int32_t> sorted, scratch, pos_scratch;
    uint32_t n_rows = 0;
    turbo_embed_options eopts{};
    turbo_rerank_options ropts{};
    Buffer out_buf, sorted_buf;
    turbo_provider_output outputs[2]{};
    std::string name0, name1;
    std::atomic<bool> in_use{false}; // one caller at a time; see BusyGuard
    uint64_t runs = 0;

    Session(Model *m, Context *c, uint32_t b, uint32_t s) : model(m), ctx(c), batch(b), seq(s) {
        require(s <= m->max_seq, TURBO_E_CAPACITY, "session max_seq " + std::to_string(s) + " exceeds the model's " + std::to_string(m->max_seq));
        require(b <= m->max_batch, TURBO_E_CAPACITY, "session max_batch " + std::to_string(b) + " exceeds the model's " + std::to_string(m->max_batch));
        const uint64_t n = static_cast<uint64_t>(b) * s;
        ids = shared_buffer(n * 4, nullptr);
        mask = shared_buffer(n * 4, nullptr);
        types = shared_buffer(n * 4, nullptr);
        std::vector<int32_t> positions(n);
        for (uint32_t r = 0; r < b; ++r) {
            for (uint32_t t = 0; t < s; ++t) {
                positions[static_cast<size_t>(r) * s + t] = static_cast<int32_t>(t);
            }
        }
        posids = shared_buffer(n * 4, positions.data());
        {
            const int32_t pad = wordpiece_pad_id(m->vocab);
            auto *p = static_cast<int32_t *>(ids.contents);
            std::fill(p, p + n, pad);
        }
        // Scratch for the whole batch: the encoder runs every row at once.
        const uint64_t H = m->hidden, I = m->intermediate;
        const uint64_t tokens = n;
        x = shared_buffer(tokens * H * 4, nullptr);
        residual = shared_buffer(tokens * H * 4, nullptr);
        q = shared_buffer(tokens * H * 4, nullptr);
        k = shared_buffer(tokens * H * 4, nullptr);
        v = shared_buffer(tokens * H * 4, nullptr);
        attn = shared_buffer(static_cast<uint64_t>(b) * m->heads * s * s * 4, nullptr);
        ctxb = shared_buffer(tokens * H * 4, nullptr);
        inter = shared_buffer(tokens * I * 4, nullptr);
        pooled = shared_buffer(static_cast<uint64_t>(b) * H * 4, nullptr);
        width = m->kind == Kind::Embedding ? m->hidden : 1;
        out = shared_buffer(static_cast<uint64_t>(b) * width * 4, nullptr);
        sorted.assign(b, 0);
        scratch.assign(static_cast<size_t>(s) * 16, 0);
        pos_scratch.assign(s, 0);
        out_buf.mtl = out;
        out_buf.bytes = static_cast<uint64_t>(b) * width * 4;
        sorted_buf.host = sorted.data();
        sorted_buf.bytes = static_cast<uint64_t>(b) * 4;
        sorted_buf.owns_host = false;
        name0 = m->kind == Kind::Embedding ? "embeddings" : "scores";
        name1 = "sorted";
    }

    uint32_t budget(uint32_t max_tokens) const {
        require(max_tokens <= seq, TURBO_E_CAPACITY,
                "max_tokens " + std::to_string(max_tokens) + " exceeds the session's max_seq " + std::to_string(seq), 3);
        const uint32_t b = max_tokens == 0 ? seq : max_tokens;
        require(b >= 2, TURBO_E_CAPACITY, "token budget must be at least 2 for [CLS] and [SEP]", 3);
        return b;
    }

    int32_t *ids_row(uint32_t r) { return static_cast<int32_t *>(ids.contents) + static_cast<size_t>(r) * seq; }
    int32_t *mask_row(uint32_t r) { return static_cast<int32_t *>(mask.contents) + static_cast<size_t>(r) * seq; }
    int32_t *types_row(uint32_t r) { return static_cast<int32_t *>(types.contents) + static_cast<size_t>(r) * seq; }

    /// Tokenize text (with optional prefix) into row r: [CLS] prefix text [SEP] pad.
    void encode_row(uint32_t r, const std::string &prefix, turbo_text text, uint32_t truncate, uint32_t budget) {
        int32_t *row_ids = ids_row(r);
        int32_t *row_mask = mask_row(r);
        const uint32_t content_budget = budget - 2;
        const wordpiece_vocab *vv = model->vocab;
        size_t n_prefix = 0;
        if (!prefix.empty()) {
            require(wordpiece_tokenize(vv, prefix.data(), prefix.size(), scratch.data(), scratch.size(), 4, &n_prefix) == WORDPIECE_OK,
                    TURBO_E_INTERNAL, "prefix tokenization failed");
            require(n_prefix <= content_budget, TURBO_E_CAPACITY, "prompt prefix alone exceeds the token budget");
        }
        size_t n_text = 0;
        require(text.len == 0 || text.ptr != nullptr, TURBO_E_INVALID_ARGUMENT, "text view has NULL ptr");
        const int cnt = wordpiece_tokenize(vv, text.ptr, static_cast<size_t>(text.len), nullptr, 0, 4, &n_text);
        require(cnt != WORDPIECE_ERR_INVALID_ARGUMENT, TURBO_E_INVALID_UTF8, "text is not valid UTF-8");
        require(cnt == WORDPIECE_OK, TURBO_E_INTERNAL, "tokenizer failed with status " + std::to_string(cnt));
        const size_t avail = content_budget - n_prefix;
        size_t skip = 0, take = n_text;
        if (n_text > avail) {
            switch (truncate) {
            case TURBO_TRUNCATE_NONE:
                fail(TURBO_E_CAPACITY, "input row " + std::to_string(r) + " tokenizes to " + std::to_string(n_text + n_prefix + 2) +
                                           " tokens but the budget is " + std::to_string(budget) + " and truncation is NONE");
            case TURBO_TRUNCATE_LEFT:
                skip = n_text - avail;
                take = avail;
                break;
            default:
                take = avail;
            }
        }
        uint32_t col = 0;
        row_ids[col++] = wordpiece_cls_id(vv);
        for (size_t i = 0; i < n_prefix; ++i) {
            row_ids[col++] = scratch[i];
        }
        if (take > 0) {
            if (skip == 0) {
                size_t n = 0;
                require(wordpiece_tokenize(vv, text.ptr, static_cast<size_t>(text.len), row_ids + col, take, 4, &n) == WORDPIECE_OK,
                        TURBO_E_INTERNAL, "text tokenization failed");
                col += static_cast<uint32_t>(std::min(n, take));
            } else {
                if (n_text > scratch.size()) {
                    scratch.resize(n_text);
                }
                size_t n = 0;
                require(wordpiece_tokenize(vv, text.ptr, static_cast<size_t>(text.len), scratch.data(), scratch.size(), 4, &n) == WORDPIECE_OK,
                        TURBO_E_INTERNAL, "text tokenization failed");
                for (size_t i = skip; i < skip + take; ++i) {
                    row_ids[col++] = scratch[i];
                }
            }
        }
        row_ids[col++] = wordpiece_sep_id(vv);
        for (uint32_t c = 0; c < col; ++c) {
            row_mask[c] = 1;
        }
        const int32_t pad = wordpiece_pad_id(vv);
        for (uint32_t c = col; c < seq; ++c) {
            row_ids[c] = pad;
            row_mask[c] = 0;
        }
        std::fill(types_row(r), types_row(r) + seq, 0);
    }

    // --- GPU encoding helpers -------------------------------------------------

    struct Enc {
        id<MTLComputeCommandEncoder> enc;
        /// One thread per grid point; Metal pads the last threadgroup.
        void dispatch(id<MTLComputePipelineState> pso, NSUInteger x, NSUInteger y = 1, NSUInteger z = 1) {
            if (x == 0 || y == 0 || z == 0) {
                return;
            }
            [enc setComputePipelineState:pso];
            const NSUInteger tw = std::min<NSUInteger>(x, pso.maxTotalThreadsPerThreadgroup);
            [enc dispatchThreads:MTLSizeMake(x, y, z) threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
        }
        /// Whole threadgroups of tg_x by tg_y threads, ceil(x / tg_x) by
        /// ceil(y / tg_y) of them: for kernels whose threads cooperate on a
        /// tile and must all exist.
        void dispatch_groups(id<MTLComputePipelineState> pso, NSUInteger x, NSUInteger y, NSUInteger tg_x, NSUInteger tg_y) {
            if (x == 0 || y == 0) {
                return;
            }
            require(pso.maxTotalThreadsPerThreadgroup >= tg_x * tg_y, TURBO_E_DEVICE_UNAVAILABLE,
                    "the tiled kernel needs " + std::to_string(tg_x * tg_y) + " threads per threadgroup but this pipeline allows " +
                        std::to_string(pso.maxTotalThreadsPerThreadgroup));
            [enc setComputePipelineState:pso];
            [enc dispatchThreadgroups:MTLSizeMake((x + tg_x - 1) / tg_x, (y + tg_y - 1) / tg_y, 1)
                threadsPerThreadgroup:MTLSizeMake(tg_x, tg_y, 1)];
        }
    };

    /// Encode the BERT forward of the first `n` rows as one batch; the
    /// final hidden states are in `x`, row r at r * seq * hidden.
    void encode_batch(Enc &e, uint32_t n) {
        const Pipelines &P = state().p;
        const Model &m = *model;
        const uint32_t H = m.hidden, I = m.intermediate, heads = m.heads, dh = H / heads;
        const uint32_t N = n * seq; // tokens in the batch
        const uint32_t hidden_n = N * H, inter_n = N * I;
        struct {
            uint32_t seq, hidden, word_rows, pos_rows, type_rows;
        } ep{N, H, m.word_rows, m.pos_rows, m.type_rows};
        [e.enc setComputePipelineState:P.embed];
        [e.enc setBuffer:x offset:0 atIndex:0];
        [e.enc setBuffer:ids offset:0 atIndex:1];
        [e.enc setBuffer:posids offset:0 atIndex:2];
        [e.enc setBuffer:types offset:0 atIndex:3];
        [e.enc setBuffer:m.word offset:0 atIndex:4];
        [e.enc setBuffer:m.pos offset:0 atIndex:5];
        [e.enc setBuffer:m.type offset:0 atIndex:6];
        [e.enc setBytes:&ep length:sizeof(ep) atIndex:7];
        e.dispatch(P.embed, N);
        struct {
            uint32_t seq, hidden;
            float eps;
        } lp{N, H, m.ln_eps};
        auto layer_norm = [&](id<MTLBuffer> w, id<MTLBuffer> bias) {
            [e.enc setComputePipelineState:P.ln];
            [e.enc setBuffer:x offset:0 atIndex:0];
            [e.enc setBuffer:w offset:0 atIndex:1];
            [e.enc setBuffer:bias offset:0 atIndex:2];
            [e.enc setBytes:&lp length:sizeof(lp) atIndex:3];
            e.dispatch(P.ln, N);
        };
        layer_norm(m.emb_ln_w, m.emb_ln_b);
        auto elem = [&](id<MTLComputePipelineState> pso, id<MTLBuffer> a, id<MTLBuffer> b2, id<MTLBuffer> c2, uint32_t count) {
            struct {
                uint32_t n;
            } p{count};
            [e.enc setComputePipelineState:pso];
            [e.enc setBuffer:a offset:0 atIndex:0];
            uint32_t idx = 1;
            if (b2 != nil) {
                [e.enc setBuffer:b2 offset:0 atIndex:idx++];
            }
            if (c2 != nil) {
                [e.enc setBuffer:c2 offset:0 atIndex:idx++];
            }
            [e.enc setBytes:&p length:sizeof(p) atIndex:idx];
            e.dispatch(pso, count);
        };
        auto linear = [&](id<MTLBuffer> in, id<MTLBuffer> w, id<MTLBuffer> bias, id<MTLBuffer> y, uint32_t kdim, uint32_t outdim) {
            struct {
                uint32_t seq, k, out, has_bias;
            } p{N, kdim, outdim, bias != nil ? 1u : 0u};
            [e.enc setComputePipelineState:P.linear];
            [e.enc setBuffer:in offset:0 atIndex:0];
            [e.enc setBuffer:w offset:0 atIndex:1];
            [e.enc setBuffer:(bias != nil ? bias : m.dummy_bias) offset:0 atIndex:2];
            [e.enc setBuffer:y offset:0 atIndex:3];
            [e.enc setBytes:&p length:sizeof(p) atIndex:4];
            e.dispatch_groups(P.linear, N, outdim, 16, 16);
        };
        struct {
            uint32_t seq, hidden, heads, dh;
            float scale;
            uint32_t batch;
        } ap{seq, H, heads, dh, 1.0f / std::sqrt(static_cast<float>(dh)), n};
        for (const Layer &L : m.layer) {
            elem(P.copyf, residual, x, nil, hidden_n);
            linear(x, L.q_w, L.q_b, q, H, H);
            linear(x, L.k_w, L.k_b, k, H, H);
            linear(x, L.v_w, L.v_b, v, H, H);
            [e.enc setComputePipelineState:P.scores];
            [e.enc setBuffer:q offset:0 atIndex:0];
            [e.enc setBuffer:k offset:0 atIndex:1];
            [e.enc setBuffer:attn offset:0 atIndex:2];
            [e.enc setBuffer:mask offset:0 atIndex:3];
            [e.enc setBytes:&ap length:sizeof(ap) atIndex:4];
            e.dispatch(P.scores, seq, seq, static_cast<NSUInteger>(n) * heads);
            [e.enc setComputePipelineState:P.softmax];
            [e.enc setBuffer:attn offset:0 atIndex:0];
            [e.enc setBytes:&ap length:sizeof(ap) atIndex:1];
            e.dispatch(P.softmax, seq, static_cast<NSUInteger>(n) * heads);
            [e.enc setComputePipelineState:P.ctx];
            [e.enc setBuffer:attn offset:0 atIndex:0];
            [e.enc setBuffer:v offset:0 atIndex:1];
            [e.enc setBuffer:ctxb offset:0 atIndex:2];
            [e.enc setBytes:&ap length:sizeof(ap) atIndex:3];
            e.dispatch(P.ctx, seq, H, n);
            linear(ctxb, L.o_w, L.o_b, q, H, H);
            elem(P.residual, x, q, residual, hidden_n);
            layer_norm(L.ln1_w, L.ln1_b);
            elem(P.copyf, residual, x, nil, hidden_n);
            linear(x, L.ff_i_w, L.ff_i_b, inter, H, I);
            elem(P.gelu, inter, nil, nil, inter_n);
            linear(inter, L.ff_o_w, L.ff_o_b, x, I, H);
            elem(P.add, x, residual, nil, hidden_n);
            layer_norm(L.ln2_w, L.ln2_b);
        }
    }

    turbo_provider_result run() {
        const Pipelines &P = state().p;
        const Model &m = *model;
        const uint32_t H = m.hidden;
        Pool pool = eopts.pooling == TURBO_POOLING_MODEL ? m.pool : eopts.pooling == TURBO_POOLING_MEAN ? Pool::Mean : eopts.pooling == TURBO_POOLING_CLS ? Pool::Cls : Pool::Last;
        const bool normalize = eopts.normalize == TURBO_NORMALIZE_MODEL ? m.normalize : eopts.normalize == TURBO_NORMALIZE_L2;
        const uint32_t dim_out = m.kind == Kind::Embedding ? (eopts.output_dim == 0 ? H : eopts.output_dim) : 1;
        require(dim_out <= H, TURBO_E_UNSUPPORTED_OPTION, "output_dim " + std::to_string(dim_out) + " exceeds the model dimension " + std::to_string(H), 7);
        @autoreleasepool {
            id<MTLCommandBuffer> cmd = [ctx->queue commandBuffer];
            require(cmd != nil, TURBO_E_RUNTIME, "Metal command buffer allocation failed");
            Enc e{[cmd computeCommandEncoder]};
            require(e.enc != nil, TURBO_E_RUNTIME, "Metal compute encoder creation failed");
            encode_batch(e, n_rows);
            if (m.kind == Kind::Embedding) {
                struct {
                    uint32_t seq, hidden, mode, batch;
                } sp{seq, H, pool == Pool::Mean ? 1u : pool == Pool::Cls ? 2u : 3u, n_rows};
                // Pool into the result rows directly when the full width is
                // wanted; otherwise pool into scratch and copy each head.
                const bool direct = dim_out == H;
                [e.enc setComputePipelineState:P.sentence_pool];
                [e.enc setBuffer:x offset:0 atIndex:0];
                [e.enc setBuffer:mask offset:0 atIndex:1];
                [e.enc setBuffer:(direct ? out : pooled) offset:0 atIndex:2];
                [e.enc setBytes:&sp length:sizeof(sp) atIndex:3];
                e.dispatch(P.sentence_pool, H, n_rows);
                if (!direct) {
                    struct {
                        uint32_t n;
                    } cp{dim_out};
                    for (uint32_t r = 0; r < n_rows; ++r) {
                        [e.enc setComputePipelineState:P.copyf];
                        [e.enc setBuffer:out offset:static_cast<NSUInteger>(r) * dim_out * 4 atIndex:0];
                        [e.enc setBuffer:pooled offset:static_cast<NSUInteger>(r) * H * 4 atIndex:1];
                        [e.enc setBytes:&cp length:sizeof(cp) atIndex:2];
                        e.dispatch(P.copyf, dim_out);
                    }
                }
                if (normalize) {
                    struct {
                        uint32_t n;
                    } np{dim_out};
                    [e.enc setComputePipelineState:P.l2];
                    [e.enc setBuffer:out offset:0 atIndex:0];
                    [e.enc setBytes:&np length:sizeof(np) atIndex:1];
                    e.dispatch(P.l2, n_rows);
                }
            } else {
                const bool sigmoid = m.activation == "sigmoid" && ropts.raw_scores == 0;
                struct {
                    uint32_t hidden;
                } pp{H};
                struct {
                    uint32_t hidden, out, activation;
                } hp{H, 1u, sigmoid ? 1u : 0u};
                for (uint32_t r = 0; r < n_rows; ++r) {
                    // The [CLS] row is the first row of this sequence.
                    const NSUInteger cls_off = static_cast<NSUInteger>(r) * seq * H * 4;
                    id<MTLBuffer> head_in = x;
                    NSUInteger head_off = cls_off;
                    if (m.has_pooler) {
                        [e.enc setComputePipelineState:P.pooler];
                        [e.enc setBuffer:x offset:cls_off atIndex:0];
                        [e.enc setBuffer:m.pool_w offset:0 atIndex:1];
                        [e.enc setBuffer:m.pool_b offset:0 atIndex:2];
                        [e.enc setBuffer:pooled offset:static_cast<NSUInteger>(r) * H * 4 atIndex:3];
                        [e.enc setBytes:&pp length:sizeof(pp) atIndex:4];
                        e.dispatch(P.pooler, H);
                        head_in = pooled;
                        head_off = static_cast<NSUInteger>(r) * H * 4;
                    }
                    [e.enc setComputePipelineState:P.classifier];
                    [e.enc setBuffer:head_in offset:head_off atIndex:0];
                    [e.enc setBuffer:m.cls_w offset:0 atIndex:1];
                    [e.enc setBuffer:m.cls_b offset:0 atIndex:2];
                    [e.enc setBuffer:out offset:static_cast<NSUInteger>(r) * 4 atIndex:3];
                    [e.enc setBytes:&hp length:sizeof(hp) atIndex:4];
                    e.dispatch(P.classifier, 1);
                }
            }
            [e.enc endEncoding];
            [cmd commit];
            [cmd waitUntilCompleted];
            require(cmd.error == nil, TURBO_E_RUNTIME,
                    std::string("Metal command buffer failed: ") + (cmd.error ? cmd.error.localizedDescription.UTF8String : "unknown"));
        }
        ++runs;
        turbo_provider_result res{};
        res.struct_size = sizeof(turbo_provider_result);
        outputs[0] = turbo_provider_output{};
        outputs[0].struct_size = sizeof(turbo_provider_output);
        outputs[0].name = turbo_text{name0.data(), name0.size()};
        if (m.kind == Kind::Embedding) {
            outputs[0].ndim = 2;
            outputs[0].shape[0] = n_rows;
            outputs[0].shape[1] = dim_out;
            out_buf.desc = packed_desc(TURBO_PLACE_SHARED, TURBO_DTYPE_F32, {batch, dim_out}, 4);
        } else {
            outputs[0].ndim = 1;
            outputs[0].shape[0] = n_rows;
            out_buf.desc = packed_desc(TURBO_PLACE_SHARED, TURBO_DTYPE_F32, {batch}, 4);
        }
        out_buf.bytes = out_buf.desc.bytes;
        outputs[0].buffer = describe(&out_buf);
        res.n_outputs = 1;
        res.outputs = outputs;
        if (m.kind == Kind::Reranker && (ropts.return_sorted != 0 || ropts.top_n != 0)) {
            const float *s = static_cast<const float *>(out.contents);
            std::iota(sorted.begin(), sorted.begin() + n_rows, 0);
            std::stable_sort(sorted.begin(), sorted.begin() + n_rows, [s](int32_t a, int32_t b) { return s[a] > s[b]; });
            const uint32_t kk = ropts.top_n == 0 ? n_rows : std::min(ropts.top_n, n_rows);
            outputs[1] = turbo_provider_output{};
            outputs[1].struct_size = sizeof(turbo_provider_output);
            outputs[1].name = turbo_text{name1.data(), name1.size()};
            outputs[1].ndim = 1;
            outputs[1].shape[0] = kk;
            sorted_buf.desc = packed_desc(TURBO_PLACE_HOST, TURBO_DTYPE_I32, {batch}, 4);
            outputs[1].buffer = describe(&sorted_buf);
            res.n_outputs = 2;
        }
        return res;
    }
};

// ---------------------------------------------------------------------------
// Vtable
// ---------------------------------------------------------------------------

extern "C" {

static int32_t x_device_count(void *, uint32_t *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        *out = state().device != nil ? 1u : 0u;
    });
}

static int32_t x_device_info(void *, uint32_t ordinal, turbo_device_info *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_device_info>("turbo_device_info", out->struct_size);
        require_device(ordinal);
        const State &s = state();
        turbo_device_info full{};
        full.struct_size = out->struct_size;
        // Apple GPUs share the SoC's memory with the CPU: an integrated GPU.
        full.kind = TURBO_DEVICE_IGPU;
        full.ordinal = 0;
        full.vendor_id = 0x106b; // Apple
        full.caps = s.ready() ? kCaps : 0;
        full.memory_total = s.device.recommendedMaxWorkingSetSize;
        full.memory_free = 0;
        put_str(full.name, std::string(s.device.name.UTF8String) + " (Metal)");
        put_str(full.vendor, std::string("Apple"));
        put_str(full.provider_id, std::string(kProviderId));
        put_str(full.provider_version, std::string(kProviderVersion));
        put_str(full.runtime_version, std::string("Metal, MSL compiled at runtime"));
        put_str(full.driver_version, "macOS " + s.os);
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_capability(void *, uint32_t ordinal, uint32_t task, uint32_t modality, turbo_capability *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_capability>("turbo_capability", out->struct_size);
        require_device(ordinal);
        turbo_capability full{};
        full.struct_size = out->struct_size;
        if (offers(task, modality)) {
            full.status = TURBO_CAP_EXPERIMENTAL;
            full.dtype = TURBO_DTYPE_F32;
            full.reference_dtype = TURBO_DTYPE_F32;
            full.deterministic = 1;
            put_str(full.notes, "Metal kernels compiled at runtime; fp32; pooling, L2, and the head on the GPU; receipt testdata/receipts/turbo/metal-2026-09-22.json");
        } else {
            full.status = TURBO_CAP_UNSUPPORTED;
            put_str(full.notes, !state().ready() ? state().why : std::string("metal provider offers EMBED and RERANK on TEXT"));
        }
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_can_run(void *, uint32_t ordinal, turbo_text bundle_dir, uint32_t task, uint32_t modality, turbo_error *err) {
    return boundary(err, [&] {
        require_device(ordinal);
        require(state().ready(), TURBO_E_DEVICE_UNAVAILABLE, state().why);
        require(offers(task, modality), TURBO_E_UNSUPPORTED_TASK,
                "metal provider does not offer task " + std::to_string(task) + " for modality " + std::to_string(modality));
        const Bundle b = Bundle::read(text_of(bundle_dir));
        require((task == TURBO_TASK_EMBED && b.kind == "embedding") || (task == TURBO_TASK_RERANK && b.kind == "reranker"),
                TURBO_E_UNSUPPORTED_TASK, "bundle kind `" + b.kind + "` does not match the task");
        require(b.artifact("safetensors") && b.artifact("hf_config"), TURBO_E_BUNDLE_NO_ARTIFACT,
                "bundle `" + b.model_id + "` needs `safetensors` and `hf_config` artifacts");
        require(b.tokenizer_kind == "wordpiece", TURBO_E_UNSUPPORTED, "metal provider needs a WordPiece tokenizer.json");
    });
}

static int32_t x_context_create(void *, uint32_t ordinal, const turbo_context_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
        if (desc != nullptr) {
            const turbo_context_desc d = read_prefix(desc, "turbo_context_desc");
            require(d.next == nullptr, TURBO_E_NOT_IMPLEMENTED, "external queue import is not implemented");
            reject_unknown(options_of(d.options, d.n_options, "context"), {}, "metal context");
        }
        require_device(ordinal);
        *out = new Context();
    });
}

static void x_context_release(void *ctx) { release<Context>(ctx); }

static int32_t x_buffer_alloc(void *ctx, const turbo_buffer_desc *desc, turbo_provider_buffer *out, turbo_error *err) {
    return boundary(err, [&] {
        require(ctx != nullptr && desc != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        const turbo_buffer_desc d = read_prefix(desc, "turbo_buffer_desc");
        check_size<turbo_provider_buffer>("turbo_provider_buffer", out->struct_size);
        require_receives(out->struct_size, TURBO_PC_FIELD_END(turbo_provider_buffer, handle), "turbo_provider_buffer", "handle");
        auto b = make_buffer(d);
        write_sized(out, describe(b.get()));
        b.release();
    });
}

static int32_t x_buffer_import(void *ctx, const turbo_buffer_desc *desc, const turbo_native_handle *handle, turbo_provider_buffer *out,
                               turbo_error *err) {
    return boundary(err, [&] {
        require(ctx != nullptr && desc != nullptr && handle != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        const turbo_buffer_desc d = read_prefix(desc, "turbo_buffer_desc");
        const turbo_native_handle h = read_prefix(handle, "turbo_native_handle");
        check_size<turbo_provider_buffer>("turbo_provider_buffer", out->struct_size);
        require_receives(out->struct_size, TURBO_PC_FIELD_END(turbo_provider_buffer, handle), "turbo_provider_buffer", "handle");
        auto b = import_buffer(d, h);
        write_sized(out, describe(b.get()));
        b.release();
    });
}

static int32_t x_buffer_read(void *buf, void *dst, uint64_t bytes, turbo_error *err) {
    return boundary(err, [&] {
        auto *b = static_cast<Buffer *>(buf);
        require(b != nullptr && dst != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        require(bytes == b->bytes, TURBO_E_CAPACITY, "destination size " + std::to_string(bytes) + " does not match the buffer's " + std::to_string(b->bytes));
        std::memcpy(dst, b->contents(), static_cast<size_t>(bytes));
    });
}

static int32_t x_buffer_export(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err) {
    return boundary(err, [&] {
        auto *b = static_cast<Buffer *>(buf);
        require(b != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        check_size<turbo_native_handle>("turbo_native_handle", out->struct_size);
        turbo_native_handle full{};
        full.offset = 0;
        full.aux = 0;
        if (b->mtl != nil) {
            require(kind == TURBO_HANDLE_MTL_BUFFER || kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED,
                    "shared buffers export TURBO_HANDLE_MTL_BUFFER (the id<MTLBuffer>) or TURBO_HANDLE_HOST_PTR (its contents)");
            full.kind = kind;
            full.handle = kind == TURBO_HANDLE_MTL_BUFFER ? reinterpret_cast<uint64_t>((__bridge void *)b->mtl)
                                                          : reinterpret_cast<uint64_t>(b->mtl.contents);
        } else {
            require(kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED, "host buffers export TURBO_HANDLE_HOST_PTR only");
            full.kind = TURBO_HANDLE_HOST_PTR;
            full.handle = reinterpret_cast<uint64_t>(b->host);
        }
        write_sized(out, full);
    });
}

static void x_buffer_release(void *buf) { release<Buffer>(buf); }

static int32_t x_model_load(void *ctx, turbo_text bundle_dir, const turbo_model_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        require(ctx != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        *out = nullptr;
        std::map<std::string, std::string> opts;
        if (desc != nullptr) {
            const turbo_model_desc d = read_prefix(desc, "turbo_model_desc");
            opts = options_of(d.options, d.n_options, "model");
        }
        @autoreleasepool {
            *out = load_model(static_cast<Context *>(ctx), text_of(bundle_dir), opts).release();
        }
    });
}

static int32_t x_model_info(void *model, turbo_model_info *out, turbo_error *err) {
    return boundary(err, [&] {
        require(model != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        check_size<turbo_model_info>("turbo_model_info", out->struct_size);
        turbo_model_info full{};
        full.struct_size = out->struct_size;
        fill_model_info(*static_cast<Model *>(model), full);
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_model_label(void *model, uint32_t, turbo_text *, turbo_error *err) {
    return boundary(err, [&] {
        require(model != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        fail(TURBO_E_INVALID_ARGUMENT, "embedding and reranker models have no labels");
    });
}

static void x_model_release(void *model) { release<Model>(model); }

static int32_t x_session_create(void *model, const turbo_session_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        auto *m = static_cast<Model *>(model);
        require(m != nullptr && desc != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        *out = nullptr;
        const turbo_session_desc d = read_prefix(desc, "turbo_session_desc");
        require(d.next == nullptr, TURBO_E_INVALID_ARGUMENT, "turbo_session_desc.next must be NULL");
        reject_unknown(options_of(d.options, d.n_options, "session"), {}, "metal session");
        require(d.max_batch >= 1 && d.max_seq >= 2, TURBO_E_INVALID_ARGUMENT, "session needs max_batch >= 1 and max_seq >= 2");
        @autoreleasepool {
            *out = new Session(m, m->ctx, d.max_batch, d.max_seq);
        }
    });
}

static Session &sess(void *s) {
    require(s != nullptr, TURBO_E_INVALID_HANDLE, "session is NULL");
    return *static_cast<Session *>(s);
}

static int32_t x_session_write_text(void *s, const turbo_text *texts, uint32_t count, const turbo_embed_options *opts, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(S.model->kind == Kind::Embedding, TURBO_E_UNSUPPORTED_TASK, "write_text needs an embedding model");
        require(texts != nullptr, TURBO_E_INVALID_ARGUMENT, "texts is NULL");
        require(count >= 1 && count <= S.batch, TURBO_E_CAPACITY,
                "count " + std::to_string(count) + " is outside 1..max_batch (" + std::to_string(S.batch) + ")");
        turbo_embed_options o{};
        o.struct_size = sizeof(o);
        if (opts != nullptr) {
            check_size<turbo_embed_options>("turbo_embed_options", opts->struct_size);
            std::memcpy(&o, opts, std::min<size_t>(opts->struct_size, sizeof(o)));
        }
        checked_truncate(o.truncate, 2);
        checked_prompt_role(o.prompt_role, 4);
        checked_normalize(o.normalize, 5);
        checked_pooling(o.pooling, 6);
        require(o.output_dim <= S.model->hidden, TURBO_E_UNSUPPORTED_OPTION,
                "output_dim " + std::to_string(o.output_dim) + " exceeds the model dimension " + std::to_string(S.model->hidden), 7);
        switch (o.output_dtype) {
        case TURBO_OUTPUT_MODEL:
        case TURBO_OUTPUT_F32:
            break;
        case TURBO_OUTPUT_F16:
        case TURBO_OUTPUT_I8:
            fail(TURBO_E_UNSUPPORTED_OPTION, "metal provider writes f32 results only", 8);
        default:
            fail(TURBO_E_INVALID_ENUM, "output_dtype " + std::to_string(o.output_dtype) + " is not a TURBO_OUTPUT_* value", 8);
        }
        S.n_rows = 0;
        S.eopts = o;
        const uint32_t budget = S.budget(o.max_tokens);
        const std::string &prefix = o.prompt_role == TURBO_PROMPT_QUERY ? S.model->bundle.prefix_query
                                    : o.prompt_role == TURBO_PROMPT_DOCUMENT ? S.model->bundle.prefix_document
                                                                             : std::string();
        for (uint32_t r = 0; r < count; ++r) {
            S.encode_row(r, prefix, texts[r], o.truncate, budget);
        }
        S.n_rows = count;
    });
}

static int32_t x_session_write_tokens(void *s, const turbo_token_batch *batch, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(batch != nullptr, TURBO_E_INVALID_ARGUMENT, "batch is NULL");
        const turbo_token_batch tb = read_prefix(batch, "turbo_token_batch");
        check_token_batch(tb);
        require(tb.batch >= 1 && tb.batch <= S.batch && tb.seq >= 1 && tb.seq <= S.seq, TURBO_E_CAPACITY,
                "token batch exceeds the session shape");
        S.n_rows = 0;
        const uint32_t stride = tb.row_stride == 0 ? tb.seq : tb.row_stride;
        // Every id (and type) is checked here, so a bad token is a write-time
        // argument error and never a silent row or a fault inside a run.
        const int32_t n_ids = static_cast<int32_t>(S.model->word_rows);
        for (uint32_t r = 0; r < tb.batch; ++r) {
            for (uint32_t c = 0; c < tb.seq; ++c) {
                const int32_t id = tb.ids[static_cast<size_t>(r) * stride + c];
                require(id >= 0 && id < n_ids, TURBO_E_INVALID_ARGUMENT,
                        "row " + std::to_string(r) + " column " + std::to_string(c) + ": token id " + std::to_string(id) +
                            " is outside the " + std::to_string(n_ids) + "-entry vocabulary");
                if (tb.types != nullptr) {
                    const int32_t ty = tb.types[static_cast<size_t>(r) * stride + c];
                    require(ty >= 0 && ty < static_cast<int32_t>(S.model->type_rows), TURBO_E_INVALID_ARGUMENT,
                            "row " + std::to_string(r) + " column " + std::to_string(c) + ": token type " + std::to_string(ty) +
                                " is outside the checkpoint's " + std::to_string(S.model->type_rows) + " token types");
                }
            }
        }
        const int32_t pad = wordpiece_pad_id(S.model->vocab);
        for (uint32_t r = 0; r < tb.batch; ++r) {
            const size_t src = static_cast<size_t>(r) * stride;
            std::memcpy(S.ids_row(r), tb.ids + src, tb.seq * 4);
            std::memcpy(S.mask_row(r), tb.mask + src, tb.seq * 4);
            if (tb.types != nullptr) {
                std::memcpy(S.types_row(r), tb.types + src, tb.seq * 4);
            } else {
                std::fill(S.types_row(r), S.types_row(r) + tb.seq, 0);
            }
            for (uint32_t c = tb.seq; c < S.seq; ++c) {
                S.ids_row(r)[c] = pad;
                S.mask_row(r)[c] = 0;
                S.types_row(r)[c] = 0;
            }
        }
        S.eopts = turbo_embed_options{};
        S.eopts.struct_size = sizeof(turbo_embed_options);
        S.ropts = turbo_rerank_options{};
        S.ropts.struct_size = sizeof(turbo_rerank_options);
        S.n_rows = tb.batch;
    });
}

static int32_t x_session_write_pairs(void *s, const turbo_text *query, const turbo_text *docs, uint32_t count, const turbo_rerank_options *opts,
                                     turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(S.model->kind == Kind::Reranker, TURBO_E_UNSUPPORTED_TASK, "write_pairs needs a reranker model");
        require(query != nullptr && docs != nullptr, TURBO_E_INVALID_ARGUMENT, "query and docs must not be NULL");
        require(count >= 1 && count <= S.batch, TURBO_E_CAPACITY,
                "count " + std::to_string(count) + " is outside 1..max_batch (" + std::to_string(S.batch) + ")");
        turbo_rerank_options o{};
        o.struct_size = sizeof(o);
        if (opts != nullptr) {
            check_size<turbo_rerank_options>("turbo_rerank_options", opts->struct_size);
            std::memcpy(&o, opts, std::min<size_t>(opts->struct_size, sizeof(o)));
        }
        require(o.raw_scores == 0 || o.raw_scores == 1, TURBO_E_INVALID_ENUM, "raw_scores must be 0 or 1", 6);
        S.n_rows = 0;
        S.ropts = o;
        const uint32_t budget = S.budget(o.max_tokens);
        uint32_t trunc = 0;
        switch (checked_truncate(o.truncate, 2)) {
        case TURBO_TRUNCATE_MODEL:
            trunc = WORDPIECE_TRUNC_LONGEST_FIRST;
            break;
        case TURBO_TRUNCATE_RIGHT:
            trunc = WORDPIECE_TRUNC_QUERY_PRIORITY;
            break;
        case TURBO_TRUNCATE_NONE:
            trunc = WORDPIECE_TRUNC_ERROR;
            break;
        default:
            fail(TURBO_E_UNSUPPORTED_OPTION, "left truncation of query/document pairs is not offered: the packer truncates from the right of the pair only", 2);
        }
        const std::string q = text_of(*query);
        for (uint32_t r = 0; r < count; ++r) {
            const std::string d = text_of(docs[r]);
            const int rc = wordpiece_pack_pair(S.model->vocab, q.data(), q.size(), d.data(), d.size(), S.ids_row(r), S.mask_row(r),
                                               S.types_row(r), S.pos_scratch.data(), S.seq, S.seq, 4, trunc, budget);
            require(rc != WORDPIECE_ERR_TOO_LONG, TURBO_E_CAPACITY,
                    "pair " + std::to_string(r) + " does not fit the budget of " + std::to_string(budget) + " tokens and truncation is NONE");
            require(rc != WORDPIECE_ERR_INVALID_ARGUMENT, TURBO_E_INVALID_UTF8, "pair " + std::to_string(r) + " is not valid UTF-8");
            require(rc == WORDPIECE_OK, TURBO_E_INTERNAL, "pair packer failed with status " + std::to_string(rc));
        }
        S.n_rows = count;
    });
}

static int32_t x_session_write_text_classify(void *s, const turbo_text *, uint32_t, const turbo_classify_options *, turbo_error *err) {
    return boundary(err, [&] {
        (void)sess(s);
        fail(TURBO_E_UNSUPPORTED_TASK, "metal provider serves embedding and reranker models; classification is not offered");
    });
}

static int32_t x_session_run(void *s, const turbo_run_options *opts, turbo_provider_result *out, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_provider_result>("turbo_provider_result", out->struct_size);
        if (opts != nullptr) {
            const turbo_run_options o = read_prefix(opts, "turbo_run_options");
            reject_unknown(options_of(o.params, o.n_params, "run"), {}, "metal run");
        }
        require(S.n_rows > 0, TURBO_E_INVALID_STATE, "no inputs written");
        write_sized(out, S.run());
    });
}

static int32_t x_session_stats(void *s, turbo_session_stats *out, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_session_stats>("turbo_session_stats", out->struct_size);
        turbo_session_stats full{};
        full.struct_size = out->struct_size;
        full.runs = S.runs;
        full.host_allocs = UINT64_MAX; // not counted
        // Unified memory: tokens are written in place and results read in
        // place; nothing crosses a bus.
        full.h2d_bytes = 0;
        full.d2h_bytes = 0;
        full.input_bytes = static_cast<uint64_t>(S.batch) * S.seq * 4 * 3;
        full.output_bytes = S.out_buf.bytes;
        full.provider_allocs = UINT64_MAX;
        std::memcpy(out, &full, out->struct_size);
    });
}

static void x_session_release(void *s) { release<Session>(s); }

static const turbo_provider_vtbl g_vtbl = {
    sizeof(turbo_provider_vtbl),
    TURBO_PROVIDER_ABI_VERSION,
    kProviderId,
    kProviderVersion,
    nullptr,
    x_device_count,
    x_device_info,
    x_capability,
    x_can_run,
    x_context_create,
    x_context_release,
    x_buffer_alloc,
    x_buffer_import,
    x_buffer_read,
    x_buffer_export,
    x_buffer_release,
    x_model_load,
    x_model_info,
    x_model_label,
    nullptr, // model_io_info
    x_model_release,
    x_session_create,
    x_session_write_text,
    x_session_write_tokens,
    x_session_write_pairs,
    x_session_write_text_classify,
    nullptr, // session_bind
    x_session_run,
    x_session_stats,
    x_session_release,
    nullptr, // generation_*
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

__attribute__((visibility("default"))) const turbo_provider_vtbl *turbo_provider_get(uint32_t core_abi_version) {
    if (core_abi_version != TURBO_PROVIDER_ABI_VERSION) {
        return nullptr;
    }
    return &g_vtbl;
}

} // extern "C"
} // namespace turbo_metal
