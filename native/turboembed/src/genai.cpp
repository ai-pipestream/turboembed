/* SPDX-License-Identifier: Apache-2.0
 *
 * CompiledModel on Intel CPU / GPU / NPU. Token ids/mask/types and
 * last_hidden_state are arena-rented (ZE SHARED on GPU).
 * InferRequest.set_tensor wraps those pointers.
 *
 * ov::genai::Tokenizer.encode has no caller-buffer hook (returns an
 * engine-owned ov::Tensor). MiniLM-compatible models write WordPiece
 * ids directly into the rented USM row — no encode→copy.
 *
 * Device string is "CPU", "GPU", or "NPU". Never "AUTO". No OVMS. No Python.
 */

#include "genai.hpp"
#include "wordpiece.h"

#ifndef TURBOEMBED_WORKSPACE_ROOT
#define TURBOEMBED_WORKSPACE_ROOT ""
#endif

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace turboembed_genai {

namespace {
std::atomic<uint64_t> g_wrap_in{0};
std::atomic<uint64_t> g_hidden_wrap{0};
std::atomic<uint64_t> g_hidden_memcpy{0};
std::atomic<uint32_t> g_last_n{0};
std::atomic<uint32_t> g_last_seq{0};
std::atomic<uint32_t> g_last_inputs{0};
} // namespace

void genai_xfer_reset() {
    g_wrap_in.store(0, std::memory_order_relaxed);
    g_hidden_wrap.store(0, std::memory_order_relaxed);
    g_hidden_memcpy.store(0, std::memory_order_relaxed);
    g_last_n.store(0, std::memory_order_relaxed);
    g_last_seq.store(0, std::memory_order_relaxed);
    g_last_inputs.store(0, std::memory_order_relaxed);
}

uint64_t genai_xfer_wrap_input_bytes() {
    return g_wrap_in.load(std::memory_order_relaxed);
}

uint64_t genai_xfer_hidden_wrap_bytes() {
    return g_hidden_wrap.load(std::memory_order_relaxed);
}

uint64_t genai_xfer_hidden_memcpy_bytes() {
    return g_hidden_memcpy.load(std::memory_order_relaxed);
}

uint32_t genai_xfer_last_n() {
    return g_last_n.load(std::memory_order_relaxed);
}

uint32_t genai_xfer_last_seq() {
    return g_last_seq.load(std::memory_order_relaxed);
}

uint32_t genai_xfer_last_inputs() {
    return g_last_inputs.load(std::memory_order_relaxed);
}

namespace {

namespace fs = std::filesystem;

constexpr uint32_t kWarmBatch = 32;
constexpr uint32_t kDefaultMaxSeq = 256;

const char* kRequiredIr[] = {
    "openvino_model.xml",
    "openvino_model.bin",
    "openvino_tokenizer.xml",
    "openvino_tokenizer.bin",
};

bool looks_like_genai_dir(const fs::path& dir) {
    if (!fs::is_directory(dir)) {
        return false;
    }
    for (const char* name : kRequiredIr) {
        if (!fs::is_regular_file(dir / name)) {
            return false;
        }
    }
    return true;
}

std::string missing_ir(const fs::path& dir) {
    std::string miss;
    for (const char* name : kRequiredIr) {
        if (!fs::is_regular_file(dir / name)) {
            if (!miss.empty()) {
                miss += ", ";
            }
            miss += name;
        }
    }
    return miss;
}

bool starts_with_gpu(const std::string& d) {
    return d.size() >= 3 && d.compare(0, 3, "GPU") == 0;
}

bool starts_with_npu(const std::string& d) {
    return d.size() >= 3 && d.compare(0, 3, "NPU") == 0;
}

bool is_cpu_device(const std::string& d) {
    return d == "CPU" || (d.size() >= 3 && d.compare(0, 3, "CPU") == 0);
}

bool listed_has(const std::vector<std::string>& listed, bool (*pred)(const std::string&)) {
    for (const auto& d : listed) {
        if (pred(d)) {
            return true;
        }
    }
    return false;
}

std::string join_devices(const std::vector<std::string>& listed) {
    std::string out;
    for (const auto& d : listed) {
        if (!out.empty()) {
            out += ", ";
        }
        out += d;
    }
    return out;
}

uint32_t dim_from_config_json(const fs::path& dir) {
    const fs::path cfg = dir / "config.json";
    if (!fs::is_regular_file(cfg)) {
        return 0;
    }
    std::ifstream in(cfg);
    if (!in) {
        return 0;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string text = ss.str();
    const char* keys[] = {
        "\"hidden_size\"",
        "\"d_model\"",
        "\"sentence_embedding_dimension\"",
    };
    for (const char* key : keys) {
        const auto pos = text.find(key);
        if (pos == std::string::npos) {
            continue;
        }
        const auto colon = text.find(':', pos + std::strlen(key));
        if (colon == std::string::npos) {
            continue;
        }
        size_t i = colon + 1;
        while (i < text.size() && (text[i] == ' ' || text[i] == '\t')) {
            ++i;
        }
        uint32_t v = 0;
        bool any = false;
        while (i < text.size() && text[i] >= '0' && text[i] <= '9') {
            any = true;
            v = v * 10u + static_cast<uint32_t>(text[i] - '0');
            ++i;
        }
        if (any && v > 0) {
            return v;
        }
    }
    return 0;
}

std::string env_or_empty(const char* key) {
    const char* v = std::getenv(key);
    return (v == nullptr || v[0] == '\0') ? std::string() : std::string(v);
}

template <typename Port>
std::string input_name_matching(const Port& in, const char* needle) {
    for (const auto& n : in.get_names()) {
        if (n.find(needle) != std::string::npos) {
            return n;
        }
    }
    const std::string any = in.get_any_name();
    if (any.find(needle) != std::string::npos) {
        return any;
    }
    return {};
}

wordpiece_vocab *load_wordpiece_or_throw(const fs::path& dir) {
    wordpiece_vocab *v = nullptr;
    const auto path = (dir / "tokenizer.json").string();
    if (wordpiece_vocab_load(path.c_str(), &v) == WORDPIECE_OK && v != nullptr) {
        return v;
    }
    throw std::runtime_error(
        "Native tokenization requires a supported uncased BERT tokenizer.json "
        "in the model bundle: " + path
    );
}

void mean_pool(
    const float *hidden,
    const int32_t *mask,
    uint32_t n,
    uint32_t seq,
    uint32_t dim,
    float *out
) {
    for (uint32_t b = 0; b < n; ++b) {
        float *row = out + static_cast<size_t>(b) * dim;
        for (uint32_t d = 0; d < dim; ++d) {
            row[d] = 0.0f;
        }
        float count = 0.0f;
        for (uint32_t s = 0; s < seq; ++s) {
            if (mask[static_cast<size_t>(b) * seq + s] == 0) {
                continue;
            }
            count += 1.0f;
            const float *h = hidden + (static_cast<size_t>(b) * seq + s) * dim;
            for (uint32_t d = 0; d < dim; ++d) {
                row[d] += h[d];
            }
        }
        if (count > 0.0f) {
            for (uint32_t d = 0; d < dim; ++d) {
                row[d] /= count;
            }
        }
    }
}

void cls_pool(const float *hidden, uint32_t n, uint32_t seq, uint32_t dim, float *out) {
    for (uint32_t b = 0; b < n; ++b) {
        const float *h = hidden + static_cast<size_t>(b) * seq * dim;
        std::memcpy(out + static_cast<size_t>(b) * dim, h, static_cast<size_t>(dim) * sizeof(float));
    }
}

void last_pool(
    const float *hidden,
    const int32_t *mask,
    uint32_t n,
    uint32_t seq,
    uint32_t dim,
    float *out
) {
    for (uint32_t b = 0; b < n; ++b) {
        uint32_t last = 0;
        for (uint32_t s = 0; s < seq; ++s) {
            if (mask[static_cast<size_t>(b) * seq + s] != 0) {
                last = s;
            }
        }
        const float *h = hidden + (static_cast<size_t>(b) * seq + last) * dim;
        std::memcpy(out + static_cast<size_t>(b) * dim, h, static_cast<size_t>(dim) * sizeof(float));
    }
}

void l2_normalize(float *rows, uint32_t n, uint32_t dim) {
    for (uint32_t b = 0; b < n; ++b) {
        float *row = rows + static_cast<size_t>(b) * dim;
        float ss = 0.0f;
        for (uint32_t d = 0; d < dim; ++d) {
            ss += row[d] * row[d];
        }
        const float norm = std::sqrt(ss);
        const float denom = norm > 1e-12f ? norm : 1e-12f;
        for (uint32_t d = 0; d < dim; ++d) {
            row[d] /= denom;
        }
    }
}

void return_view(turbo_buffer_arena *arena, turbo_buffer_view *view) {
    if (arena == nullptr || view == nullptr || view->ptr == nullptr) {
        return;
    }
    (void)turbo_buffer_arena_return(arena, view);
}

} // namespace

std::vector<std::string> available_devices() {
    ov::Core core;
    return core.get_available_devices();
}

bool runtime_has_gpu() {
    return listed_has(available_devices(), starts_with_gpu);
}

bool runtime_has_cpu() {
    return listed_has(available_devices(), is_cpu_device);
}

bool runtime_has_npu() {
    return listed_has(available_devices(), starts_with_npu);
}

std::string require_ov_device(
    const std::string& requested,
    const std::vector<std::string>& available
) {
    const std::string listed_join = join_devices(available);
    if (requested != "CPU" && requested != "GPU" && requested != "NPU") {
        throw std::runtime_error(
            "unsupported OpenVINO GenAI device string '" + requested +
            "' (pass \"CPU\", \"GPU\", or \"NPU\"; never AUTO)"
        );
    }
    if (requested == "GPU" && !listed_has(available, starts_with_gpu)) {
        throw std::runtime_error(
            "OpenVINO GPU plugin unavailable (listed: [" + listed_join +
            "]); GPU was requested so CPU fallback is refused. "
            "Need libopenvino_intel_gpu_plugin + Level Zero, or create "
            "the engine with TURBOEMBED_DEVICE_OPENVINO_CPU / "
            "TextEmbeddingPipeline(..., \"CPU\", config)."
        );
    }
    if (requested == "NPU" && !listed_has(available, starts_with_npu)) {
        throw std::runtime_error(
            "OpenVINO NPU plugin unavailable (listed: [" + listed_join +
            "]); NPU was requested — refusing CPU fallback. "
            "Need Intel NPU silicon + intel-npu/accel driver + "
            "libopenvino_intel_npu_plugin.so. Live probe of "
            "TextEmbeddingPipeline(..., \"NPU\") fails with: Device with "
            "\"NPU\" name is not registered in the OpenVINO Runtime. "
            "Battlemage dGPU + AMD CPU hosts do not provide an NPU."
        );
    }
    if (requested == "CPU" && !listed_has(available, is_cpu_device)) {
        throw std::runtime_error(
            "OpenVINO CPU plugin unavailable (listed: [" + listed_join +
            "]); CPU was requested. Need libopenvino_intel_cpu_plugin."
        );
    }
    return requested;
}

std::string device_full_name(const std::string& ov_device) {
    try {
        ov::Core core;
        return core.get_property(ov_device, ov::device::full_name);
    } catch (...) {
        return {};
    }
}

struct Pipeline::Impl {
    ov::Core core;
    ov::CompiledModel compiled;
    ov::InferRequest request;
    wordpiece_vocab *vocab = nullptr;
    uint8_t pooling = 1;
    bool normalize = true;
    uint32_t max_seq = kDefaultMaxSeq;
    uint32_t warm_batch = 0;
    uint32_t hidden_dim = 0;
    bool has_types = false;
    std::string in_ids;
    std::string in_mask;
    std::string in_types;
    std::string out_hidden;
    turbo_buffer_arena *arena = nullptr;
    turbo_buffer_placement place = TURBO_BUFFER_PLACE_HOST;
    turbo_buffer_view ids {};
    turbo_buffer_view mask {};
    turbo_buffer_view types {};
    turbo_buffer_view hidden {};

    Impl() = default;

    ~Impl() {
        return_view(arena, &ids);
        return_view(arena, &mask);
        return_view(arena, &types);
        return_view(arena, &hidden);
        if (vocab != nullptr) {
            wordpiece_vocab_destroy(vocab);
            vocab = nullptr;
        }
    }

    void ensure_workspace(uint32_t n, uint32_t seq) {
        if (arena == nullptr) {
            throw std::runtime_error("GenAI embed requires a turbo_buffer arena");
        }
        if (n == 0 || seq == 0 || hidden_dim == 0) {
            throw std::runtime_error("ensure_workspace: empty shape");
        }
        const uint32_t need_batch = n > warm_batch ? n : warm_batch;
        const uint32_t need_seq = seq > max_seq ? seq : max_seq;
        const bool have = ids.ptr != nullptr && mask.ptr != nullptr &&
                          hidden.ptr != nullptr &&
                          ids.rows >= need_batch && ids.cols >= need_seq &&
                          hidden.rows >= need_batch &&
                          hidden.cols >= need_seq * hidden_dim &&
                          (!has_types || types.ptr != nullptr);
        if (have) {
            return;
        }
        return_view(arena, &ids);
        return_view(arena, &mask);
        return_view(arena, &types);
        return_view(arena, &hidden);
        auto rent_i32 = [&](turbo_buffer_view *v) {
            const turbo_buffer_status st = turbo_buffer_arena_rent(
                arena,
                TURBO_BUFFER_DTYPE_I32,
                place,
                need_batch,
                need_seq,
                need_seq,
                v
            );
            if (st != TURBO_BUFFER_OK || v->ptr == nullptr) {
                throw std::runtime_error(
                    std::string("GenAI token arena rent failed: ") +
                    turbo_buffer_last_error(arena)
                );
            }
        };
        rent_i32(&ids);
        rent_i32(&mask);
        if (has_types) {
            rent_i32(&types);
        }
        const turbo_buffer_status hst = turbo_buffer_arena_rent(
            arena,
            TURBO_BUFFER_DTYPE_F32,
            place,
            need_batch,
            need_seq * hidden_dim,
            need_seq * hidden_dim,
            &hidden
        );
        if (hst != TURBO_BUFFER_OK || hidden.ptr == nullptr) {
            throw std::runtime_error(
                std::string("GenAI hidden arena rent failed: ") +
                turbo_buffer_last_error(arena)
            );
        }
        warm_batch = need_batch;
        max_seq = need_seq;
    }
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)), dim_(0), last_hidden_arena_(false) {}

Pipeline::~Pipeline() = default;

turbo_buffer_device Pipeline::arena_device() const {
    return impl_ && impl_->ids.ptr != nullptr ? impl_->ids.device
                                              : TURBO_BUFFER_DEVICE_CPU;
}

turbo_buffer_placement Pipeline::token_placement() const {
    return impl_ ? impl_->place : TURBO_BUFFER_PLACE_HOST;
}

const void *Pipeline::token_ids_ptr() const {
    return impl_ ? impl_->ids.ptr : nullptr;
}

const void *Pipeline::hidden_ptr() const {
    return impl_ ? impl_->hidden.ptr : nullptr;
}

bool Pipeline::owns_tokens() const {
    return impl_ && impl_->arena != nullptr &&
           turbo_buffer_arena_owns(impl_->arena, impl_->ids.ptr) != 0;
}

bool Pipeline::owns_hidden() const {
    return impl_ && impl_->arena != nullptr &&
           turbo_buffer_arena_owns(impl_->arena, impl_->hidden.ptr) != 0;
}

std::unique_ptr<Pipeline> load_pipeline(
    const std::string& models_path,
    const std::string& ov_device,
    const LoadConfig& config,
    turbo_buffer_arena *arena,
    turbo_buffer_placement place
) {
    const fs::path dir(models_path);
    if (!fs::is_directory(dir)) {
        throw std::runtime_error(
            "OpenVINO GenAI model directory not found: " + models_path
        );
    }
    const std::string miss = missing_ir(dir);
    if (!miss.empty()) {
        throw std::runtime_error(
            "OpenVINO GenAI model directory is incomplete (" + models_path +
            " missing " + miss +
            "); need openvino_model.xml/.bin + openvino_tokenizer.xml/.bin"
        );
    }
    if (arena == nullptr) {
        throw std::runtime_error(
            "load_pipeline requires a turbo_buffer arena; refusing a "
            "private-alloc GenAI path"
        );
    }
    if (ov_device == "GPU" && place != TURBO_BUFFER_PLACE_SHARED) {
        throw std::runtime_error(
            "OpenVINO GPU embed requires ZE SHARED USM for tokens/hidden; "
            "refusing HOST/CPU remap"
        );
    }

    std::vector<std::string> listed;
    try {
        listed = available_devices();
    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("failed to query OpenVINO devices: ") + e.what() +
            " (source OpenVINO setupvars.sh)"
        );
    }

    require_ov_device(ov_device, listed);

    auto impl = std::unique_ptr<Pipeline::Impl>(new Pipeline::Impl());
    impl->arena = arena;
    impl->place = place;
    impl->pooling = config.pooling;
    impl->normalize = config.normalize;
    impl->max_seq = config.max_length > 0 ? config.max_length : kDefaultMaxSeq;
    impl->vocab = load_wordpiece_or_throw(dir);

    std::shared_ptr<ov::Model> model =
        impl->core.read_model((dir / "openvino_model.xml").string());
    ov::preprocess::PrePostProcessor ppp(model);
    for (const auto& in : model->inputs()) {
        ppp.input(in.get_any_name()).tensor().set_element_type(ov::element::i32);
    }
    model = ppp.build();

    std::map<std::string, ov::PartialShape> shapes;
    const ov::PartialShape token_shape(std::vector<ov::Dimension>{
        ov::Dimension::dynamic(),
        ov::Dimension(static_cast<int64_t>(impl->max_seq))
    });
    for (const auto& in : model->inputs()) {
        shapes[in.get_any_name()] = token_shape;
    }
    model->reshape(shapes);

    impl->has_types = false;
    for (const auto& in : model->inputs()) {
        if (input_name_matching(in, "input_ids").size() > 0 && impl->in_ids.empty()) {
            impl->in_ids = in.get_any_name();
        } else if (input_name_matching(in, "attention_mask").size() > 0) {
            impl->in_mask = in.get_any_name();
        } else if (input_name_matching(in, "token_type").size() > 0) {
            impl->in_types = in.get_any_name();
            impl->has_types = true;
        }
    }
    if (impl->in_ids.empty() && !model->inputs().empty()) {
        impl->in_ids = model->input(0).get_any_name();
    }
    if (impl->in_mask.empty() && model->inputs().size() > 1) {
        impl->in_mask = model->input(1).get_any_name();
    }
    if (!impl->has_types && model->inputs().size() > 2) {
        impl->in_types = model->input(2).get_any_name();
        impl->has_types = true;
    }
    impl->out_hidden = model->output(0).get_any_name();
    if (impl->out_hidden.empty()) {
        impl->out_hidden = "last_hidden_state";
    }

    ov::AnyMap props;
    props[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    props[ov::hint::inference_precision.name()] = ov::element::f32;
    props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    props[ov::hint::dynamic_quantization_group_size.name()] = static_cast<uint64_t>(0);
    impl->compiled = impl->core.compile_model(model, ov_device, props);
    impl->request = impl->compiled.create_infer_request();

    uint32_t dim = dim_from_config_json(dir);
    try {
        const ov::PartialShape osh = impl->compiled.output(0).get_partial_shape();
        if (osh.rank().is_static() && osh.rank().get_length() >= 3 && osh[osh.rank().get_length() - 1].is_static()) {
            dim = static_cast<uint32_t>(osh[osh.rank().get_length() - 1].get_length());
        }
    } catch (...) {
    }
    if (dim == 0) {
        throw std::runtime_error("could not resolve embedding hidden size");
    }
    impl->hidden_dim = dim;
    impl->ensure_workspace(kWarmBatch, impl->max_seq);

    auto out = std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
    out->models_path_ = models_path;
    out->device_ = ov_device;
    out->available_ = std::move(listed);
    out->device_full_name_ = device_full_name(ov_device);
    out->dim_ = dim;
    out->last_hidden_arena_ = false;
    return out;
}

void Pipeline::embed_into(const std::vector<std::string>& texts, float *out) {
    last_hidden_arena_ = false;
    if (texts.empty()) {
        throw std::invalid_argument("embed_into called with no texts");
    }
    if (out == nullptr) {
        throw std::invalid_argument("embed_into: null result pointer");
    }
    if (device_ != "CPU" && device_ != "GPU" && device_ != "NPU") {
        throw std::runtime_error(
            "internal error: pipeline device is " + device_ +
            " (must be CPU, GPU, or NPU)"
        );
    }
    if (!impl_) {
        throw std::runtime_error("pipeline is not loaded");
    }

    if (impl_->vocab == nullptr || !wordpiece_vocab_is_loaded(impl_->vocab)) {
        throw std::runtime_error("GenAI WordPiece vocab is not loaded");
    }
    const uint32_t n = static_cast<uint32_t>(texts.size());
    const uint32_t seq = impl_->max_seq;
    impl_->ensure_workspace(n, seq);

    int32_t *ids = turbo_buffer_view_i32(&impl_->ids);
    int32_t *mask = turbo_buffer_view_i32(&impl_->mask);
    int32_t *types = impl_->has_types ? turbo_buffer_view_i32(&impl_->types) : nullptr;
    const uint32_t id_stride = impl_->ids.row_stride;
    const uint32_t mask_stride = impl_->mask.row_stride;
    const uint32_t type_stride = impl_->has_types ? impl_->types.row_stride : seq;
    for (uint32_t b = 0; b < n; ++b) {
        const int st = wordpiece_encode_sentence(
            impl_->vocab,
            texts[b].data(),
            texts[b].size(),
            ids + static_cast<size_t>(b) * id_stride,
            mask + static_cast<size_t>(b) * mask_stride,
            types ? types + static_cast<size_t>(b) * type_stride : nullptr,
            nullptr,
            seq,
            id_stride,
            4
        );
        if (st != WORDPIECE_OK) {
            throw std::runtime_error("WordPiece write-through into USM failed");
        }
    }

    const ov::Shape token_shape{static_cast<size_t>(n), static_cast<size_t>(seq)};
    ov::Tensor t_ids(ov::element::i32, token_shape, ids);
    ov::Tensor t_mask(ov::element::i32, token_shape, mask);
    impl_->request.set_tensor(impl_->in_ids, t_ids);
    impl_->request.set_tensor(impl_->in_mask, t_mask);
    if (impl_->has_types) {
        ov::Tensor t_types(
            ov::element::i32, token_shape, turbo_buffer_view_i32(&impl_->types)
        );
        impl_->request.set_tensor(impl_->in_types, t_types);
    }

    float *hidden = turbo_buffer_view_f32(&impl_->hidden);
    const ov::Shape hidden_shape{
        static_cast<size_t>(n),
        static_cast<size_t>(seq),
        static_cast<size_t>(impl_->hidden_dim)
    };
    bool used_arena_hidden = false;
    try {
        ov::Tensor t_hidden(ov::element::f32, hidden_shape, hidden);
        impl_->request.set_tensor(impl_->out_hidden, t_hidden);
        used_arena_hidden = true;
    } catch (const std::exception&) {
        used_arena_hidden = false;
    }

    impl_->request.infer();

    const ov::Tensor got = impl_->request.get_tensor(impl_->out_hidden);
    const float *hptr = got.data<float>();
    if (used_arena_hidden && hptr == hidden) {
        last_hidden_arena_ = true;
    } else if (used_arena_hidden && hptr != hidden) {
        /* Plugin accepted the set but served a different buffer — copy
         * into the rented hidden so pooling stays on arena memory. */
        const size_t nbytes =
            static_cast<size_t>(n) * seq * impl_->hidden_dim * sizeof(float);
        std::memcpy(hidden, hptr, nbytes);
        last_hidden_arena_ = true;
        hptr = hidden;
        g_hidden_memcpy.fetch_add(nbytes, std::memory_order_relaxed);
    } else {
        /* set_tensor(output) rejected — API did not allow a caller result
         * tensor. Pool from the plugin buffer; tokens still arena USM. */
        last_hidden_arena_ = false;
    }

    const uint32_t dim = impl_->hidden_dim;
    switch (impl_->pooling) {
        case 0:
            cls_pool(hptr, n, seq, dim, out);
            break;
        case 2:
            last_pool(hptr, mask, n, seq, dim, out);
            break;
        case 1:
        default:
            mean_pool(hptr, mask, n, seq, dim, out);
            break;
    }
    if (impl_->normalize) {
        l2_normalize(out, n, dim);
    }
    dim_ = dim;

    const uint32_t n_in = impl_->has_types ? 3u : 2u;
    g_wrap_in.fetch_add(
        static_cast<uint64_t>(n) * seq * sizeof(int32_t) * n_in,
        std::memory_order_relaxed
    );
    if (used_arena_hidden) {
        g_hidden_wrap.fetch_add(
            static_cast<uint64_t>(n) * seq * impl_->hidden_dim * sizeof(float),
            std::memory_order_relaxed
        );
    }
    g_last_n.store(n, std::memory_order_relaxed);
    g_last_seq.store(seq, std::memory_order_relaxed);
    g_last_inputs.store(n_in, std::memory_order_relaxed);
}

std::string resolve_models_path(
    const std::string& alias,
    const std::string& config_path,
    const std::string& workspace_root
) {
    if (alias.empty()) {
        throw std::invalid_argument("alias is empty");
    }

    std::vector<fs::path> candidates;

    const std::string exact = env_or_empty("INFERSTREAM_OV_MODEL");
    if (!exact.empty()) {
        candidates.emplace_back(exact);
    }
    const std::string te_exact = env_or_empty("TURBOEMBED_OV_MODEL");
    if (!te_exact.empty()) {
        candidates.emplace_back(te_exact);
    }

    const std::string roots[] = {
        env_or_empty("TURBOEMBED_OV_ROOT"),
        env_or_empty("INFERSTREAM_OV_ROOT"),
    };
    for (const auto& root : roots) {
        if (!root.empty()) {
            candidates.emplace_back(fs::path(root) / alias);
        }
    }

    if (!config_path.empty()) {
        const fs::path cfg(config_path);
        if (looks_like_genai_dir(cfg)) {
            candidates.push_back(cfg);
        }
        if (looks_like_genai_dir(cfg / alias)) {
            candidates.push_back(cfg / alias);
        }
        if (fs::is_directory(cfg / "models" / "ov" / alias)) {
            candidates.push_back(cfg / "models" / "ov" / alias);
        }
    }

    if (!workspace_root.empty()) {
        candidates.push_back(fs::path(workspace_root) / "models" / "ov" / alias);
    }

    fs::path cwd = fs::current_path();
    for (int i = 0; i < 8; ++i) {
        candidates.push_back(cwd / "models" / "ov" / alias);
        if (!cwd.has_parent_path() || cwd.parent_path() == cwd) {
            break;
        }
        cwd = cwd.parent_path();
    }

    for (const auto& c : candidates) {
        if (looks_like_genai_dir(c)) {
            return fs::weakly_canonical(c).string();
        }
    }

    throw std::runtime_error(
        "no GenAI-layout directory for alias '" + alias +
        "' (need openvino_model.xml/.bin + openvino_tokenizer.xml/.bin). "
        "Set TURBOEMBED_OV_MODEL / INFERSTREAM_OV_MODEL or place IR at "
        "models/ov/" +
        alias + "/"
    );
}

} // namespace turboembed_genai
