// SPDX-License-Identifier: Apache-2.0
#include "turboembed_prepared.h"
#include "vocab.hpp"
#include "nlohmann/json.hpp"
#include "picosha2/picosha2.h"
#include "utf8proc/utf8proc.h"
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace te { struct Context; struct Model; struct Slot; }
struct te_context { std::shared_ptr<te::Context> state; };
struct te_model { std::shared_ptr<te::Model> state; };
struct te_slot { std::shared_ptr<te::Slot> state; };
struct te_result { std::shared_ptr<te::Slot> owner; };

namespace te {
namespace fs = std::filesystem;
namespace op = ov::opset13;
using Json = nlohmann::json;
struct Failure : std::runtime_error {
    uint32_t code;
    Failure(uint32_t c, const std::string &s) : std::runtime_error(s), code(c) {}
};
void require(bool ok, const char *why, uint32_t code = TE_INVALID_ARGUMENT) {
    if (!ok) { throw Failure(code, why); }
}
void error(te_error *out, uint32_t code, const char *message) noexcept {
    if (!out) { return; }
    out->code = code;
    size_t len = 0;
    while (len < sizeof(out->message) - 1 && message[len]) { ++len; }
    std::memcpy(out->message, message, len);
    out->message[len] = 0;
}
template<typename F> uint32_t boundary(te_error *out, F &&fn) noexcept {
    error(out, TE_OK, "");
    try { fn(); return TE_OK; }
    catch (const Failure &e) { error(out, e.code, e.what()); return e.code; }
    catch (const std::bad_alloc &) { error(out, TE_OUT_OF_MEMORY, "native allocation failed"); return TE_OUT_OF_MEMORY; }
    catch (const cl::Error &e) { error(out, TE_UNAVAILABLE, e.what()); return TE_UNAVAILABLE; }
    catch (const ov::Exception &e) { error(out, TE_UNAVAILABLE, e.what()); return TE_UNAVAILABLE; }
    catch (const Json::exception &e) { error(out, TE_INVALID_ARGUMENT, e.what()); return TE_INVALID_ARGUMENT; }
    catch (const std::exception &e) { error(out, TE_INTERNAL, e.what()); return TE_INTERNAL; }
    catch (...) { error(out, TE_INTERNAL, "unknown native failure"); return TE_INTERNAL; }
}
template<typename T> void descriptor(const T *value) {
    require(value != nullptr, "descriptor is null");
    require(value->struct_size == sizeof(T) && value->version == TE_PREPARED_VERSION,
            "descriptor size or version mismatch", TE_ABI_MISMATCH);
}
template<size_t N> void copy_text(char (&out)[N], const std::string &text) {
    require(text.size() < N, "metadata text exceeds ABI capacity");
    require(text.find('\0') == std::string::npos, "metadata text contains NUL");
    std::memset(out, 0, N);
    std::memcpy(out, text.data(), text.size());
}
bool utf8(const char *p, size_t n) {
    for (size_t i = 0; i < n;) {
        int32_t cp;
        const auto k = utf8proc_iterate(reinterpret_cast<const uint8_t *>(p + i),
            static_cast<utf8proc_ssize_t>(std::min(n - i, size_t{4})), &cp);
        if (k <= 0) { return false; }
        i += static_cast<size_t>(k);
    }
    return true;
}
std::string path_span(const char *p, uint64_t n) {
    require(p && n && n <= 1024 * 1024, "invalid bundle path span");
    require(!std::memchr(p, 0, static_cast<size_t>(n)) && utf8(p, static_cast<size_t>(n)), "bundle path is not NUL-free UTF-8");
    return std::string(p, static_cast<size_t>(n));
}
size_t product(uint64_t a, uint64_t b, size_t width = 1) {
    const auto bound = static_cast<uint64_t>(PTRDIFF_MAX) / width;
    require(b && a <= bound / b, "tensor dimensions overflow");
    return static_cast<size_t>(a * b);
}
uint32_t bounded_integer(const Json &j, uint32_t lo, uint32_t hi) {
    require(j.is_number_integer() && j >= lo && j <= hi, "invalid integer in model contract");
    return j.get<uint32_t>();
}
std::vector<uint8_t> read_bytes(const fs::path &path, uint64_t max_bytes) {
    std::error_code ec;
    const auto size = fs::file_size(path, ec);
    require(!ec, "bundle file not found", TE_NOT_FOUND);
    require(size > 0 && size <= max_bytes && size <= static_cast<uint64_t>(PTRDIFF_MAX), "bundle file exceeds supported size");
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    std::ifstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "cannot open bundle file", TE_NOT_FOUND);
    stream.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(size));
    require(stream.gcount() == static_cast<std::streamsize>(size), "incomplete bundle read", TE_INTEGRITY_ERROR);
    return bytes;
}
Json parse_json(const std::vector<uint8_t> &bytes) {
    std::vector<std::set<std::string>> keys;
    return Json::parse(bytes.begin(), bytes.end(), [&](int, Json::parse_event_t event, Json &value) {
        if (event == Json::parse_event_t::object_start) { keys.emplace_back(); }
        else if (event == Json::parse_event_t::key) {
            require(keys.back().insert(value.get<std::string>()).second,
                    "duplicate JSON object key", TE_INTEGRITY_ERROR);
        } else if (event == Json::parse_event_t::object_end) { keys.pop_back(); }
        return true;
    });
}
fs::path canonical(const fs::path &path) {
    std::error_code ec;
    auto resolved = fs::canonical(path, ec);
    require(!ec, "bundle path not found or inaccessible", TE_NOT_FOUND);
    return resolved;
}
bool hex(const std::string &s, size_t n) {
    return s.size() == n && std::all_of(s.begin(), s.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}
void keys(const Json &object, std::initializer_list<const char *> expected) {
    require(object.is_object() && object.size() == expected.size(), "unexpected bundle fields", TE_INTEGRITY_ERROR);
    for (const auto *key : expected) {
        require(object.contains(key), "missing bundle field", TE_INTEGRITY_ERROR);
    }
}

struct Context {
    ov::Core core;
    std::mutex compiler;
    bool gpu;
    uint32_t ordinal;
    std::string device, name, driver, runtime;
    cl::Context opencl;
    cl::Device cl_device;
    explicit Context(const te_context_options &opts) : gpu(opts.device != TE_DEVICE_OPENVINO_CPU), ordinal(opts.ordinal) {
        require(opts.device <= TE_DEVICE_OPENVINO_CPU, "unsupported device selection", TE_NOT_IMPLEMENTED);
        require(gpu || ordinal == 0, "CPU ordinal must be zero");
        device = gpu ? "GPU." + std::to_string(ordinal) : "CPU";
        const auto available = core.get_available_devices();
        require(std::any_of(available.begin(), available.end(), [&](const std::string &v) {
            return v == device || (gpu && ordinal == 0 && v == "GPU") || (!gpu && v == "CPU");
        }), "requested device is unavailable; CPU is never an automatic fallback", TE_UNAVAILABLE);
        name = core.get_property(device, ov::device::full_name);
        runtime = ov::get_openvino_version().buildNumber;
        if (gpu) {
            auto shared = core.get_default_context(device).as<ov::intel_gpu::ocl::ClContext>();
            opencl = cl::Context(shared.get(), true);
            const auto devices = opencl.getInfo<CL_CONTEXT_DEVICES>();
            require(devices.size() == 1, "multi-device OpenCL contexts are not supported", TE_NOT_IMPLEMENTED);
            cl_device = devices.front();
            driver = cl_device.getInfo<CL_DRIVER_VERSION>();
        }
    }
};

struct Model {
    std::shared_ptr<Context> context;
    te_model_info info{};
    std::vector<uint8_t> weights;
    std::shared_ptr<ov::Model> graph;
    std::unique_ptr<wordpiece_vocab, decltype(&wordpiece_vocab_destroy)> vocab{nullptr, wordpiece_vocab_destroy};
    Model(std::shared_ptr<Context> ctx, const std::string &path) : context(std::move(ctx)) {
        const fs::path root = te::canonical(path);
        const Json manifest = parse_json(read_bytes(root / "bundle.json", 1024 * 1024));
        keys(manifest, {"schema_version", "model", "conversion", "contract", "files"});
        require(manifest.at("schema_version").is_number_integer() && manifest.at("schema_version") == 1,
                "unsupported bundle schema", TE_NOT_IMPLEMENTED);
        const auto &meta = manifest.at("model");
        const auto &contract = manifest.at("contract");
        const auto &conversion = manifest.at("conversion");
        keys(meta, {"id", "revision", "license", "source_onnx_sha256"});
        keys(conversion, {"tool", "tool_version", "command"});
        keys(contract, {"pooling", "normalize", "precision", "dimension", "vocab_size",
                        "max_sequence_length", "max_batch_size", "query_prefix", "document_prefix"});
        keys(manifest.at("files"), {"openvino_model.xml", "openvino_model.bin", "tokenizer.json", "config.json", "MODEL_CARD.md"});
        require(conversion.at("tool") == "turboembed-export-model", "unsupported model converter", TE_NOT_IMPLEMENTED);
        require(!meta.at("license").get<std::string>().empty() &&
                hex(meta.at("revision").get<std::string>(), 40) &&
                hex(meta.at("source_onnx_sha256").get<std::string>(), 64), "missing model license or source provenance");
        for (const char *field : {"tool", "tool_version", "command"}) {
            require(!conversion.at(field).get<std::string>().empty(), "missing conversion provenance");
        }
        require(contract.at("pooling") == "mean" && contract.at("normalize") == true &&
                contract.at("precision") == "f32" && contract.at("query_prefix") == "" &&
                contract.at("document_prefix") == "", "unsupported model execution contract", TE_NOT_IMPLEMENTED);
        info.struct_size = sizeof(info); info.version = 1;
        info.dimension = bounded_integer(contract.at("dimension"), 384, 384);
        info.vocab_size = bounded_integer(contract.at("vocab_size"), 1, INT32_MAX);
        info.max_sequence_length = bounded_integer(contract.at("max_sequence_length"), 2, 512);
        info.max_batch_size = bounded_integer(contract.at("max_batch_size"), 1, 32);
        info.normalized = 1;
        copy_text(info.model_id, meta.at("id").get<std::string>());
        require(info.model_id[0] != 0, "model ID is empty");
        copy_text(info.revision, meta.at("revision").get<std::string>());
        copy_text(info.pooling, "mean");
        const auto capture = [&](const char *name, uint64_t limit) {
            require(te::canonical(root / name).parent_path() == root, "bundle file resolves outside its directory", TE_INTEGRITY_ERROR);
            auto bytes = read_bytes(root / name, limit);
            const auto expected = manifest.at("files").at(name).get<std::string>();
            require(hex(expected, 64) && picosha2::hash256_hex_string(bytes.begin(), bytes.end()) == expected,
                    "bundle file SHA-256 mismatch", TE_INTEGRITY_ERROR);
            return bytes;
        };
        auto xml = capture("openvino_model.xml", 32ull * 1024 * 1024);
        weights = capture("openvino_model.bin", 1024ull * 1024 * 1024);
        auto tokenizer = capture("tokenizer.json", 16ull * 1024 * 1024);
        const auto config = parse_json(capture("config.json", 1024 * 1024));
        (void)capture("MODEL_CARD.md", 16ull * 1024 * 1024);
        require(config.at("model_type") == "bert" &&
                bounded_integer(config.at("hidden_size"), 384, 384) == info.dimension &&
                bounded_integer(config.at("vocab_size"), 1, INT32_MAX) == info.vocab_size &&
                bounded_integer(config.at("max_position_embeddings"), 2, INT32_MAX) >= info.max_sequence_length &&
                bounded_integer(config.at("type_vocab_size"), 2, 2) == 2,
                "bundle contract disagrees with BERT configuration", TE_INTEGRITY_ERROR);
        const auto tok = parse_json(tokenizer);
        const auto &entries = tok.at("model").at("vocab");
        require(entries.is_object() && entries.size() == info.vocab_size, "tokenizer vocabulary size mismatch", TE_INTEGRITY_ERROR);
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            (void)bounded_integer(it.value(), 0, info.vocab_size - 1);
        }
        wordpiece_vocab *native = nullptr;
        require(wordpiece_vocab_load_json_bytes(reinterpret_cast<const char *>(tokenizer.data()), tokenizer.size(), &native) == WORDPIECE_OK,
                "native tokenizer configuration is unsupported", TE_NOT_IMPLEMENTED);
        vocab.reset(native);
        copy_text(info.tokenizer_sha256, manifest.at("files").at("tokenizer.json").get<std::string>());
        std::lock_guard<std::mutex> lock(context->compiler);
        ov::Tensor weight_tensor(ov::element::u8, ov::Shape{weights.size()}, weights.data());
        graph = context->core.read_model(std::string(xml.begin(), xml.end()), weight_tensor);
        require(graph->inputs().size() == 3 && graph->outputs().size() == 1, "expected a three-input BERT embedding graph", TE_NOT_IMPLEMENTED);
        std::set<std::string> input_names;
        for (const auto &input : graph->inputs()) {
            const auto name = input.get_any_name();
            require(input_names.insert(name).second, "duplicate model input name", TE_NOT_IMPLEMENTED);
            require(name == "input_ids" || name == "attention_mask" || name == "token_type_ids", "unsupported model input", TE_NOT_IMPLEMENTED);
            require(input.get_partial_shape().rank() == 2 &&
                    (input.get_element_type() == ov::element::i32 || input.get_element_type() == ov::element::i64),
                    "unsupported model input layout", TE_NOT_IMPLEMENTED);
        }
        bool embedding_table = false;
        for (const auto &node : graph->get_ordered_ops()) {
            if (auto constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(node)) {
                embedding_table |= constant->get_shape() == ov::Shape{info.vocab_size, info.dimension};
            }
        }
        require(embedding_table, "model has no embedding table matching vocabulary and dimension", TE_INTEGRITY_ERROR);
    }
    std::shared_ptr<ov::Model> shaped(uint32_t batch, uint32_t sequence) const {
        auto result = graph->clone();
        std::map<std::string, ov::PartialShape> shapes;
        ov::Output<ov::Node> mask;
        for (const auto &input : result->inputs()) {
            const auto name = input.get_any_name();
            shapes[name] = ov::PartialShape{batch, sequence};
            if (name == "attention_mask") { mask = input; }
        }
        require(mask.get_node_shared_ptr() != nullptr, "attention mask input missing");
        result->reshape(shapes);
        const auto hidden = result->get_results().at(0)->input_value(0);
        require(hidden.get_element_type() == ov::element::f32 && hidden.get_shape() == ov::Shape{batch, sequence, info.dimension},
                "unsupported hidden-state output", TE_NOT_IMPLEMENTED);
        const auto axis1 = op::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto axis2 = op::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        const auto fmask = std::make_shared<op::Convert>(mask, ov::element::f32);
        const auto expanded = std::make_shared<op::Unsqueeze>(fmask, axis2);
        const auto sum = std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(hidden, expanded), axis1, false);
        const auto count = std::make_shared<op::ReduceSum>(fmask, axis1, true);
        const auto denominator = std::make_shared<op::Maximum>(count, op::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
        const auto mean = std::make_shared<op::Divide>(sum, denominator);
        const auto square = std::make_shared<op::Multiply>(mean, mean);
        const auto norm = std::make_shared<op::Sqrt>(std::make_shared<op::ReduceSum>(square, axis1, true));
        const auto norm_denominator = std::make_shared<op::Maximum>(norm, op::Constant::create(ov::element::f32, ov::Shape{}, {1e-12f}));
        const auto normalized = std::make_shared<op::Divide>(mean, norm_denominator);
        normalized->output(0).set_names({"embeddings"});
        result = std::make_shared<ov::Model>(ov::OutputVector{normalized}, result->get_parameters());
        ov::preprocess::PrePostProcessor prep(result);
        for (size_t i = 0; i < result->inputs().size(); ++i) { prep.input(i).tensor().set_element_type(ov::element::i32); }
        return prep.build();
    }
};

struct Slot : std::enable_shared_from_this<Slot> {
    std::shared_ptr<Model> model;
    mutable std::mutex mutex;
    uint32_t batch, sequence;
    size_t token_count, output_count;
    std::array<std::vector<int32_t>, 3> host_inputs;
    std::vector<float> host_output;
    cl::CommandQueue queue;
    std::optional<ov::intel_gpu::ocl::ClContext> remote;
    std::array<cl::Buffer, 3> device_inputs;
    cl::Buffer device_output;
    ov::CompiledModel compiled;
    ov::InferRequest request;
    te_result result;
    bool inputs_ready = false, exposed = false;
    uint64_t executions = 0, input_write_bytes = 0, output_read_bytes = 0;
    Slot(std::shared_ptr<Model> source, const te_slot_options &opts)
        : model(std::move(source)), batch(opts.batch), sequence(opts.sequence_length),
          token_count(product(batch, sequence, 4)), output_count(product(batch, model->info.dimension, 4)) {
        require(batch >= 1 && batch <= model->info.max_batch_size && sequence >= 2 && sequence <= model->info.max_sequence_length,
                "slot shape exceeds model limits");
        for (auto &input : host_inputs) { input.resize(token_count); }
        auto &ctx = *model->context;
        std::lock_guard<std::mutex> lock(ctx.compiler);
        auto graph = model->shaped(batch, sequence);
        const ov::AnyMap props = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                 ov::hint::inference_precision(ov::element::f32)};
        if (ctx.gpu) {
            queue = cl::CommandQueue(ctx.opencl, ctx.cl_device);
            remote.emplace(ctx.core, queue.get());
            for (auto &input : device_inputs) { input = cl::Buffer(ctx.opencl, CL_MEM_READ_WRITE, token_count * 4); }
            device_output = cl::Buffer(ctx.opencl, CL_MEM_READ_WRITE, output_count * 4);
            compiled = ctx.core.compile_model(graph, *remote, props);
        } else {
            host_output.resize(output_count);
            compiled = ctx.core.compile_model(graph, "CPU", props);
        }
        request = compiled.create_infer_request();
        for (const auto &port : compiled.inputs()) {
            const auto name = port.get_any_name();
            size_t index = name == "input_ids" ? 0 : name == "attention_mask" ? 1 : 2;
            if (ctx.gpu) { request.set_tensor(port, remote->create_tensor(ov::element::i32, ov::Shape{batch, sequence}, device_inputs[index])); }
            else { request.set_tensor(port, ov::Tensor(ov::element::i32, ov::Shape{batch, sequence}, host_inputs[index].data())); }
        }
        if (ctx.gpu) { request.set_output_tensor(remote->create_tensor(ov::element::f32, ov::Shape{batch, model->info.dimension}, device_output)); }
        else { request.set_output_tensor(ov::Tensor(ov::element::f32, ov::Shape{batch, model->info.dimension}, host_output.data())); }
    }
    std::unique_lock<std::mutex> lock(bool allow_result = false) const {
        std::unique_lock<std::mutex> guard(mutex, std::try_to_lock);
        require(guard.owns_lock(), "slot is busy", TE_BUSY);
        require(allow_result || !result.owner, "release the result before reusing the slot", TE_BUSY);
        return guard;
    }
    void upload() {
        if (model->context->gpu) {
            for (size_t i = 0; i < 3; ++i) {
                queue.enqueueWriteBuffer(device_inputs[i], CL_TRUE, 0, token_count * 4, host_inputs[i].data());
            }
            input_write_bytes += token_count * 4 * 3;
        }
        inputs_ready = true;
    }
};
} // namespace te

extern "C" {
uint32_t turboembed_prepared_v1_version(void) { return TE_PREPARED_VERSION; }
uint32_t turboembed_prepared_v1_context_create(const te_context_options *opts, te_context **out, te_error *err) {
    if (out) { *out = nullptr; }
    return te::boundary(err, [&] {
        te::require(out != nullptr, "context output is null"); te::descriptor(opts);
        auto state = std::make_shared<te::Context>(*opts);
        *out = new te_context{std::move(state)};
    });
}
uint32_t turboembed_prepared_v1_context_info(const te_context *ctx, te_context_info *out, te_error *err) {
    return te::boundary(err, [&] {
        te::require(ctx != nullptr, "context is null"); te::descriptor(out);
        const auto &state = *ctx->state;
        out->device = state.gpu ? TE_DEVICE_OPENVINO_GPU : TE_DEVICE_OPENVINO_CPU;
        out->ordinal = state.ordinal;
        out->capabilities = TE_CAP_TEXT | TE_CAP_PREPARED_I32 | TE_CAP_HOST_READ | (state.gpu ? TE_CAP_OPENCL_RESULT : 0);
        te::copy_text(out->device_name, state.name); te::copy_text(out->runtime_version, state.runtime); te::copy_text(out->driver_version, state.driver);
    });
}
void turboembed_prepared_v1_context_release(te_context *ctx) { delete ctx; }
uint32_t turboembed_prepared_v1_model_load(const te_context *ctx, const char *path, uint64_t length, te_model **out, te_error *err) {
    if (out) { *out = nullptr; }
    return te::boundary(err, [&] {
        te::require(ctx && out, "model context/output is null");
        auto state = std::make_shared<te::Model>(ctx->state, te::path_span(path, length));
        *out = new te_model{std::move(state)};
    });
}
uint32_t turboembed_prepared_v1_model_info(const te_model *model, te_model_info *out, te_error *err) {
    return te::boundary(err, [&] { te::require(model != nullptr, "model is null"); te::descriptor(out); *out = model->state->info; });
}
void turboembed_prepared_v1_model_release(te_model *model) { delete model; }
uint32_t turboembed_prepared_v1_slot_create(const te_model *model, const te_slot_options *opts, te_slot **out, te_error *err) {
    if (out) { *out = nullptr; }
    return te::boundary(err, [&] {
        te::require(model && out, "slot model/output is null"); te::descriptor(opts);
        auto state = std::make_shared<te::Slot>(model->state, *opts);
        *out = new te_slot{std::move(state)};
    });
}
void turboembed_prepared_v1_slot_release(te_slot *slot) { delete slot; }
uint32_t turboembed_prepared_v1_slot_write_tokens(te_slot *handle, const int32_t *ids, const int32_t *mask, const int32_t *types, uint64_t count, te_error *err) {
    return te::boundary(err, [&] {
        te::require(handle != nullptr, "slot is null"); auto &slot = *handle->state; auto lock = slot.lock();
        slot.inputs_ready = false;
        te::require(ids && mask && count == slot.token_count, "input shape or pointer mismatch");
        for (size_t i = 0; i < slot.token_count; ++i) {
            te::require(ids[i] >= 0 && static_cast<uint32_t>(ids[i]) < slot.model->info.vocab_size, "token ID outside vocabulary");
            te::require((mask[i] == 0 || mask[i] == 1) && (!types || types[i] == 0 || types[i] == 1), "mask/type must be zero or one");
        }
        std::copy_n(ids, slot.token_count, slot.host_inputs[0].data());
        std::copy_n(mask, slot.token_count, slot.host_inputs[1].data());
        if (types) { std::copy_n(types, slot.token_count, slot.host_inputs[2].data()); }
        else { std::fill(slot.host_inputs[2].begin(), slot.host_inputs[2].end(), 0); }
        slot.upload();
    });
}
uint32_t turboembed_prepared_v1_slot_write_text(te_slot *handle, const te_text *texts, uint64_t count, te_error *err) {
    return te::boundary(err, [&] {
        te::require(handle != nullptr, "slot is null"); auto &slot = *handle->state; auto lock = slot.lock();
        slot.inputs_ready = false;
        te::require(texts && count == slot.batch, "text batch size or pointer mismatch");
        for (size_t i = 0; i < slot.batch; ++i) {
            const auto &text = texts[i];
            te::require(text.byte_length <= static_cast<uint64_t>(PTRDIFF_MAX), "text span exceeds addressable memory");
            const size_t offset = i * slot.sequence;
            const auto status = wordpiece_encode_sentence(slot.model->vocab.get(), text.ptr, static_cast<size_t>(text.byte_length),
                slot.host_inputs[0].data() + offset, slot.host_inputs[1].data() + offset, slot.host_inputs[2].data() + offset,
                nullptr, slot.sequence, slot.sequence, 4);
            te::require(status == WORDPIECE_OK, "invalid text input or unsupported tokenization");
        }
        slot.upload();
    });
}
uint32_t turboembed_prepared_v1_slot_execute(te_slot *handle, te_result **out, te_error *err) {
    if (out) { *out = nullptr; }
    return te::boundary(err, [&] {
        te::require(handle && out, "slot/result output is null"); auto &slot = *handle->state; auto lock = slot.lock();
        te::require(slot.inputs_ready, "write valid inputs before execution");
        try { slot.request.infer(); }
        catch (...) { slot.inputs_ready = false; throw; }
        ++slot.executions;
        slot.exposed = false; slot.result.owner = handle->state; *out = &slot.result;
    });
}
uint32_t turboembed_prepared_v1_slot_stats(const te_slot *handle, te_slot_stats *out, te_error *err) {
    return te::boundary(err, [&] {
        te::require(handle != nullptr, "slot is null"); te::descriptor(out); const auto &slot = *handle->state; auto lock = slot.lock(true);
        out->executions = slot.executions; out->input_write_bytes = slot.input_write_bytes; out->output_read_bytes = slot.output_read_bytes;
        out->owned_input_bytes = slot.token_count * 4 * 3; out->owned_output_bytes = slot.output_count * 4;
    });
}
uint32_t turboembed_prepared_v1_result_info(const te_result *result, te_result_info *out, te_error *err) {
    return te::boundary(err, [&] {
        te::require(result && result->owner, "result is null or released"); te::descriptor(out); const auto &slot = *result->owner; auto lock = slot.lock(true);
        out->batch = slot.batch; out->dimension = slot.model->info.dimension; out->byte_size = slot.output_count * 4;
    });
}
uint32_t turboembed_prepared_v1_result_read(const te_result *result, float *out, uint64_t capacity, te_error *err) {
    return te::boundary(err, [&] {
        te::require(result && result->owner, "result is null or released"); auto &slot = *result->owner; auto lock = slot.lock(true);
        te::require(out && capacity >= slot.output_count, "output capacity is too small or pointer is null");
        if (slot.model->context->gpu) {
            slot.queue.enqueueReadBuffer(slot.device_output, CL_TRUE, 0, slot.output_count * 4, out);
            slot.output_read_bytes += slot.output_count * 4;
        } else { std::copy(slot.host_output.begin(), slot.host_output.end(), out); }
    });
}
uint32_t turboembed_prepared_v1_result_opencl(const te_result *result, te_opencl_view *out, te_error *err) {
    return te::boundary(err, [&] {
        te::require(result && result->owner, "result is null or released"); te::descriptor(out); auto &slot = *result->owner; auto lock = slot.lock(true);
        te::require(slot.model->context->gpu, "CPU results have no OpenCL resource", TE_NOT_IMPLEMENTED);
        out->context = reinterpret_cast<uintptr_t>(slot.model->context->opencl());
        out->queue = reinterpret_cast<uintptr_t>(slot.queue()); out->buffer = reinterpret_cast<uintptr_t>(slot.device_output());
        out->byte_size = slot.output_count * 4; out->batch = slot.batch; out->dimension = slot.model->info.dimension; slot.exposed = true;
    });
}
uint32_t turboembed_prepared_v1_result_release(te_result *result, te_error *err) {
    if (!result) { te::error(err, TE_OK, ""); return TE_OK; }
    return te::boundary(err, [&] {
        auto owner = result->owner;
        te::require(static_cast<bool>(owner), "result is already released");
        auto lock = owner->lock(true);
        try {
            if (owner->exposed) { owner->queue.finish(); }
        } catch (...) { owner->inputs_ready = false; result->owner.reset(); throw; }
        result->owner.reset();
    });
}
} // extern C
