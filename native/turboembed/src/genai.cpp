/* SPDX-License-Identifier: Apache-2.0
 *
 * ov::genai::TextEmbeddingPipeline on Intel CPU or GPU.
 *
 * Official C++ usage (OpenVINO GenAI samples/cpp/rag/text_embeddings.cpp):
 *
 *   std::string device = "CPU";  // GPU can be used as well
 *   ov::genai::TextEmbeddingPipeline::Config config;
 *   config.pooling_type = ov::genai::TextEmbeddingPipeline::PoolingType::MEAN;
 *   ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);
 *   ov::genai::EmbeddingResults rows = pipeline.embed_documents(documents);
 *
 * The `device` argument is the OpenVINO plugin name passed through to
 * ov::Core::compile_model (non-NPU). We pass the caller's string
 * unchanged: "CPU" or "GPU". Never "AUTO". Asking for "GPU" when the
 * GPU plugin is missing does not compile "CPU".
 * No OVMS. No Python.
 */

#include "genai.hpp"

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/runtime/properties.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace turboembed_genai {
namespace {

namespace fs = std::filesystem;

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

ov::genai::TextEmbeddingPipeline::PoolingType pooling_from_u8(uint8_t pooling) {
    using PT = ov::genai::TextEmbeddingPipeline::PoolingType;
    switch (pooling) {
        case 0:
            return PT::CLS;
        case 1:
            return PT::MEAN;
        case 2:
            return PT::LAST_TOKEN;
        default:
            throw std::invalid_argument(
                "unknown pooling id (expected 0=CLS, 1=MEAN, 2=LAST_TOKEN)"
            );
    }
}

std::vector<std::vector<float>> as_float_rows(ov::genai::EmbeddingResults results) {
    if (auto* floats = std::get_if<std::vector<std::vector<float>>>(&results)) {
        return std::move(*floats);
    }
    throw std::runtime_error(
        "TextEmbeddingPipeline returned a non-float embedding "
        "(int8/uint8 outputs are not accepted; use an fp16/fp32 IR)"
    );
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

std::string require_ov_device(
    const std::string& requested,
    const std::vector<std::string>& available
) {
    const std::string listed_join = join_devices(available);
    if (requested != "CPU" && requested != "GPU") {
        throw std::runtime_error(
            "unsupported OpenVINO GenAI device string '" + requested +
            "' (pass \"CPU\" or \"GPU\"; never AUTO)"
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
    ov::genai::TextEmbeddingPipeline pipe;
    Impl(const fs::path& dir, const std::string& device, const ov::genai::TextEmbeddingPipeline::Config& cfg)
        : pipe(dir, device, cfg) {}
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)), dim_(0) {}

Pipeline::~Pipeline() = default;

std::unique_ptr<Pipeline> load_pipeline(
    const std::string& models_path,
    const std::string& ov_device,
    const LoadConfig& config
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

    ov::genai::TextEmbeddingPipeline::Config cfg;
    cfg.pooling_type = pooling_from_u8(config.pooling);
    cfg.normalize = config.normalize;
    if (config.max_length > 0) {
        cfg.max_length = static_cast<size_t>(config.max_length);
        cfg.pad_to_max_length = true;
    }

    /* Exact plugin name from the official sample / docs: "CPU" or "GPU". */
    auto out = std::unique_ptr<Pipeline>(new Pipeline(
        std::unique_ptr<Pipeline::Impl>(new Pipeline::Impl(dir, ov_device, cfg))
    ));
    out->models_path_ = models_path;
    out->device_ = ov_device;
    out->available_ = std::move(listed);
    out->device_full_name_ = device_full_name(ov_device);
    out->dim_ = dim_from_config_json(dir);
    return out;
}

std::vector<float> Pipeline::embed_documents(const std::vector<std::string>& texts) const {
    if (texts.empty()) {
        throw std::invalid_argument("embed_documents called with no texts");
    }
    if (device_ != "CPU" && device_ != "GPU") {
        throw std::runtime_error(
            "internal error: TextEmbeddingPipeline device is " + device_ +
            " (must be CPU or GPU)"
        );
    }
    auto rows = as_float_rows(impl_->pipe.embed_documents(texts));
    if (rows.size() != texts.size()) {
        throw std::runtime_error("embedding row count does not match input batch");
    }
    const size_t dim = rows.front().size();
    if (dim == 0) {
        throw std::runtime_error("embedding dimension is 0");
    }
    for (const auto& row : rows) {
        if (row.size() != dim) {
            throw std::runtime_error("ragged embedding batch");
        }
    }
    dim_ = static_cast<uint32_t>(dim);
    std::vector<float> flat;
    flat.reserve(rows.size() * dim);
    for (const auto& row : rows) {
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return flat;
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
