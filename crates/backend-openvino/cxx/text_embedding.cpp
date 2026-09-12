#include "text_embedding.hpp"

#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/runtime/core.hpp"

#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace inferstream_ov {
namespace {

struct NativeConfig {
    uint8_t pooling;
    bool normalize;
    uint32_t max_length;
    bool pad_to_max_length;
};

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

ov::genai::TextEmbeddingPipeline make_pipeline(
    const std::string& models_path,
    const std::string& device,
    const NativeConfig& config
) {
    ov::genai::TextEmbeddingPipeline::Config cfg;
    cfg.pooling_type = pooling_from_u8(config.pooling);
    cfg.normalize = config.normalize;
    if (config.max_length > 0) {
        cfg.max_length = static_cast<size_t>(config.max_length);
    }
    if (config.pad_to_max_length) {
        cfg.pad_to_max_length = true;
    }
    return ov::genai::TextEmbeddingPipeline(models_path, device, cfg);
}

std::vector<std::vector<float>> as_float_rows(ov::genai::EmbeddingResults results) {
    if (auto* floats = std::get_if<std::vector<std::vector<float>>>(&results)) {
        return std::move(*floats);
    }
    throw std::runtime_error(
        "TextEmbeddingPipeline returned a non-float embedding "
        "(int8/uint8 outputs are not served on this path; use an fp16/fp32 IR)"
    );
}

} // namespace

struct Pipeline::Impl {
    ov::genai::TextEmbeddingPipeline pipe;
    std::string device;
    mutable size_t dim;

    Impl(std::string models_path, std::string device_, const NativeConfig& config)
        : pipe(make_pipeline(models_path, device_, config)),
          device(std::move(device_)),
          dim(0) {}
};

Pipeline::Pipeline(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Pipeline::~Pipeline() = default;

std::unique_ptr<Pipeline> load_pipeline(
    rust::Str models_path,
    rust::Str device,
    uint8_t pooling,
    bool normalize,
    uint32_t max_length,
    bool pad_to_max_length
) {
    NativeConfig config{pooling, normalize, max_length, pad_to_max_length};
    auto impl = std::make_unique<Pipeline::Impl>(
        std::string(models_path),
        std::string(device),
        config
    );
    return std::unique_ptr<Pipeline>(new Pipeline(std::move(impl)));
}

rust::Vec<float> Pipeline::embed_documents(const rust::Vec<rust::String>& texts) const {
    if (texts.empty()) {
        throw std::invalid_argument("embed_documents called with no texts");
    }
    std::vector<std::string> input;
    input.reserve(texts.size());
    for (const auto& t : texts) {
        input.emplace_back(std::string(t));
    }
    auto rows = as_float_rows(impl_->pipe.embed_documents(input));
    if (rows.size() != input.size()) {
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
    impl_->dim = dim;
    rust::Vec<float> flat;
    flat.reserve(rows.size() * dim);
    for (const auto& row : rows) {
        for (float v : row) {
            flat.push_back(v);
        }
    }
    return flat;
}

size_t Pipeline::embedding_dim() const {
    return impl_->dim;
}

rust::String Pipeline::device() const {
    return rust::String(impl_->device);
}

rust::Vec<rust::String> available_devices() {
    ov::Core core;
    rust::Vec<rust::String> out;
    for (const auto& d : core.get_available_devices()) {
        out.push_back(rust::String(d));
    }
    return out;
}

} // namespace inferstream_ov
