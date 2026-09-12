// Thin C++ façade over ov::genai::TextEmbeddingPipeline.
// Included by the cxx bridge (src/ffi.rs). Header stays free of OpenVINO
// types so the generated rust/cxx glue compiles against this file alone;
// the .cpp is the only translation unit that includes GenAI headers.

#pragma once

#include "rust/cxx.h"

#include <cstdint>
#include <memory>

namespace inferstream_ov {

class Pipeline {
public:
    Pipeline() = delete;
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    rust::Vec<float> embed_documents(const rust::Vec<rust::String>& texts) const;
    size_t embedding_dim() const;
    rust::String device() const;

private:
    friend std::unique_ptr<Pipeline> load_pipeline(
        rust::Str models_path,
        rust::Str device,
        uint8_t pooling,
        bool normalize,
        uint32_t max_length,
        bool pad_to_max_length
    );
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<Pipeline> load_pipeline(
    rust::Str models_path,
    rust::Str device,
    uint8_t pooling,
    bool normalize,
    uint32_t max_length,
    bool pad_to_max_length
);

rust::Vec<rust::String> available_devices();

} // namespace inferstream_ov
