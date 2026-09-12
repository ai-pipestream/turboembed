/* SPDX-License-Identifier: Apache-2.0
 *
 * Internal façade over ov::genai::TextEmbeddingPipeline.
 * The .cpp is the only translation unit that includes GenAI headers.
 *
 * GPU only. CPU / AUTO / NPU are rejected here — callers must not treat
 * a CPU compile as success.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace turboembed_genai {

struct LoadConfig {
    uint8_t pooling; /* 0=CLS, 1=MEAN, 2=LAST_TOKEN */
    bool normalize;
    uint32_t max_length; /* 0 = provider default */
};

class Pipeline {
public:
    Pipeline() = delete;
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    std::vector<float> embed_documents(const std::vector<std::string>& texts) const;
    uint32_t embedding_dim() const { return dim_; }
    const std::string& device() const { return device_; }
    const std::string& models_path() const { return models_path_; }
    const std::string& gpu_full_name() const { return gpu_full_name_; }
    const std::vector<std::string>& available_devices() const { return available_; }

private:
    friend std::unique_ptr<Pipeline> load_gpu_pipeline(
        const std::string& models_path,
        const LoadConfig& config
    );
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    std::string models_path_;
    std::string device_;
    std::string gpu_full_name_;
    std::vector<std::string> available_;
    mutable uint32_t dim_;
};

/** Runtime OpenVINO device list (`CPU`, `GPU`, `GPU.0`, …). Throws on Core failure. */
std::vector<std::string> available_devices();

/** True iff any listed device starts with `GPU`. */
bool runtime_has_gpu();

/**
 * Full GPU name from the OpenVINO GPU plugin (empty if query fails).
 * Does not compile a model.
 */
std::string gpu_full_name();

/**
 * Construct TextEmbeddingPipeline on device `"GPU"`.
 *
 * Fails (throws std::runtime_error) if the GPU plugin is missing.
 * Never falls back to CPU.
 */
std::unique_ptr<Pipeline> load_gpu_pipeline(
    const std::string& models_path,
    const LoadConfig& config
);

/** Resolve a GenAI-layout directory for `alias` (see impl for search order). */
std::string resolve_models_path(
    const std::string& alias,
    const std::string& config_path,
    const std::string& workspace_root
);

} // namespace turboembed_genai
