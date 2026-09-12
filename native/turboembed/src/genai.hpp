/* SPDX-License-Identifier: Apache-2.0
 *
 * Internal façade over ov::genai::TextEmbeddingPipeline.
 * The .cpp is the only translation unit that includes GenAI headers.
 *
 * Device string is the OpenVINO GenAI constructor argument: "CPU" or
 * "GPU" (see samples/cpp/rag/text_embeddings.cpp). The requested string
 * is what we pass through — no silent swap. Asking for "GPU" when the
 * GPU plugin is missing fails; it does not compile "CPU".
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
    const std::string& device_full_name() const { return device_full_name_; }
    const std::vector<std::string>& available_devices() const { return available_; }

private:
    friend std::unique_ptr<Pipeline> load_pipeline(
        const std::string& models_path,
        const std::string& ov_device,
        const LoadConfig& config
    );
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    std::string models_path_;
    std::string device_;
    std::string device_full_name_;
    std::vector<std::string> available_;
    mutable uint32_t dim_;
};

/** Runtime OpenVINO device list (`CPU`, `GPU`, `GPU.0`, …). Throws on Core failure. */
std::vector<std::string> available_devices();

bool runtime_has_gpu();
bool runtime_has_cpu();

/** `ov::device::full_name` for `ov_device` (`"CPU"` / `"GPU"`), or empty. */
std::string device_full_name(const std::string& ov_device);

/**
 * Policy: pass through `"CPU"` or `"GPU"` exactly. Never `"AUTO"`.
 * `"GPU"` with no listed GPU plugin throws (no CPU fallback).
 * `"CPU"` with no listed CPU plugin throws.
 */
std::string require_ov_device(
    const std::string& requested,
    const std::vector<std::string>& available
);

/**
 * Construct TextEmbeddingPipeline on the exact OpenVINO device string
 * (`"CPU"` or `"GPU"`). Matches the official C++ sample:
 *
 *   ov::genai::TextEmbeddingPipeline pipeline(models_path, device, config);
 *
 * `"GPU"` with no GPU plugin throws (no CPU fallback).
 * `"CPU"` with no CPU plugin throws.
 * Any other string is rejected here — we never pass `"AUTO"`.
 */
std::unique_ptr<Pipeline> load_pipeline(
    const std::string& models_path,
    const std::string& ov_device,
    const LoadConfig& config
);

/** Resolve a GenAI-layout directory for `alias` (see impl for search order). */
std::string resolve_models_path(
    const std::string& alias,
    const std::string& config_path,
    const std::string& workspace_root
);

} // namespace turboembed_genai
