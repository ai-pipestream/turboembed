/* SPDX-License-Identifier: Apache-2.0
 *
 * Internal façade: ov::genai::Tokenizer + CompiledModel on the official
 * device string ("CPU" / "GPU" / "NPU"). Token rows and hidden-state
 * scratch are rented from the engine turbo_buffer arena (ZE SHARED on
 * GPU). embed_into writes the pooled row into a caller-rented result
 * slot — it does not return a private std::vector.
 *
 * TextEmbeddingPipeline.embed_documents is not on the hot path: that
 * API private-allocs token_type_ids and EmbeddingResults every call
 * and has no hook for caller USM.
 *
 * Device string is passed through — no silent swap. Asking for "GPU"
 * / "NPU" when that plugin is missing fails; it does not compile "CPU".
 */

#pragma once

#include "turbo_buffer.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace turboembed_genai {

struct LoadConfig {
    uint8_t pooling; /* 0=CLS, 1=MEAN, 2=LAST_TOKEN */
    bool normalize;
    uint32_t max_length; /* 0 = 256 */
};

class Pipeline {
public:
    Pipeline() = delete;
    ~Pipeline();
    Pipeline(const Pipeline&) = delete;
    Pipeline& operator=(const Pipeline&) = delete;

    /**
     * Tokenize, wrap arena USM as ov::Tensor, infer, pool+L2 into `out`.
     * `out` is [n_texts, dim] row-major, caller-rented. No result vector.
     */
    void embed_into(const std::vector<std::string>& texts, float *out);

    uint32_t embedding_dim() const { return dim_; }
    const std::string& device() const { return device_; }
    const std::string& models_path() const { return models_path_; }
    const std::string& device_full_name() const { return device_full_name_; }
    const std::vector<std::string>& available_devices() const { return available_; }

    turbo_buffer_device arena_device() const;
    turbo_buffer_placement token_placement() const;
    const void *token_ids_ptr() const;
    const void *hidden_ptr() const;
    bool owns_tokens() const;
    bool owns_hidden() const;
    bool last_hidden_used_arena() const { return last_hidden_arena_; }

private:
    friend std::unique_ptr<Pipeline> load_pipeline(
        const std::string& models_path,
        const std::string& ov_device,
        const LoadConfig& config,
        turbo_buffer_arena *arena,
        turbo_buffer_placement place
    );
    struct Impl;
    explicit Pipeline(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
    std::string models_path_;
    std::string device_;
    std::string device_full_name_;
    std::vector<std::string> available_;
    uint32_t dim_;
    bool last_hidden_arena_;
};

/** Runtime OpenVINO device list (`CPU`, `GPU`, `GPU.0`, …). Throws on Core failure. */
std::vector<std::string> available_devices();

bool runtime_has_gpu();
bool runtime_has_cpu();
bool runtime_has_npu();

/** `ov::device::full_name` for `ov_device` (`"CPU"` / `"GPU"` / `"NPU"`), or empty. */
std::string device_full_name(const std::string& ov_device);

/**
 * Policy: pass through `"CPU"`, `"GPU"`, or `"NPU"` exactly. Never `"AUTO"`.
 * `"GPU"` / `"NPU"` with no listed plugin throws (no CPU fallback).
 * `"CPU"` with no listed CPU plugin throws.
 */
std::string require_ov_device(
    const std::string& requested,
    const std::vector<std::string>& available
);

/**
 * Compile the MiniLM IR on `ov_device` and rent token/hidden slabs from
 * `arena` at `place` (SHARED on GPU, HOST on CPU). Never constructs
 * TextEmbeddingPipeline for embed — that path private-allocs.
 */
std::unique_ptr<Pipeline> load_pipeline(
    const std::string& models_path,
    const std::string& ov_device,
    const LoadConfig& config,
    turbo_buffer_arena *arena,
    turbo_buffer_placement place
);

/** Resolve a GenAI-layout directory for `alias` (see impl for search order). */
std::string resolve_models_path(
    const std::string& alias,
    const std::string& config_path,
    const std::string& workspace_root
);

} // namespace turboembed_genai
