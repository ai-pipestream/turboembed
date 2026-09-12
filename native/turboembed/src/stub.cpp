/* SPDX-License-Identifier: Apache-2.0
 *
 * TurboEmbed C ABI.
 *
 * Always: deterministic `mock-embed` / `mock`.
 * With -DTURBOEMBED_GENAI: catalog aliases (minilm, …) load
 * ov::genai::TextEmbeddingPipeline on the literal device string "GPU".
 * CPU / AUTO compile is rejected — not accepted as success.
 *
 * No Python. No OVMS. Does not replace inferstream servers.
 */

#include "turboembed.h"

#ifdef TURBOEMBED_GENAI
#include "genai.hpp"
#endif

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {

thread_local std::string g_create_error = "";

constexpr uint32_t kMockDim = 8;
constexpr const char *kMockAlias = "mock-embed";
constexpr const char *kMockAliasShort = "mock";

bool is_mock_alias(const char *alias, size_t len) {
    if (alias == nullptr) {
        return false;
    }
    if (len == 0) {
        len = std::strlen(alias);
    }
    return (len == 10 && std::memcmp(alias, kMockAlias, 10) == 0) ||
           (len == 4 && std::memcmp(alias, kMockAliasShort, 4) == 0);
}

uint64_t fnv1a(const char *ptr, size_t len) {
    uint64_t h = 14695981039346656037ull;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<unsigned char>(ptr[i]);
        h *= 1099511628211ull;
    }
    return h;
}

/* Deterministic unit-ish vector from text bytes. Independent of
 * inferstream's Rust mock — ABI smoke only. */
void mock_embed_row(const char *ptr, size_t len, float *out, uint32_t dim) {
    uint64_t h = fnv1a(ptr == nullptr ? "" : ptr, len);
    double sum_sq = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        h ^= h >> 30;
        h *= 0xbf58476d1ce4e5b9ull;
        h ^= h >> 27;
        h *= 0x94d049bb133111ebull;
        h ^= h >> 31;
        /* map to (-1, 1) */
        const int32_t raw = static_cast<int32_t>(h & 0xffffu) - 32768;
        const float v = static_cast<float>(raw) / 32768.0f;
        out[i] = v;
        sum_sq += static_cast<double>(v) * static_cast<double>(v);
    }
    if (sum_sq > 0.0) {
        const float inv = static_cast<float>(1.0 / std::sqrt(sum_sq));
        for (uint32_t i = 0; i < dim; ++i) {
            out[i] *= inv;
        }
    }
}

} // namespace

#ifndef TURBOEMBED_WORKSPACE_ROOT
#define TURBOEMBED_WORKSPACE_ROOT ""
#endif

struct turboembed_engine {
    turboembed_device device;
    std::string config_path;
    std::string last_error;
    bool mock_loaded;
#ifdef TURBOEMBED_GENAI
    std::unique_ptr<turboembed_genai::Pipeline> genai;
    std::string genai_alias;
#endif

    explicit turboembed_engine(turboembed_device device_, std::string config)
        : device(device_),
          config_path(std::move(config)),
          last_error(),
          mock_loaded(device_ == TURBOEMBED_DEVICE_MOCK ||
                      device_ == TURBOEMBED_DEVICE_AUTO ||
                      device_ == TURBOEMBED_DEVICE_CPU) {}

    void set_error(const char *msg) { last_error = msg ? msg : ""; }
};

#ifdef TURBOEMBED_GENAI
/* Map the ABI device to the OpenVINO GenAI constructor string.
 * AUTO picks GPU if the plugin is listed, else CPU — then that exact
 * string is compiled. OPENVINO_GPU never becomes "CPU". */
bool resolve_ov_device(turboembed_device device, std::string *out, std::string *err) {
    switch (device) {
        case TURBOEMBED_DEVICE_OPENVINO_GPU:
            *out = "GPU";
            return true;
        case TURBOEMBED_DEVICE_OPENVINO_CPU:
        case TURBOEMBED_DEVICE_CPU:
            *out = "CPU";
            return true;
        case TURBOEMBED_DEVICE_AUTO:
            try {
                *out = turboembed_genai::runtime_has_gpu() ? "GPU" : "CPU";
                return true;
            } catch (const std::exception &e) {
                *err = std::string("OpenVINO device query failed: ") + e.what();
                return false;
            }
        default:
            *err =
                "catalog alias needs TURBOEMBED_DEVICE_OPENVINO_GPU, "
                "TURBOEMBED_DEVICE_OPENVINO_CPU, TURBOEMBED_DEVICE_CPU, or AUTO";
            return false;
    }
}

turboembed_device abi_device_from_ov(const std::string &ov) {
    if (ov == "CPU") {
        return TURBOEMBED_DEVICE_OPENVINO_CPU;
    }
    return TURBOEMBED_DEVICE_OPENVINO_GPU;
}

uint8_t pooling_for_alias(const char *alias, size_t len, uint8_t requested) {
    if (requested == TURBOEMBED_POOLING_CLS) {
        return 0;
    }
    if (requested == TURBOEMBED_POOLING_LAST) {
        return 2;
    }
    if (requested == TURBOEMBED_POOLING_MEAN) {
        return 1;
    }
    /* DEFAULT: catalog convention — BGE = CLS, everything else MEAN. */
    if (len >= 3 && std::memcmp(alias, "bge", 3) == 0) {
        return 0;
    }
    return 1;
}

bool alias_eq(const char *alias, size_t len, const std::string &loaded) {
    return len == loaded.size() && std::memcmp(alias, loaded.data(), len) == 0;
}
#endif

extern "C" {

uint32_t turboembed_abi_version(void) { return TURBOEMBED_ABI_VERSION; }

const char *turboembed_status_name(turboembed_status status) {
    switch (status) {
        case TURBOEMBED_OK:
            return "OK";
        case TURBOEMBED_ERR_INVALID_ARGUMENT:
            return "INVALID_ARGUMENT";
        case TURBOEMBED_ERR_NOT_FOUND:
            return "NOT_FOUND";
        case TURBOEMBED_ERR_NOT_IMPLEMENTED:
            return "NOT_IMPLEMENTED";
        case TURBOEMBED_ERR_UNAVAILABLE:
            return "UNAVAILABLE";
        case TURBOEMBED_ERR_INTERNAL:
            return "INTERNAL";
        case TURBOEMBED_ERR_OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case TURBOEMBED_ERR_UNSUPPORTED_DEVICE:
            return "UNSUPPORTED_DEVICE";
        default:
            return "UNKNOWN";
    }
}

const char *turboembed_device_name(turboembed_device device) {
    switch (device) {
        case TURBOEMBED_DEVICE_AUTO:
            return "auto";
        case TURBOEMBED_DEVICE_CPU:
            return "cpu";
        case TURBOEMBED_DEVICE_CUDA:
            return "cuda";
        case TURBOEMBED_DEVICE_TENSORRT:
            return "tensorrt";
        case TURBOEMBED_DEVICE_OPENVINO_CPU:
            return "openvino-cpu";
        case TURBOEMBED_DEVICE_OPENVINO_GPU:
            return "openvino-gpu";
        case TURBOEMBED_DEVICE_OPENVINO_NPU:
            return "openvino-npu";
        case TURBOEMBED_DEVICE_METAL:
            return "metal";
        case TURBOEMBED_DEVICE_MOCK:
            return "mock";
        default:
            return "unknown";
    }
}

const char *turboembed_last_error(const turboembed_engine *engine) {
    if (engine == nullptr) {
        return g_create_error.c_str();
    }
    return engine->last_error.empty() ? "" : engine->last_error.c_str();
}

turboembed_status turboembed_engine_create(
    turboembed_device device,
    const char *config_path,
    turboembed_engine **out
) {
    if (out == nullptr) {
        g_create_error = "out pointer is null";
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    switch (device) {
        case TURBOEMBED_DEVICE_AUTO:
        case TURBOEMBED_DEVICE_CPU:
        case TURBOEMBED_DEVICE_CUDA:
        case TURBOEMBED_DEVICE_TENSORRT:
        case TURBOEMBED_DEVICE_OPENVINO_CPU:
        case TURBOEMBED_DEVICE_OPENVINO_GPU:
        case TURBOEMBED_DEVICE_OPENVINO_NPU:
        case TURBOEMBED_DEVICE_METAL:
        case TURBOEMBED_DEVICE_MOCK:
            break;
        default:
            g_create_error = "unknown device enum";
            return TURBOEMBED_ERR_UNSUPPORTED_DEVICE;
    }
#ifdef TURBOEMBED_GENAI
    if (device == TURBOEMBED_DEVICE_OPENVINO_GPU) {
        try {
            if (!turboembed_genai::runtime_has_gpu()) {
                std::string listed;
                for (const auto &d : turboembed_genai::available_devices()) {
                    if (!listed.empty()) {
                        listed += ", ";
                    }
                    listed += d;
                }
                g_create_error =
                    "TURBOEMBED_DEVICE_OPENVINO_GPU requested but OpenVINO "
                    "listed no GPU ([" +
                    listed +
                    "]); refusing CPU fallback. "
                    "Need libopenvino_intel_gpu_plugin + Level Zero.";
                return TURBOEMBED_ERR_UNSUPPORTED_DEVICE;
            }
        } catch (const std::exception &e) {
            g_create_error =
                std::string("OpenVINO device query failed: ") + e.what();
            return TURBOEMBED_ERR_UNAVAILABLE;
        }
    }
#endif
    try {
        *out = new turboembed_engine(device, config_path ? config_path : "");
    } catch (const std::bad_alloc &) {
        g_create_error = "engine allocation failed";
        return TURBOEMBED_ERR_OUT_OF_MEMORY;
    }
    g_create_error.clear();
    return TURBOEMBED_OK;
}

void turboembed_engine_destroy(turboembed_engine *engine) { delete engine; }

turboembed_status turboembed_list_models(
    turboembed_engine *engine,
    turboembed_model_info **out_infos,
    size_t *out_count
) {
    if (engine == nullptr || out_infos == nullptr || out_count == nullptr) {
        if (engine != nullptr) {
            engine->set_error("null list_models argument");
        }
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    *out_infos = nullptr;
    *out_count = 0;

#ifdef TURBOEMBED_GENAI
    if (engine->genai) {
        auto *infos = static_cast<turboembed_model_info *>(
            std::calloc(1, sizeof(turboembed_model_info))
        );
        if (infos == nullptr) {
            engine->set_error("model list allocation failed");
            return TURBOEMBED_ERR_OUT_OF_MEMORY;
        }
        const std::string &name = engine->genai_alias;
        char *alias = static_cast<char *>(std::malloc(name.size() + 1));
        if (alias == nullptr) {
            std::free(infos);
            engine->set_error("alias allocation failed");
            return TURBOEMBED_ERR_OUT_OF_MEMORY;
        }
        std::memcpy(alias, name.data(), name.size());
        alias[name.size()] = '\0';
        infos[0].alias.ptr = alias;
        infos[0].alias.len = name.size();
        infos[0].dim = engine->genai->embedding_dim();
        infos[0].device = abi_device_from_ov(engine->genai->device());
        infos[0].ready = 1;
        *out_infos = infos;
        *out_count = 1;
        engine->set_error("");
        return TURBOEMBED_OK;
    }
#endif

    auto *infos = static_cast<turboembed_model_info *>(
        std::calloc(1, sizeof(turboembed_model_info))
    );
    if (infos == nullptr) {
        engine->set_error("model list allocation failed");
        return TURBOEMBED_ERR_OUT_OF_MEMORY;
    }
    char *alias = static_cast<char *>(std::malloc(11));
    if (alias == nullptr) {
        std::free(infos);
        engine->set_error("alias allocation failed");
        return TURBOEMBED_ERR_OUT_OF_MEMORY;
    }
    std::memcpy(alias, kMockAlias, 11);
    infos[0].alias.ptr = alias;
    infos[0].alias.len = 10;
    infos[0].dim = kMockDim;
    infos[0].device = TURBOEMBED_DEVICE_MOCK;
    infos[0].ready = engine->mock_loaded ? 1 : 0;
    *out_infos = infos;
    *out_count = 1;
    engine->set_error("");
    return TURBOEMBED_OK;
}

void turboembed_model_list_free(turboembed_model_info *infos, size_t count) {
    if (infos == nullptr) {
        return;
    }
    for (size_t i = 0; i < count; ++i) {
        std::free(const_cast<char *>(infos[i].alias.ptr));
    }
    std::free(infos);
}

turboembed_status turboembed_load_model(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len
) {
    if (engine == nullptr || alias == nullptr) {
        if (engine != nullptr) {
            engine->set_error("null load_model argument");
        }
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    if (alias_len == 0) {
        alias_len = std::strlen(alias);
    }
    if (alias_len == 0) {
        engine->set_error("alias is empty");
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    if (is_mock_alias(alias, alias_len)) {
        engine->mock_loaded = true;
        engine->set_error("");
        return TURBOEMBED_OK;
    }

#ifdef TURBOEMBED_GENAI
    if (engine->device == TURBOEMBED_DEVICE_OPENVINO_NPU) {
        engine->set_error(
            "NPU is not wired (GenAI has a distinct NPU compile path); "
            "use TURBOEMBED_DEVICE_OPENVINO_GPU or "
            "TURBOEMBED_DEVICE_OPENVINO_CPU / CPU"
        );
        return TURBOEMBED_ERR_NOT_IMPLEMENTED;
    }
    if (engine->device == TURBOEMBED_DEVICE_CUDA ||
        engine->device == TURBOEMBED_DEVICE_TENSORRT ||
        engine->device == TURBOEMBED_DEVICE_METAL) {
        engine->set_error(
            "ORT CUDA / TensorRT / Metal providers are not wired in this "
            "Intel GenAI build; use OPENVINO_GPU or OPENVINO_CPU / CPU"
        );
        return TURBOEMBED_ERR_NOT_IMPLEMENTED;
    }
    std::string ov_device;
    std::string resolve_err;
    if (!resolve_ov_device(engine->device, &ov_device, &resolve_err)) {
        engine->set_error(resolve_err.c_str());
        return TURBOEMBED_ERR_UNSUPPORTED_DEVICE;
    }
    try {
        const std::string alias_s(alias, alias_len);
        const std::string path = turboembed_genai::resolve_models_path(
            alias_s,
            engine->config_path,
            TURBOEMBED_WORKSPACE_ROOT
        );
        turboembed_genai::LoadConfig cfg;
        cfg.pooling = pooling_for_alias(alias, alias_len, TURBOEMBED_POOLING_DEFAULT);
        cfg.normalize = true;
        cfg.max_length = 256;
        engine->genai = turboembed_genai::load_pipeline(path, ov_device, cfg);
        if (engine->genai->device() != ov_device) {
            const std::string got = engine->genai->device();
            engine->genai.reset();
            engine->last_error =
                "TextEmbeddingPipeline compiled for " + got + " but requested " +
                ov_device + "; refusing a silent device swap";
            return TURBOEMBED_ERR_INTERNAL;
        }
        if (ov_device == "GPU" && engine->genai->device() != "GPU") {
            engine->genai.reset();
            engine->set_error(
                "GPU was requested but the pipeline is not on GPU; "
                "refusing CPU fallback"
            );
            return TURBOEMBED_ERR_INTERNAL;
        }
        engine->genai_alias = alias_s;
        engine->set_error("");
        return TURBOEMBED_OK;
    } catch (const std::exception &e) {
        engine->genai.reset();
        engine->genai_alias.clear();
        engine->set_error(e.what());
        return TURBOEMBED_ERR_UNAVAILABLE;
    }
#else
    /* Catalog aliases (minilm, bge-*, …) need --features genai on Intel. */
    engine->set_error(
        "OpenVINO GenAI is not compiled into this TurboEmbed build; "
        "rebuild crates/turboembed with --features genai "
        "(TextEmbeddingPipeline on GPU; see docs/intel-genai-embed.md)"
    );
    return TURBOEMBED_ERR_NOT_IMPLEMENTED;
#endif
}

static turboembed_status embed_impl(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const turboembed_str *texts,
    size_t n_texts,
    const turboembed_embed_options *opts,
    turboembed_embed_result **out
) {
    if (engine == nullptr || out == nullptr) {
        if (engine != nullptr) {
            engine->set_error("null embed argument");
        }
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (alias == nullptr) {
        engine->set_error("alias is null");
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    if (alias_len == 0) {
        alias_len = std::strlen(alias);
    }
    if (n_texts == 0) {
        engine->set_error("texts must not be empty");
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    if (texts == nullptr) {
        engine->set_error("texts pointer is null");
        return TURBOEMBED_ERR_INVALID_ARGUMENT;
    }
    for (size_t i = 0; i < n_texts; ++i) {
        if (texts[i].len > 0 && texts[i].ptr == nullptr) {
            engine->set_error("text view has null ptr with non-zero len");
            return TURBOEMBED_ERR_INVALID_ARGUMENT;
        }
    }
#ifdef TURBOEMBED_GENAI
    if (engine->genai && alias_eq(alias, alias_len, engine->genai_alias)) {
        if (engine->device == TURBOEMBED_DEVICE_OPENVINO_GPU &&
            engine->genai->device() != "GPU") {
            engine->last_error =
                "GPU was requested but the loaded pipeline is on " +
                engine->genai->device() + "; refusing CPU fallback";
            return TURBOEMBED_ERR_INTERNAL;
        }
        if (engine->genai->device() != "CPU" &&
            engine->genai->device() != "GPU") {
            engine->set_error(
                "loaded pipeline device is not CPU or GPU"
            );
            return TURBOEMBED_ERR_INTERNAL;
        }
        if (opts != nullptr) {
            if (opts->pooling == TURBOEMBED_POOLING_CLS ||
                opts->pooling == TURBOEMBED_POOLING_LAST) {
                const uint8_t want =
                    pooling_for_alias(alias, alias_len, opts->pooling);
                const uint8_t have =
                    pooling_for_alias(alias, alias_len, TURBOEMBED_POOLING_DEFAULT);
                if (want != have) {
                    engine->set_error(
                        "embed pooling overrides the catalog default; "
                        "reload the pipeline with that pooling "
                        "(constructor-time Config only)"
                    );
                    return TURBOEMBED_ERR_INVALID_ARGUMENT;
                }
            }
            if (opts->normalize == 0) {
                engine->set_error(
                    "normalize=false is not the catalog MiniLM path "
                    "(goldens are L2-normalized)"
                );
                return TURBOEMBED_ERR_INVALID_ARGUMENT;
            }
        }
        try {
            std::vector<std::string> input;
            input.reserve(n_texts);
            for (size_t i = 0; i < n_texts; ++i) {
                input.emplace_back(
                    texts[i].ptr == nullptr ? "" : texts[i].ptr,
                    texts[i].len
                );
            }
            std::vector<float> flat = engine->genai->embed_documents(input);
            const uint32_t dim = engine->genai->embedding_dim();
            if (dim == 0 || flat.size() != n_texts * static_cast<size_t>(dim)) {
                engine->set_error("ragged GenAI embedding batch");
                return TURBOEMBED_ERR_INTERNAL;
            }
            auto *result = static_cast<turboembed_embed_result *>(
                std::calloc(1, sizeof(turboembed_embed_result))
            );
            if (result == nullptr) {
                engine->set_error("result allocation failed");
                return TURBOEMBED_ERR_OUT_OF_MEMORY;
            }
            auto *values = static_cast<float *>(
                std::malloc(flat.size() * sizeof(float))
            );
            if (values == nullptr) {
                std::free(result);
                engine->set_error("values allocation failed");
                return TURBOEMBED_ERR_OUT_OF_MEMORY;
            }
            std::memcpy(values, flat.data(), flat.size() * sizeof(float));
            result->dim = dim;
            result->count = static_cast<uint32_t>(n_texts);
            result->values = values;
            result->packed = reinterpret_cast<const uint8_t *>(values);
            result->packed_len = flat.size() * sizeof(float);
            *out = result;
            engine->set_error("");
            return TURBOEMBED_OK;
        } catch (const std::exception &e) {
            engine->set_error(e.what());
            return TURBOEMBED_ERR_INTERNAL;
        }
    }
    if (!is_mock_alias(alias, alias_len)) {
        engine->set_error(
            engine->genai
                ? "alias is not the loaded GenAI model"
                : "catalog alias is not loaded; call turboembed_load_model first"
        );
        return TURBOEMBED_ERR_NOT_FOUND;
    }
#endif

    if (!is_mock_alias(alias, alias_len)) {
#ifndef TURBOEMBED_GENAI
        engine->set_error(
            "embed on catalog aliases needs --features genai "
            "(TextEmbeddingPipeline on GPU)"
        );
        return TURBOEMBED_ERR_NOT_IMPLEMENTED;
#else
        engine->set_error("catalog alias is not loaded");
        return TURBOEMBED_ERR_NOT_FOUND;
#endif
    }
    if (!engine->mock_loaded) {
        engine->set_error("mock-embed is not loaded");
        return TURBOEMBED_ERR_NOT_FOUND;
    }

    (void)opts; /* pooling / normalize ignored on the mock path */

    const size_t n_floats = n_texts * static_cast<size_t>(kMockDim);
    auto *result = static_cast<turboembed_embed_result *>(
        std::calloc(1, sizeof(turboembed_embed_result))
    );
    if (result == nullptr) {
        engine->set_error("result allocation failed");
        return TURBOEMBED_ERR_OUT_OF_MEMORY;
    }
    auto *values = static_cast<float *>(std::malloc(n_floats * sizeof(float)));
    if (values == nullptr) {
        std::free(result);
        engine->set_error("values allocation failed");
        return TURBOEMBED_ERR_OUT_OF_MEMORY;
    }
    for (size_t i = 0; i < n_texts; ++i) {
        mock_embed_row(
            texts[i].ptr,
            texts[i].len,
            values + i * kMockDim,
            kMockDim
        );
    }
    result->dim = kMockDim;
    result->count = static_cast<uint32_t>(n_texts);
    result->values = values;
    result->packed = reinterpret_cast<const uint8_t *>(values);
    result->packed_len = n_floats * sizeof(float);
    *out = result;
    engine->set_error("");
    return TURBOEMBED_OK;
}

turboembed_status turboembed_embed_one(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const char *text,
    size_t text_len,
    const turboembed_embed_options *opts,
    turboembed_embed_result **out
) {
    turboembed_str view;
    view.ptr = text;
    view.len = text_len;
    return embed_impl(engine, alias, alias_len, &view, 1, opts, out);
}

turboembed_status turboembed_embed(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const turboembed_str *texts,
    size_t n_texts,
    const turboembed_embed_options *opts,
    turboembed_embed_result **out
) {
    return embed_impl(engine, alias, alias_len, texts, n_texts, opts, out);
}

turboembed_status turboembed_embed_stream(
    turboembed_engine *engine,
    const char *alias,
    size_t alias_len,
    const turboembed_str *texts,
    size_t n_texts,
    const turboembed_embed_options *opts,
    turboembed_stream_cb cb,
    void *user_data,
    turboembed_embed_result **out
) {
    turboembed_embed_result *result = nullptr;
    const turboembed_status st =
        embed_impl(engine, alias, alias_len, texts, n_texts, opts, &result);
    if (st != TURBOEMBED_OK) {
        return st;
    }
    if (cb != nullptr && result != nullptr) {
        for (uint32_t i = 0; i < result->count; ++i) {
            const int32_t is_final = (i + 1 == result->count) ? 1 : 0;
            cb(user_data, i, result->values + (i * result->dim), result->dim, is_final);
        }
    }
    if (out != nullptr) {
        *out = result;
    } else {
        turboembed_embed_result_free(result);
    }
    return TURBOEMBED_OK;
}

void turboembed_embed_result_free(turboembed_embed_result *result) {
    if (result == nullptr) {
        return;
    }
    std::free(const_cast<float *>(result->values));
    std::free(result);
}

void turboembed_buffer_free(void *ptr) { std::free(ptr); }

turboembed_status turboembed_register_provider(
    const turboembed_provider_vtbl *vtbl
) {
    (void)vtbl;
    g_create_error =
        "turboembed_register_provider is reserved for ORT / GenAI / MLX / "
        "model2vec plugins";
    return TURBOEMBED_ERR_NOT_IMPLEMENTED;
}

} // extern "C"
