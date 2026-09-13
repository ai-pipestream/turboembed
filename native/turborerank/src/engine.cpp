// SPDX-License-Identifier: Apache-2.0
//
// Engine lifecycle + frozen C ABI.

#include "cuda_api.hpp"
#include "internal.hpp"
#include "metal_api.hpp"
#include "ov_api.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifndef TURBORERANK_WORKSPACE_ROOT
#define TURBORERANK_WORKSPACE_ROOT ""
#endif

namespace {

bool is_dir(const std::string &p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool is_file(const std::string &p) {
    struct stat st {};
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

std::string join_path(const std::string &a, const std::string &b) {
    if (a.empty()) {
        return b;
    }
    if (a.back() == '/') {
        return a + b;
    }
    return a + "/" + b;
}

std::string alias_view(const char *alias, size_t len) {
    if (alias == nullptr) {
        return {};
    }
    if (len == 0) {
        return std::string(alias);
    }
    return std::string(alias, len);
}

bool alias_is_ce(const std::string &a) {
    return a == "ms-marco-minilm-l6" || a == "ms-marco-minilm-l-6-v2" ||
           a == "ms-marco-minilm-l6-v2" || a == "minilm-ce" ||
           a == "cross-encoder/ms-marco-MiniLM-L6-v2";
}

void free_work_buffer(turborerank_engine *e) {
    if (e != nullptr && e->work != nullptr) {
        turborerank_buffer_free(e->work);
        e->work = nullptr;
    }
}

} // namespace

namespace turborerank {
namespace impl {

std::string resolve_model_dir(
    const char *alias,
    size_t alias_len,
    const char *config_path,
    const char *workspace_root
) {
    const std::string name = alias_view(alias, alias_len);
    if (name.empty()) {
        return {};
    }
    if (is_dir(name)) {
        return name;
    }
    if (is_file(name)) {
        // A file path: use parent directory.
        const auto slash = name.find_last_of('/');
        if (slash != std::string::npos) {
            return name.substr(0, slash);
        }
    }
    std::vector<std::string> roots;
    if (config_path && *config_path) {
        if (is_dir(config_path)) {
            roots.push_back(config_path);
            roots.push_back(join_path(config_path, name));
        }
    }
    if (workspace_root && *workspace_root) {
        roots.push_back(join_path(join_path(workspace_root, "models/rerank"), name));
        roots.push_back(join_path(workspace_root, name));
    }
    if (const char *env = std::getenv("TURBORERANK_MODEL_DIR")) {
        roots.push_back(env);
        roots.push_back(join_path(env, name));
    }
    roots.push_back(join_path("models/rerank", name));
    // Canonical alias folder.
    if (alias_is_ce(name)) {
        if (workspace_root && *workspace_root) {
            roots.push_back(join_path(workspace_root, "models/rerank/ms-marco-minilm-l6"));
        }
        roots.push_back("models/rerank/ms-marco-minilm-l6");
    }
    for (const auto &r : roots) {
        if (is_dir(r) && is_file(join_path(r, "model.safetensors"))) {
            return r;
        }
    }
    return {};
}

bool read_bert_config_json(const char *path, BertConfig *cfg, std::string *err) {
    std::ifstream in(path);
    if (!in) {
        // Optional: keep compiled defaults (MiniLM-L6).
        return true;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const std::string j = ss.str();
    auto grab_u32 = [&](const char *key, uint32_t *dst) {
        const std::string k = std::string("\"") + key + "\"";
        size_t p = j.find(k);
        if (p == std::string::npos) {
            return;
        }
        p = j.find(':', p);
        if (p == std::string::npos) {
            return;
        }
        *dst = static_cast<uint32_t>(std::strtoul(j.c_str() + p + 1, nullptr, 10));
    };
    auto grab_f = [&](const char *key, float *dst) {
        const std::string k = std::string("\"") + key + "\"";
        size_t p = j.find(k);
        if (p == std::string::npos) {
            return;
        }
        p = j.find(':', p);
        if (p == std::string::npos) {
            return;
        }
        *dst = std::strtof(j.c_str() + p + 1, nullptr);
    };
    grab_u32("hidden_size", &cfg->hidden);
    grab_u32("intermediate_size", &cfg->intermediate);
    grab_u32("num_hidden_layers", &cfg->layers);
    grab_u32("num_attention_heads", &cfg->heads);
    grab_u32("max_position_embeddings", &cfg->max_position);
    grab_u32("vocab_size", &cfg->vocab_size);
    grab_u32("type_vocab_size", &cfg->type_vocab);
    grab_f("layer_norm_eps", &cfg->ln_eps);
    if (cfg->layers == 0 || cfg->hidden == 0 || cfg->heads == 0 ||
        cfg->hidden % cfg->heads != 0) {
        if (err) {
            *err = std::string("invalid BERT config: ") + path;
        }
        return false;
    }
    if (cfg->layers > 12) {
        if (err) {
            *err = "Phase 1 CPU kernel supports at most 12 layers";
        }
        return false;
    }
    return true;
}

} // namespace impl
} // namespace turborerank

extern "C" {

uint32_t turborerank_abi_version(void) {
    return TURBORERANK_ABI_VERSION;
}

const char *turborerank_status_name(turborerank_status status) {
    switch (status) {
    case TURBORERANK_OK:
        return "OK";
    case TURBORERANK_ERR_INVALID_ARGUMENT:
        return "INVALID_ARGUMENT";
    case TURBORERANK_ERR_NOT_FOUND:
        return "NOT_FOUND";
    case TURBORERANK_ERR_NOT_IMPLEMENTED:
        return "NOT_IMPLEMENTED";
    case TURBORERANK_ERR_UNAVAILABLE:
        return "UNAVAILABLE";
    case TURBORERANK_ERR_INTERNAL:
        return "INTERNAL";
    case TURBORERANK_ERR_OUT_OF_MEMORY:
        return "OUT_OF_MEMORY";
    case TURBORERANK_ERR_UNSUPPORTED_DEVICE:
        return "UNSUPPORTED_DEVICE";
    default:
        return "UNKNOWN";
    }
}

const char *turborerank_device_name(turborerank_device device) {
    switch (device) {
    case TURBORERANK_DEVICE_AUTO:
        return "AUTO";
    case TURBORERANK_DEVICE_CPU:
        return "CPU";
    case TURBORERANK_DEVICE_CUDA:
        return "CUDA";
    case TURBORERANK_DEVICE_TENSORRT:
        return "TENSORRT";
    case TURBORERANK_DEVICE_OPENVINO_CPU:
        return "OPENVINO_CPU";
    case TURBORERANK_DEVICE_OPENVINO_GPU:
        return "OPENVINO_GPU";
    case TURBORERANK_DEVICE_OPENVINO_NPU:
        return "OPENVINO_NPU";
    case TURBORERANK_DEVICE_METAL:
        return "METAL";
    case TURBORERANK_DEVICE_MOCK:
        return "MOCK";
    default:
        return "UNKNOWN";
    }
}

const char *turborerank_last_error(const turborerank_engine *engine) {
    if (engine != nullptr) {
        return engine->last_error.empty() ? "" : engine->last_error.c_str();
    }
    return turborerank::impl::create_error();
}

turborerank_status turborerank_engine_create(
    turborerank_device device,
    const char *config_path,
    turborerank_engine **out
) {
    if (out == nullptr) {
        turborerank::impl::set_create_error("engine_create: null out");
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;

    if (device == TURBORERANK_DEVICE_OPENVINO_CPU ||
        turborerank::impl::device_is_accelerator(device)) {
        std::string why;
        if (turborerank::impl::accelerator_unavailable(device, &why)) {
            if (why.empty()) {
                why = "device is not available; refusing CPU fallback";
            }
            turborerank::impl::set_create_error(why);
            if (device == TURBORERANK_DEVICE_OPENVINO_CPU &&
                !turborerank::impl::ov_compiled()) {
                return TURBORERANK_ERR_NOT_IMPLEMENTED;
            }
            return TURBORERANK_ERR_UNAVAILABLE;
        }
        // CUDA / AUTO-with-CUDA continue; other accelerators still fail above.
    }

    const turborerank_device resolved =
        turborerank::impl::resolve_create_device(device);

    if (resolved != TURBORERANK_DEVICE_CPU &&
        resolved != TURBORERANK_DEVICE_MOCK &&
        resolved != TURBORERANK_DEVICE_CUDA &&
        resolved != TURBORERANK_DEVICE_OPENVINO_CPU &&
        resolved != TURBORERANK_DEVICE_OPENVINO_GPU &&
        resolved != TURBORERANK_DEVICE_METAL) {
        turborerank::impl::set_create_error(
            "unsupported device; refusing CPU fallback"
        );
        return TURBORERANK_ERR_UNSUPPORTED_DEVICE;
    }

    auto *engine = new (std::nothrow) turborerank_engine();
    if (engine == nullptr) {
        turborerank::impl::set_create_error("engine_create: OOM");
        return TURBORERANK_ERR_OUT_OF_MEMORY;
    }
    engine->device = resolved;
    if (config_path != nullptr) {
        engine->config_path = config_path;
    }
    const char *ws = std::getenv("INFERSTREAM_ROOT");
    if (ws == nullptr || ws[0] == '\0') {
        ws = std::getenv("TURBORERANK_WORKSPACE_ROOT");
    }
    if (ws == nullptr || ws[0] == '\0') {
        const char *baked = TURBORERANK_WORKSPACE_ROOT;
        ws = (baked != nullptr && baked[0] != '\0') ? baked : nullptr;
    }
    if (ws != nullptr) {
        engine->workspace_root = ws;
    }
    if (device == TURBORERANK_DEVICE_MOCK) {
        engine->last_error =
            "MOCK engine created; catalog cross-encoders will not load. "
            "Mock does not produce relevance scores.";
    }
    *out = engine;
    return TURBORERANK_OK;
}

void turborerank_engine_destroy(turborerank_engine *engine) {
    if (engine == nullptr) {
        return;
    }
    free_work_buffer(engine);
    turborerank::impl::cuda_resources_free(&engine->cuda);
    turborerank::impl::ov_resources_free(&engine->ov);
    turborerank::impl::metal_resources_free(&engine->metal);
    turborerank::impl::free_scratch(&engine->scratch);
    turborerank::impl::free_owned(&engine->owned_weights);
    turborerank::impl::free_mapped(&engine->mapped);
    delete engine;
}

turborerank_status turborerank_list_models(
    turborerank_engine *engine,
    turborerank_model_info **out_infos,
    size_t *out_count
) {
    if (engine == nullptr || out_infos == nullptr || out_count == nullptr) {
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    auto *infos = new (std::nothrow) turborerank_model_info[1];
    if (infos == nullptr) {
        engine->last_error = "list_models: OOM";
        return TURBORERANK_ERR_OUT_OF_MEMORY;
    }
    static const char kAlias[] = "ms-marco-minilm-l6";
    infos[0].alias.ptr = kAlias;
    infos[0].alias.len = sizeof(kAlias) - 1;
    infos[0].max_length = engine->cfg.max_position ? engine->cfg.max_position : 512;
    infos[0].hidden_size = engine->cfg.hidden ? engine->cfg.hidden : 384;
    infos[0].device = engine->device;
    infos[0].ready =
        engine->ready &&
                (engine->device == TURBORERANK_DEVICE_CPU ||
                 engine->device == TURBORERANK_DEVICE_CUDA ||
                 engine->device == TURBORERANK_DEVICE_OPENVINO_CPU ||
                 engine->device == TURBORERANK_DEVICE_OPENVINO_GPU ||
                 engine->device == TURBORERANK_DEVICE_METAL)
            ? 1
            : 0;
    *out_infos = infos;
    *out_count = 1;
    return TURBORERANK_OK;
}

void turborerank_model_list_free(turborerank_model_info *infos, size_t count) {
    (void)count;
    delete[] infos;
}

turborerank_status turborerank_load_model(
    turborerank_engine *engine,
    const char *alias,
    size_t alias_len
) {
    if (engine == nullptr) {
        turborerank::impl::set_create_error("load_model: null engine");
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    const std::string name = alias_view(alias, alias_len);
    if (name.empty()) {
        engine->last_error = "load_model: empty alias";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (engine->device == TURBORERANK_DEVICE_MOCK) {
        engine->last_error =
            "MOCK device refuses catalog cross-encoder '" + name +
            "'; mock does not produce relevance scores";
        return TURBORERANK_ERR_NOT_IMPLEMENTED;
    }
    const bool ov_dev = engine->device == TURBORERANK_DEVICE_OPENVINO_CPU ||
                        engine->device == TURBORERANK_DEVICE_OPENVINO_GPU;
    if (engine->device != TURBORERANK_DEVICE_CPU &&
        engine->device != TURBORERANK_DEVICE_CUDA &&
        engine->device != TURBORERANK_DEVICE_METAL && !ov_dev) {
        engine->last_error = "load_model: engine device cannot run MiniLM CE";
        return TURBORERANK_ERR_UNAVAILABLE;
    }

    const char *cfg_c =
        engine->config_path.empty() ? nullptr : engine->config_path.c_str();
    const char *ws_c =
        engine->workspace_root.empty() ? nullptr : engine->workspace_root.c_str();

    std::string dir;
    std::string ir_xml;
    if (ov_dev) {
        dir = turborerank::impl::resolve_ov_ir_dir(name.c_str(), name.size(), cfg_c, ws_c);
        if (dir.empty()) {
            engine->last_error =
                "OpenVINO IR missing for alias '" + name +
                "'; expected openvino_model.xml under models/ov-rerank/" + name +
                " (export ONNX then make convert-rerank-ov). Refusing a mock score.";
            return TURBORERANK_ERR_UNAVAILABLE;
        }
        if (is_file(join_path(dir, "openvino_model.xml"))) {
            ir_xml = join_path(dir, "openvino_model.xml");
        } else {
            ir_xml = join_path(dir, "model.xml");
        }
    } else {
        dir = turborerank::impl::resolve_model_dir(name.c_str(), name.size(), cfg_c, ws_c);
        if (dir.empty()) {
            engine->last_error =
                "weights missing for alias '" + name +
                "'; expected model.safetensors under models/rerank/" + name +
                " (make fetch-rerankers). Refusing a mock score.";
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }

    std::string cfg_path = join_path(dir, "config.json");
    std::string vocab_path = join_path(dir, "vocab.txt");
    if (!is_file(vocab_path) || !is_file(cfg_path)) {
        const std::string rerank = turborerank::impl::resolve_model_dir(
            name.c_str(), name.size(), cfg_c, ws_c
        );
        if (!rerank.empty()) {
            if (!is_file(vocab_path)) {
                vocab_path = join_path(rerank, "vocab.txt");
            }
            if (!is_file(cfg_path)) {
                cfg_path = join_path(rerank, "config.json");
            }
        }
    }
    if (!ov_dev) {
        const std::string weight_path = join_path(dir, "model.safetensors");
        if (!is_file(weight_path)) {
            engine->last_error = "weights missing: " + weight_path;
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }
    if (!is_file(vocab_path)) {
        engine->last_error = "vocab.txt missing: " + vocab_path;
        return TURBORERANK_ERR_UNAVAILABLE;
    }

    engine->cfg = turborerank::impl::BertConfig{};
    std::string err;
    if (!turborerank::impl::read_bert_config_json(cfg_path.c_str(), &engine->cfg, &err)) {
        engine->last_error = err;
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (!turborerank::impl::load_vocab_txt(vocab_path.c_str(), &engine->vocab, &err)) {
        engine->last_error = err;
        return TURBORERANK_ERR_UNAVAILABLE;
    }
    if (!ov_dev) {
        const std::string weight_path = join_path(dir, "model.safetensors");
        if (!turborerank::impl::load_safetensors(
                weight_path.c_str(),
                engine->cfg,
                &engine->weights,
                &engine->mapped,
                &engine->owned_weights,
                &err
            )) {
            engine->last_error = err;
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }
    if (!turborerank::impl::alloc_scratch(&engine->scratch, engine->cfg, &err)) {
        engine->last_error = err;
        return TURBORERANK_ERR_OUT_OF_MEMORY;
    }
    if (engine->device == TURBORERANK_DEVICE_CUDA) {
        if (!turborerank::impl::cuda_resources_init(
                &engine->cuda, engine->cfg, engine->weights, &err
            )) {
            engine->last_error =
                err.empty() ? "CUDA MiniLM CE init failed; refusing CPU fallback"
                            : err + "; refusing CPU fallback";
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }
    if (ov_dev) {
        if (!turborerank::impl::ov_resources_init(
                &engine->ov, engine->device, ir_xml.c_str(), engine->cfg, &err
            )) {
            engine->last_error =
                err.empty()
                    ? "OpenVINO MiniLM CE init failed; refusing CPU fallback"
                    : err;
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }
    if (engine->device == TURBORERANK_DEVICE_METAL) {
        if (!turborerank::impl::metal_resources_init(
                &engine->metal, engine->cfg, engine->weights, &err
            )) {
            engine->last_error =
                err.empty() ? "Metal MiniLM CE init failed; refusing CPU fallback"
                            : err + "; refusing CPU fallback";
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }
    free_work_buffer(engine);
    turborerank_status st = turborerank_buffer_alloc(
        engine->device,
        engine->cfg.max_batch,
        engine->cfg.max_position,
        &engine->work
    );
    if (st != TURBORERANK_OK) {
        engine->last_error = "failed to reserve engine work buffer";
        return st;
    }

    engine->model_dir = dir;
    engine->loaded_alias = name;
    engine->ready = true;
    engine->last_error.clear();
    return TURBORERANK_OK;
}

turborerank_status turborerank_buffer_alloc(
    turborerank_device device,
    uint32_t batch,
    uint32_t seq,
    turborerank_buffer **out
) {
    if (out == nullptr) {
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    *out = nullptr;
    if (batch == 0 || seq < turborerank::kSpecials) {
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    turborerank_device resolved = device;
    if (device == TURBORERANK_DEVICE_AUTO) {
        resolved = turborerank::impl::resolve_create_device(device);
    }
    if (resolved == TURBORERANK_DEVICE_CUDA) {
        std::string why;
        if (!turborerank::impl::cuda_device_present(&why)) {
            turborerank::impl::set_create_error(
                why.empty() ? "buffer_alloc: CUDA missing; refusing CPU"
                            : why
            );
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    } else if (resolved == TURBORERANK_DEVICE_OPENVINO_GPU) {
        std::string why;
        if (!turborerank::impl::ov_gpu_present(&why)) {
            turborerank::impl::set_create_error(
                why.empty() ? "buffer_alloc: OpenVINO GPU missing; refusing CPU"
                            : why
            );
            return TURBORERANK_ERR_UNAVAILABLE;
        }
        if (!turborerank::impl::ov_usm_available(&why)) {
            turborerank::impl::set_create_error(
                why.empty()
                    ? "buffer_alloc: Level Zero USM missing; refusing CPU"
                    : why
            );
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    } else if (resolved == TURBORERANK_DEVICE_OPENVINO_CPU) {
        std::string why;
        if (!turborerank::impl::ov_cpu_present(&why)) {
            turborerank::impl::set_create_error(
                why.empty() ? "buffer_alloc: OpenVINO CPU missing"
                            : why
            );
            return turborerank::impl::ov_compiled()
                       ? TURBORERANK_ERR_UNAVAILABLE
                       : TURBORERANK_ERR_NOT_IMPLEMENTED;
        }
    } else if (resolved == TURBORERANK_DEVICE_METAL) {
        std::string why;
        if (!turborerank::impl::metal_device_present(&why)) {
            turborerank::impl::set_create_error(
                why.empty() ? "buffer_alloc: Metal missing; refusing CPU"
                            : why
            );
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    } else if (resolved != TURBORERANK_DEVICE_CPU &&
               resolved != TURBORERANK_DEVICE_MOCK) {
        std::string why;
        turborerank::impl::accelerator_unavailable(resolved, &why);
        turborerank::impl::set_create_error(
            why.empty() ? "buffer_alloc: device not implemented; refusing CPU"
                        : why
        );
        return TURBORERANK_ERR_NOT_IMPLEMENTED;
    }
    if (resolved == TURBORERANK_DEVICE_MOCK) {
        // Allow mock to allocate CPU-shaped buffers for pack tests, but
        // forward will still refuse to score.
        resolved = TURBORERANK_DEVICE_CPU;
    }

    const size_t n = static_cast<size_t>(batch) * static_cast<size_t>(seq);
    const size_t bytes = n * sizeof(int32_t);
    turborerank::Status st = turborerank::Status::Ok;
    auto *buf = new (std::nothrow) turborerank_buffer();
    if (buf == nullptr) {
        return TURBORERANK_ERR_OUT_OF_MEMORY;
    }
    std::memset(buf, 0, sizeof(*buf));
    buf->device = resolved;
    auto alloc_field = [&](int32_t **slot) -> bool {
        if (resolved == TURBORERANK_DEVICE_CUDA) {
            *slot = static_cast<int32_t *>(turborerank::impl::pinned_alloc_bytes(bytes, &st));
        } else if (resolved == TURBORERANK_DEVICE_METAL) {
            *slot = static_cast<int32_t *>(
                turborerank::impl::metal_shared_alloc_bytes(bytes, &st)
            );
        } else if (resolved == TURBORERANK_DEVICE_OPENVINO_GPU) {
            *slot = static_cast<int32_t *>(
                turborerank::impl::usm_alloc_bytes(bytes, true, &st)
            );
        } else if (resolved == TURBORERANK_DEVICE_OPENVINO_CPU) {
            if (turborerank::impl::ov_usm_available(nullptr)) {
                *slot = static_cast<int32_t *>(
                    turborerank::impl::usm_alloc_bytes(bytes, false, &st)
                );
            } else {
                *slot = static_cast<int32_t *>(
                    turborerank::aligned_alloc_bytes(bytes, turborerank::kCpuAlignment, &st)
                );
            }
        } else {
            *slot = static_cast<int32_t *>(
                turborerank::aligned_alloc_bytes(bytes, turborerank::kCpuAlignment, &st)
            );
        }
        return *slot != nullptr;
    };
    if (!alloc_field(&buf->input_ids) || !alloc_field(&buf->attention_mask) ||
        !alloc_field(&buf->token_type_ids) || !alloc_field(&buf->position_ids)) {
        turborerank_buffer_free(buf);
        return TURBORERANK_ERR_OUT_OF_MEMORY;
    }
    buf->batch = batch;
    buf->seq = seq;
    buf->row_stride = seq;
    *out = buf;
    return TURBORERANK_OK;
}

void turborerank_buffer_free(turborerank_buffer *buffer) {
    if (buffer == nullptr) {
        return;
    }
    if (buffer->device == TURBORERANK_DEVICE_CUDA) {
        turborerank::impl::pinned_free_bytes(buffer->input_ids);
        turborerank::impl::pinned_free_bytes(buffer->attention_mask);
        turborerank::impl::pinned_free_bytes(buffer->token_type_ids);
        turborerank::impl::pinned_free_bytes(buffer->position_ids);
    } else if (buffer->device == TURBORERANK_DEVICE_METAL) {
        turborerank::impl::metal_shared_free_bytes(buffer->input_ids);
        turborerank::impl::metal_shared_free_bytes(buffer->attention_mask);
        turborerank::impl::metal_shared_free_bytes(buffer->token_type_ids);
        turborerank::impl::metal_shared_free_bytes(buffer->position_ids);
    } else if (buffer->device == TURBORERANK_DEVICE_OPENVINO_GPU ||
               (buffer->device == TURBORERANK_DEVICE_OPENVINO_CPU &&
                turborerank::impl::ov_usm_available(nullptr))) {
        turborerank::impl::usm_free_bytes(buffer->input_ids);
        turborerank::impl::usm_free_bytes(buffer->attention_mask);
        turborerank::impl::usm_free_bytes(buffer->token_type_ids);
        turborerank::impl::usm_free_bytes(buffer->position_ids);
    } else {
        turborerank::aligned_free_bytes(buffer->input_ids);
        turborerank::aligned_free_bytes(buffer->attention_mask);
        turborerank::aligned_free_bytes(buffer->token_type_ids);
        turborerank::aligned_free_bytes(buffer->position_ids);
    }
    delete buffer;
}

turborerank_status turborerank_pack_ids(
    turborerank_buffer *buffer,
    uint32_t row,
    const int32_t *query_ids,
    size_t n_query,
    const int32_t *doc_ids,
    size_t n_doc,
    turborerank_truncation truncation,
    uint32_t max_length
) {
    std::string err;
    const auto st = turborerank::impl::pack_ids_into(
        buffer, row, query_ids, n_query, doc_ids, n_doc, truncation, max_length, &err
    );
    if (st != turborerank::Status::Ok) {
        turborerank::impl::set_create_error(err);
        return static_cast<turborerank_status>(st);
    }
    return TURBORERANK_OK;
}

turborerank_status turborerank_pack_text(
    turborerank_engine *engine,
    turborerank_buffer *buffer,
    uint32_t row,
    turborerank_str query,
    turborerank_str document,
    turborerank_truncation truncation,
    uint32_t max_length
) {
    if (engine == nullptr) {
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (!engine->vocab.loaded) {
        engine->last_error = "pack_text: vocab not loaded (call load_model)";
        return TURBORERANK_ERR_UNAVAILABLE;
    }
    if ((query.ptr == nullptr && query.len > 0) ||
        (document.ptr == nullptr && document.len > 0)) {
        engine->last_error = "pack_text: null string with len>0";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (query.len == 0 && document.len == 0) {
        engine->last_error = "pack_text: empty query and empty document";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    std::string err;
    const size_t nq = turborerank::impl::tokenize_wordpiece(
        engine->vocab,
        query.ptr,
        query.len,
        engine->scratch.tok_q,
        engine->scratch.tok_cap,
        &err
    );
    if (!err.empty() && nq == 0 && query.len > 0) {
        engine->last_error = err;
        return TURBORERANK_ERR_INTERNAL;
    }
    err.clear();
    const size_t nd = turborerank::impl::tokenize_wordpiece(
        engine->vocab,
        document.ptr,
        document.len,
        engine->scratch.tok_d,
        engine->scratch.tok_cap,
        &err
    );
    if (!err.empty() && nd == 0 && document.len > 0) {
        engine->last_error = err;
        return TURBORERANK_ERR_INTERNAL;
    }
    const auto st = turborerank::impl::pack_ids_into(
        buffer,
        row,
        engine->scratch.tok_q,
        nq,
        engine->scratch.tok_d,
        nd,
        truncation,
        max_length,
        &err
    );
    if (st != turborerank::Status::Ok) {
        engine->last_error = err;
        return static_cast<turborerank_status>(st);
    }
    return TURBORERANK_OK;
}

turborerank_status turborerank_forward(
    turborerank_engine *engine,
    const turborerank_buffer *buffer,
    uint32_t n_rows,
    turborerank_activation activation,
    float *scores_out
) {
    if (engine == nullptr || buffer == nullptr || scores_out == nullptr) {
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (engine->device == TURBORERANK_DEVICE_MOCK) {
        engine->last_error =
            "MOCK forward refuses to emit relevance scores (not a stand-in "
            "for MiniLM CE)";
        return TURBORERANK_ERR_NOT_IMPLEMENTED;
    }
    if (!engine->ready) {
        engine->last_error = "forward: model not loaded";
        return TURBORERANK_ERR_UNAVAILABLE;
    }
    if (n_rows == 0 || n_rows > buffer->batch) {
        engine->last_error = "forward: n_rows out of range";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (buffer->input_ids == nullptr || buffer->attention_mask == nullptr) {
        engine->last_error = "forward: token pointers are null";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }

    if (engine->device == TURBORERANK_DEVICE_OPENVINO_CPU ||
        engine->device == TURBORERANK_DEVICE_OPENVINO_GPU) {
        if (buffer->row_stride != buffer->seq) {
            engine->last_error = "forward: OpenVINO requires row_stride == seq";
            return TURBORERANK_ERR_INVALID_ARGUMENT;
        }
        std::string err;
        if (!turborerank::impl::bert_forward_ov(
                &engine->ov,
                buffer->input_ids,
                buffer->attention_mask,
                buffer->token_type_ids,
                n_rows,
                buffer->seq,
                scores_out,
                &err
            )) {
            engine->last_error =
                err.empty() ? "OpenVINO forward failed; refusing CPU fallback"
                            : err;
            return TURBORERANK_ERR_INTERNAL;
        }
        if (activation != TURBORERANK_ACT_IDENTITY) {
            for (uint32_t r = 0; r < n_rows; ++r) {
                scores_out[r] = turborerank::sigmoid(scores_out[r]);
            }
        }
        return TURBORERANK_OK;
    }

    for (uint32_t r = 0; r < n_rows; ++r) {
        const size_t off = static_cast<size_t>(r) * buffer->row_stride;
        const int32_t *mask = buffer->attention_mask + off;
        uint32_t used = buffer->seq;
        while (used > turborerank::kSpecials && mask[used - 1] == 0) {
            --used;
        }
        float logit = 0.0f;
        if (engine->device == TURBORERANK_DEVICE_METAL) {
            std::string err;
            if (!turborerank::impl::bert_forward_row_metal(
                    &engine->metal,
                    engine->cfg,
                    buffer->input_ids + off,
                    mask,
                    buffer->token_type_ids + off,
                    buffer->position_ids + off,
                    used,
                    &logit,
                    &err
                )) {
                engine->last_error =
                    err.empty() ? "Metal forward failed; refusing CPU fallback"
                                : err + "; refusing CPU fallback";
                return TURBORERANK_ERR_INTERNAL;
            }
        } else if (engine->device == TURBORERANK_DEVICE_CUDA) {
            std::string err;
            if (!turborerank::impl::bert_forward_row_cuda(
                    &engine->cuda,
                    engine->cfg,
                    buffer->input_ids + off,
                    mask,
                    buffer->token_type_ids + off,
                    buffer->position_ids + off,
                    used,
                    &logit,
                    &err
                )) {
                engine->last_error =
                    err.empty() ? "CUDA forward failed; refusing CPU fallback"
                                : err + "; refusing CPU fallback";
                return TURBORERANK_ERR_INTERNAL;
            }
        } else {
            logit = turborerank::impl::bert_forward_row(
                engine->cfg,
                engine->weights,
                &engine->scratch,
                buffer->input_ids + off,
                mask,
                buffer->token_type_ids + off,
                buffer->position_ids + off,
                used
            );
        }
        scores_out[r] = activation == TURBORERANK_ACT_IDENTITY
                            ? logit
                            : turborerank::sigmoid(logit);
    }
    return TURBORERANK_OK;
}

turborerank_status turborerank_score(
    turborerank_engine *engine,
    const char *alias,
    size_t alias_len,
    turborerank_str query,
    const turborerank_str *documents,
    size_t n_documents,
    const turborerank_score_options *opts,
    float *scores_out
) {
    if (engine == nullptr || scores_out == nullptr) {
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (n_documents == 0 || documents == nullptr) {
        engine->last_error = "score: documents must not be empty";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }
    if (!engine->ready) {
        if (alias != nullptr) {
            const turborerank_status st = turborerank_load_model(engine, alias, alias_len);
            if (st != TURBORERANK_OK) {
                return st;
            }
        } else {
            engine->last_error = "score: model not loaded";
            return TURBORERANK_ERR_UNAVAILABLE;
        }
    }
    if (engine->work == nullptr) {
        engine->last_error = "score: work buffer missing";
        return TURBORERANK_ERR_INTERNAL;
    }

    const turborerank_truncation trunc =
        opts ? opts->truncation : TURBORERANK_TRUNC_LONGEST_FIRST;
    const turborerank_activation act =
        opts ? opts->activation : TURBORERANK_ACT_SIGMOID;
    const uint32_t max_len = opts && opts->max_length ? opts->max_length : engine->cfg.max_position;

    if (n_documents > engine->work->batch) {
        engine->last_error = "score: batch exceeds engine max_batch";
        return TURBORERANK_ERR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < n_documents; ++i) {
        const turborerank_status st = turborerank_pack_text(
            engine, engine->work, static_cast<uint32_t>(i), query, documents[i], trunc, max_len
        );
        if (st != TURBORERANK_OK) {
            return st;
        }
    }
    return turborerank_forward(
        engine, engine->work, static_cast<uint32_t>(n_documents), act, scores_out
    );
}

} // extern "C"
