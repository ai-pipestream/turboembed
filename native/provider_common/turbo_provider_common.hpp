// SPDX-License-Identifier: Apache-2.0
//
// Shared helpers for the C++ providers (OpenVINO, Hailo): error containment
// at the plugin boundary, string copies into ABI arrays, descriptor size
// checks, key/value options, and the bundle manifest reader. The core has
// already verified bundle hashes before a provider sees a bundle directory;
// a provider re-reads only what it needs.
//
// Everything here is header-only and depends on nlohmann/json and the ABI
// headers only.
#pragma once

#include "turbo/turbo_types.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace turbo_pc {

/// An error with a Turbo status code and optional field index.
struct Failure : std::runtime_error {
    int32_t code;
    uint32_t field;
    Failure(int32_t c, const std::string &m, uint32_t f = 0) : std::runtime_error(m), code(c), field(f) {}
};

[[noreturn]] inline void fail(int32_t code, const std::string &message, uint32_t field = 0) {
    throw Failure(code, message, field);
}

inline void require(bool ok, int32_t code, const std::string &message, uint32_t field = 0) {
    if (!ok) {
        fail(code, message, field);
    }
}

/// Copy a UTF-8 string into a fixed char array, NUL-terminated, without
/// cutting a multi-byte sequence.
template <size_t N>
inline void put_str(char (&dst)[N], const std::string &s) {
    size_t n = s.size() < N - 1 ? s.size() : N - 1;
    while (n > 0 && n < s.size() && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) {
        --n;
    }
    std::memcpy(dst, s.data(), n);
    dst[n] = '\0';
}

/// Write an error into the caller-owned record (if any) and return the code.
inline int32_t report(turbo_error *err, int32_t code, uint32_t field, const std::string &message) {
    if (err != nullptr) {
        const size_t size = err->struct_size;
        if (size >= 8) {
            err->code = code;
        }
        if (size >= 12) {
            err->field = field;
        }
        if (size > 12) {
            size_t cap = size - 12;
            if (cap > TURBO_ERROR_MESSAGE_LEN) {
                cap = TURBO_ERROR_MESSAGE_LEN;
            }
            size_t n = message.size() < cap - 1 ? message.size() : cap - 1;
            while (n > 0 && n < message.size() && (static_cast<unsigned char>(message[n]) & 0xC0) == 0x80) {
                --n;
            }
            std::memcpy(err->message, message.data(), n);
            err->message[n] = '\0';
        }
    }
    return code;
}

/// Reset an error record to success.
inline void clear(turbo_error *err) {
    if (err != nullptr) {
        if (err->struct_size >= 8) {
            err->code = TURBO_OK;
        }
        if (err->struct_size >= 12) {
            err->field = 0;
        }
        if (err->struct_size > 12) {
            err->message[0] = '\0';
        }
    }
}

/// Run `fn` and convert every exception into a status code. Nothing unwinds
/// past this function.
template <typename F>
inline int32_t boundary(turbo_error *err, F &&fn) noexcept {
    clear(err);
    try {
        fn();
        return TURBO_OK;
    } catch (const Failure &e) {
        return report(err, e.code, e.field, e.what());
    } catch (const std::bad_alloc &) {
        return report(err, TURBO_E_OUT_OF_MEMORY, 0, "native allocation failed");
    } catch (const std::exception &e) {
        // Runtime exceptions (OpenVINO, OpenCL, and the standard library)
        // derive from std::exception.
        return report(err, TURBO_E_RUNTIME, 0, e.what());
    } catch (...) {
        return report(err, TURBO_E_PANIC, 0, "unknown native exception at the provider boundary");
    }
}

/// A view as std::string.
inline std::string text_of(turbo_text t) {
    if (t.len == 0) {
        return {};
    }
    require(t.ptr != nullptr, TURBO_E_INVALID_ARGUMENT, "text view has NULL ptr with non-zero len");
    return std::string(t.ptr, static_cast<size_t>(t.len));
}

/// Accept a descriptor of the exact current size or an older, smaller one.
template <typename T>
inline void check_size(const char *what, uint32_t got) {
    const size_t expected = sizeof(T);
    if (got == expected) {
        return;
    }
    if (got < 4 || got > expected) {
        fail(TURBO_E_INVALID_STRUCT_SIZE,
             std::string(what) + ".struct_size is " + std::to_string(got) + "; this provider understands " +
                 std::to_string(expected));
    }
}

/// Key/value options; unknown keys are rejected by the receiver.
inline std::map<std::string, std::string> options_of(const turbo_kv *kv, uint32_t n, const char *what) {
    std::map<std::string, std::string> out;
    if (n == 0) {
        return out;
    }
    require(kv != nullptr, TURBO_E_INVALID_ARGUMENT, std::string(what) + " options pointer is NULL");
    for (uint32_t i = 0; i < n; ++i) {
        out[text_of(kv[i].key)] = text_of(kv[i].value);
    }
    return out;
}

inline void reject_unknown(const std::map<std::string, std::string> &opts, const std::vector<std::string> &known,
                           const char *what) {
    for (const auto &[k, v] : opts) {
        bool ok = false;
        for (const auto &kn : known) {
            if (kn == k) {
                ok = true;
            }
        }
        if (!ok) {
            std::string list;
            for (const auto &kn : known) {
                list += (list.empty() ? "" : ", ") + kn;
            }
            fail(TURBO_E_INVALID_ARGUMENT,
                 std::string(what) + " option `" + k + "` is not recognized; known: " + (list.empty() ? "(none)" : list));
        }
    }
}

// ---------------------------------------------------------------------------
// Bundle manifest (bundle.json version 2), read-only subset the provider needs.
// ---------------------------------------------------------------------------

struct Bundle {
    std::string dir;
    std::string model_id;
    std::string revision;
    std::string task;      // embed, rerank, classify, token_classify, generate, run
    std::string kind;      // embedding, reranker, classifier, token_classifier, generative, generic
    std::string modality;
    std::string pooling;   // mean, cls, last
    std::string normalize; // l2, none
    std::string activation; // softmax, sigmoid, none
    std::string aggregation; // none, simple, first, max
    uint32_t max_seq = 0;
    uint32_t dim = 0;
    uint32_t vocab_size = 0;
    uint32_t max_batch = 0;
    bool fixed_shape = false;
    std::vector<std::string> labels;
    std::string prefix_query;
    std::string prefix_document;
    std::string tokenizer_kind;
    std::string tokenizer_path;   // absolute, or empty
    std::string tokenizer_sha256;
    std::map<std::string, std::string> artifacts; // format -> absolute path

    static Bundle read(const std::string &dir) {
        using nlohmann::json;
        Bundle b;
        b.dir = dir;
        std::ifstream in(dir + "/bundle.json");
        require(in.good(), TURBO_E_BUNDLE_NOT_FOUND, "bundle.json is missing in " + dir);
        json j;
        try {
            in >> j;
        } catch (const std::exception &e) {
            fail(TURBO_E_BUNDLE_INVALID, std::string("bundle.json: ") + e.what());
        }
        require(j.value("bundle_version", 0) == 2, TURBO_E_BUNDLE_INVALID, "bundle_version must be 2");
        b.model_id = j.value("model_id", "");
        b.revision = j.value("revision", "");
        b.task = j.value("task", "");
        b.kind = j.value("kind", "");
        b.modality = j.value("modality", "text");
        const json c = j.value("contract", json::object());
        b.pooling = c.value("pooling", "");
        b.normalize = c.value("normalize", "");
        b.activation = c.value("activation", "");
        b.aggregation = c.value("aggregation", "");
        b.max_seq = c.value("max_seq", 0u);
        b.dim = c.value("dim", 0u);
        b.vocab_size = c.value("vocab_size", 0u);
        if (c.contains("labels")) {
            for (const auto &l : c.at("labels")) {
                b.labels.push_back(l.get<std::string>());
            }
        }
        const json prompts = c.value("prompts", json::object());
        b.prefix_query = prompts.value("query", "");
        b.prefix_document = prompts.value("document", "");
        const json limits = j.value("limits", json::object());
        b.max_batch = limits.value("max_batch", 0u);
        b.fixed_shape = limits.value("fixed_shape", false);
        if (j.contains("tokenizer") && !j.at("tokenizer").is_null()) {
            const json t = j.at("tokenizer");
            b.tokenizer_kind = t.value("kind", "");
            const json files = t.value("files", json::object());
            if (files.contains("tokenizer.json")) {
                b.tokenizer_path = dir + "/" + files.at("tokenizer.json").value("path", "");
                b.tokenizer_sha256 = files.at("tokenizer.json").value("sha256", "");
            }
        }
        const json arts = j.value("artifacts", json::object());
        for (auto it = arts.begin(); it != arts.end(); ++it) {
            b.artifacts[it.key()] = dir + "/" + it.value().value("path", "");
        }
        return b;
    }

    std::optional<std::string> artifact(const char *format) const {
        auto it = artifacts.find(format);
        if (it == artifacts.end()) {
            return std::nullopt;
        }
        return it->second;
    }
};

} // namespace turbo_pc
