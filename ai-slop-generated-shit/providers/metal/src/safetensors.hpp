// SPDX-License-Identifier: Apache-2.0
//
// A read-only safetensors view: the file is mapped, the JSON header parsed
// once, and tensors are handed out as (pointer, shape) into the mapping.
// Only F32 tensors are served; anything else is a load failure naming the
// tensor and its dtype, never a silent conversion. The bundle's hash was
// verified by the core before the provider sees the file.
#pragma once

#include "turbo_provider_common.hpp"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace turbo_metal {

struct Tensor {
    const float *data = nullptr;
    std::vector<uint64_t> shape;
    uint64_t rows() const { return shape.empty() ? 0 : shape[0]; }
    uint64_t cols() const { return shape.size() < 2 ? 1 : shape[1]; }
    uint64_t elements() const {
        uint64_t n = 1;
        for (uint64_t d : shape) {
            turbo_pc::require(d == 0 || n <= UINT64_MAX / d, TURBO_E_BUNDLE_INVALID, "tensor shape product overflows");
            n *= d;
        }
        return n;
    }
};

class SafeTensors {
  public:
    SafeTensors() = default;
    SafeTensors(const SafeTensors &) = delete;
    SafeTensors &operator=(const SafeTensors &) = delete;

    ~SafeTensors() {
        if (map_ != nullptr && map_ != MAP_FAILED) {
            munmap(map_, size_);
        }
    }

    void open(const std::string &path) {
        using turbo_pc::fail;
        using turbo_pc::require;
        const int fd = ::open(path.c_str(), O_RDONLY);
        require(fd >= 0, TURBO_E_BUNDLE_NOT_FOUND, "cannot open safetensors artifact " + path);
        struct stat st {};
        if (fstat(fd, &st) != 0 || st.st_size < 8) {
            ::close(fd);
            fail(TURBO_E_BUNDLE_INVALID, path + ": not a safetensors file (too short)");
        }
        size_ = static_cast<size_t>(st.st_size);
        map_ = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        require(map_ != MAP_FAILED, TURBO_E_RUNTIME, path + ": mmap failed");
        const auto *bytes = static_cast<const uint8_t *>(map_);
        uint64_t header_len = 0;
        std::memcpy(&header_len, bytes, 8);
        require(header_len > 0 && header_len < size_ - 8, TURBO_E_BUNDLE_INVALID, path + ": bad safetensors header length");
        nlohmann::json header;
        try {
            header = nlohmann::json::parse(bytes + 8, bytes + 8 + header_len);
        } catch (const std::exception &e) {
            fail(TURBO_E_BUNDLE_INVALID, path + ": safetensors header is not JSON: " + e.what());
        }
        const uint8_t *data = bytes + 8 + header_len;
        const uint64_t data_len = size_ - 8 - header_len;
        require(header.is_object(), TURBO_E_BUNDLE_INVALID, path + ": safetensors header is not a JSON object");
        for (auto it = header.begin(); it != header.end(); ++it) {
            if (it.key() == "__metadata__") {
                continue;
            }
            // A malformed entry is a bad bundle, named by tensor, never a
            // runtime fault from the JSON library.
            const nlohmann::json &t = it.value();
            Entry e;
            try {
                require(t.is_object() && t.contains("dtype") && t.contains("shape") && t.contains("data_offsets"), TURBO_E_BUNDLE_INVALID,
                        path + ": tensor `" + it.key() + "` lacks dtype, shape, or data_offsets");
                e.dtype = t.at("dtype").get<std::string>();
                for (const auto &d : t.at("shape")) {
                    require(d.is_number_unsigned(), TURBO_E_BUNDLE_INVALID,
                            path + ": tensor `" + it.key() + "` has a shape entry that is not a non-negative integer");
                    e.shape.push_back(d.get<uint64_t>());
                }
                const auto &off = t.at("data_offsets");
                require(off.is_array() && off.size() == 2 && off.at(0).is_number_unsigned() && off.at(1).is_number_unsigned(),
                        TURBO_E_BUNDLE_INVALID, path + ": tensor `" + it.key() + "` data_offsets must be two non-negative integers");
                e.begin = off.at(0).get<uint64_t>();
                e.end = off.at(1).get<uint64_t>();
            } catch (const turbo_pc::Failure &) {
                throw;
            } catch (const std::exception &ex) {
                fail(TURBO_E_BUNDLE_INVALID, path + ": tensor `" + it.key() + "`: " + ex.what());
            }
            require(e.begin <= e.end && e.end <= data_len, TURBO_E_BUNDLE_INVALID,
                    path + ": tensor `" + it.key() + "` has offsets outside the file");
            e.ptr = data + e.begin;
            entries_[it.key()] = e;
        }
        path_ = path;
    }

    bool has(const std::string &name) const { return entries_.count(name) != 0; }

    /// The tensor by its first present name; F32 only, `expect_rank` when non-zero.
    Tensor get(std::initializer_list<const char *> names, size_t expect_rank = 0) const {
        using turbo_pc::fail;
        using turbo_pc::require;
        for (const char *n : names) {
            auto it = entries_.find(n);
            if (it == entries_.end()) {
                continue;
            }
            const Entry &e = it->second;
            require(e.dtype == "F32", TURBO_E_UNSUPPORTED_DTYPE,
                    path_ + ": tensor `" + std::string(n) + "` is " + e.dtype + "; the metal provider takes F32 checkpoints");
            require(reinterpret_cast<uintptr_t>(e.ptr) % 4 == 0, TURBO_E_BUNDLE_INVALID,
                    path_ + ": tensor `" + std::string(n) + "` is not 4-byte aligned");
            Tensor t;
            t.data = reinterpret_cast<const float *>(e.ptr);
            t.shape = e.shape;
            require(t.elements() * 4 == e.end - e.begin, TURBO_E_BUNDLE_INVALID,
                    path_ + ": tensor `" + std::string(n) + "` byte length does not match its shape");
            require(expect_rank == 0 || t.shape.size() == expect_rank, TURBO_E_BUNDLE_INVALID,
                    path_ + ": tensor `" + std::string(n) + "` has rank " + std::to_string(t.shape.size()) + ", expected " +
                        std::to_string(expect_rank));
            return t;
        }
        std::string list;
        for (const char *n : names) {
            list += (list.empty() ? "" : " or ") + std::string(n);
        }
        fail(TURBO_E_BUNDLE_INVALID, path_ + ": missing tensor " + list);
    }

    /// Number of `encoder.layer.N.` groups present.
    uint32_t encoder_layers() const {
        uint32_t n = 0;
        while (has("encoder.layer." + std::to_string(n) + ".attention.self.query.weight") ||
               has("bert.encoder.layer." + std::to_string(n) + ".attention.self.query.weight")) {
            ++n;
        }
        return n;
    }

  private:
    struct Entry {
        std::string dtype;
        std::vector<uint64_t> shape;
        uint64_t begin = 0, end = 0;
        const uint8_t *ptr = nullptr;
    };
    void *map_ = nullptr;
    size_t size_ = 0;
    std::string path_;
    std::map<std::string, Entry> entries_;
};

} // namespace turbo_metal
