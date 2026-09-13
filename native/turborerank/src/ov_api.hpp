// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "internal.hpp"

#include <string>

namespace turborerank {
namespace impl {

/** True when this binary was compiled with OpenVINO (`TURBORERANK_OPENVINO`). */
bool ov_compiled();

/** Runtime: compiled + `ov::Core` lists CPU. */
bool ov_cpu_present(std::string *why);

/**
 * Runtime: compiled + `ov::Core` lists GPU*. Requesting GPU without GPU
 * always fails loud — never a silent CPU engine.
 */
bool ov_gpu_present(std::string *why);

/** Full name of GPU.0 (or empty). */
bool ov_gpu_name(std::string *name);

/** Level Zero USM is initialized and can allocate host buffers. */
bool ov_usm_available(std::string *why);

/** ZE SHARED USM (GPU). Missing GPU → false, never a HOST stand-in. */
bool ov_usm_shared_available(std::string *why);

/**
 * SOLIDIFY (7): try wrapping a turbo_buffer ZE SHARED pointer as
 * `USM_USER_BUFFER` on the compiled GPU RemoteContext.
 * Returns true only if create_tensor accepts the L0 pointer.
 * On this Machine B stack the OCL engine reports size 0.
 */
bool ov_probe_remote_usm_wrap(std::string *why);

void *usm_alloc_bytes(size_t bytes, bool shared_ok, Status *status);
void usm_free_bytes(void *ptr);

bool ov_resources_init(
    OvResources *r,
    turborerank_device device,
    const char *ir_xml,
    const BertConfig &cfg,
    std::string *err
);

void ov_resources_free(OvResources *r);

/**
 * Batch MiniLM CE via CompiledModel. Token pointers are caller-owned
 * Level Zero USM (or OV-CPU aligned). Wrapped as
 * `ov::Tensor(element::i32, {n_rows, seq}, usm_pointer)` — no
 * std::vector on this path. Remote OCL `USM_USER_BUFFER` wrap of
 * those ZE pointers is probed at init and is unavailable on
 * Machine B (OCL reports size 0).
 */
bool bert_forward_ov(
    OvResources *r,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    uint32_t n_rows,
    uint32_t seq,
    float *logits_out,
    std::string *err
);

/** Bench: bytes of ov::Tensor host wraps on the last / cumulative forwards. */
void ov_xfer_reset();
uint64_t ov_xfer_wrap_input_bytes();
uint64_t ov_xfer_result_bytes();
uint32_t ov_xfer_last_n_rows();
uint32_t ov_xfer_last_seq();

std::string resolve_ov_ir_dir(
    const char *alias,
    size_t alias_len,
    const char *config_path,
    const char *workspace_root
);

} // namespace impl
} // namespace turborerank
