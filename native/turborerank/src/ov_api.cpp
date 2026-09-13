// SPDX-License-Identifier: Apache-2.0
//
// OpenVINO presence, Level Zero USM token workspace, CompiledModel CE.

#include "ov_api.hpp"
#include "turbo_buffer.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <vector>

#ifdef TURBORERANK_OPENVINO
#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/remote_context.hpp>
#endif


namespace turborerank {
namespace impl {

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

#ifdef TURBORERANK_OPENVINO
struct OvHold {
    ov::Core core;
    ov::CompiledModel compiled;
    ov::InferRequest request;
    std::string in_ids;
    std::string in_mask;
    std::string in_types;
    std::string out_logits;
    bool has_types = false;
};

bool listed_has_gpu(const std::vector<std::string> &devs) {
    for (const auto &d : devs) {
        if (d == "GPU" || d.rfind("GPU.", 0) == 0) {
            return true;
        }
    }
    return false;
}

bool listed_has_cpu(const std::vector<std::string> &devs) {
    for (const auto &d : devs) {
        if (d == "CPU") {
            return true;
        }
    }
    return false;
}

ov::AnyMap ov_accuracy_props() {
    ov::AnyMap props;
    props[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    props[ov::hint::inference_precision.name()] = ov::element::f32;
    props[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
    props[ov::hint::dynamic_quantization_group_size.name()] = static_cast<uint64_t>(0);
    return props;
}

bool ov_list_devices(std::vector<std::string> *devs, std::string *err) {
    try {
        ov::Core core;
        auto listed = core.get_available_devices();
        if (devs) {
            *devs = listed;
        }
        return true;
    } catch (const std::exception &e) {
        if (err) {
            *err = std::string("OpenVINO Core failed: ") + e.what();
        }
        return false;
    }
}
#endif

} // namespace

bool ov_compiled() {
#ifdef TURBORERANK_OPENVINO
    return true;
#else
    return false;
#endif
}

bool ov_cpu_present(std::string *why) {
#ifdef TURBORERANK_OPENVINO
    std::string err;
    std::vector<std::string> devs;
    if (!ov_list_devices(&devs, &err)) {
        if (why) {
            *why = err + " Refusing a silent stand-in.";
        }
        return false;
    }
    if (!listed_has_cpu(devs)) {
        if (why) {
            *why = "TURBORERANK_DEVICE_OPENVINO_CPU requested but OpenVINO "
                   "lists no CPU plugin. Refusing a silent stand-in.";
        }
        return false;
    }
    return true;
#else
    if (why) {
        *why = "TURBORERANK_DEVICE_OPENVINO_CPU is not compiled into this "
               "binary (OpenVINO CompiledModel). Use TURBORERANK_DEVICE_CPU "
               "for the first-party MiniLM CE kernel. Refusing a silent stand-in.";
    }
    return false;
#endif
}

bool ov_gpu_present(std::string *why) {
#ifdef TURBORERANK_OPENVINO
    std::string err;
    std::vector<std::string> devs;
    if (!ov_list_devices(&devs, &err)) {
        if (why) {
            *why = err + " Refusing CPU fallback.";
        }
        return false;
    }
    if (!listed_has_gpu(devs)) {
        if (why) {
            *why = "TURBORERANK_DEVICE_OPENVINO_GPU requested but OpenVINO "
                   "lists no GPU plugin (need libopenvino_intel_gpu_plugin + "
                   "Level Zero). Refusing CPU fallback.";
        }
        return false;
    }
    return true;
#else
    if (why) {
        *why = "TURBORERANK_DEVICE_OPENVINO_GPU requested but this binary was "
               "built without OpenVINO (Level Zero USM + ov::Tensor). "
               "Refusing CPU fallback.";
    }
    return false;
#endif
}

bool ov_gpu_name(std::string *name) {
#ifdef TURBORERANK_OPENVINO
    try {
        ov::Core core;
        const std::string n =
            core.get_property("GPU", ov::device::full_name);
        if (name) {
            *name = n;
        }
        return !n.empty();
    } catch (...) {
        if (name) {
            *name = {};
        }
        return false;
    }
#else
    if (name) {
        *name = {};
    }
    return false;
#endif
}

bool ov_usm_available(std::string *why) {
    const turbo_buffer_status st = turbo_buffer_backend_probe(
        TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_HOST
    );
    if (st == TURBO_BUFFER_OK) {
        return true;
    }
    if (why) {
        const char *msg = turbo_buffer_last_error(nullptr);
        if (msg != nullptr && msg[0] != '\0') {
            *why = msg;
        } else {
            *why = "Level Zero USM is not available. Refusing CPU fallback "
                   "for OpenVINO GPU.";
        }
    }
    return false;
}

bool ov_probe_remote_usm_wrap(std::string *why) {
#ifdef TURBORERANK_OPENVINO
    std::string present_why;
    if (!ov_gpu_present(&present_why)) {
        if (why) {
            *why = present_why;
        }
        return false;
    }
    if (!ov_usm_shared_available(&present_why)) {
        if (why) {
            *why = present_why;
        }
        return false;
    }
    void *usm = nullptr;
    const size_t bytes = 64 * sizeof(int32_t);
    const turbo_buffer_status st = turbo_buffer_raw_alloc(
        TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED, bytes, &usm
    );
    if (st != TURBO_BUFFER_OK || usm == nullptr) {
        if (why) {
            const char *msg = turbo_buffer_last_error(nullptr);
            *why = msg && msg[0] ? msg : "ZE SHARED alloc failed for remote-USM probe";
        }
        return false;
    }
    std::memset(usm, 0, bytes);
    bool ok = false;
    try {
        ov::Core core;
        auto param = std::make_shared<ov::op::v0::Parameter>(
            ov::element::i32, ov::Shape{1, 8}
        );
        param->set_friendly_name("input_ids");
        param->output(0).set_names({"input_ids"});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        auto model = std::make_shared<ov::Model>(
            ov::ResultVector{result}, ov::ParameterVector{param}, "remote_usm_probe"
        );
        auto compiled = core.compile_model(model, "GPU", ov_accuracy_props());
        ov::RemoteContext ctx = compiled.get_context();
        ov::AnyMap tparams = {
            {ov::intel_gpu::shared_mem_type.name(),
             ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
            {ov::intel_gpu::mem_handle.name(),
             static_cast<ov::intel_gpu::gpu_handle_param>(usm)},
        };
        (void)ctx.create_tensor(ov::element::i32, ov::Shape{1, 8}, tparams);
        ok = true;
        if (why) {
            *why = "USM_USER_BUFFER wrap accepted for turbo_buffer ZE SHARED";
        }
    } catch (const std::exception &e) {
        if (why) {
            *why = e.what();
        }
        ok = false;
    }
    turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED, usm);
    return ok;
#else
    if (why) {
        *why = "binary compiled without OpenVINO";
    }
    return false;
#endif
}

bool ov_usm_shared_available(std::string *why) {
    const turbo_buffer_status st = turbo_buffer_backend_probe(
        TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED
    );
    if (st == TURBO_BUFFER_OK) {
        return true;
    }
    if (why) {
        const char *msg = turbo_buffer_last_error(nullptr);
        if (msg != nullptr && msg[0] != '\0') {
            *why = msg;
        } else {
            *why = "ZE SHARED USM is not available. Refusing HOST/CPU remap "
                   "for OpenVINO GPU.";
        }
    }
    return false;
}

void *usm_alloc_bytes(size_t bytes, bool shared_ok, Status *status) {
    const turbo_buffer_placement place =
        shared_ok ? TURBO_BUFFER_PLACE_SHARED : TURBO_BUFFER_PLACE_HOST;
    void *ptr = nullptr;
    const turbo_buffer_status st =
        turbo_buffer_raw_alloc(TURBO_BUFFER_DEVICE_ZE, place, bytes, &ptr);
    if (st != TURBO_BUFFER_OK || ptr == nullptr) {
        if (status) {
            *status = st == TURBO_BUFFER_ERR_INVALID_ARGUMENT
                          ? Status::InvalidArgument
                          : Status::Unavailable;
        }
        return nullptr;
    }
    if (status) {
        *status = Status::Ok;
    }
    return ptr;
}

void usm_free_bytes(void *ptr) {
    turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_HOST, ptr);
}

bool ov_resources_init(
    OvResources *r,
    turborerank_device device,
    const char *ir_xml,
    const BertConfig &cfg,
    std::string *err
) {
    if (r == nullptr || ir_xml == nullptr) {
        if (err) {
            *err = "ov_resources_init: null argument";
        }
        return false;
    }
    ov_resources_free(r);
#ifdef TURBORERANK_OPENVINO
    const bool want_gpu = device == TURBORERANK_DEVICE_OPENVINO_GPU;
    std::string why;
    if (want_gpu) {
        if (!ov_gpu_present(&why)) {
            if (err) {
                *err = why;
            }
            return false;
        }
    } else if (device == TURBORERANK_DEVICE_OPENVINO_CPU) {
        if (!ov_cpu_present(&why)) {
            if (err) {
                *err = why;
            }
            return false;
        }
    } else {
        if (err) {
            *err = "ov_resources_init: device is not OpenVINO";
        }
        return false;
    }

    try {
        auto *hold = new OvHold();
        std::shared_ptr<ov::Model> model = hold->core.read_model(ir_xml);
        ov::preprocess::PrePostProcessor ppp(model);
        for (const auto &in : model->inputs()) {
            ppp.input(in.get_any_name()).tensor().set_element_type(ov::element::i32);
        }
        model = ppp.build();

        hold->has_types = false;
        for (const auto &in : model->inputs()) {
            const std::string n = in.get_any_name();
            if (n.find("input_ids") != std::string::npos) {
                hold->in_ids = n;
            } else if (n.find("attention_mask") != std::string::npos) {
                hold->in_mask = n;
            } else if (n.find("token_type") != std::string::npos) {
                hold->in_types = n;
                hold->has_types = true;
            }
        }
        if (hold->in_ids.empty() && !model->inputs().empty()) {
            hold->in_ids = model->input(0).get_any_name();
        }
        if (hold->in_mask.empty() && model->inputs().size() > 1) {
            hold->in_mask = model->input(1).get_any_name();
        }
        if (!hold->has_types && model->inputs().size() > 2) {
            hold->in_types = model->input(2).get_any_name();
            hold->has_types = true;
        }
        hold->out_logits = model->output(0).get_any_name();

        const char *ov_dev = want_gpu ? "GPU" : "CPU";
        hold->compiled = hold->core.compile_model(model, ov_dev, ov_accuracy_props());
        hold->request = hold->compiled.create_infer_request();

        r->enabled = true;
        r->gpu = want_gpu;
        r->token_usm = ov_usm_available(nullptr);
        r->remote_wrap = false;
        if (want_gpu) {
            std::string wrap_why;
            r->remote_wrap = ov_probe_remote_usm_wrap(&wrap_why);
            r->remote_wrap_why = wrap_why;
        } else {
            r->remote_wrap_why = "CPU path; remote GPU wrap not applicable";
        }
        r->ov_device = ov_dev;
        r->hold = hold;
        r->max_batch = cfg.max_batch;
        r->max_seq = cfg.max_position;
        return true;
    } catch (const std::exception &e) {
        if (err) {
            *err = std::string("OpenVINO compile failed on ") +
                   (want_gpu ? "GPU" : "CPU") + ": " + e.what() +
                   "; refusing CPU fallback";
        }
        return false;
    }
#else
    (void)device;
    (void)cfg;
    if (err) {
        *err = "OpenVINO MiniLM CE is not compiled into this binary";
    }
    return false;
#endif
}

void ov_resources_free(OvResources *r) {
    if (r == nullptr) {
        return;
    }
#ifdef TURBORERANK_OPENVINO
    if (r->hold != nullptr) {
        delete static_cast<OvHold *>(r->hold);
        r->hold = nullptr;
    }
#endif
    *r = OvResources{};
}

bool bert_forward_ov(
    OvResources *r,
    const int32_t *input_ids,
    const int32_t *attention_mask,
    const int32_t *token_type_ids,
    uint32_t n_rows,
    uint32_t seq,
    float *logits_out,
    std::string *err
) {
#ifdef TURBORERANK_OPENVINO
    if (r == nullptr || !r->enabled || r->hold == nullptr) {
        if (err) {
            *err = "OpenVINO resources are not initialized";
        }
        return false;
    }
    if (input_ids == nullptr || attention_mask == nullptr || logits_out == nullptr ||
        n_rows == 0 || seq == 0) {
        if (err) {
            *err = "bert_forward_ov: invalid pointers / shape";
        }
        return false;
    }
    auto *hold = static_cast<OvHold *>(r->hold);
    try {
        const ov::Shape shape{static_cast<size_t>(n_rows), static_cast<size_t>(seq)};
        // Caller-written USM (or aligned) pointers. No std::vector copy.
        ov::Tensor t_ids(ov::element::i32, shape, const_cast<int32_t *>(input_ids));
        ov::Tensor t_mask(
            ov::element::i32, shape, const_cast<int32_t *>(attention_mask)
        );
        hold->request.set_tensor(hold->in_ids, t_ids);
        hold->request.set_tensor(hold->in_mask, t_mask);
        if (hold->has_types) {
            if (token_type_ids == nullptr) {
                if (err) {
                    *err = "bert_forward_ov: token_type_ids required";
                }
                return false;
            }
            ov::Tensor t_types(
                ov::element::i32, shape, const_cast<int32_t *>(token_type_ids)
            );
            hold->request.set_tensor(hold->in_types, t_types);
        }
        hold->request.infer();
        const ov::Tensor out = hold->request.get_tensor(hold->out_logits);
        const ov::Shape osh = out.get_shape();
        const float *data = out.data<float>();
        const size_t stride = osh.size() >= 2 ? osh[1] : 1;
        for (uint32_t i = 0; i < n_rows; ++i) {
            logits_out[i] = data[static_cast<size_t>(i) * stride];
        }
        return true;
    } catch (const std::exception &e) {
        if (err) {
            *err = std::string("OpenVINO infer failed: ") + e.what() +
                   "; refusing CPU fallback";
        }
        return false;
    }
#else
    (void)r;
    (void)input_ids;
    (void)attention_mask;
    (void)token_type_ids;
    (void)n_rows;
    (void)seq;
    (void)logits_out;
    if (err) {
        *err = "OpenVINO MiniLM CE is not compiled into this binary";
    }
    return false;
#endif
}

std::string resolve_ov_ir_dir(
    const char *alias,
    size_t alias_len,
    const char *config_path,
    const char *workspace_root
) {
    const std::string name = alias_view(alias, alias_len);
    if (name.empty()) {
        return {};
    }
    auto has_ir = [](const std::string &d) {
        return is_dir(d) && (is_file(join_path(d, "openvino_model.xml")) ||
                             is_file(join_path(d, "model.xml")));
    };
    std::vector<std::string> roots;
    if (config_path && *config_path) {
        if (is_dir(config_path)) {
            roots.push_back(config_path);
            roots.push_back(join_path(config_path, name));
        }
    }
    if (const char *env = std::getenv("TURBORERANK_OV_MODEL_DIR")) {
        roots.push_back(env);
        roots.push_back(join_path(env, name));
    }
    if (workspace_root && *workspace_root) {
        roots.push_back(join_path(join_path(workspace_root, "models/ov-rerank"), name));
        if (alias_is_ce(name)) {
            roots.push_back(
                join_path(workspace_root, "models/ov-rerank/ms-marco-minilm-l6")
            );
        }
    }
    roots.push_back(join_path("models/ov-rerank", name));
    if (alias_is_ce(name)) {
        roots.push_back("models/ov-rerank/ms-marco-minilm-l6");
    }
    for (const auto &r : roots) {
        if (has_ir(r)) {
            return r;
        }
    }
    return {};
}

} // namespace impl
} // namespace turborerank
