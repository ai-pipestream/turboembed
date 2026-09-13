// SPDX-License-Identifier: Apache-2.0
//
// Machine B probe: can OpenVINO wrap turbo_buffer ZE SHARED USM as a
// remote tensor (USM_USER_BUFFER via RemoteContext)? This path does not
// include intel_gpu/ocl/ocl.hpp — that header requires CL/cl2.hpp, which
// this host does not ship. Prints stack facts and the exact exception.

#include "turbo_buffer.h"

#include <openvino/core/version.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/intel_gpu/remote_properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/remote_context.hpp>

#include <cstdio>
#include <cstring>
#include <string>

static void say(const char *k, const char *v) {
    std::printf("%-32s %s\n", k, v);
}

static std::string any_str(const ov::Any &a) {
    try {
        return a.as<std::string>();
    } catch (...) {
        try {
            return std::to_string(a.as<int>());
        } catch (...) {
            try {
                return std::to_string(a.as<int64_t>());
            } catch (...) {
                return "<unprintable>";
            }
        }
    }
}

int main() {
    std::printf("=== SOLIDIFY (7) remote-USM wrap probe ===\n");
    const ov::Version ver = ov::get_openvino_version();
    std::printf("openvino.build                 %s %s\n", ver.buildNumber, ver.description);

    ov::Core core;
    const auto devices = core.get_available_devices();
    std::string listed;
    for (const auto &d : devices) {
        if (!listed.empty()) {
            listed += ", ";
        }
        listed += d;
    }
    say("ov.devices", listed.c_str());

    bool has_gpu = false;
    for (const auto &d : devices) {
        if (d == "GPU" || d.rfind("GPU.", 0) == 0) {
            has_gpu = true;
        }
    }
    if (!has_gpu) {
        say("result", "UNAVAILABLE: OpenVINO lists no GPU plugin");
        return 2;
    }

    try {
        const std::string name = core.get_property("GPU", ov::device::full_name);
        say("gpu.full_name", name.c_str());
    } catch (const std::exception &e) {
        say("gpu.full_name.error", e.what());
    }

    const char *gpu_props[] = {
        "FULL_DEVICE_NAME",
        "DEVICE_TYPE",
        "OPTIMIZATION_CAPABILITIES",
        "INFERENCE_PRECISION_HINT",
        "EXECUTION_MODE_HINT",
        "GPU_UARCH_VERSION",
        "GPU_EXECUTION_UNITS_COUNT",
        "CONTEXT_TYPE",
    };
    for (const char *p : gpu_props) {
        try {
            const std::string v = any_str(core.get_property("GPU", p));
            std::printf("gpu.prop.%-22s %s\n", p, v.c_str());
        } catch (const std::exception &e) {
            std::printf("gpu.prop.%-22s ERR %s\n", p, e.what());
        }
    }

    void *usm = nullptr;
    const size_t bytes = 64 * sizeof(int32_t);
    const turbo_buffer_status zst =
        turbo_buffer_raw_alloc(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED, bytes, &usm);
    if (zst != TURBO_BUFFER_OK || usm == nullptr) {
        const char *err = turbo_buffer_last_error(nullptr);
        std::printf("ze.shared.alloc                FAIL %s\n", err ? err : "unknown");
        say("result", "UNAVAILABLE: ZE SHARED USM alloc failed");
        return 3;
    }
    std::memset(usm, 0, bytes);
    turbo_buffer_placement place = TURBO_BUFFER_PLACE_HOST;
    (void)turbo_buffer_ze_query(usm, &place);
    std::printf("ze.shared.ptr                  %p place=%d (3=SHARED)\n", usm, (int)place);

    std::shared_ptr<ov::Model> model;
    try {
        auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::i32, ov::Shape{1, 8});
        param->set_friendly_name("input_ids");
        param->output(0).set_names({"input_ids"});
        auto result = std::make_shared<ov::op::v0::Result>(param);
        model = std::make_shared<ov::Model>(
            ov::ResultVector{result}, ov::ParameterVector{param}, "probe"
        );
    } catch (const std::exception &e) {
        std::printf("model.build                    FAIL %s\n", e.what());
        turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED, usm);
        return 4;
    }

    ov::AnyMap props;
    props[ov::hint::execution_mode.name()] = ov::hint::ExecutionMode::ACCURACY;
    props[ov::hint::inference_precision.name()] = ov::element::f32;
    ov::CompiledModel compiled;
    try {
        compiled = core.compile_model(model, "GPU", props);
        say("compile", "OK GPU ACCURACY+f32");
    } catch (const std::exception &e) {
        std::printf("compile                        FAIL %s\n", e.what());
        turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED, usm);
        return 5;
    }

    ov::RemoteContext rctx;
    bool have_ctx = false;
    try {
        rctx = compiled.get_context();
        have_ctx = true;
        say("compiled.get_context", "OK");
        try {
            const auto params = rctx.get_params();
            for (const auto &kv : params) {
                std::printf("ctx.param.%-20s %s\n", kv.first.c_str(), any_str(kv.second).c_str());
            }
        } catch (const std::exception &e) {
            std::printf("ctx.params                     ERR %s\n", e.what());
        }
    } catch (const std::exception &e) {
        std::printf("compiled.get_context           FAIL %s\n", e.what());
    }

    bool usm_wrap_ok = false;
    std::string usm_wrap_err;
    if (have_ctx) {
        ov::AnyMap tparams = {
            {ov::intel_gpu::shared_mem_type.name(), ov::intel_gpu::SharedMemType::USM_USER_BUFFER},
            {ov::intel_gpu::mem_handle.name(), static_cast<ov::intel_gpu::gpu_handle_param>(usm)},
        };
        try {
            ov::RemoteTensor t = rctx.create_tensor(ov::element::i32, ov::Shape{1, 8}, tparams);
            usm_wrap_ok = true;
            say("create_tensor(USM_USER_BUFFER)", "OK");
            try {
                const auto tp = t.get_params();
                for (const auto &kv : tp) {
                    std::printf("usm.param.%-20s %s\n", kv.first.c_str(), any_str(kv.second).c_str());
                }
            } catch (const std::exception &e) {
                std::printf("usm.params                     ERR %s\n", e.what());
            }
            try {
                auto req = compiled.create_infer_request();
                req.set_tensor("input_ids", t);
                req.infer();
                say("infer.remote_usm", "OK");
            } catch (const std::exception &e) {
                usm_wrap_err = e.what();
                usm_wrap_ok = false;
                std::printf("infer.remote_usm               FAIL %s\n", e.what());
            }
        } catch (const std::exception &e) {
            usm_wrap_err = e.what();
            std::printf("create_tensor(USM_USER_BUFFER)  FAIL %s\n", e.what());
        }
    }

    // Plugin-allocated USM (not a wrap of turbo_buffer). Proves the remote
    // API itself works on this GPU when the pointer is OCL-owned.
    bool plugin_usm_ok = false;
    std::string plugin_usm_err;
    if (have_ctx) {
        ov::AnyMap plugin_params = {
            {ov::intel_gpu::shared_mem_type.name(),
             ov::intel_gpu::SharedMemType::USM_HOST_BUFFER},
        };
        try {
            ov::RemoteTensor t =
                rctx.create_tensor(ov::element::i32, ov::Shape{1, 8}, plugin_params);
            auto req = compiled.create_infer_request();
            req.set_tensor("input_ids", t);
            req.infer();
            plugin_usm_ok = true;
            say("infer.plugin_usm_host", "OK (OV-allocated, not turbo_buffer)");
        } catch (const std::exception &e) {
            plugin_usm_err = e.what();
            std::printf("infer.plugin_usm_host          FAIL %s\n", e.what());
        }
    }

    // Host ov::Tensor wrap (today's path) as control.
    try {
        ov::Tensor host(ov::element::i32, ov::Shape{1, 8}, usm);
        auto req = compiled.create_infer_request();
        req.set_tensor("input_ids", host);
        req.infer();
        say("infer.host_tensor_usm", "OK (current path)");
    } catch (const std::exception &e) {
        std::printf("infer.host_tensor_usm          FAIL %s\n", e.what());
    }

    turbo_buffer_raw_free(TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED, usm);

    std::printf("\n=== verdict ===\n");
    std::printf("remote_usm_user_buffer         %s%s\n",
                usm_wrap_ok ? "LIVE" : "FAIL",
                usm_wrap_err.empty() ? "" : (std::string(" : ") + usm_wrap_err).c_str());
    std::printf("plugin_usm_host_buffer         %s%s\n",
                plugin_usm_ok ? "LIVE" : "FAIL",
                plugin_usm_err.empty() ? "" : (std::string(" : ") + plugin_usm_err).c_str());
    std::printf("ocl.hpp                        UNAVAILABLE (CL/cl2.hpp missing on this host)\n");
    if (usm_wrap_ok) {
        say("result", "WRAP_POSSIBLE");
        return 0;
    }
    say("result", "WRAP_IMPOSSIBLE_ON_THIS_STACK");
    return 1;
}
