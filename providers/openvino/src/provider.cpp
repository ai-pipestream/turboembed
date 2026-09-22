// SPDX-License-Identifier: Apache-2.0
//
// Turbo OpenVINO provider.
//
// Devices: every OpenVINO GPU (ordinal 0..n-1) then CPU (ordinal n). NPUs are
// listed with no offered capability until qualified. Models are read from a
// bundle's `openvino_ir` or `onnx` artifact and compiled per session at a
// fixed [max_batch, max_seq] with the task's post-processing fused into the
// graph: masked mean / CLS / last-token pooling plus L2 for embedders,
// sigmoid for rerankers, softmax for classifiers, per-token softmax for token
// classifiers. Tokens are written by the native WordPiece encoder straight
// into the session's host staging rows; on GPU they are uploaded with one
// explicit OpenCL write per input tensor (counted in h2d_bytes) and results
// stay resident in a device buffer until the caller reads or exports them.
//
// Everything this provider does not offer is reported through the capability
// matrix and rejected with a status code; there is no silent fallback.

#include "turbo/turbo_provider.h"
#include "turbo/turbo_types.h"

#include "common.hpp"
#include "wordpiece.h"

#include <openvino/core/preprocess/pre_post_process.hpp>
#include <openvino/openvino.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#endif

namespace turbo_ov {
namespace op = ov::opset13;

// ---------------------------------------------------------------------------
// Provider state and devices
// ---------------------------------------------------------------------------

struct Device {
    std::string ov_name; // "GPU.0", "CPU", "NPU"
    uint32_t kind;       // TURBO_DEVICE_*
    uint32_t ordinal;
};

struct State {
    ov::Core core;
    std::mutex compile_mu;
    std::vector<Device> devices;

    State() {
        std::vector<std::string> gpus;
        bool cpu = false;
        std::vector<std::string> npus;
        for (const auto &name : core.get_available_devices()) {
            if (name == "CPU") {
                cpu = true;
            } else if (name == "GPU" || name.rfind("GPU.", 0) == 0) {
                gpus.push_back(name);
            } else if (name == "NPU" || name.rfind("NPU.", 0) == 0) {
                npus.push_back(name);
            }
        }
        std::sort(gpus.begin(), gpus.end());
        uint32_t ordinal = 0;
        for (const auto &g : gpus) {
            uint32_t kind = TURBO_DEVICE_GPU;
            try {
                const ov::device::Type t = core.get_property(g, ov::device::type);
                if (t == ov::device::Type::INTEGRATED) {
                    kind = TURBO_DEVICE_IGPU;
                }
            } catch (const std::exception &) {
                kind = TURBO_DEVICE_GPU;
            }
            devices.push_back(Device{g, kind, ordinal++});
        }
        if (cpu) {
            devices.push_back(Device{"CPU", TURBO_DEVICE_CPU, ordinal++});
        }
        for (const auto &n : npus) {
            devices.push_back(Device{n, TURBO_DEVICE_NPU, ordinal++});
        }
    }
};

State &state() {
    static State s;
    return s;
}

const Device &device_at(uint32_t ordinal) {
    auto &s = state();
    require(ordinal < s.devices.size(), TURBO_E_DEVICE_NOT_FOUND,
            "openvino provider has no device ordinal " + std::to_string(ordinal) + " (" +
                std::to_string(s.devices.size()) + " devices)");
    return s.devices[ordinal];
}

/// PCI vendor id and name of the host CPU, read from the CPU itself. An
/// unrecognized vendor is reported as 0/"unknown" rather than guessed: the
/// OpenVINO CPU plugin runs on AMD and on non-x86 hosts too.
struct Vendor {
    uint32_t id;
    std::string name;
};

Vendor cpu_vendor() {
#if defined(__x86_64__) || defined(__i386__)
    unsigned eax = 0, ebx = 0, ecx = 0, edx = 0;
    if (__get_cpuid(0, &eax, &ebx, &ecx, &edx) != 0) {
        char id[13];
        std::memcpy(id + 0, &ebx, 4);
        std::memcpy(id + 4, &edx, 4);
        std::memcpy(id + 8, &ecx, 4);
        id[12] = '\0';
        const std::string vendor(id);
        if (vendor == "GenuineIntel") {
            return Vendor{0x8086, "Intel"};
        }
        if (vendor == "AuthenticAMD") {
            return Vendor{0x1022, "AMD"};
        }
        return Vendor{0, vendor.empty() ? std::string("unknown") : vendor};
    }
#endif
    return Vendor{0, "unknown"};
}

/// `delete` at a vtable release entry. The release slots return void and
/// run during teardown, so there is nowhere to report a failing clRelease;
/// with CL_HPP_ENABLE_EXCEPTIONS a throwing `cl::Buffer`/`cl::Context`
/// destructor would otherwise unwind out of `noexcept` and terminate the
/// caller's process.
template <typename T>
void release(void *p) noexcept {
    try {
        delete static_cast<T *>(p);
    } catch (...) {
    }
}

constexpr uint64_t kCapsCommon = TURBO_CAP_OPT_TRUNCATE | TURBO_CAP_OPT_MAX_TOKENS | TURBO_CAP_OPT_PROMPT_ROLE |
                                 TURBO_CAP_OPT_TOP_N | TURBO_CAP_OPT_AGGREGATION;

/// True when this device returns the same bits for the same input on every
/// run. The OpenVINO CPU plugin does; the GPU plugin does not, because its
/// kernels reduce in an order the driver picks per dispatch. Measured on
/// `krick-1` (Intel Battlemage B70, driver 26.05.037020, OpenVINO 2026.3.1):
/// twenty repeats of one identical MiniLM batch in one session differ by up
/// to 2.3e-7 absolute, on every repeat, with or without padding columns,
/// while the same repeats on the CPU device are bit-identical
/// (`crates/turbo-conformance/tests/live_openvino.rs`). The bit says
/// "deterministic across runs", so a GPU device must not claim it.
bool deterministic_device(const Device &d) {
    return d.kind == TURBO_DEVICE_CPU;
}

uint64_t caps_of(const Device &d) {
    if (d.kind == TURBO_DEVICE_NPU) {
        return 0;
    }
    // Host-pointer import is implemented for every device this provider
    // serves (see `import_buffer`); only GPUs keep results device-resident.
    uint64_t caps = kCapsCommon | TURBO_CAP_HOST_PTR_IMPORT;
    if (deterministic_device(d)) {
        caps |= TURBO_CAP_DETERMINISTIC;
    }
    if (d.kind != TURBO_DEVICE_CPU) {
        caps |= TURBO_CAP_DEVICE_RESULT;
    }
    return caps;
}

bool offers(const Device &d, uint32_t task, uint32_t modality) {
    if (d.kind == TURBO_DEVICE_NPU || modality != TURBO_MODALITY_TEXT) {
        return false;
    }
    switch (task) {
    case TURBO_TASK_EMBED:
    case TURBO_TASK_RERANK:
    case TURBO_TASK_CLASSIFY:
    case TURBO_TASK_TOKEN_CLASSIFY:
        return true;
    default:
        return false;
    }
}

// ---------------------------------------------------------------------------
// Option enumerations
//
// ABI enumerations are open `uint32_t` values. A value this provider does
// not recognize is `TURBO_E_INVALID_ENUM` naming the 1-based field index; it
// is never mapped to a default, because a caller that asked for something
// this build does not know about must not silently get something else.
// ---------------------------------------------------------------------------

uint32_t checked_truncate(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_TRUNCATE_MODEL:
    case TURBO_TRUNCATE_NONE:
    case TURBO_TRUNCATE_RIGHT:
    case TURBO_TRUNCATE_LEFT:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "truncate " + std::to_string(v) + " is not a TURBO_TRUNCATE_* value", field);
    }
}

uint32_t checked_prompt_role(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_PROMPT_NONE:
    case TURBO_PROMPT_QUERY:
    case TURBO_PROMPT_DOCUMENT:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "prompt_role " + std::to_string(v) + " is not a TURBO_PROMPT_* value", field);
    }
}

uint32_t checked_aggregation(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_AGGREGATE_MODEL:
    case TURBO_AGGREGATE_NONE:
    case TURBO_AGGREGATE_SIMPLE:
    case TURBO_AGGREGATE_FIRST:
    case TURBO_AGGREGATE_MAX:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "aggregation " + std::to_string(v) + " is not a TURBO_AGGREGATE_* value", field);
    }
}

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct Context {
    Device dev;
    bool gpu = false;
    std::optional<ov::intel_gpu::ocl::ClContext> remote;
    cl::Context cl;
    cl::Device cl_dev;
    cl::CommandQueue queue;
    std::string driver;

    explicit Context(const Device &d) : dev(d), gpu(d.kind == TURBO_DEVICE_GPU || d.kind == TURBO_DEVICE_IGPU) {
        if (gpu) {
            remote.emplace(state().core.get_default_context(dev.ov_name).as<ov::intel_gpu::ocl::ClContext>());
            cl = cl::Context(remote->get(), true);
            const auto devs = cl.getInfo<CL_CONTEXT_DEVICES>();
            require(devs.size() == 1, TURBO_E_NOT_IMPLEMENTED, "multi-device OpenCL contexts are not supported");
            cl_dev = cl::Device(devs.front());
            queue = cl::CommandQueue(cl, cl_dev);
            driver = cl_dev.getInfo<CL_DRIVER_VERSION>();
        }
    }
};

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

struct Buffer {
    Context *ctx = nullptr;
    turbo_buffer_desc desc{};
    bool device = false;
    cl::Buffer cl_buf;
    void *host = nullptr;
    uint64_t bytes = 0;
    /// False for imported host pointers (the caller owns the memory) and for
    /// the session's own result staging.
    bool owns_host = true;

    ~Buffer() {
        if (host != nullptr && owns_host) {
            std::free(host);
        }
    }
};

turbo_buffer_desc packed_desc(uint32_t placement, uint32_t dtype, const std::vector<uint64_t> &shape, uint64_t elem) {
    turbo_buffer_desc d{};
    d.struct_size = sizeof(turbo_buffer_desc);
    d.placement = placement;
    d.dtype = dtype;
    d.ndim = static_cast<uint32_t>(shape.size());
    uint64_t acc = elem;
    for (size_t i = shape.size(); i-- > 0;) {
        d.shape[i] = shape[i];
        d.strides[i] = acc;
        acc *= shape[i];
    }
    d.bytes = acc;
    return d;
}

uint64_t dtype_size(uint32_t dtype) {
    switch (dtype) {
    case TURBO_DTYPE_BOOL:
    case TURBO_DTYPE_U8:
    case TURBO_DTYPE_I8:
        return 1;
    case TURBO_DTYPE_U16:
    case TURBO_DTYPE_I16:
    case TURBO_DTYPE_F16:
    case TURBO_DTYPE_BF16:
        return 2;
    case TURBO_DTYPE_U32:
    case TURBO_DTYPE_I32:
    case TURBO_DTYPE_F32:
        return 4;
    case TURBO_DTYPE_U64:
    case TURBO_DTYPE_I64:
    case TURBO_DTYPE_F64:
        return 8;
    default:
        fail(TURBO_E_UNSUPPORTED_DTYPE, "dtype " + std::to_string(dtype) + " is not supported by the openvino provider");
    }
}

/// Fill `b`'s descriptor from `in`, packing the shape when the caller
/// declared no byte count, and return the byte count.
uint64_t describe_into(Buffer *b, const turbo_buffer_desc &in) {
    b->desc = in;
    b->desc.next = nullptr;
    const uint64_t elem = dtype_size(in.dtype);
    check_buffer_desc(in, elem);
    uint64_t bytes = in.bytes;
    if (bytes == 0) {
        bytes = packed_bytes(elem, in.ndim, in.shape);
        b->desc = packed_desc(in.placement, in.dtype, std::vector<uint64_t>(in.shape, in.shape + in.ndim), elem);
    }
    b->bytes = bytes;
    b->desc.bytes = bytes;
    return bytes;
}

std::unique_ptr<Buffer> make_buffer(Context *ctx, const turbo_buffer_desc &in) {
    auto b = std::make_unique<Buffer>();
    b->ctx = ctx;
    const uint64_t bytes = describe_into(b.get(), in);
    require(bytes > 0, TURBO_E_INVALID_SHAPE, "zero-byte buffers are not allocated");
    switch (in.placement) {
    case TURBO_PLACE_HOST: {
        const size_t sz = (static_cast<size_t>(bytes) + 63) & ~static_cast<size_t>(63);
        b->host = std::aligned_alloc(64, sz);
        require(b->host != nullptr, TURBO_E_OUT_OF_MEMORY, "host allocation of " + std::to_string(bytes) + " bytes failed");
        std::memset(b->host, 0, sz);
        break;
    }
    case TURBO_PLACE_DEVICE:
        require(ctx->gpu, TURBO_E_UNSUPPORTED_PLACEMENT, "TURBO_PLACE_DEVICE needs an OpenVINO GPU device; CPU offers HOST only");
        b->device = true;
        b->cl_buf = cl::Buffer(ctx->cl, CL_MEM_READ_WRITE, static_cast<size_t>(bytes));
        break;
    default:
        fail(TURBO_E_UNSUPPORTED_PLACEMENT,
             "openvino provider supports TURBO_PLACE_HOST and TURBO_PLACE_DEVICE (GPU); PINNED and SHARED are not offered");
    }
    return b;
}

/// Wrap caller memory (`TURBO_CAP_HOST_PTR_IMPORT`). The pointer is used as
/// it is: nothing is copied and the caller keeps ownership, so the memory
/// must outlive the buffer handle. Only host memory is imported; a device
/// pointer would have to belong to this context's OpenCL context, which the
/// provider cannot verify, so `TURBO_HANDLE_CL_MEM` is rejected here.
std::unique_ptr<Buffer> import_buffer(Context *ctx, const turbo_buffer_desc &in, const turbo_native_handle &h) {
    require(h.kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED,
            "openvino provider imports TURBO_HANDLE_HOST_PTR only; handle kind " + std::to_string(h.kind) +
                " is not offered");
    require(in.placement == TURBO_PLACE_HOST, TURBO_E_UNSUPPORTED_PLACEMENT,
            "an imported host pointer is TURBO_PLACE_HOST; placement " + std::to_string(in.placement) +
                " cannot describe caller memory");
    require(h.offset == 0, TURBO_E_UNSUPPORTED, "openvino provider imports handles with offset 0 only");
    require(h.handle != 0, TURBO_E_INVALID_ARGUMENT, "imported host pointer is NULL");
    auto b = std::make_unique<Buffer>();
    b->ctx = ctx;
    const uint64_t bytes = describe_into(b.get(), in);
    require(bytes > 0, TURBO_E_INVALID_SHAPE, "an imported buffer must describe at least one byte");
    b->host = reinterpret_cast<void *>(static_cast<uintptr_t>(h.handle));
    b->owns_host = false;
    return b;
}

turbo_provider_buffer describe(Buffer *b) {
    turbo_provider_buffer out{};
    out.struct_size = sizeof(turbo_provider_buffer);
    out.handle = b;
    out.host_ptr = b->host;
    out.desc = b->desc;
    return out;
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

enum class Kind { Embedding, Reranker, Classifier, TokenClassifier };
enum class Pool { Mean, Cls, Last };

struct Model {
    Context *ctx = nullptr;
    Bundle bundle;
    Kind kind = Kind::Embedding;
    Pool pool = Pool::Mean;
    bool normalize = false;
    std::string activation; // softmax | sigmoid | none
    std::shared_ptr<ov::Model> graph;
    std::string in_ids, in_mask, in_types;
    bool has_types = false;
    uint32_t width = 0; // dim or n_labels
    uint32_t dim = 0;   // contract.dim as the bundle states it, for every kind
    uint32_t max_seq = 0;
    uint32_t max_batch = 0;
    uint32_t aggregation = TURBO_AGGREGATE_SIMPLE; // what TURBO_AGGREGATE_MODEL resolves to
    wordpiece_vocab *vocab = nullptr;
    std::vector<std::string> labels;

    ~Model() {
        if (vocab != nullptr) {
            wordpiece_vocab_destroy(vocab);
        }
    }

    /// Clone, reshape to [batch, seq], and fuse the task's post-processing.
    std::shared_ptr<ov::Model> shaped(uint32_t batch, uint32_t seq) const {
        auto m = graph->clone();
        std::map<std::string, ov::PartialShape> shapes;
        ov::Output<ov::Node> mask;
        for (const auto &input : m->inputs()) {
            const auto name = input.get_any_name();
            shapes[name] = ov::PartialShape{batch, seq};
            if (name == in_mask) {
                mask = input;
            }
        }
        m->reshape(shapes);
        const auto out = m->get_results().at(0)->input_value(0);
        const auto axis1 = op::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        const auto axis2 = op::Constant::create(ov::element::i64, ov::Shape{1}, {2});
        ov::Output<ov::Node> y;
        switch (kind) {
        case Kind::Embedding: {
            require(out.get_partial_shape().rank().get_length() == 3, TURBO_E_UNSUPPORTED,
                    "embedding model output must be [batch, seq, hidden]");
            const auto fmask = std::make_shared<op::Convert>(mask, ov::element::f32);
            switch (pool) {
            case Pool::Mean: {
                const auto expanded = std::make_shared<op::Unsqueeze>(fmask, axis2);
                const auto sum = std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(out, expanded), axis1, false);
                const auto count = std::make_shared<op::ReduceSum>(fmask, axis1, true);
                const auto denom = std::make_shared<op::Maximum>(count, op::Constant::create(ov::element::f32, ov::Shape{}, {1.0f}));
                y = std::make_shared<op::Divide>(sum, denom);
                break;
            }
            case Pool::Cls: {
                const auto idx = op::Constant::create(ov::element::i64, ov::Shape{}, {0});
                y = std::make_shared<op::Gather>(out, idx, axis1);
                break;
            }
            case Pool::Last: {
                // index = max(sum(mask) - 1, 0) per row; gather with batch_dims = 1.
                const auto count = std::make_shared<op::ReduceSum>(std::make_shared<op::Convert>(mask, ov::element::i64), axis1, true);
                const auto last = std::make_shared<op::Maximum>(
                    std::make_shared<op::Subtract>(count, op::Constant::create(ov::element::i64, ov::Shape{}, {1})),
                    op::Constant::create(ov::element::i64, ov::Shape{}, {0}));
                const auto g = std::make_shared<op::Gather>(out, last, axis1, 1);
                y = std::make_shared<op::Squeeze>(g, axis1);
                break;
            }
            }
            if (normalize) {
                const auto sq = std::make_shared<op::Multiply>(y, y);
                const auto norm = std::make_shared<op::Sqrt>(std::make_shared<op::ReduceSum>(sq, axis1, true));
                const auto d = std::make_shared<op::Maximum>(norm, op::Constant::create(ov::element::f32, ov::Shape{}, {1e-12f}));
                y = std::make_shared<op::Divide>(y, d);
            }
            break;
        }
        case Kind::Reranker: {
            require(out.get_partial_shape().rank().get_length() == 2, TURBO_E_UNSUPPORTED,
                    "reranker model output must be [batch, 1] logits");
            ov::Output<ov::Node> s = out;
            if (activation == "sigmoid") {
                s = std::make_shared<op::Sigmoid>(s);
            } else if (activation != "none" && !activation.empty()) {
                fail(TURBO_E_UNSUPPORTED, "reranker activation `" + activation + "` is not supported (sigmoid, none)");
            }
            y = std::make_shared<op::Squeeze>(s, axis1);
            break;
        }
        case Kind::Classifier: {
            require(out.get_partial_shape().rank().get_length() == 2, TURBO_E_UNSUPPORTED,
                    "classifier model output must be [batch, labels] logits");
            if (activation == "softmax") {
                y = std::make_shared<op::Softmax>(out, 1);
            } else if (activation == "sigmoid") {
                y = std::make_shared<op::Sigmoid>(out);
            } else if (activation == "none" || activation.empty()) {
                y = out;
            } else {
                fail(TURBO_E_UNSUPPORTED, "classifier activation `" + activation + "` is not supported");
            }
            break;
        }
        case Kind::TokenClassifier: {
            require(out.get_partial_shape().rank().get_length() == 3, TURBO_E_UNSUPPORTED,
                    "token classifier model output must be [batch, seq, labels] logits");
            y = std::make_shared<op::Softmax>(out, 2);
            break;
        }
        }
        y.get_node_shared_ptr()->output(0).set_names({"turbo_output"});
        auto fused = std::make_shared<ov::Model>(ov::OutputVector{y}, m->get_parameters());
        ov::preprocess::PrePostProcessor prep(fused);
        for (size_t i = 0; i < fused->inputs().size(); ++i) {
            prep.input(i).tensor().set_element_type(ov::element::i32);
        }
        prep.output(0).tensor().set_element_type(ov::element::f32);
        return prep.build();
    }
};

std::unique_ptr<Model> load_model(Context *ctx, const std::string &dir, const std::map<std::string, std::string> &opts) {
    reject_unknown(opts, {}, "openvino model");
    auto m = std::make_unique<Model>();
    m->ctx = ctx;
    m->bundle = Bundle::read(dir);
    const Bundle &b = m->bundle;
    require(b.modality == "text", TURBO_E_UNSUPPORTED_MODALITY, "openvino provider serves text bundles only");
    m->dim = b.dim;
    if (b.kind == "embedding") {
        m->kind = Kind::Embedding;
        require(b.dim > 0, TURBO_E_BUNDLE_INVALID, "embedding bundle must declare contract.dim");
        if (b.pooling == "mean") {
            m->pool = Pool::Mean;
        } else if (b.pooling == "cls") {
            m->pool = Pool::Cls;
        } else if (b.pooling == "last") {
            m->pool = Pool::Last;
        } else {
            fail(TURBO_E_BUNDLE_INVALID, "embedding bundle pooling `" + b.pooling + "` must be mean, cls, or last");
        }
        require(b.normalize == "l2" || b.normalize == "none", TURBO_E_BUNDLE_INVALID, "contract.normalize must be l2 or none");
        m->normalize = b.normalize == "l2";
        m->width = b.dim;
    } else if (b.kind == "reranker") {
        m->kind = Kind::Reranker;
        m->activation = b.activation.empty() ? "sigmoid" : b.activation;
        m->width = 1;
    } else if (b.kind == "classifier") {
        m->kind = Kind::Classifier;
        require(!b.labels.empty(), TURBO_E_BUNDLE_INVALID, "classifier bundle must declare labels");
        m->activation = b.activation.empty() ? "softmax" : b.activation;
        m->width = static_cast<uint32_t>(b.labels.size());
    } else if (b.kind == "token_classifier") {
        m->kind = Kind::TokenClassifier;
        require(!b.labels.empty(), TURBO_E_BUNDLE_INVALID, "token classifier bundle must declare labels");
        m->width = static_cast<uint32_t>(b.labels.size());
        // contract.aggregation is what TURBO_AGGREGATE_MODEL resolves to. It
        // names a strategy or is absent (then: simple); it cannot name
        // `model` itself, and an unknown name is a broken bundle.
        const std::string &a = b.aggregation;
        if (a.empty() || a == "simple") {
            m->aggregation = TURBO_AGGREGATE_SIMPLE;
        } else if (a == "none") {
            m->aggregation = TURBO_AGGREGATE_NONE;
        } else if (a == "first") {
            m->aggregation = TURBO_AGGREGATE_FIRST;
        } else if (a == "max") {
            m->aggregation = TURBO_AGGREGATE_MAX;
        } else {
            fail(TURBO_E_BUNDLE_INVALID,
                 "contract.aggregation `" + a + "` must be none, simple, first, or max");
        }
    } else {
        fail(TURBO_E_UNSUPPORTED_TASK, "openvino provider does not serve `" + b.kind + "` bundles (embedding, reranker, classifier, token_classifier)");
    }
    m->labels = b.labels;
    m->max_seq = b.max_seq == 0 ? 512 : b.max_seq;
    m->max_batch = b.max_batch == 0 ? 32 : b.max_batch;

    // Artifact.
    std::string path;
    if (auto ir = b.artifact("openvino_ir")) {
        path = *ir;
    } else if (auto onnx = b.artifact("onnx")) {
        path = *onnx;
    } else {
        std::string have;
        for (const auto &[k, v] : b.artifacts) {
            have += (have.empty() ? "" : ", ") + k;
        }
        fail(TURBO_E_BUNDLE_NO_ARTIFACT, "bundle `" + b.model_id + "` has no openvino_ir or onnx artifact; it has: " + (have.empty() ? "(none)" : have));
    }
    m->graph = state().core.read_model(path);
    require(m->graph->outputs().size() == 1, TURBO_E_UNSUPPORTED,
            "model must have exactly one output; `" + b.model_id + "` has " + std::to_string(m->graph->outputs().size()));
    for (const auto &input : m->graph->inputs()) {
        const auto name = input.get_any_name();
        require(input.get_partial_shape().rank().is_static() && input.get_partial_shape().rank().get_length() == 2,
                TURBO_E_UNSUPPORTED, "model input `" + name + "` must be rank 2 [batch, seq]");
        if (name == "input_ids") {
            m->in_ids = name;
        } else if (name == "attention_mask") {
            m->in_mask = name;
        } else if (name == "token_type_ids") {
            m->in_types = name;
            m->has_types = true;
        } else {
            fail(TURBO_E_UNSUPPORTED, "model input `" + name + "` is not one of input_ids, attention_mask, token_type_ids");
        }
    }
    require(!m->in_ids.empty() && !m->in_mask.empty(), TURBO_E_UNSUPPORTED, "model must have input_ids and attention_mask inputs");
    if (m->kind == Kind::Embedding) {
        const auto &ps = m->graph->output(0).get_partial_shape();
        require(ps.rank().is_static() && ps.rank().get_length() == 3 && ps[2].is_static() &&
                    static_cast<uint32_t>(ps[2].get_length()) == b.dim,
                TURBO_E_BUNDLE_INVALID,
                "model hidden size does not match contract.dim " + std::to_string(b.dim));
    }

    // Tokenizer: native WordPiece from tokenizer.json (gated loader).
    require(!b.tokenizer_path.empty(), TURBO_E_BUNDLE_INVALID, "bundle has no tokenizer.json; the openvino provider tokenizes natively");
    require(b.tokenizer_kind == "wordpiece", TURBO_E_UNSUPPORTED,
            "openvino provider tokenizes WordPiece (BERT) natively; tokenizer kind `" + b.tokenizer_kind +
                "` is not supported here yet (use turbo_session_write_tokens with the core tokenizer)");
    const int rc = wordpiece_vocab_load(b.tokenizer_path.c_str(), &m->vocab);
    require(rc == WORDPIECE_OK && m->vocab != nullptr, TURBO_E_BUNDLE_INVALID,
            "tokenizer.json is not a supported uncased BERT WordPiece configuration (wordpiece status " + std::to_string(rc) + ")");
    return m;
}

void fill_model_info(const Model &m, turbo_model_info &out) {
    out.task = m.kind == Kind::Embedding ? TURBO_TASK_EMBED
               : m.kind == Kind::Reranker ? TURBO_TASK_RERANK
               : m.kind == Kind::Classifier ? TURBO_TASK_CLASSIFY
                                            : TURBO_TASK_TOKEN_CLASSIFY;
    out.kind = m.kind == Kind::Embedding ? TURBO_MODEL_EMBEDDING
               : m.kind == Kind::Reranker ? TURBO_MODEL_RERANKER
               : m.kind == Kind::Classifier ? TURBO_MODEL_CLASSIFIER
                                            : TURBO_MODEL_TOKEN_CLASSIFIER;
    out.modality = TURBO_MODALITY_TEXT;
    out.dim = m.dim; // contract.dim as the bundle states it, for every kind
    out.n_labels = static_cast<uint32_t>(m.labels.size());
    out.pooling = m.kind == Kind::Embedding ? (m.pool == Pool::Mean ? TURBO_POOLING_MEAN : m.pool == Pool::Cls ? TURBO_POOLING_CLS : TURBO_POOLING_LAST) : 0;
    out.normalize = m.kind == Kind::Embedding ? (m.normalize ? TURBO_NORMALIZE_L2 : TURBO_NORMALIZE_NONE) : 0;
    out.max_seq = m.max_seq;
    out.max_batch = m.max_batch;
    out.dtype_used = TURBO_DTYPE_F32;
    const bool gpu = m.ctx->gpu;
    const uint32_t on_dev = gpu ? TURBO_STAGE_FUSED : TURBO_STAGE_HOST;
    out.stage_placement[TURBO_STAGE_TOKENIZE] = TURBO_STAGE_HOST;
    out.stage_placement[TURBO_STAGE_ENCODE] = gpu ? TURBO_STAGE_DEVICE : TURBO_STAGE_HOST;
    out.stage_placement[TURBO_STAGE_POOL] = m.kind == Kind::Embedding ? on_dev : TURBO_STAGE_UNUSED;
    out.stage_placement[TURBO_STAGE_NORMALIZE] = (m.kind == Kind::Embedding && m.normalize) ? on_dev : TURBO_STAGE_UNUSED;
    out.stage_placement[TURBO_STAGE_POSTPROCESS] =
        m.kind == Kind::Embedding ? TURBO_STAGE_UNUSED : (m.kind == Kind::TokenClassifier ? TURBO_STAGE_HOST : on_dev);
    // Tokenization is host work, so a GPU model is never fully accelerated.
    out.fully_accelerated = 0;
    out.n_inputs = 0;
    out.n_outputs = 0;
    out.vocab_size = m.bundle.vocab_size;
    put_str(out.model_id, m.bundle.model_id);
    put_str(out.revision, m.bundle.revision);
    put_str(out.tokenizer_sha256, m.bundle.tokenizer_sha256);
    put_str(out.provider_id, std::string("openvino"));
    put_str(out.prefix_query, m.bundle.prefix_query);
    put_str(out.prefix_document, m.bundle.prefix_document);
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

struct WordSpan {
    uint64_t start, end; // bytes
    uint32_t first_token; // column in the row
    uint32_t n_tokens;
};

struct Session {
    Model *model = nullptr;
    uint32_t batch = 0, seq = 0;
    ov::CompiledModel compiled;
    ov::InferRequest request;
    std::vector<int32_t> ids, mask, types; // host staging [batch*seq]
    std::vector<float> host_out;            // [batch*width] (CPU) or read-back scratch (GPU)
    std::vector<int32_t> sorted;            // rerank order
    std::vector<int32_t> scratch;           // left-truncation staging
    std::vector<int32_t> pos_scratch;       // position ids the pair packer writes (unused by the graph)
    std::vector<int32_t> types_scratch;     // type ids for models without a token_type_ids input
    cl::Buffer d_ids, d_mask, d_types, d_out;
    uint32_t n_rows = 0;
    turbo_embed_options eopts{};
    turbo_rerank_options ropts{};
    turbo_classify_options copts{};
    std::vector<std::vector<WordSpan>> words; // per row, token classification
    std::vector<turbo_span> spans;
    // Result descriptors, stable until the next run.
    Buffer out_buf, sorted_buf;
    turbo_provider_output outputs[2]{};
    std::string name0, name1;
    std::atomic<bool> in_use{false}; // one caller at a time; see BusyGuard
    uint64_t runs = 0, h2d = 0, d2h = 0;
    bool host_out_valid = false;

    Session(Model *m, uint32_t b, uint32_t s) : model(m), batch(b), seq(s) {
        const size_t n = static_cast<size_t>(b) * s;
        ids.assign(n, 0);
        mask.assign(n, 0);
        if (m->has_types) {
            types.assign(n, 0);
        }
        const uint64_t out_elems = m->kind == Kind::TokenClassifier ? static_cast<uint64_t>(b) * s * m->width
                                                                    : static_cast<uint64_t>(b) * m->width;
        host_out.assign(out_elems, 0.0f);
        sorted.assign(b, 0);
        scratch.assign(static_cast<size_t>(s) * 16, 0);
        pos_scratch.assign(s, 0);
        types_scratch.assign(s, 0);
        words.resize(b);
        for (auto &w : words) {
            w.reserve(s);
        }
        spans.reserve(static_cast<size_t>(b) * s);
        auto &st = state();
        std::lock_guard<std::mutex> lock(st.compile_mu);
        auto graph = m->shaped(b, s);
        const ov::AnyMap props = {ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY),
                                  ov::hint::inference_precision(ov::element::f32)};
        Context &ctx = *m->ctx;
        if (ctx.gpu) {
            d_ids = cl::Buffer(ctx.cl, CL_MEM_READ_WRITE, n * 4);
            d_mask = cl::Buffer(ctx.cl, CL_MEM_READ_WRITE, n * 4);
            if (m->has_types) {
                d_types = cl::Buffer(ctx.cl, CL_MEM_READ_WRITE, n * 4);
            }
            d_out = cl::Buffer(ctx.cl, CL_MEM_READ_WRITE, static_cast<size_t>(out_elems) * 4);
            compiled = st.core.compile_model(graph, *ctx.remote, props);
        } else {
            compiled = st.core.compile_model(graph, "CPU", props);
        }
        request = compiled.create_infer_request();
        const ov::Shape tshape{b, s};
        for (const auto &port : compiled.inputs()) {
            const auto name = port.get_any_name();
            if (ctx.gpu) {
                const cl::Buffer &buf = name == m->in_ids ? d_ids : name == m->in_mask ? d_mask : d_types;
                request.set_tensor(port, ctx.remote->create_tensor(ov::element::i32, tshape, buf));
            } else {
                std::vector<int32_t> &v = name == m->in_ids ? ids : name == m->in_mask ? mask : types;
                request.set_tensor(port, ov::Tensor(ov::element::i32, tshape, v.data()));
            }
        }
        const ov::Shape oshape = m->kind == Kind::TokenClassifier ? ov::Shape{b, s, m->width} : (m->kind == Kind::Reranker ? ov::Shape{b} : ov::Shape{b, m->width});
        if (ctx.gpu) {
            request.set_output_tensor(ctx.remote->create_tensor(ov::element::f32, oshape, d_out));
        } else {
            request.set_output_tensor(ov::Tensor(ov::element::f32, oshape, host_out.data()));
        }
        // Result buffer descriptors: output 0 (device on GPU, host on CPU), output 1 sorted (host).
        // Both descriptors borrow memory this session owns.
        out_buf.ctx = &ctx;
        out_buf.device = ctx.gpu;
        out_buf.cl_buf = d_out;
        out_buf.host = ctx.gpu ? nullptr : host_out.data();
        out_buf.bytes = out_elems * 4;
        out_buf.owns_host = false;
        sorted_buf.ctx = &ctx;
        sorted_buf.host = sorted.data();
        sorted_buf.bytes = static_cast<uint64_t>(b) * 4;
        sorted_buf.owns_host = false;
        name0 = m->kind == Kind::Embedding ? "embeddings" : "scores";
        name1 = "sorted";
    }

    uint32_t budget(uint32_t max_tokens) const {
        const uint32_t b = max_tokens == 0 ? seq : std::min(max_tokens, seq);
        require(b >= 2, TURBO_E_CAPACITY, "token budget must be at least 2 for [CLS] and [SEP]");
        return b;
    }

    /// Tokenize text (with optional prefix) into row r: [CLS] prefix text [SEP] pad.
    void encode_row(uint32_t r, const std::string &prefix, turbo_text text, uint32_t truncate, uint32_t budget,
                    std::vector<WordSpan> *word_spans) {
        const size_t base = static_cast<size_t>(r) * seq;
        int32_t *row_ids = ids.data() + base;
        int32_t *row_mask = mask.data() + base;
        const uint32_t content_budget = budget - 2;
        const wordpiece_vocab *v = model->vocab;
        // Prefix tokens go first, never truncated away on the right.
        size_t n_prefix = 0;
        if (!prefix.empty()) {
            require(wordpiece_tokenize(v, prefix.data(), prefix.size(), scratch.data(), scratch.size(), 4, &n_prefix) == WORDPIECE_OK,
                    TURBO_E_INTERNAL, "prefix tokenization failed");
            require(n_prefix <= content_budget, TURBO_E_CAPACITY, "prompt prefix alone exceeds the token budget");
        }
        // Count the text tokens, then place them.
        size_t n_text = 0;
        require(text.len == 0 || text.ptr != nullptr, TURBO_E_INVALID_ARGUMENT, "text view has NULL ptr");
        const int cnt = wordpiece_tokenize(v, text.ptr, static_cast<size_t>(text.len), nullptr, 0, 4, &n_text);
        require(cnt != WORDPIECE_ERR_INVALID_ARGUMENT, TURBO_E_INVALID_UTF8, "text is not valid UTF-8");
        require(cnt == WORDPIECE_OK, TURBO_E_INTERNAL, "tokenizer failed with status " + std::to_string(cnt));
        const size_t avail = content_budget - n_prefix;
        size_t skip = 0, take = n_text;
        if (n_text > avail) {
            switch (truncate) {
            case TURBO_TRUNCATE_NONE:
                fail(TURBO_E_CAPACITY, "input row " + std::to_string(r) + " tokenizes to " + std::to_string(n_text + n_prefix + 2) +
                                           " tokens but the budget is " + std::to_string(budget) + " and truncation is NONE");
            case TURBO_TRUNCATE_LEFT:
                skip = n_text - avail;
                take = avail;
                break;
            default:
                take = avail;
            }
        }
        uint32_t col = 0;
        row_ids[col++] = wordpiece_cls_id(v);
        for (size_t i = 0; i < n_prefix; ++i) {
            row_ids[col++] = scratch[i];
        }
        if (take > 0) {
            if (skip == 0) {
                size_t n = 0;
                require(wordpiece_tokenize(v, text.ptr, static_cast<size_t>(text.len), row_ids + col, take, 4, &n) == WORDPIECE_OK,
                        TURBO_E_INTERNAL, "text tokenization failed");
                col += static_cast<uint32_t>(std::min(n, take));
            } else {
                if (n_text > scratch.size()) {
                    scratch.resize(n_text);
                }
                size_t n = 0;
                require(wordpiece_tokenize(v, text.ptr, static_cast<size_t>(text.len), scratch.data(), scratch.size(), 4, &n) == WORDPIECE_OK,
                        TURBO_E_INTERNAL, "text tokenization failed");
                for (size_t i = skip; i < skip + take; ++i) {
                    row_ids[col++] = scratch[i];
                }
            }
        }
        row_ids[col++] = wordpiece_sep_id(v);
        for (uint32_t c = 0; c < col; ++c) {
            row_mask[c] = 1;
        }
        const int32_t pad = wordpiece_pad_id(v);
        for (uint32_t c = col; c < seq; ++c) {
            row_ids[c] = pad;
            row_mask[c] = 0;
        }
        if (model->has_types) {
            std::fill(types.begin() + base, types.begin() + base + seq, 0);
        }
        if (word_spans != nullptr) {
            collect_words(word_spans, text, n_prefix, skip, take);
        }
    }

    /// Word boundaries for span aggregation, in row columns.
    ///
    /// Words are whitespace-delimited runs with each ASCII punctuation
    /// character its own word, each tokenized on its own so the sub-token
    /// counts line up with the row (WordPiece never crosses whitespace or
    /// punctuation).
    ///
    /// The row holds the text tokens `[skip, skip + take)` at the columns
    /// `[1 + n_prefix, 1 + n_prefix + take)`. A word whose sub-tokens only
    /// partly fall inside that window - truncation cut it in half, on either
    /// end - is dropped, not clipped: its label would otherwise be read from
    /// a fragment, and a clipped count would no longer describe the word.
    /// Every emitted span therefore satisfies
    /// `first_token + n_tokens <= 1 + n_prefix + take`, which is inside the
    /// row's live tokens, so aggregation never reads past the row.
    void collect_words(std::vector<WordSpan> *out, turbo_text text, size_t n_prefix, size_t skip, size_t take) const {
        out->clear();
        const wordpiece_vocab *v = model->vocab;
        const char *p = text.ptr;
        const size_t len = static_cast<size_t>(text.len);
        const size_t first_col = 1 + n_prefix;
        auto is_punct = [](unsigned char c) {
            return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
        };
        size_t i = 0;
        size_t seen = 0; // text tokens before this word
        while (i < len) {
            while (i < len && static_cast<unsigned char>(p[i]) <= ' ') {
                ++i;
            }
            if (i >= len) {
                break;
            }
            const size_t start = i;
            if (is_punct(static_cast<unsigned char>(p[i]))) {
                ++i; // one punctuation character is one word
            } else {
                while (i < len && static_cast<unsigned char>(p[i]) > ' ' && !is_punct(static_cast<unsigned char>(p[i]))) {
                    ++i;
                }
            }
            size_t nw = 0;
            require(wordpiece_tokenize(v, p + start, i - start, nullptr, 0, 4, &nw) == WORDPIECE_OK, TURBO_E_INTERNAL,
                    "word tokenization failed");
            if (nw == 0) {
                continue; // the normalizer dropped it; it occupies no column
            }
            if (seen >= skip + take) {
                break; // this word and every later one are past the right edge
            }
            if (seen >= skip && seen + nw <= skip + take) {
                out->push_back(
                    WordSpan{start, i, static_cast<uint32_t>(first_col + (seen - skip)), static_cast<uint32_t>(nw)});
            }
            seen += nw;
        }
    }

    void upload() {
        Context &ctx = *model->ctx;
        if (!ctx.gpu) {
            return;
        }
        const size_t bytes = static_cast<size_t>(batch) * seq * 4;
        ctx.queue.enqueueWriteBuffer(d_ids, CL_FALSE, 0, bytes, ids.data());
        ctx.queue.enqueueWriteBuffer(d_mask, CL_FALSE, 0, bytes, mask.data());
        h2d += 2 * bytes;
        if (model->has_types) {
            ctx.queue.enqueueWriteBuffer(d_types, CL_FALSE, 0, bytes, types.data());
            h2d += bytes;
        }
        ctx.queue.finish();
    }

    void read_back_output() {
        Context &ctx = *model->ctx;
        if (ctx.gpu && !host_out_valid) {
            ctx.queue.enqueueReadBuffer(d_out, CL_TRUE, 0, host_out.size() * 4, host_out.data());
            d2h += host_out.size() * 4;
            host_out_valid = true;
        }
    }

    turbo_provider_result run() {
        upload();
        request.infer();
        host_out_valid = !model->ctx->gpu;
        ++runs;
        turbo_provider_result r{};
        r.struct_size = sizeof(turbo_provider_result);
        outputs[0] = turbo_provider_output{};
        outputs[0].struct_size = sizeof(turbo_provider_output);
        outputs[0].name = turbo_text{name0.data(), name0.size()};
        outputs[0].buffer = describe(&out_buf);
        const uint32_t w = model->width;
        switch (model->kind) {
        case Kind::Embedding:
        case Kind::Classifier:
            outputs[0].ndim = 2;
            outputs[0].shape[0] = n_rows;
            outputs[0].shape[1] = w;
            out_buf.desc = packed_desc(model->ctx->gpu ? TURBO_PLACE_DEVICE : TURBO_PLACE_HOST, TURBO_DTYPE_F32, {batch, w}, 4);
            break;
        case Kind::Reranker:
            outputs[0].ndim = 1;
            outputs[0].shape[0] = n_rows;
            out_buf.desc = packed_desc(model->ctx->gpu ? TURBO_PLACE_DEVICE : TURBO_PLACE_HOST, TURBO_DTYPE_F32, {batch}, 4);
            break;
        case Kind::TokenClassifier:
            outputs[0].ndim = 3;
            outputs[0].shape[0] = n_rows;
            outputs[0].shape[1] = seq;
            outputs[0].shape[2] = w;
            out_buf.desc = packed_desc(model->ctx->gpu ? TURBO_PLACE_DEVICE : TURBO_PLACE_HOST, TURBO_DTYPE_F32, {batch, seq, w}, 4);
            break;
        }
        outputs[0].buffer.desc = out_buf.desc;
        r.n_outputs = 1;
        r.outputs = outputs;
        spans.clear();
        if (model->kind == Kind::Reranker && (ropts.return_sorted != 0 || ropts.top_n != 0)) {
            read_back_output();
            std::iota(sorted.begin(), sorted.begin() + n_rows, 0);
            const float *s = host_out.data();
            std::stable_sort(sorted.begin(), sorted.begin() + n_rows, [s](int32_t a, int32_t b) { return s[a] > s[b]; });
            // top_n is a request for at most k of the rows that were
            // written; a larger k would publish indices the sort never
            // touched. The core rejects top_n > rows before this point, so
            // the clamp only guards a caller that drives the vtable itself.
            const uint32_t k = ropts.top_n == 0 ? n_rows : std::min(ropts.top_n, n_rows);
            outputs[1] = turbo_provider_output{};
            outputs[1].struct_size = sizeof(turbo_provider_output);
            outputs[1].name = turbo_text{name1.data(), name1.size()};
            outputs[1].ndim = 1;
            outputs[1].shape[0] = k;
            sorted_buf.desc = packed_desc(TURBO_PLACE_HOST, TURBO_DTYPE_I32, {batch}, 4);
            outputs[1].buffer = describe(&sorted_buf);
            r.n_outputs = 2;
        }
        if (model->kind == Kind::TokenClassifier) {
            read_back_output();
            aggregate_spans();
            r.n_spans = static_cast<uint32_t>(spans.size());
            r.spans = spans.empty() ? nullptr : spans.data();
        }
        return r;
    }

    /// Label of a token: argmax over the fused softmax.
    uint32_t token_label(uint32_t row, uint32_t col, float *score) const {
        const uint32_t w = model->width;
        const float *p = host_out.data() + (static_cast<size_t>(row) * seq + col) * w;
        uint32_t best = 0;
        for (uint32_t l = 1; l < w; ++l) {
            if (p[l] > p[best]) {
                best = l;
            }
        }
        *score = p[best];
        return best;
    }

    static std::string entity_of(const std::string &label) {
        if (label.size() > 2 && (label[1] == '-') && (label[0] == 'B' || label[0] == 'I' || label[0] == 'L' || label[0] == 'U' || label[0] == 'E' || label[0] == 'S')) {
            return label.substr(2);
        }
        return label;
    }

    /// Word-aligned span aggregation over the fused per-token softmax.
    ///
    /// The word's label is the first sub-token's (`SIMPLE` and `FIRST`: both
    /// are word-aligned here, so they agree by construction) or the
    /// highest-scoring sub-token's (`MAX`). Consecutive words carrying the
    /// same entity merge unless a `B-`/`U-`/`S-` tag starts a new one, and a
    /// group's score is the mean of its word scores. Spans are word-aligned:
    /// an entity change inside one word is not representable, which is where
    /// `SIMPLE` differs from Hugging Face's token-level grouping. The same
    /// rules run in the CUDA provider's `aggregate_spans`.
    void aggregate_spans() {
        const uint32_t agg = copts.aggregation == TURBO_AGGREGATE_MODEL ? model->aggregation : copts.aggregation;
        const auto &labels = model->labels;
        for (uint32_t r = 0; r < n_rows; ++r) {
            std::optional<turbo_span> open;
            std::string open_entity;
            float open_sum = 0.0f;
            uint32_t open_words = 0;
            auto close = [&] {
                open->score = open_sum / static_cast<float>(open_words);
                spans.push_back(*open);
                open.reset();
            };
            for (const WordSpan &w : words[r]) {
                // encode_row drops every word that truncation cut, so this
                // holds by construction; a violation would read another
                // row's logits, so it is an error, not a silent skip.
                require(w.first_token + w.n_tokens <= seq, TURBO_E_INTERNAL,
                        "word span at column " + std::to_string(w.first_token) + " spans " +
                            std::to_string(w.n_tokens) + " tokens but the row holds " + std::to_string(seq));
                float score = 0.0f;
                uint32_t label = token_label(r, w.first_token, &score);
                if (agg == TURBO_AGGREGATE_MAX) {
                    for (uint32_t t = 1; t < w.n_tokens; ++t) {
                        float s2 = 0.0f;
                        const uint32_t l2 = token_label(r, w.first_token + t, &s2);
                        if (s2 > score) {
                            score = s2;
                            label = l2;
                        }
                    }
                }
                const std::string &name = labels[label];
                const bool outside = name == "O";
                if (agg == TURBO_AGGREGATE_NONE) {
                    if (!outside) {
                        spans.push_back(turbo_span{w.start, w.end, r, label, score, 0});
                    }
                    continue;
                }
                const std::string entity = entity_of(name);
                const bool begins = name.size() > 1 && (name[0] == 'B' || name[0] == 'U' || name[0] == 'S') && name[1] == '-';
                if (open && (outside || entity != open_entity || begins)) {
                    close();
                }
                if (outside) {
                    continue;
                }
                if (open) {
                    open->byte_end = w.end;
                    open_sum += score;
                    ++open_words;
                } else {
                    open = turbo_span{w.start, w.end, r, label, score, 0};
                    open_entity = entity;
                    open_sum = score;
                    open_words = 1;
                }
            }
            if (open) {
                close();
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Vtable functions
// ---------------------------------------------------------------------------

extern "C" {

static int32_t x_device_count(void *, uint32_t *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        *out = static_cast<uint32_t>(state().devices.size());
    });
}

static int32_t x_device_info(void *, uint32_t ordinal, turbo_device_info *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_device_info>("turbo_device_info", out->struct_size);
        const Device &d = device_at(ordinal);
        auto &core = state().core;
        turbo_device_info full{};
        full.struct_size = out->struct_size;
        full.kind = d.kind;
        full.ordinal = d.ordinal;
        full.caps = caps_of(d);
        // The CPU plugin runs on whatever CPU the host has, so the vendor is
        // read from the CPU; a GPU reports the OpenCL device's vendor. Only
        // the NPU plugin is Intel silicon by construction.
        Vendor vendor = d.kind == TURBO_DEVICE_NPU ? Vendor{0x8086, "Intel"} : cpu_vendor();
        std::string name;
        try {
            name = core.get_property(d.ov_name, ov::device::full_name);
        } catch (const std::exception &) {
            name = d.ov_name;
        }
        if (d.kind == TURBO_DEVICE_GPU || d.kind == TURBO_DEVICE_IGPU) {
            try {
                full.memory_total = core.get_property(d.ov_name, ov::intel_gpu::device_total_mem_size);
            } catch (const std::exception &) {
                full.memory_total = 0;
            }
            try {
                auto shared = core.get_default_context(d.ov_name).as<ov::intel_gpu::ocl::ClContext>();
                cl::Context c(shared.get(), true);
                const auto devs = c.getInfo<CL_CONTEXT_DEVICES>();
                if (!devs.empty()) {
                    const cl::Device cl_dev(devs.front());
                    put_str(full.driver_version, cl_dev.getInfo<CL_DRIVER_VERSION>());
                    // OpenCL string properties carry their terminator.
                    std::string cl_vendor = cl_dev.getInfo<CL_DEVICE_VENDOR>();
                    while (!cl_vendor.empty() && cl_vendor.back() == '\0') {
                        cl_vendor.pop_back();
                    }
                    vendor = Vendor{cl_dev.getInfo<CL_DEVICE_VENDOR_ID>(), cl_vendor};
                }
            } catch (const std::exception &e) {
                put_str(full.driver_version, std::string("unavailable: ") + e.what());
                vendor = Vendor{0, "unknown"};
            }
        }
        full.vendor_id = vendor.id;
        put_str(full.name, name + " (" + d.ov_name + ")");
        put_str(full.vendor, vendor.name);
        put_str(full.provider_id, std::string("openvino"));
        put_str(full.provider_version, std::string("2.0.0-alpha.0"));
        put_str(full.runtime_version, std::string(ov::get_openvino_version().buildNumber));
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_capability(void *, uint32_t ordinal, uint32_t task, uint32_t modality, turbo_capability *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_capability>("turbo_capability", out->struct_size);
        const Device &d = device_at(ordinal);
        turbo_capability full{};
        full.struct_size = out->struct_size;
        if (offers(d, task, modality)) {
            full.dtype = TURBO_DTYPE_F32;
            full.reference_dtype = TURBO_DTYPE_F32;
            full.deterministic = deterministic_device(d) ? 1 : 0;
            // SUPPORTED needs a conformance receipt, a precision receipt and a
            // matched-native benchmark from a named machine (AGENTS.md rule
            // 7); the cell names them. Embeddings on an Intel GPU have all
            // three from krick-1 (Battlemage B70, 2026-09-22). Every other
            // cell is EXPERIMENTAL until its receipts exist.
            if (task == TURBO_TASK_EMBED && d.kind == TURBO_DEVICE_GPU) {
                full.status = TURBO_CAP_SUPPORTED;
                full.cosine_floor = 0.9995f;
                put_str(full.notes, "receipts openvino-krick-1-2026-09-22, openvino-minilm-2026-09-21, compare-openvino-krick-1-gpu-embed-2026-09-22b");
            } else {
                full.status = TURBO_CAP_EXPERIMENTAL;
                put_str(full.notes, std::string("openvino ") + d.ov_name + ": fused pooling/activation in graph; no matched-native benchmark for this cell yet");
            }
        } else {
            full.status = TURBO_CAP_UNSUPPORTED;
            put_str(full.notes, d.kind == TURBO_DEVICE_NPU ? std::string("NPU: planned (static shapes); not qualified")
                                                              : std::string("not offered by the openvino provider"));
        }
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_can_run(void *, uint32_t ordinal, turbo_text bundle_dir, uint32_t task, uint32_t modality, turbo_error *err) {
    return boundary(err, [&] {
        const Device &d = device_at(ordinal);
        require(offers(d, task, modality), TURBO_E_UNSUPPORTED_TASK,
                "openvino device " + d.ov_name + " does not offer task " + std::to_string(task) + " for modality " + std::to_string(modality));
        const Bundle b = Bundle::read(text_of(bundle_dir));
        require(b.artifact("openvino_ir") || b.artifact("onnx"), TURBO_E_BUNDLE_NO_ARTIFACT,
                "bundle `" + b.model_id + "` has no openvino_ir or onnx artifact");
        require(b.tokenizer_kind == "wordpiece", TURBO_E_UNSUPPORTED, "openvino provider needs a WordPiece tokenizer.json");
    });
}

static int32_t x_context_create(void *, uint32_t ordinal, const turbo_context_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
        if (desc != nullptr) {
            const turbo_context_desc d = read_prefix(desc, "turbo_context_desc");
            require(d.next == nullptr, TURBO_E_NOT_IMPLEMENTED, "external queue import is not implemented");
            reject_unknown(options_of(d.options, d.n_options, "context"), {}, "openvino context");
        }
        const Device &d = device_at(ordinal);
        require(d.kind != TURBO_DEVICE_NPU, TURBO_E_DEVICE_UNAVAILABLE, "NPU devices are not qualified in this provider");
        *out = new Context(d);
    });
}

static void x_context_release(void *ctx) { release<Context>(ctx); }

static int32_t x_buffer_alloc(void *ctx, const turbo_buffer_desc *desc, turbo_provider_buffer *out, turbo_error *err) {
    return boundary(err, [&] {
        require(ctx != nullptr && desc != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        const turbo_buffer_desc d = read_prefix(desc, "turbo_buffer_desc");
        check_size<turbo_provider_buffer>("turbo_provider_buffer", out->struct_size);
        require_receives(out->struct_size, TURBO_PC_FIELD_END(turbo_provider_buffer, handle), "turbo_provider_buffer", "handle");
        auto b = make_buffer(static_cast<Context *>(ctx), d);
        write_sized(out, describe(b.get()));
        b.release();
    });
}

static int32_t x_buffer_import(void *ctx, const turbo_buffer_desc *desc, const turbo_native_handle *handle,
                               turbo_provider_buffer *out, turbo_error *err) {
    return boundary(err, [&] {
        require(ctx != nullptr && desc != nullptr && handle != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT,
                "NULL argument");
        const turbo_buffer_desc d = read_prefix(desc, "turbo_buffer_desc");
        const turbo_native_handle h = read_prefix(handle, "turbo_native_handle");
        check_size<turbo_provider_buffer>("turbo_provider_buffer", out->struct_size);
        require_receives(out->struct_size, TURBO_PC_FIELD_END(turbo_provider_buffer, handle), "turbo_provider_buffer", "handle");
        auto b = import_buffer(static_cast<Context *>(ctx), d, h);
        write_sized(out, describe(b.get()));
        b.release();
    });
}

static int32_t x_buffer_read(void *buf, void *dst, uint64_t bytes, turbo_error *err) {
    return boundary(err, [&] {
        auto *b = static_cast<Buffer *>(buf);
        require(b != nullptr && dst != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        require(bytes == b->bytes, TURBO_E_CAPACITY, "destination size " + std::to_string(bytes) + " does not match the buffer's " + std::to_string(b->bytes));
        if (b->device) {
            b->ctx->queue.enqueueReadBuffer(b->cl_buf, CL_TRUE, 0, static_cast<size_t>(bytes), dst);
        } else {
            std::memcpy(dst, b->host, static_cast<size_t>(bytes));
        }
    });
}

static int32_t x_buffer_export(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err) {
    return boundary(err, [&] {
        auto *b = static_cast<Buffer *>(buf);
        require(b != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        check_size<turbo_native_handle>("turbo_native_handle", out->struct_size);
        turbo_native_handle full{};
        full.offset = 0;
        full.aux = 0;
        if (b->device) {
            require(kind == TURBO_HANDLE_CL_MEM, TURBO_E_UNSUPPORTED, "device buffers export TURBO_HANDLE_CL_MEM only");
            full.kind = TURBO_HANDLE_CL_MEM;
            full.handle = reinterpret_cast<uint64_t>(b->cl_buf.get());
            full.aux = reinterpret_cast<uint64_t>(b->ctx->cl.get());
        } else {
            require(kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED, "host buffers export TURBO_HANDLE_HOST_PTR only");
            full.kind = TURBO_HANDLE_HOST_PTR;
            full.handle = reinterpret_cast<uint64_t>(b->host);
        }
        write_sized(out, full);
    });
}

static void x_buffer_release(void *buf) { release<Buffer>(buf); }

static int32_t x_model_load(void *ctx, turbo_text bundle_dir, const turbo_model_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        require(ctx != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        *out = nullptr;
        std::map<std::string, std::string> opts;
        if (desc != nullptr) {
            const turbo_model_desc d = read_prefix(desc, "turbo_model_desc");
            opts = options_of(d.options, d.n_options, "model");
        }
        *out = load_model(static_cast<Context *>(ctx), text_of(bundle_dir), opts).release();
    });
}

static int32_t x_model_info(void *model, turbo_model_info *out, turbo_error *err) {
    return boundary(err, [&] {
        require(model != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        check_size<turbo_model_info>("turbo_model_info", out->struct_size);
        turbo_model_info full{};
        full.struct_size = out->struct_size;
        fill_model_info(*static_cast<Model *>(model), full);
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_model_label(void *model, uint32_t index, turbo_text *out, turbo_error *err) {
    return boundary(err, [&] {
        auto *m = static_cast<Model *>(model);
        require(m != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        require(index < m->labels.size(), TURBO_E_INVALID_ARGUMENT, "label index out of range");
        out->ptr = m->labels[index].data();
        out->len = m->labels[index].size();
    });
}

static void x_model_release(void *model) { release<Model>(model); }

static int32_t x_session_create(void *model, const turbo_session_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        auto *m = static_cast<Model *>(model);
        require(m != nullptr && desc != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        *out = nullptr;
        const turbo_session_desc d = read_prefix(desc, "turbo_session_desc");
        require(d.next == nullptr, TURBO_E_INVALID_ARGUMENT, "turbo_session_desc.next must be NULL");
        reject_unknown(options_of(d.options, d.n_options, "session"), {}, "openvino session");
        require(d.max_batch >= 1 && d.max_seq >= 2, TURBO_E_INVALID_ARGUMENT, "session needs max_batch >= 1 and max_seq >= 2");
        *out = new Session(m, d.max_batch, d.max_seq);
    });
}

static Session &sess(void *s) {
    require(s != nullptr, TURBO_E_INVALID_HANDLE, "session is NULL");
    return *static_cast<Session *>(s);
}

static int32_t x_session_write_text(void *s, const turbo_text *texts, uint32_t count, const turbo_embed_options *opts, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(S.model->kind == Kind::Embedding, TURBO_E_UNSUPPORTED_TASK, "write_text needs an embedding model");
        require(texts != nullptr, TURBO_E_INVALID_ARGUMENT, "texts is NULL");
        require(count >= 1 && count <= S.batch, TURBO_E_CAPACITY,
                "count " + std::to_string(count) + " is outside 1..max_batch (" + std::to_string(S.batch) + ")");
        turbo_embed_options o{};
        o.struct_size = sizeof(o);
        if (opts != nullptr) {
            check_size<turbo_embed_options>("turbo_embed_options", opts->struct_size);
            std::memcpy(&o, opts, std::min<size_t>(opts->struct_size, sizeof(o)));
        }
        checked_truncate(o.truncate, 2);
        checked_prompt_role(o.prompt_role, 4);
        S.eopts = o;
        const uint32_t budget = S.budget(o.max_tokens);
        const std::string &prefix = o.prompt_role == TURBO_PROMPT_QUERY ? S.model->bundle.prefix_query
                                    : o.prompt_role == TURBO_PROMPT_DOCUMENT ? S.model->bundle.prefix_document
                                                                             : std::string();
        for (uint32_t r = 0; r < count; ++r) {
            S.encode_row(r, prefix, texts[r], o.truncate, budget, nullptr);
        }
        S.n_rows = count;
        S.host_out_valid = false;
    });
}

static int32_t x_session_write_tokens(void *s, const turbo_token_batch *batch, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(batch != nullptr, TURBO_E_INVALID_ARGUMENT, "batch is NULL");
        const turbo_token_batch tb = read_prefix(batch, "turbo_token_batch");
        check_token_batch(tb);
        require(tb.batch >= 1 && tb.batch <= S.batch && tb.seq >= 1 && tb.seq <= S.seq, TURBO_E_CAPACITY, "token batch exceeds the session shape");
        const uint32_t stride = tb.row_stride == 0 ? tb.seq : tb.row_stride;
        // Every id (and type) is checked here, so a bad token is a write-time
        // argument error and never a silent row or a fault inside a run. The
        // message is built only for the failing id: a `require` with a
        // string argument builds its message on every call, which cost about
        // 1 us per token on this path (measured on krick, 2026-09-22).
        const int32_t n_ids = wordpiece_vocab_size(S.model->vocab);
        for (uint32_t r = 0; r < tb.batch; ++r) {
            for (uint32_t c = 0; c < tb.seq; ++c) {
                const int32_t id = tb.ids[static_cast<size_t>(r) * stride + c];
                if (id < 0 || id >= n_ids) {
                    fail(TURBO_E_INVALID_ARGUMENT, "row " + std::to_string(r) + " column " + std::to_string(c) + ": token id " +
                                                       std::to_string(id) + " is outside the " + std::to_string(n_ids) + "-entry vocabulary");
                }
            }
        }
        const int32_t pad = wordpiece_pad_id(S.model->vocab);
        for (uint32_t r = 0; r < tb.batch; ++r) {
            const size_t src = static_cast<size_t>(r) * stride;
            const size_t dst = static_cast<size_t>(r) * S.seq;
            std::memcpy(S.ids.data() + dst, tb.ids + src, tb.seq * 4);
            std::memcpy(S.mask.data() + dst, tb.mask + src, tb.seq * 4);
            for (uint32_t c = tb.seq; c < S.seq; ++c) {
                S.ids[dst + c] = pad;
                S.mask[dst + c] = 0;
            }
            if (S.model->has_types) {
                if (tb.types != nullptr) {
                    std::memcpy(S.types.data() + dst, tb.types + src, tb.seq * 4);
                    std::fill(S.types.begin() + dst + tb.seq, S.types.begin() + dst + S.seq, 0);
                } else {
                    std::fill(S.types.begin() + dst, S.types.begin() + dst + S.seq, 0);
                }
            }
            S.words[r].clear();
        }
        S.n_rows = tb.batch;
        S.host_out_valid = false;
    });
}

static int32_t x_session_write_pairs(void *s, const turbo_text *query, const turbo_text *docs, uint32_t count, const turbo_rerank_options *opts, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(S.model->kind == Kind::Reranker, TURBO_E_UNSUPPORTED_TASK, "write_pairs needs a reranker model");
        require(query != nullptr && docs != nullptr, TURBO_E_INVALID_ARGUMENT, "query and docs must not be NULL");
        require(count >= 1 && count <= S.batch, TURBO_E_CAPACITY,
                "count " + std::to_string(count) + " is outside 1..max_batch (" + std::to_string(S.batch) + ")");
        turbo_rerank_options o{};
        o.struct_size = sizeof(o);
        if (opts != nullptr) {
            check_size<turbo_rerank_options>("turbo_rerank_options", opts->struct_size);
            std::memcpy(&o, opts, std::min<size_t>(opts->struct_size, sizeof(o)));
        }
        require(o.raw_scores == 0 || S.model->activation == "none", TURBO_E_UNSUPPORTED_OPTION,
                "raw_scores needs logits, but this bundle's `" + S.model->activation +
                    "` activation is fused into the compiled graph and the openvino provider compiles one graph per "
                    "session",
                6);
        S.ropts = o;
        const uint32_t budget = S.budget(o.max_tokens);
        // Pair truncation policies, mapped one to one onto the packer:
        // MODEL is the tokenizer's own longest-first rule, RIGHT truncates
        // from the right of the pair, which for a cross-encoder means the
        // query is kept whole and the document is cut, NONE fails instead of
        // dropping tokens, and LEFT has no packer equivalent (it would drop
        // the [CLS] and the query) so it is rejected naming the field.
        uint32_t trunc = 0;
        switch (checked_truncate(o.truncate, 2)) {
        case TURBO_TRUNCATE_MODEL:
            trunc = WORDPIECE_TRUNC_LONGEST_FIRST;
            break;
        case TURBO_TRUNCATE_RIGHT:
            trunc = WORDPIECE_TRUNC_QUERY_PRIORITY;
            break;
        case TURBO_TRUNCATE_NONE:
            trunc = WORDPIECE_TRUNC_ERROR;
            break;
        default:
            fail(TURBO_E_UNSUPPORTED_OPTION,
                 "left truncation of query/document pairs is not offered: the packer truncates from the right of the "
                 "pair only",
                 2);
        }
        const std::string q = text_of(*query);
        for (uint32_t r = 0; r < count; ++r) {
            const size_t base = static_cast<size_t>(r) * S.seq;
            const std::string d = text_of(docs[r]);
            int32_t *types_row = S.model->has_types ? S.types.data() + base : S.types_scratch.data();
            const int rc = wordpiece_pack_pair(S.model->vocab, q.data(), q.size(), d.data(), d.size(), S.ids.data() + base,
                                               S.mask.data() + base, types_row, S.pos_scratch.data(), S.seq, S.seq, 4, trunc,
                                               budget);
            require(rc != WORDPIECE_ERR_TOO_LONG, TURBO_E_CAPACITY,
                    "pair " + std::to_string(r) + " does not fit the budget of " + std::to_string(budget) + " tokens and truncation is NONE");
            require(rc != WORDPIECE_ERR_INVALID_ARGUMENT, TURBO_E_INVALID_UTF8, "pair " + std::to_string(r) + " is not valid UTF-8");
            require(rc == WORDPIECE_OK, TURBO_E_INTERNAL, "pair packer failed with status " + std::to_string(rc));
        }
        S.n_rows = count;
        S.host_out_valid = false;
    });
}

static int32_t x_session_write_text_classify(void *s, const turbo_text *texts, uint32_t count, const turbo_classify_options *opts, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(S.model->kind == Kind::Classifier || S.model->kind == Kind::TokenClassifier, TURBO_E_UNSUPPORTED_TASK, "write_text_classify needs a classifier model");
        require(texts != nullptr, TURBO_E_INVALID_ARGUMENT, "texts is NULL");
        require(count >= 1 && count <= S.batch, TURBO_E_CAPACITY,
                "count " + std::to_string(count) + " is outside 1..max_batch (" + std::to_string(S.batch) + ")");
        turbo_classify_options o{};
        o.struct_size = sizeof(o);
        if (opts != nullptr) {
            check_size<turbo_classify_options>("turbo_classify_options", opts->struct_size);
            std::memcpy(&o, opts, std::min<size_t>(opts->struct_size, sizeof(o)));
        }
        checked_truncate(o.truncate, 2);
        checked_aggregation(o.aggregation, 4);
        // The activation is fused into the compiled graph, so logits are
        // only available when the bundle declares no activation at all.
        require(o.raw_scores == 0 || (S.model->kind == Kind::Classifier && S.model->activation == "none"),
                TURBO_E_UNSUPPORTED_OPTION,
                S.model->kind == Kind::TokenClassifier
                    ? std::string("raw_scores is not offered for token classification: the per-token softmax is fused "
                                  "into the compiled graph")
                    : "raw_scores needs logits, but this bundle's `" + S.model->activation +
                          "` activation is fused into the compiled graph and the openvino provider compiles one graph "
                          "per session",
                5);
        S.copts = o;
        const uint32_t budget = S.budget(o.max_tokens);
        const bool want_words = S.model->kind == Kind::TokenClassifier;
        for (uint32_t r = 0; r < count; ++r) {
            S.encode_row(r, std::string(), texts[r], o.truncate, budget, want_words ? &S.words[r] : nullptr);
        }
        S.n_rows = count;
        S.host_out_valid = false;
    });
}

static int32_t x_session_run(void *s, const turbo_run_options *opts, turbo_provider_result *out, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_provider_result>("turbo_provider_result", out->struct_size);
        if (opts != nullptr) {
            const turbo_run_options o = read_prefix(opts, "turbo_run_options");
            reject_unknown(options_of(o.params, o.n_params, "run"), {}, "openvino run");
        }
        require(S.n_rows > 0, TURBO_E_INVALID_STATE, "no inputs written");
        write_sized(out, S.run());
    });
}

static int32_t x_session_stats(void *s, turbo_session_stats *out, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_session_stats>("turbo_session_stats", out->struct_size);
        turbo_session_stats full{};
        full.struct_size = out->struct_size;
        full.runs = S.runs;
        full.host_allocs = UINT64_MAX; // not counted: the result path allocates and the provider keeps no tally
        full.h2d_bytes = S.h2d;
        full.d2h_bytes = S.d2h;
        full.input_bytes = (S.ids.size() + S.mask.size() + S.types.size()) * 4;
        full.output_bytes = S.host_out.size() * 4;
        // WordPiece encodes into preallocated rows; OpenVINO's own request
        // allocations are not observable from here.
        full.provider_allocs = UINT64_MAX;
        std::memcpy(out, &full, out->struct_size);
    });
}

static void x_session_release(void *s) { release<Session>(s); }

static const turbo_provider_vtbl g_vtbl = {
    sizeof(turbo_provider_vtbl),
    TURBO_PROVIDER_ABI_VERSION,
    "openvino",
    "2.0.0-alpha.0",
    nullptr,
    x_device_count,
    x_device_info,
    x_capability,
    x_can_run,
    x_context_create,
    x_context_release,
    x_buffer_alloc,
    x_buffer_import,
    x_buffer_read,
    x_buffer_export,
    x_buffer_release,
    x_model_load,
    x_model_info,
    x_model_label,
    nullptr, // model_io_info: no generic RUN models
    x_model_release,
    x_session_create,
    x_session_write_text,
    x_session_write_tokens,
    x_session_write_pairs,
    x_session_write_text_classify,
    nullptr, // session_bind: no generic RUN models
    x_session_run,
    x_session_stats,
    x_session_release,
    nullptr, // generation_*: this provider does not generate
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

__attribute__((visibility("default"))) const turbo_provider_vtbl *turbo_provider_get(uint32_t core_abi_version) {
    if (core_abi_version != TURBO_PROVIDER_ABI_VERSION) {
        return nullptr;
    }
    return &g_vtbl;
}

} // extern "C"
} // namespace turbo_ov
