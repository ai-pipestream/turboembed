// SPDX-License-Identifier: Apache-2.0
//
// Turbo Hailo provider: text embeddings on Hailo-8 / Hailo-8L accelerators
// through the HailoRT C API (`hailo/hailort.h`, the 4.x line shipped by the
// Raspberry Pi `hailo-all` packages).
//
// Split pipeline (the public HEFs compile only the transformer encoder body,
// at one fixed shape, batch 1; the vocabulary gather does not fit the
// Dataflow Compiler):
//
//   text --> WordPiece (host, native/wordpiece)
//        --> word-embedding rows gathered from the bundle's `hailo_tables`
//            artifact (host)
//        --> encoder body on the NPU: model.hef through vstreams
//            (one hidden-state input [seq, dim]; the official Model Zoo HEFs
//            add an additive attention-bias input [seq, seq])
//        --> masked pooling, L2 normalization, output_dim (host)
//
// Front ends, inferred for two-input HEFs or named by the `front_end` model
// option (a single-input HEF must name one):
//   `word`            the official Hailo Model Zoo contract: the raw word
//                     embedding row at every position (the PAD row on
//                     padding); position and token-type embeddings and the
//                     embeddings LayerNorm run inside the HEF.
//   `bert_embeddings` community single-input HEFs: the host computes
//                     word + position + token-type(0) + LayerNorm for the
//                     live tokens and zero-fills the padding.
//
// The HEF is quantized (INT8 on the NPU; HailoRT dequantizes to f32 in the
// vstream). Absolute cosine against an FP32 reference is not preserved,
// ranking is; the capability cell reports the measured cosine floor and the
// stage placement says which stages run on the host. Nothing here falls
// back to the CPU: without a device, a driver, or a HEF for this chip, the
// call fails with the reason.
//
// Threading: HailoRT vstreams are not reentrant, and one model owns one set
// of vstreams, so sessions of the same model serialize their runs on the
// model's mutex. The core already serializes calls on one session.

#include "turbo/turbo_provider.h"
#include "turbo/turbo_types.h"
#include "turbo_provider_common.hpp"
#include "wordpiece.h"

#include <hailo/hailort.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace turbo_hailo {
using namespace turbo_pc; // NOLINT(google-build-using-namespace)

constexpr const char *kProviderId = "hailo";
constexpr const char *kProviderVersion = "2.0.0-alpha.0";

/// Measured cosine floor of the INT8 MiniLM HEF against the FP32 ONNX
/// reference vectors (testdata/receipts/turbo/hailo-*.json). The live
/// embedding suite gates on the value the capability cell reports, so it
/// must not be raised without a receipt that shows the new floor.
constexpr float kCosineFloorVsF32 = 0.30f;

constexpr size_t kMaxDevices = 8;
constexpr size_t kMaxNetworkGroups = HAILO_MAX_NETWORK_GROUPS;
constexpr size_t kMaxStreams = HAILO_MAX_STREAMS_COUNT;

// ---------------------------------------------------------------------------
// HailoRT status handling
// ---------------------------------------------------------------------------

[[noreturn]] void hailo_fail(hailo_status st, const std::string &what, int32_t code = TURBO_E_RUNTIME) {
    fail(code, what + ": " + hailo_get_status_message(st) + " (hailo_status " + std::to_string(st) + ")");
}

void hailo_check(hailo_status st, const std::string &what, int32_t code = TURBO_E_RUNTIME) {
    if (st != HAILO_SUCCESS) {
        hailo_fail(st, what, code);
    }
}

std::string arch_name(hailo_device_architecture_t a) {
    switch (a) {
    case HAILO_ARCH_HAILO8_A0:
        return "Hailo-8 A0";
    case HAILO_ARCH_HAILO8:
        return "Hailo-8";
    case HAILO_ARCH_HAILO8L:
        return "Hailo-8L";
    case HAILO_ARCH_HAILO15H:
        return "Hailo-15H";
    case HAILO_ARCH_HAILO15L:
        return "Hailo-15L";
    case HAILO_ARCH_HAILO15M:
        return "Hailo-15M";
    case HAILO_ARCH_HAILO10H:
        return "Hailo-10H";
    default:
        return "Hailo (architecture " + std::to_string(static_cast<int>(a)) + ")";
    }
}

std::string fixed_str(const char *s, size_t len_field, size_t cap) {
    const size_t n = std::min(len_field, cap);
    std::string out(s, n);
    while (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return out;
}

// ---------------------------------------------------------------------------
// Provider state and devices
// ---------------------------------------------------------------------------

struct Device {
    uint32_t ordinal = 0;
    hailo_device_id_t id{};     // BDF of the PCIe device, as HailoRT names it
    std::string id_str;
    bool identified = false;    // identity read at scan time
    std::string identify_error; // why not, when `identified` is false
    hailo_device_architecture_t arch = HAILO_ARCH_MAX_ENUM;
    std::string board_name;
    std::string firmware;       // "major.minor.revision"
};

struct State {
    std::vector<Device> devices;
    std::string scan_error; // non-empty when the scan itself failed
    std::string library_version;

    State() {
        hailo_version_t v{};
        if (hailo_get_library_version(&v) == HAILO_SUCCESS) {
            library_version = std::to_string(v.major) + "." + std::to_string(v.minor) + "." + std::to_string(v.revision);
        } else {
            library_version = "unknown";
        }
        hailo_device_id_t ids[kMaxDevices];
        size_t count = kMaxDevices;
        const hailo_status st = hailo_scan_devices(nullptr, ids, &count);
        if (st != HAILO_SUCCESS) {
            scan_error = std::string("hailo_scan_devices failed: ") + hailo_get_status_message(st) +
                         " (is the hailo PCIe driver loaded? `sudo apt install dkms hailo-all` on Raspberry Pi OS)";
            return;
        }
        for (size_t i = 0; i < count; ++i) {
            Device d;
            d.ordinal = static_cast<uint32_t>(i);
            d.id = ids[i];
            d.id_str = fixed_str(ids[i].id, HAILO_MAX_DEVICE_ID_LENGTH, HAILO_MAX_DEVICE_ID_LENGTH);
            hailo_device dev = nullptr;
            hailo_status ds = hailo_create_device_by_id(&d.id, &dev);
            if (ds != HAILO_SUCCESS) {
                d.identify_error = std::string("hailo_create_device_by_id failed: ") + hailo_get_status_message(ds);
            } else {
                hailo_device_identity_t ident{};
                ds = hailo_identify(dev, &ident);
                if (ds != HAILO_SUCCESS) {
                    d.identify_error = std::string("hailo_identify failed: ") + hailo_get_status_message(ds);
                } else {
                    d.identified = true;
                    d.arch = ident.device_architecture;
                    d.board_name = fixed_str(ident.board_name, ident.board_name_length, HAILO_MAX_BOARD_NAME_LENGTH);
                    d.firmware = std::to_string(ident.fw_version.major) + "." + std::to_string(ident.fw_version.minor) +
                                 "." + std::to_string(ident.fw_version.revision);
                }
                (void)hailo_release_device(dev);
            }
            devices.push_back(std::move(d));
        }
    }
};

State &state() {
    static State s;
    return s;
}

const Device &device_at(uint32_t ordinal) {
    auto &s = state();
    require(s.scan_error.empty(), TURBO_E_DEVICE_UNAVAILABLE, s.scan_error);
    require(ordinal < s.devices.size(), TURBO_E_DEVICE_NOT_FOUND,
            "hailo provider has no device ordinal " + std::to_string(ordinal) + " (" +
                std::to_string(s.devices.size()) + " devices)");
    return s.devices[ordinal];
}

constexpr uint64_t kCaps = TURBO_CAP_DETERMINISTIC | TURBO_CAP_HOST_PTR_IMPORT | TURBO_CAP_OPT_TRUNCATE |
                           TURBO_CAP_OPT_MAX_TOKENS | TURBO_CAP_OPT_PROMPT_ROLE | TURBO_CAP_OPT_NORMALIZE |
                           TURBO_CAP_OPT_POOLING_OVERRIDE | TURBO_CAP_OPT_OUTPUT_DIM;

bool offers(const Device &d, uint32_t task, uint32_t modality) {
    return d.identified && task == TURBO_TASK_EMBED && modality == TURBO_MODALITY_TEXT;
}

template <typename T>
void release(void *p) noexcept {
    try {
        delete static_cast<T *>(p);
    } catch (...) {
    }
}

// ---------------------------------------------------------------------------
// Option enumerations: unknown values are errors naming the field, never a
// default.
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

uint32_t checked_normalize(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_NORMALIZE_MODEL:
    case TURBO_NORMALIZE_NONE:
    case TURBO_NORMALIZE_L2:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "normalize " + std::to_string(v) + " is not a TURBO_NORMALIZE_* value", field);
    }
}

uint32_t checked_pooling(uint32_t v, uint32_t field) {
    switch (v) {
    case TURBO_POOLING_MODEL:
    case TURBO_POOLING_MEAN:
    case TURBO_POOLING_CLS:
    case TURBO_POOLING_LAST:
        return v;
    default:
        fail(TURBO_E_INVALID_ENUM, "pooling " + std::to_string(v) + " is not a TURBO_POOLING_* value", field);
    }
}

// ---------------------------------------------------------------------------
// Context: one virtual device bound to exactly one physical device. The
// scheduler stays on (HailoRT's default) so several models can be
// configured on one context; HailoRT then owns network-group activation.
// ---------------------------------------------------------------------------

struct Context {
    Device dev;
    hailo_vdevice vdevice = nullptr;

    explicit Context(const Device &d) : dev(d) {
        require(d.identified, TURBO_E_DEVICE_UNAVAILABLE,
                "hailo device " + d.id_str + " could not be identified at scan time: " + d.identify_error);
        hailo_vdevice_params_t params{};
        hailo_check(hailo_init_vdevice_params(&params), "hailo_init_vdevice_params");
        params.device_count = 1;
        params.device_ids = &dev.id;
        hailo_check(hailo_create_vdevice(&params, &vdevice), "hailo_create_vdevice on " + d.id_str,
                    TURBO_E_DEVICE_UNAVAILABLE);
    }

    ~Context() {
        if (vdevice != nullptr) {
            (void)hailo_release_vdevice(vdevice);
        }
    }

    Context(const Context &) = delete;
    Context &operator=(const Context &) = delete;
};

// ---------------------------------------------------------------------------
// Buffers: host memory only. The NPU is fed through HailoRT's own DMA
// pipeline, so there is no device placement to hand out.
// ---------------------------------------------------------------------------

struct Buffer {
    Context *ctx = nullptr;
    turbo_buffer_desc desc{};
    void *host = nullptr;
    uint64_t bytes = 0;
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
        fail(TURBO_E_UNSUPPORTED_DTYPE, "dtype " + std::to_string(dtype) + " is not supported by the hailo provider");
    }
}

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
    require(in.placement == TURBO_PLACE_HOST, TURBO_E_UNSUPPORTED_PLACEMENT,
            "hailo provider allocates TURBO_PLACE_HOST only; the NPU is fed through HailoRT's DMA pipeline and has no "
            "caller-visible device memory");
    const size_t sz = (static_cast<size_t>(bytes) + 63) & ~static_cast<size_t>(63);
    b->host = std::aligned_alloc(64, sz);
    require(b->host != nullptr, TURBO_E_OUT_OF_MEMORY, "host allocation of " + std::to_string(bytes) + " bytes failed");
    std::memset(b->host, 0, sz);
    return b;
}

std::unique_ptr<Buffer> import_buffer(Context *ctx, const turbo_buffer_desc &in, const turbo_native_handle &h) {
    require(h.kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED,
            "hailo provider imports TURBO_HANDLE_HOST_PTR only; handle kind " + std::to_string(h.kind) +
                " is not offered");
    require(in.placement == TURBO_PLACE_HOST, TURBO_E_UNSUPPORTED_PLACEMENT,
            "an imported host pointer is TURBO_PLACE_HOST; placement " + std::to_string(in.placement) +
                " cannot describe caller memory");
    require(h.offset == 0, TURBO_E_UNSUPPORTED, "hailo provider imports handles with offset 0 only");
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
// Embedding tables (`hailo_tables` artifact)
//
// Little-endian header followed by fp32 arrays:
//   u32 magic "TEMB" | u32 version (1) | u32 vocab_rows | u32 max_pos |
//   u32 dim | f32 layer_norm_eps |
//   f32 word[vocab_rows][dim] | f32 pos[max_pos][dim] | f32 token_type[2][dim]
//   | f32 ln_gamma[dim] | f32 ln_beta[dim]
// Written by scripts/export-hailo-tables.py from the checkpoint the HEF was
// compiled from; the core has verified the file's hash before this runs.
// ---------------------------------------------------------------------------

constexpr uint32_t kTablesMagic = 0x54454d42u; // "TEMB"
constexpr uint32_t kTablesVersion = 1u;

struct Tables {
    uint32_t vocab_rows = 0;
    uint32_t max_pos = 0;
    uint32_t dim = 0;
    float ln_eps = 1e-12f;
    std::vector<float> word, pos, token_type, ln_gamma, ln_beta;

    static bool read_exact(std::FILE *f, void *dst, size_t bytes) {
        return bytes == 0 || std::fread(dst, 1, bytes, f) == bytes;
    }

    static Tables load(const std::string &path) {
        std::FILE *f = std::fopen(path.c_str(), "rb");
        require(f != nullptr, TURBO_E_BUNDLE_NOT_FOUND, "cannot open hailo_tables artifact " + path);
        struct Closer {
            std::FILE *f;
            ~Closer() { std::fclose(f); }
        } closer{f};
        uint32_t header[5] = {0, 0, 0, 0, 0};
        Tables t;
        require(read_exact(f, header, sizeof(header)) && read_exact(f, &t.ln_eps, sizeof(t.ln_eps)),
                TURBO_E_BUNDLE_INVALID, path + ": short read on the embedding tables header");
        require(header[0] == kTablesMagic, TURBO_E_BUNDLE_INVALID, path + ": not an embedding tables file (bad magic)");
        require(header[1] == kTablesVersion, TURBO_E_BUNDLE_INVALID,
                path + ": embedding tables version " + std::to_string(header[1]) + "; this provider reads version 1");
        t.vocab_rows = header[2];
        t.max_pos = header[3];
        t.dim = header[4];
        require(t.vocab_rows > 0 && t.max_pos > 0 && t.dim > 0 && t.vocab_rows <= (1u << 22) && t.dim <= 8192 &&
                    t.max_pos <= 65536,
                TURBO_E_BUNDLE_INVALID,
                path + ": implausible table shape vocab=" + std::to_string(t.vocab_rows) + " max_pos=" +
                    std::to_string(t.max_pos) + " dim=" + std::to_string(t.dim));
        t.word.resize(static_cast<size_t>(t.vocab_rows) * t.dim);
        t.pos.resize(static_cast<size_t>(t.max_pos) * t.dim);
        t.token_type.resize(2u * t.dim);
        t.ln_gamma.resize(t.dim);
        t.ln_beta.resize(t.dim);
        const bool ok = read_exact(f, t.word.data(), t.word.size() * sizeof(float)) &&
                        read_exact(f, t.pos.data(), t.pos.size() * sizeof(float)) &&
                        read_exact(f, t.token_type.data(), t.token_type.size() * sizeof(float)) &&
                        read_exact(f, t.ln_gamma.data(), t.dim * sizeof(float)) &&
                        read_exact(f, t.ln_beta.data(), t.dim * sizeof(float));
        require(ok, TURBO_E_BUNDLE_INVALID, path + ": truncated embedding tables");
        unsigned char extra = 0;
        require(std::fread(&extra, 1, 1, f) == 0, TURBO_E_BUNDLE_INVALID,
                path + ": trailing bytes after the embedding tables");
        return t;
    }
};

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

enum class Pool { Mean, Cls, Last };
enum class FrontEnd { Word, BertEmbeddings };

struct Model {
    Context *ctx = nullptr;
    Bundle bundle;
    Pool pool = Pool::Mean;
    bool normalize = false;
    uint32_t dim = 0;
    uint32_t seq_len = 0;   // the HEF's fixed frame length
    uint32_t max_batch = 0; // host loops rows; the HEF is batch 1
    FrontEnd front_end = FrontEnd::Word;
    Tables tables;
    wordpiece_vocab *vocab = nullptr;

    hailo_hef hef = nullptr;
    hailo_configured_network_group group = nullptr; // owned by the vdevice
    hailo_input_vstream inputs[2] = {nullptr, nullptr};
    size_t n_inputs = 0;
    hailo_input_vstream hidden = nullptr;
    hailo_input_vstream mask = nullptr; // nullptr on single-input HEFs
    hailo_output_vstream output = nullptr;
    size_t hidden_bytes = 0, mask_bytes = 0, output_bytes = 0;
    std::mutex run_mu; // vstreams are not reentrant
    /// Non-empty after a vstream write or read failed mid-run: the pipeline
    /// may hold a frame the next read would return as this run's result,
    /// so every later run is refused until the model is reloaded.
    std::string poisoned;

    ~Model() {
        if (n_inputs > 0) {
            (void)hailo_release_input_vstreams(inputs, n_inputs);
        }
        if (output != nullptr) {
            (void)hailo_release_output_vstreams(&output, 1);
        }
        if (hef != nullptr) {
            (void)hailo_release_hef(hef);
        }
        if (vocab != nullptr) {
            wordpiece_vocab_destroy(vocab);
        }
    }

    Model() = default;
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
};

/// Configure the HEF on the context's vdevice and open its vstreams.
void open_network(Model &m, const std::string &hef_path) {
    hailo_check(hailo_create_hef_file(&m.hef, hef_path.c_str()), "hailo_create_hef_file " + hef_path,
                TURBO_E_BUNDLE_INVALID);
    hailo_configure_params_t params{};
    hailo_check(hailo_init_configure_params_by_vdevice(m.hef, m.ctx->vdevice, &params),
                "hailo_init_configure_params_by_vdevice");
    hailo_configured_network_group groups[kMaxNetworkGroups];
    size_t n_groups = kMaxNetworkGroups;
    const hailo_status cs = hailo_configure_vdevice(m.ctx->vdevice, m.hef, &params, groups, &n_groups);
    if (cs != HAILO_SUCCESS) {
        fail(TURBO_E_UNSUPPORTED, std::string("hailo_configure_vdevice failed: ") + hailo_get_status_message(cs) +
                                      " (a HEF is compiled for one chip: this device is " +
                                      arch_name(m.ctx->dev.arch) + "; hailo8, hailo8l, and hailo10h HEFs are not " +
                                      "interchangeable)");
    }
    require(n_groups == 1, TURBO_E_UNSUPPORTED,
            "the HEF holds " + std::to_string(n_groups) + " network groups; the encoder HEF must hold exactly 1");
    m.group = groups[0];

    hailo_input_vstream_params_by_name_t in_params[kMaxStreams];
    size_t n_in = kMaxStreams;
    hailo_check(hailo_hef_make_input_vstream_params(m.hef, nullptr, false, HAILO_FORMAT_TYPE_FLOAT32, in_params, &n_in),
                "hailo_hef_make_input_vstream_params");
    hailo_output_vstream_params_by_name_t out_params[kMaxStreams];
    size_t n_out = kMaxStreams;
    hailo_check(hailo_make_output_vstream_params(m.group, false, HAILO_FORMAT_TYPE_FLOAT32, out_params, &n_out),
                "hailo_make_output_vstream_params");
    require(n_in >= 1 && n_in <= 2 && n_out == 1, TURBO_E_UNSUPPORTED,
            "the encoder HEF must have 1 or 2 input vstreams and 1 output; this one has " + std::to_string(n_in) +
                " inputs and " + std::to_string(n_out) + " outputs");

    hailo_check(hailo_create_input_vstreams(m.group, in_params, n_in, m.inputs), "hailo_create_input_vstreams");
    m.n_inputs = n_in;
    hailo_output_vstream outs[1];
    hailo_check(hailo_create_output_vstreams(m.group, out_params, 1, outs), "hailo_create_output_vstreams");
    m.output = outs[0];

    // Tell the inputs apart by frame size: hidden is [seq, dim] f32, the
    // attention bias is [seq, seq] f32. seq is not known yet, so classify by
    // divisibility: the hidden frame is a multiple of dim*4 whose quotient
    // (seq) squared times 4 matches the other frame, if there is one.
    size_t frames[2] = {0, 0};
    for (size_t i = 0; i < n_in; ++i) {
        hailo_check(hailo_get_input_vstream_frame_size(m.inputs[i], &frames[i]), "hailo_get_input_vstream_frame_size");
    }
    hailo_check(hailo_get_output_vstream_frame_size(m.output, &m.output_bytes), "hailo_get_output_vstream_frame_size");
    const size_t row = static_cast<size_t>(m.dim) * sizeof(float);
    require(m.output_bytes % row == 0 && m.output_bytes > 0, TURBO_E_BUNDLE_INVALID,
            "HEF output frame is " + std::to_string(m.output_bytes) + " bytes, not a multiple of dim " +
                std::to_string(m.dim) + " * 4; the encoder must end at the token-level hidden state [seq, dim]");
    m.seq_len = static_cast<uint32_t>(m.output_bytes / row);
    const size_t expect_hidden = static_cast<size_t>(m.seq_len) * row;
    const size_t expect_mask = static_cast<size_t>(m.seq_len) * m.seq_len * sizeof(float);
    require(expect_hidden != expect_mask, TURBO_E_UNSUPPORTED,
            "dim equals seq_len (" + std::to_string(m.dim) + "), which makes the hidden and mask inputs ambiguous");
    for (size_t i = 0; i < n_in; ++i) {
        if (frames[i] == expect_hidden && m.hidden == nullptr) {
            m.hidden = m.inputs[i];
            m.hidden_bytes = frames[i];
        } else if (frames[i] == expect_mask && m.mask == nullptr) {
            m.mask = m.inputs[i];
            m.mask_bytes = frames[i];
        } else {
            fail(TURBO_E_BUNDLE_INVALID, "HEF input " + std::to_string(i) + " has a " + std::to_string(frames[i]) +
                                             " byte frame; expected the hidden state [" + std::to_string(m.seq_len) +
                                             ", " + std::to_string(m.dim) + "] f32 (" + std::to_string(expect_hidden) +
                                             " bytes) or the attention bias [" + std::to_string(m.seq_len) + ", " +
                                             std::to_string(m.seq_len) + "] f32 (" + std::to_string(expect_mask) +
                                             " bytes)");
        }
    }
    require(m.hidden != nullptr, TURBO_E_BUNDLE_INVALID,
            "the HEF has no hidden-state input of [" + std::to_string(m.seq_len) + ", " + std::to_string(m.dim) +
                "] f32");
}

std::unique_ptr<Model> load_model(Context *ctx, const std::string &dir, const std::map<std::string, std::string> &opts) {
    reject_unknown(opts, {"front_end"}, "hailo model");
    auto m = std::make_unique<Model>();
    m->ctx = ctx;
    m->bundle = Bundle::read(dir);
    const Bundle &b = m->bundle;
    require(b.modality == "text", TURBO_E_UNSUPPORTED_MODALITY, "hailo provider serves text bundles only");
    require(b.kind == "embedding", TURBO_E_UNSUPPORTED_TASK,
            "hailo provider does not serve `" + b.kind + "` bundles (embedding only)");
    require(b.dim > 0, TURBO_E_BUNDLE_INVALID, "embedding bundle must declare contract.dim");
    m->dim = b.dim;
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
    require(b.fixed_shape, TURBO_E_BUNDLE_INVALID,
            "a HEF is compiled for one shape; the bundle must declare limits.fixed_shape (turbo-bundle import "
            "--fixed-shape --max-seq <frame length>)");
    m->max_batch = b.max_batch == 0 ? 32 : b.max_batch;

    const auto hef = b.artifact("hef");
    const auto tables = b.artifact("hailo_tables");
    if (!hef || !tables) {
        std::string have;
        for (const auto &[k, v] : b.artifacts) {
            have += (have.empty() ? "" : ", ") + k;
        }
        fail(TURBO_E_BUNDLE_NO_ARTIFACT, "bundle `" + b.model_id + "` needs both a `hef` and a `hailo_tables` artifact; it has: " +
                                             (have.empty() ? "(none)" : have));
    }

    require(!b.tokenizer_path.empty(), TURBO_E_BUNDLE_INVALID, "bundle has no tokenizer.json; the hailo provider tokenizes natively");
    require(b.tokenizer_kind == "wordpiece", TURBO_E_UNSUPPORTED,
            "hailo provider tokenizes WordPiece (BERT) natively; tokenizer kind `" + b.tokenizer_kind + "` is not supported");
    const int rc = wordpiece_vocab_load(b.tokenizer_path.c_str(), &m->vocab);
    require(rc == WORDPIECE_OK && m->vocab != nullptr, TURBO_E_BUNDLE_INVALID,
            "tokenizer.json is not a supported uncased BERT WordPiece configuration (wordpiece status " + std::to_string(rc) + ")");

    m->tables = Tables::load(*tables);
    require(m->tables.dim == m->dim, TURBO_E_BUNDLE_INVALID,
            "hailo_tables dim " + std::to_string(m->tables.dim) + " does not match contract.dim " + std::to_string(m->dim));
    require(b.vocab_size == 0 || m->tables.vocab_rows == b.vocab_size, TURBO_E_BUNDLE_INVALID,
            "hailo_tables has " + std::to_string(m->tables.vocab_rows) + " vocabulary rows but contract.vocab_size is " +
                std::to_string(b.vocab_size));
    const int32_t pad = wordpiece_pad_id(m->vocab);
    require(pad >= 0 && static_cast<uint32_t>(pad) < m->tables.vocab_rows, TURBO_E_BUNDLE_INVALID,
            "the tokenizer's [PAD] id is outside the hailo_tables vocabulary");

    open_network(*m, *hef);
    require(m->seq_len == b.max_seq, TURBO_E_BUNDLE_INVALID,
            "contract.max_seq is " + std::to_string(b.max_seq) + " but the HEF frame holds " + std::to_string(m->seq_len) +
                " tokens; import the bundle with --max-seq " + std::to_string(m->seq_len));

    // Front end: the official Model Zoo HEFs take the attention bias as a
    // second input and do the position/token-type/LayerNorm work inside;
    // single-input HEFs are the community export that expects the full BERT
    // embeddings from the host. The option overrides the inference.
    const auto fe = opts.find("front_end");
    if (fe == opts.end()) {
        // Only the two-input Model Zoo contract is inferred; it is the one
        // the receipts measure. A single-input HEF must say which front
        // end it expects, and `bert_embeddings` carries no receipt yet.
        require(m->mask != nullptr, TURBO_E_BUNDLE_INVALID,
                "the HEF has one input, so the provider cannot tell which front end it expects; load it with the "
                "model option front_end=word or front_end=bert_embeddings (the latter has no precision receipt)");
        m->front_end = FrontEnd::Word;
    } else if (fe->second == "word") {
        m->front_end = FrontEnd::Word;
    } else if (fe->second == "bert_embeddings") {
        m->front_end = FrontEnd::BertEmbeddings;
    } else {
        fail(TURBO_E_INVALID_ARGUMENT, "model option front_end must be `word` or `bert_embeddings`, not `" + fe->second + "`");
    }
    if (m->front_end == FrontEnd::BertEmbeddings) {
        require(m->tables.max_pos >= m->seq_len, TURBO_E_BUNDLE_INVALID,
                "hailo_tables has " + std::to_string(m->tables.max_pos) + " position rows; the HEF frame needs " +
                    std::to_string(m->seq_len));
    }
    return m;
}

void fill_model_info(const Model &m, turbo_model_info &out) {
    out.task = TURBO_TASK_EMBED;
    out.kind = TURBO_MODEL_EMBEDDING;
    out.modality = TURBO_MODALITY_TEXT;
    out.dim = m.dim;
    out.n_labels = 0;
    out.pooling = m.pool == Pool::Mean ? TURBO_POOLING_MEAN : m.pool == Pool::Cls ? TURBO_POOLING_CLS : TURBO_POOLING_LAST;
    out.normalize = m.normalize ? TURBO_NORMALIZE_L2 : TURBO_NORMALIZE_NONE;
    out.max_seq = m.seq_len;
    out.max_batch = m.max_batch;
    // Results are f32 (HailoRT dequantizes in the output vstream); the
    // encoder itself runs quantized, which the capability cell reports.
    out.dtype_used = TURBO_DTYPE_F32;
    out.stage_placement[TURBO_STAGE_TOKENIZE] = TURBO_STAGE_HOST;
    out.stage_placement[TURBO_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
    out.stage_placement[TURBO_STAGE_POOL] = TURBO_STAGE_HOST;
    out.stage_placement[TURBO_STAGE_NORMALIZE] = m.normalize ? TURBO_STAGE_HOST : TURBO_STAGE_UNUSED;
    out.stage_placement[TURBO_STAGE_POSTPROCESS] = TURBO_STAGE_UNUSED;
    out.fully_accelerated = 0;
    out.n_inputs = 0;
    out.n_outputs = 0;
    out.vocab_size = m.bundle.vocab_size;
    put_str(out.model_id, m.bundle.model_id);
    put_str(out.revision, m.bundle.revision);
    put_str(out.tokenizer_sha256, m.bundle.tokenizer_sha256);
    put_str(out.provider_id, std::string(kProviderId));
    put_str(out.prefix_query, m.bundle.prefix_query);
    put_str(out.prefix_document, m.bundle.prefix_document);
}

// ---------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------

struct Session {
    Model *model = nullptr;
    uint32_t batch = 0, seq = 0;      // session shape; seq <= model seq_len
    std::vector<int32_t> ids, mask;   // [batch * seq]
    std::vector<int32_t> scratch;     // left-truncation staging
    std::vector<float> in_frame;      // [seq_len * dim]
    std::vector<float> mask_frame;    // [seq_len * seq_len] on two-input HEFs
    std::vector<float> out_frame;     // [seq_len * dim]
    std::vector<float> host_out;      // [batch * dim]
    uint32_t n_rows = 0;
    turbo_embed_options eopts{};
    Buffer out_buf;
    turbo_provider_output outputs[1]{};
    std::string name0;
    std::atomic<bool> in_use{false}; // one caller at a time; see BusyGuard
    uint64_t runs = 0, h2d = 0, d2h = 0;

    Session(Model *m, uint32_t b, uint32_t s) : model(m), batch(b), seq(s) {
        require(s <= m->seq_len, TURBO_E_CAPACITY,
                "session max_seq " + std::to_string(s) + " exceeds the HEF frame of " + std::to_string(m->seq_len) + " tokens");
        require(b <= m->max_batch, TURBO_E_CAPACITY,
                "session max_batch " + std::to_string(b) + " exceeds the model's max_batch " + std::to_string(m->max_batch));
        const size_t n = static_cast<size_t>(b) * s;
        ids.assign(n, wordpiece_pad_id(m->vocab));
        mask.assign(n, 0);
        scratch.assign(static_cast<size_t>(s) * 16, 0);
        const size_t frame = static_cast<size_t>(m->seq_len) * m->dim;
        in_frame.assign(frame, 0.0f);
        out_frame.assign(frame, 0.0f);
        if (m->mask != nullptr) {
            mask_frame.assign(static_cast<size_t>(m->seq_len) * m->seq_len, 0.0f);
        }
        host_out.assign(static_cast<size_t>(b) * m->dim, 0.0f);
        out_buf.ctx = m->ctx;
        out_buf.host = host_out.data();
        out_buf.bytes = host_out.size() * sizeof(float);
        out_buf.owns_host = false;
        name0 = "embeddings";
    }

    uint32_t budget(uint32_t max_tokens) const {
        require(max_tokens <= seq, TURBO_E_CAPACITY,
                "max_tokens " + std::to_string(max_tokens) + " exceeds the session's max_seq " + std::to_string(seq), 3);
        const uint32_t b = max_tokens == 0 ? seq : max_tokens;
        require(b >= 2, TURBO_E_CAPACITY, "token budget must be at least 2 for [CLS] and [SEP]", 3);
        return b;
    }

    /// Tokenize text (with optional prefix) into row r: [CLS] prefix text [SEP] pad.
    void encode_row(uint32_t r, const std::string &prefix, turbo_text text, uint32_t truncate, uint32_t budget) {
        const size_t base = static_cast<size_t>(r) * seq;
        int32_t *row_ids = ids.data() + base;
        int32_t *row_mask = mask.data() + base;
        const uint32_t content_budget = budget - 2;
        const wordpiece_vocab *v = model->vocab;
        size_t n_prefix = 0;
        if (!prefix.empty()) {
            require(wordpiece_tokenize(v, prefix.data(), prefix.size(), scratch.data(), scratch.size(), 4, &n_prefix) ==
                        WORDPIECE_OK,
                    TURBO_E_INTERNAL, "prefix tokenization failed");
            require(n_prefix <= content_budget, TURBO_E_CAPACITY, "prompt prefix alone exceeds the token budget");
        }
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
                fail(TURBO_E_CAPACITY, "input row " + std::to_string(r) + " tokenizes to " +
                                           std::to_string(n_text + n_prefix + 2) + " tokens but the budget is " +
                                           std::to_string(budget) + " and truncation is NONE");
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
                require(wordpiece_tokenize(v, text.ptr, static_cast<size_t>(text.len), row_ids + col, take, 4, &n) ==
                            WORDPIECE_OK,
                        TURBO_E_INTERNAL, "text tokenization failed");
                col += static_cast<uint32_t>(std::min(n, take));
            } else {
                if (n_text > scratch.size()) {
                    scratch.resize(n_text);
                }
                size_t n = 0;
                require(wordpiece_tokenize(v, text.ptr, static_cast<size_t>(text.len), scratch.data(), scratch.size(), 4,
                                           &n) == WORDPIECE_OK,
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
    }

    /// Row of the word table for a token id; an id outside the table is an
    /// error, not a silent substitution.
    const float *word_row(int32_t id, uint32_t r, uint32_t t) const {
        const Tables &tb = model->tables;
        require(id >= 0 && static_cast<uint32_t>(id) < tb.vocab_rows, TURBO_E_INVALID_ARGUMENT,
                "row " + std::to_string(r) + " column " + std::to_string(t) + ": token id " + std::to_string(id) +
                    " is outside the " + std::to_string(tb.vocab_rows) + "-row embedding table");
        return tb.word.data() + static_cast<size_t>(id) * tb.dim;
    }

    /// Fill the hidden-state frame for row r from its ids and mask.
    void build_frame(uint32_t r) {
        const Tables &tb = model->tables;
        const uint32_t dim = model->dim;
        const uint32_t seq_len = model->seq_len;
        const int32_t *row_ids = ids.data() + static_cast<size_t>(r) * seq;
        const int32_t *row_mask = mask.data() + static_cast<size_t>(r) * seq;
        float *frame = in_frame.data();
        if (model->front_end == FrontEnd::Word) {
            const float *pad_row = word_row(wordpiece_pad_id(model->vocab), r, 0);
            for (uint32_t t = 0; t < seq_len; ++t) {
                const bool live = t < seq && row_mask[t] != 0;
                const float *src = live ? word_row(row_ids[t], r, t) : pad_row;
                std::memcpy(frame + static_cast<size_t>(t) * dim, src, dim * sizeof(float));
            }
            return;
        }
        std::memset(frame, 0, in_frame.size() * sizeof(float));
        for (uint32_t t = 0; t < seq; ++t) {
            if (row_mask[t] == 0) {
                continue;
            }
            const float *word = word_row(row_ids[t], r, t);
            const float *pos = tb.pos.data() + static_cast<size_t>(t) * dim;
            const float *tt = tb.token_type.data(); // type 0
            float *row = frame + static_cast<size_t>(t) * dim;
            double mean = 0.0;
            for (uint32_t j = 0; j < dim; ++j) {
                row[j] = word[j] + pos[j] + tt[j];
                mean += row[j];
            }
            mean /= dim;
            double var = 0.0;
            for (uint32_t j = 0; j < dim; ++j) {
                const double d = row[j] - mean;
                var += d * d;
            }
            var /= dim;
            const float inv = static_cast<float>(1.0 / std::sqrt(var + tb.ln_eps));
            for (uint32_t j = 0; j < dim; ++j) {
                row[j] = (row[j] - static_cast<float>(mean)) * inv * tb.ln_gamma[j] + tb.ln_beta[j];
            }
        }
    }

    /// Additive attention bias for row r: 0 where both query and key are
    /// live tokens, -10000 elsewhere (the Model Zoo contract).
    void build_mask(uint32_t r) {
        const uint32_t seq_len = model->seq_len;
        const int32_t *row_mask = mask.data() + static_cast<size_t>(r) * seq;
        float *m = mask_frame.data();
        for (uint32_t q = 0; q < seq_len; ++q) {
            const bool ql = q < seq && row_mask[q] != 0;
            float *mrow = m + static_cast<size_t>(q) * seq_len;
            for (uint32_t k = 0; k < seq_len; ++k) {
                const bool kl = k < seq && row_mask[k] != 0;
                mrow[k] = (ql && kl) ? 0.0f : -10000.0f;
            }
        }
    }

    /// Pool the output frame of row r into host_out, then normalize and cut
    /// to output_dim as the options ask.
    void pool_row(uint32_t r, Pool pool, bool normalize, uint32_t dim_out) {
        const uint32_t dim = model->dim;
        const int32_t *row_mask = mask.data() + static_cast<size_t>(r) * seq;
        float *dst = host_out.data() + static_cast<size_t>(r) * dim_out;
        uint32_t n_live = 0;
        uint32_t last = 0;
        for (uint32_t t = 0; t < seq; ++t) {
            if (row_mask[t] != 0) {
                ++n_live;
                last = t;
            }
        }
        require(n_live > 0, TURBO_E_INVALID_STATE, "row " + std::to_string(r) + " has no live tokens to pool");
        switch (pool) {
        case Pool::Cls:
            std::memcpy(dst, out_frame.data(), dim_out * sizeof(float));
            break;
        case Pool::Last:
            std::memcpy(dst, out_frame.data() + static_cast<size_t>(last) * dim, dim_out * sizeof(float));
            break;
        case Pool::Mean:
            for (uint32_t j = 0; j < dim_out; ++j) {
                double sum = 0.0;
                for (uint32_t t = 0; t < seq; ++t) {
                    if (row_mask[t] != 0) {
                        sum += out_frame[static_cast<size_t>(t) * dim + j];
                    }
                }
                dst[j] = static_cast<float>(sum / n_live);
            }
            break;
        }
        if (normalize) {
            double sum_sq = 0.0;
            for (uint32_t j = 0; j < dim_out; ++j) {
                sum_sq += static_cast<double>(dst[j]) * dst[j];
            }
            if (sum_sq > 1e-24) {
                const float inv = static_cast<float>(1.0 / std::sqrt(sum_sq));
                for (uint32_t j = 0; j < dim_out; ++j) {
                    dst[j] *= inv;
                }
            }
        }
    }

    turbo_provider_result run() {
        Model &m = *model;
        const Pool pool = eopts.pooling == TURBO_POOLING_MODEL ? m.pool
                          : eopts.pooling == TURBO_POOLING_MEAN ? Pool::Mean
                          : eopts.pooling == TURBO_POOLING_CLS  ? Pool::Cls
                                                                 : Pool::Last;
        const bool normalize = eopts.normalize == TURBO_NORMALIZE_MODEL ? m.normalize : eopts.normalize == TURBO_NORMALIZE_L2;
        const uint32_t dim_out = eopts.output_dim == 0 ? m.dim : eopts.output_dim;
        require(dim_out <= m.dim, TURBO_E_UNSUPPORTED_OPTION,
                "output_dim " + std::to_string(dim_out) + " exceeds the model dimension " + std::to_string(m.dim), 7);
        {
            std::lock_guard<std::mutex> lock(m.run_mu);
            require(m.poisoned.empty(), TURBO_E_INVALID_STATE,
                    "the HailoRT pipeline is out of step after an earlier failure (" + m.poisoned +
                        "); reload the model");
            // Poisoning is for a pipeline left out of step: a frame went in
            // and the matching read did not complete. Host-side failures
            // before the first write, or after the read, leave the vstreams
            // paired and are ordinary errors.
            bool in_flight = false;
            try {
                for (uint32_t r = 0; r < n_rows; ++r) {
                    build_frame(r);
                    in_flight = true;
                    hailo_check(hailo_vstream_write_raw_buffer(m.hidden, in_frame.data(), m.hidden_bytes),
                                "hailo_vstream_write_raw_buffer (hidden state)");
                    h2d += m.hidden_bytes;
                    if (m.mask != nullptr) {
                        build_mask(r);
                        hailo_check(hailo_vstream_write_raw_buffer(m.mask, mask_frame.data(), m.mask_bytes),
                                    "hailo_vstream_write_raw_buffer (attention bias)");
                        h2d += m.mask_bytes;
                    }
                    hailo_check(hailo_vstream_read_raw_buffer(m.output, out_frame.data(), m.output_bytes),
                                "hailo_vstream_read_raw_buffer");
                    d2h += m.output_bytes;
                    in_flight = false;
                    pool_row(r, pool, normalize, dim_out);
                }
            } catch (const Failure &e) {
                // A write that failed after the hidden frame went in, or a
                // read that timed out, leaves frames in flight; nothing
                // here can drain them, so the model is marked unusable.
                if (in_flight) {
                    m.poisoned = e.what();
                }
                throw;
            }
        }
        ++runs;
        turbo_provider_result res{};
        res.struct_size = sizeof(turbo_provider_result);
        outputs[0] = turbo_provider_output{};
        outputs[0].struct_size = sizeof(turbo_provider_output);
        outputs[0].name = turbo_text{name0.data(), name0.size()};
        outputs[0].ndim = 2;
        outputs[0].shape[0] = n_rows;
        outputs[0].shape[1] = dim_out;
        out_buf.desc = packed_desc(TURBO_PLACE_HOST, TURBO_DTYPE_F32, {batch, dim_out}, 4);
        out_buf.bytes = static_cast<uint64_t>(batch) * dim_out * 4;
        outputs[0].buffer = describe(&out_buf);
        res.n_outputs = 1;
        res.outputs = outputs;
        return res;
    }
};

// ---------------------------------------------------------------------------
// Vtable functions
// ---------------------------------------------------------------------------

extern "C" {

static int32_t x_device_count(void *, uint32_t *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        const State &s = state();
        require(s.scan_error.empty(), TURBO_E_DEVICE_UNAVAILABLE, s.scan_error);
        *out = static_cast<uint32_t>(s.devices.size());
    });
}

static int32_t x_device_info(void *, uint32_t ordinal, turbo_device_info *out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_device_info>("turbo_device_info", out->struct_size);
        const Device &d = device_at(ordinal);
        turbo_device_info full{};
        full.struct_size = out->struct_size;
        full.kind = TURBO_DEVICE_NPU;
        full.ordinal = d.ordinal;
        full.vendor_id = 0x1e60; // Hailo Technologies
        full.caps = d.identified ? kCaps : 0;
        full.memory_total = 0;
        full.memory_free = 0;
        if (d.identified) {
            put_str(full.name, arch_name(d.arch) + " (" + d.board_name + ", " + d.id_str + ")");
            put_str(full.driver_version, "firmware " + d.firmware);
        } else {
            put_str(full.name, "Hailo device " + d.id_str + " (not identified: " + d.identify_error + ")");
            put_str(full.driver_version, std::string("unavailable"));
        }
        put_str(full.vendor, std::string("Hailo"));
        put_str(full.provider_id, std::string(kProviderId));
        put_str(full.provider_version, std::string(kProviderVersion));
        put_str(full.runtime_version, "HailoRT " + state().library_version);
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_capability(void *, uint32_t ordinal, uint32_t task, uint32_t modality, turbo_capability *out,
                            turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        check_size<turbo_capability>("turbo_capability", out->struct_size);
        const Device &d = device_at(ordinal);
        turbo_capability full{};
        full.struct_size = out->struct_size;
        if (offers(d, task, modality)) {
            full.status = TURBO_CAP_EXPERIMENTAL;
            full.dtype = TURBO_DTYPE_I8;
            full.reference_dtype = TURBO_DTYPE_F32;
            full.cosine_floor = kCosineFloorVsF32;
            full.max_abs_error = 0.0f;
            full.deterministic = 1;
            put_str(full.notes, "INT8 encoder on " + arch_name(d.arch) +
                                    "; host tokenize/gather/pool; ranking parity with FP32, absolute cosine is not preserved");
        } else {
            full.status = TURBO_CAP_UNSUPPORTED;
            put_str(full.notes, !d.identified ? "device not identified: " + d.identify_error
                                              : std::string("hailo provider offers EMBED on TEXT only"));
        }
        std::memcpy(out, &full, out->struct_size);
    });
}

static int32_t x_can_run(void *, uint32_t ordinal, turbo_text bundle_dir, uint32_t task, uint32_t modality,
                         turbo_error *err) {
    return boundary(err, [&] {
        const Device &d = device_at(ordinal);
        require(offers(d, task, modality), TURBO_E_UNSUPPORTED_TASK,
                "hailo device " + d.id_str + " does not offer task " + std::to_string(task) + " for modality " +
                    std::to_string(modality));
        const Bundle b = Bundle::read(text_of(bundle_dir));
        require(b.kind == "embedding", TURBO_E_UNSUPPORTED_TASK, "hailo provider serves embedding bundles only");
        require(b.artifact("hef") && b.artifact("hailo_tables"), TURBO_E_BUNDLE_NO_ARTIFACT,
                "bundle `" + b.model_id + "` needs both a `hef` and a `hailo_tables` artifact");
        require(b.tokenizer_kind == "wordpiece", TURBO_E_UNSUPPORTED, "hailo provider needs a WordPiece tokenizer.json");
        require(b.fixed_shape, TURBO_E_BUNDLE_INVALID, "a HEF bundle must declare limits.fixed_shape");
    });
}

static int32_t x_context_create(void *, uint32_t ordinal, const turbo_context_desc *desc, void **out, turbo_error *err) {
    return boundary(err, [&] {
        require(out != nullptr, TURBO_E_INVALID_ARGUMENT, "out is NULL");
        *out = nullptr;
        if (desc != nullptr) {
            const turbo_context_desc d = read_prefix(desc, "turbo_context_desc");
            require(d.next == nullptr, TURBO_E_NOT_IMPLEMENTED, "external queue import is not implemented");
            reject_unknown(options_of(d.options, d.n_options, "context"), {}, "hailo context");
        }
        *out = new Context(device_at(ordinal));
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
        require(bytes == b->bytes, TURBO_E_CAPACITY,
                "destination size " + std::to_string(bytes) + " does not match the buffer's " + std::to_string(b->bytes));
        std::memcpy(dst, b->host, static_cast<size_t>(bytes));
    });
}

static int32_t x_buffer_export(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err) {
    return boundary(err, [&] {
        auto *b = static_cast<Buffer *>(buf);
        require(b != nullptr && out != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        check_size<turbo_native_handle>("turbo_native_handle", out->struct_size);
        turbo_native_handle full{};
        require(kind == TURBO_HANDLE_HOST_PTR, TURBO_E_UNSUPPORTED, "host buffers export TURBO_HANDLE_HOST_PTR only");
        full.kind = TURBO_HANDLE_HOST_PTR;
        full.handle = reinterpret_cast<uint64_t>(b->host);
        full.offset = 0;
        full.aux = 0;
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

static int32_t x_model_label(void *model, uint32_t, turbo_text *, turbo_error *err) {
    return boundary(err, [&] {
        require(model != nullptr, TURBO_E_INVALID_ARGUMENT, "NULL argument");
        fail(TURBO_E_INVALID_ARGUMENT, "embedding models have no labels");
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
        reject_unknown(options_of(d.options, d.n_options, "session"), {}, "hailo session");
        require(d.max_batch >= 1 && d.max_seq >= 2, TURBO_E_INVALID_ARGUMENT,
                "session needs max_batch >= 1 and max_seq >= 2");
        *out = new Session(m, d.max_batch, d.max_seq);
    });
}

static Session &sess(void *s) {
    require(s != nullptr, TURBO_E_INVALID_HANDLE, "session is NULL");
    return *static_cast<Session *>(s);
}

static int32_t x_session_write_text(void *s, const turbo_text *texts, uint32_t count, const turbo_embed_options *opts,
                                    turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
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
        checked_normalize(o.normalize, 5);
        checked_pooling(o.pooling, 6);
        require(o.output_dim <= S.model->dim, TURBO_E_UNSUPPORTED_OPTION,
                "output_dim " + std::to_string(o.output_dim) + " exceeds the model dimension " + std::to_string(S.model->dim),
                7);
        switch (o.output_dtype) {
        case TURBO_OUTPUT_MODEL:
        case TURBO_OUTPUT_F32:
            break;
        case TURBO_OUTPUT_F16:
        case TURBO_OUTPUT_I8:
            fail(TURBO_E_UNSUPPORTED_OPTION, "hailo provider writes f32 results only; output_dtype " +
                                                 std::to_string(o.output_dtype) + " is not offered", 8);
        default:
            fail(TURBO_E_INVALID_ENUM, "output_dtype " + std::to_string(o.output_dtype) + " is not a TURBO_OUTPUT_* value", 8);
        }
        S.n_rows = 0; // a failed write leaves nothing runnable
        S.eopts = o;
        const uint32_t budget = S.budget(o.max_tokens);
        const std::string &prefix = o.prompt_role == TURBO_PROMPT_QUERY      ? S.model->bundle.prefix_query
                                    : o.prompt_role == TURBO_PROMPT_DOCUMENT ? S.model->bundle.prefix_document
                                                                             : std::string();
        for (uint32_t r = 0; r < count; ++r) {
            S.encode_row(r, prefix, texts[r], o.truncate, budget);
        }
        S.n_rows = count;
    });
}

static int32_t x_session_write_tokens(void *s, const turbo_token_batch *batch, turbo_error *err) {
    return boundary(err, [&] {
        Session &S = sess(s);
        BusyGuard busy(S.in_use, "session");
        require(batch != nullptr, TURBO_E_INVALID_ARGUMENT, "batch is NULL");
        const turbo_token_batch tb = read_prefix(batch, "turbo_token_batch");
        check_token_batch(tb);
        require(tb.batch >= 1 && tb.batch <= S.batch && tb.seq >= 1 && tb.seq <= S.seq, TURBO_E_CAPACITY,
                "token batch exceeds the session shape");
        S.n_rows = 0; // a failed write leaves nothing runnable
        const uint32_t stride = tb.row_stride == 0 ? tb.seq : tb.row_stride;
        // Every id (and type) is checked here, so a bad token is a write-time
        // argument error and never a silent row or a fault inside a run. The
        // message is built only for a failing id: a `require` with a string
        // argument builds its message on every call, about 1 us per token.
        const int32_t n_ids = static_cast<int32_t>(S.model->tables.vocab_rows);
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
            if (tb.types != nullptr) {
                for (uint32_t c = 0; c < tb.seq; ++c) {
                    // The HEF folds token type 0 into its embeddings; another
                    // type cannot be honored, so it is refused rather than dropped.
                    require(tb.types[src + c] == 0, TURBO_E_UNSUPPORTED,
                            "row " + std::to_string(r) + " column " + std::to_string(c) + " has token_type_id " +
                                std::to_string(tb.types[src + c]) + "; the HEF supports type 0 only");
                }
            }
            std::memcpy(S.ids.data() + dst, tb.ids + src, tb.seq * sizeof(int32_t));
            std::memcpy(S.mask.data() + dst, tb.mask + src, tb.seq * sizeof(int32_t));
            for (uint32_t c = tb.seq; c < S.seq; ++c) {
                S.ids[dst + c] = pad;
                S.mask[dst + c] = 0;
            }
        }
        S.eopts = turbo_embed_options{};
        S.eopts.struct_size = sizeof(turbo_embed_options);
        S.n_rows = tb.batch;
    });
}

static int32_t x_session_write_pairs(void *s, const turbo_text *, const turbo_text *, uint32_t, const turbo_rerank_options *,
                                     turbo_error *err) {
    return boundary(err, [&] {
        (void)sess(s);
        fail(TURBO_E_UNSUPPORTED_TASK, "hailo provider serves embedding models only; write_pairs needs a reranker");
    });
}

static int32_t x_session_write_text_classify(void *s, const turbo_text *, uint32_t, const turbo_classify_options *,
                                             turbo_error *err) {
    return boundary(err, [&] {
        (void)sess(s);
        fail(TURBO_E_UNSUPPORTED_TASK, "hailo provider serves embedding models only; write_text_classify needs a classifier");
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
            reject_unknown(options_of(o.params, o.n_params, "run"), {}, "hailo run");
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
        // Frames written to and read from the NPU through HailoRT's DMA pipeline.
        full.h2d_bytes = S.h2d;
        full.d2h_bytes = S.d2h;
        full.input_bytes = (S.ids.size() + S.mask.size()) * sizeof(int32_t);
        full.output_bytes = S.host_out.size() * sizeof(float);
        // HailoRT's own pipeline buffers are not observable from here.
        full.provider_allocs = UINT64_MAX;
        std::memcpy(out, &full, out->struct_size);
    });
}

static void x_session_release(void *s) { release<Session>(s); }

static const turbo_provider_vtbl g_vtbl = {
    sizeof(turbo_provider_vtbl),
    TURBO_PROVIDER_ABI_VERSION,
    kProviderId,
    kProviderVersion,
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
} // namespace turbo_hailo
