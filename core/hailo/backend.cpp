/* SPDX-License-Identifier: Apache-2.0
 *
 * The Hailo backend: Hailo accelerators through HailoRT, behind
 * include/turbo/turbo_backend.h. It lists the devices HailoRT finds, each
 * identified once through the C API; one that does not answer is refused
 * by device_info with the reason, which the runtime logs as it leaves the
 * device out. It runs embed on a HEF (FORMAT_HEF, INPUT_EMBEDDINGS to
 * OUTPUT_HIDDEN_STATES) through the C++ InferModel API in embed.cpp: the
 * word rows are gathered on the host, the encoder runs on the device, and
 * pooling and normalize run on the host.
 *
 * Every function here is called from any thread. The device list is made
 * once per process, on first need, under std::call_once, and kept: HailoRT
 * opens a device to identify it, and a device that failed to answer is
 * not asked again until the process restarts.
 */

#include <turbo/turbo_backend.h>

#include <hailo/hailort.h>

#include "embed.h"

#include <cstdarg>
#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <new>
#include <string>
#include <vector>

namespace {

// ---- Errors ------------------------------------------------------------------
//
// Nothing throws across the table: each entry runs its body inside
// guarded(), which turns an exception into a status.

__attribute__((format(printf, 3, 4))) int32_t refuse(turbo_error *err, int32_t code, const char *fmt, ...) {
    if (err) {
        va_list ap;
        va_start(ap, fmt);
        err->code = code;
        err->field = 0;
        vsnprintf(err->message, TURBO_ERROR_MESSAGE_LEN, fmt, ap);
        va_end(ap);
    }
    return code;
}

template <typename F> int32_t guarded(turbo_error *err, F f) noexcept {
    try {
        return f();
    } catch (const std::bad_alloc &) {
        return refuse(err, TURBO_E_OUT_OF_MEMORY, "host memory for the hailo backend");
    } catch (...) {
        return refuse(err, TURBO_E_INTERNAL, "an exception inside the hailo backend");
    }
}

/* HailoRT's name for a status, with its number. */
std::string status_text(hailo_status s) {
    const char *m = hailo_get_status_message(s);
    char n[16];
    snprintf(n, sizeof n, "%d", (int)s);
    return std::string(m ? m : "unknown status") + " (" + n + ")";
}

void copy_str(char *dst, size_t len, const char *src) {
    if (len == 0) return;
    size_t i = 0;
    for (; src[i] && i + 1 < len; i++) dst[i] = src[i];
    dst[i] = 0;
}

/* A fixed-width field HailoRT fills with a length beside it, which need not
 * be NUL-terminated, and may be padded with blanks: a Hailo-8's board name
 * comes back as "Hailo-8" and trailing spaces. The blanks are dropped. */
std::string counted(const char *s, size_t len, size_t cap) {
    if (len > cap) len = cap;
    size_t n = 0;
    while (n < len && s[n]) n++;
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t')) n--;
    return std::string(s, n);
}

// ---- Devices -------------------------------------------------------------------

/* The label benchmark records for a device are filed under (turbo.h's
 * turbo_device_info.arch), and the name the device is shown with, from the
 * architecture HailoRT's identify reports. A Hailo-10H names no board, so
 * the architecture is the name. */
struct Arch {
    const char *label;
    const char *name;
};

Arch arch_of(hailo_device_architecture_t a) {
    switch (a) {
    case HAILO_ARCH_HAILO8_A0:
        return {"hailo8a0", "Hailo-8 A0"};
    case HAILO_ARCH_HAILO8:
        return {"hailo8", "Hailo-8"};
    case HAILO_ARCH_HAILO8L:
        return {"hailo8l", "Hailo-8L"};
    case HAILO_ARCH_HAILO15H:
        return {"hailo15h", "Hailo-15H"};
    case HAILO_ARCH_HAILO15L:
        return {"hailo15l", "Hailo-15L"};
    case HAILO_ARCH_HAILO15M:
        return {"hailo15m", "Hailo-15M"};
    case HAILO_ARCH_HAILO10H:
        return {"hailo10h", "Hailo-10H"};
    default:
        return {"", ""};
    }
}

struct Device {
    hailo_device_id_t id;
    hailo_device_architecture_t architecture;
    std::string name;
    std::string firmware;
    std::string failure;   // non-empty: the device did not answer, and this says how
};

struct Listing {
    std::vector<Device> devices;
    std::string failure;   // non-empty: HailoRT failed to scan, and this says how
    std::string library;   // HailoRT's version, "5.1.1"
};

/* The kernel module HailoRT talks to, and its version, from sysfs. Empty
 * when neither module is loaded. */
std::string kernel_module() {
    for (const char *m : {"hailo1x_pci", "hailo_pci"}) {
        const std::string path = std::string("/sys/module/") + m + "/version";
        FILE *f = fopen(path.c_str(), "r");
        if (!f) continue;
        char v[64] = {0};
        const bool read = fgets(v, sizeof v, f) != nullptr;
        fclose(f);
        if (!read) continue;
        v[strcspn(v, "\r\n")] = 0;
        return std::string(m) + " " + v;
    }
    return "";
}

/* One device's identity. A device HailoRT found that does not answer, or
 * whose architecture this backend has no label for, keeps the reason in
 * failure: its device_info refuses with it, so the runtime leaves it out
 * and its log says why. */
Device identify(const hailo_device_id_t &id) {
    Device out;
    out.id = id;
    out.architecture = HAILO_ARCH_MAX_ENUM;
    const std::string where = counted(id.id, sizeof id.id, sizeof id.id);
    hailo_device d = nullptr;
    hailo_status s = hailo_create_device_by_id(&id, &d);
    if (s != HAILO_SUCCESS) {
        out.failure = "hailo device " + where + ": hailo_create_device_by_id: " + status_text(s);
        return out;
    }
    hailo_device_identity_t who;
    memset(&who, 0, sizeof who);
    s = hailo_identify(d, &who);
    (void)hailo_release_device(d);
    if (s != HAILO_SUCCESS) {
        out.failure = "hailo device " + where + ": hailo_identify: " + status_text(s);
        return out;
    }
    const Arch a = arch_of(who.device_architecture);
    if (!a.label[0]) {
        out.failure = "hailo device " + where + ": architecture " + std::to_string((int)who.device_architecture) +
                      " is not one this backend knows";
        return out;
    }
    out.architecture = who.device_architecture;
    const std::string product = counted(who.product_name, who.product_name_length, HAILO_MAX_PRODUCT_NAME_LENGTH);
    const std::string board = counted(who.board_name, who.board_name_length, HAILO_MAX_BOARD_NAME_LENGTH);
    out.name = !product.empty() ? product : !board.empty() ? board : a.name;
    char fw[48];
    snprintf(fw, sizeof fw, "%u.%u.%u", who.fw_version.major, who.fw_version.minor, who.fw_version.revision);
    out.firmware = fw;
    return out;
}

Listing make_listing() {
    Listing l;
    hailo_version_t v;
    if (hailo_get_library_version(&v) == HAILO_SUCCESS) {
        char s[48];
        snprintf(s, sizeof s, "%u.%u.%u", v.major, v.minor, v.revision);
        l.library = s;
    }
    // On HAILO_INSUFFICIENT_BUFFER, HailoRT sets n to the devices it found,
    // and the scan is repeated with room for them.
    std::vector<hailo_device_id_t> ids(32);
    size_t n = ids.size();
    hailo_status s = hailo_scan_devices(nullptr, ids.data(), &n);
    while (s == HAILO_INSUFFICIENT_BUFFER && n > ids.size()) {
        ids.resize(n);
        s = hailo_scan_devices(nullptr, ids.data(), &n);
    }
    // On Linux, HailoRT scans the driver's class in sysfs: with no driver
    // loaded it returns HAILO_SUCCESS and no devices. It is not known to
    // return HAILO_DRIVER_NOT_INSTALLED there; that status also means
    // nothing to list.
    if (s == HAILO_DRIVER_NOT_INSTALLED) return l;
    if (s != HAILO_SUCCESS) {
        l.failure = "hailo_scan_devices: " + status_text(s);
        return l;
    }
    for (size_t i = 0; i < n; i++) l.devices.push_back(identify(ids[i]));
    return l;
}

const Listing &listing() {
    static std::once_flag once;
    static const Listing *l = nullptr;
    std::call_once(once, [] { l = new Listing(make_listing()); });
    return *l;
}

int32_t device_count(uint32_t *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Listing &l = listing();
        if (!l.failure.empty()) return refuse(err, TURBO_E_DEVICE_UNAVAILABLE, "%s", l.failure.c_str());
        *out = (uint32_t)l.devices.size();
        return TURBO_OK;
    });
}

int32_t device_info(uint32_t ordinal, turbo_device_info *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Listing &l = listing();
        const Device &d = l.devices.at(ordinal);
        if (!d.failure.empty()) return refuse(err, TURBO_E_DEVICE_UNAVAILABLE, "%s", d.failure.c_str());
        out->kind = TURBO_DEVICE_NPU;
        out->ordinal = ordinal;
        // Each Hailo device has its own memory, if any, and HailoRT reports
        // neither its size nor what is free.
        out->unified_memory = 0;
        out->memory_total = 0;
        out->memory_free = 0;
        copy_str(out->arch, sizeof out->arch, arch_of(d.architecture).label);
        copy_str(out->name, sizeof out->name, d.name.c_str());
        copy_str(out->vendor, sizeof out->vendor, "Hailo");
        copy_str(out->runtime_version, sizeof out->runtime_version, l.library.c_str());
        const std::string module = kernel_module();
        const std::string driver = (module.empty() ? std::string() : module + ", ") + "firmware " + d.firmware;
        copy_str(out->driver_version, sizeof out->driver_version, driver.c_str());
        return TURBO_OK;
    });
}

/* Fields of turbo_embed_options a run honors: normalize (4), pooling (5)
 * and output_dim (6), every value of each. */
constexpr uint32_t EMBED_HONORED = 0x38;

/* Embed, in the dtype a HEF is compiled in: I8 at MODEL and FASTEST. A HEF
 * never computes in F32, so EXACT is refused. */
int32_t capability(uint32_t, uint32_t, uint32_t precision, uint32_t *status, uint32_t *dtype,
                   uint32_t *options_honored, char *reason, uint32_t reason_len, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        if (precision == TURBO_PRECISION_EXACT) {
            *status = TURBO_CAP_UNSUPPORTED;
            *dtype = 0;
            *options_honored = 0;
            if (reason_len)
                copy_str(reason, reason_len, "a HEF computes in the I8 it was compiled in; EXACT asks for F32");
            return TURBO_OK;
        }
        *status = TURBO_CAP_EXPERIMENTAL;
        *dtype = TURBO_DTYPE_I8;
        *options_honored = EMBED_HONORED;
        if (reason_len) reason[0] = 0;
        return TURBO_OK;
    });
}

__attribute__((format(printf, 4, 5))) int32_t refuse_field(turbo_error *err, int32_t code, uint32_t field,
                                                           const char *fmt, ...) {
    if (err) {
        va_list ap;
        va_start(ap, fmt);
        err->code = code;
        err->field = field;
        vsnprintf(err->message, TURBO_ERROR_MESSAGE_LEN, fmt, ap);
        va_end(ap);
    }
    return code;
}

int32_t failed(turbo_error *err, const turbo_hailo::Failure &f) {
    return refuse_field(err, f.code, f.field, "%s", f.message.c_str());
}

// ---- Contexts and buffers ------------------------------------------------------
//
// A context is the device's HailoRT vdevice. HailoRT opens a device for one
// vdevice at a time in a process, so every context on a device shares one,
// made by the first and released with the last. Buffers are host memory:
// the device's own memory is HailoRT's, reached only through the frames it
// sends, so HOST is the one placement.

struct Context {
    std::shared_ptr<hailort::VDevice> vdevice;
};

std::mutex vdevices_lock;

std::shared_ptr<hailort::VDevice> shared_vdevice(const hailo_device_id_t &id, hailo_status &status) {
    static std::vector<std::pair<std::string, std::weak_ptr<hailort::VDevice>>> open;
    std::lock_guard<std::mutex> g(vdevices_lock);
    const std::string key = counted(id.id, sizeof id.id, sizeof id.id);
    for (auto &e : open)
        if (e.first == key)
            if (auto v = e.second.lock()) return v;
    hailo_vdevice_params_t params;
    status = hailo_init_vdevice_params(&params);
    if (status != HAILO_SUCCESS) return nullptr;
    hailo_device_id_t ids[1] = {id};
    params.device_ids = ids;
    params.device_count = 1;
    auto v = hailort::VDevice::create(params);
    if (!v) {
        status = v.status();
        return nullptr;
    }
    std::shared_ptr<hailort::VDevice> shared(v.release());
    bool placed = false;
    for (auto &e : open)
        if (e.first == key) {
            e.second = shared;
            placed = true;
        }
    if (!placed) open.emplace_back(key, shared);
    return shared;
}

int32_t context_create(uint32_t ordinal, turbo_log_fn, void *, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        const Device &d = listing().devices.at(ordinal);
        if (!d.failure.empty()) return refuse(err, TURBO_E_DEVICE_UNAVAILABLE, "%s", d.failure.c_str());
        hailo_status s = HAILO_SUCCESS;
        auto v = shared_vdevice(d.id, s);
        if (!v)
            return refuse(err, TURBO_E_DEVICE_UNAVAILABLE, "hailo device %s: VDevice::create: %s",
                          counted(d.id.id, sizeof d.id.id, sizeof d.id.id).c_str(), status_text(s).c_str());
        *out = new Context{v};
        return TURBO_OK;
    });
}

void context_release(void *ctx) { delete static_cast<Context *>(ctx); }

/* Every buffer starts on a 64-byte boundary, a cache line. */
constexpr size_t ALIGN = 64;

struct Buffer {
    void *ptr;
};

void *aligned(uint64_t bytes) {
    const size_t n = (size_t)((bytes + ALIGN - 1) / ALIGN * ALIGN);
    return aligned_alloc(ALIGN, n ? n : ALIGN);
}

int32_t buffer_alloc(void *, const turbo_buffer_desc *desc, void **out, void **host, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        if (desc->placement != TURBO_PLACE_HOST)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "placement: %u: a hailo context allocates TURBO_PLACE_HOST only; the device's memory is "
                          "HailoRT's",
                          desc->placement);
        void *p = aligned(desc->bytes);
        if (!p) return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes of host memory", (unsigned long long)desc->bytes);
        *host = p;
        *out = new Buffer{p};
        return TURBO_OK;
    });
}

void buffer_release(void *buf) {
    Buffer *b = static_cast<Buffer *>(buf);
    free(b->ptr);
    delete b;
}

int32_t buffer_export(void *buf, uint32_t kind, turbo_native_handle *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        if (kind != TURBO_HANDLE_HOST_PTR)
            return refuse(err, TURBO_E_UNSUPPORTED, "kind: %u is not TURBO_HANDLE_HOST_PTR, the one kind hailo exports",
                          kind);
        out->kind = TURBO_HANDLE_HOST_PTR;
        out->handle = (uint64_t)(uintptr_t) static_cast<Buffer *>(buf)->ptr;
        out->aux = 0;
        out->offset = 0;
        return TURBO_OK;
    });
}

// ---- Models --------------------------------------------------------------------
//
// A model is a HEF that takes word-embedding rows (INPUT_EMBEDDINGS) and
// gives hidden states, configured on the context's vdevice. Its bytes and
// the word table stay where the core holds them until model_release.

struct Model {
    std::shared_ptr<hailort::VDevice> vdevice;
    std::unique_ptr<turbo_hailo::Model> hef;
};

int32_t model_load(void *ctx, const turbo_backend_model *desc, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        if (desc->format != TURBO_FORMAT_HEF)
            return refuse(err, TURBO_E_UNSUPPORTED, "format %u: the hailo backend loads FORMAT_HEF", desc->format);
        if (desc->graph_input != TURBO_INPUT_EMBEDDINGS || desc->graph_output != TURBO_OUTPUT_HIDDEN_STATES)
            return refuse(err, TURBO_E_UNSUPPORTED,
                          "graph_input %u, graph_output %u: the hailo backend runs a HEF from INPUT_EMBEDDINGS to "
                          "OUTPUT_HIDDEN_STATES",
                          desc->graph_input, desc->graph_output);
        if (desc->compute_dtype != TURBO_DTYPE_I8)
            return refuse(err, TURBO_E_UNSUPPORTED, "compute_dtype %u: the hailo backend runs I8 HEFs",
                          desc->compute_dtype);
        if (desc->fixed_seq == 0 || desc->fixed_batch == 0)
            return refuse(err, TURBO_E_BUNDLE_INVALID,
                          "fixed_seq %u, fixed_batch %u: a HEF's shape is compiled in, and the manifest must give it",
                          desc->fixed_seq, desc->fixed_batch);
        if (desc->fixed_batch != 1)
            return refuse(err, TURBO_E_UNSUPPORTED, "fixed_batch %u: the hailo backend runs HEFs of one row a frame",
                          desc->fixed_batch);
        if (desc->family != TURBO_FAMILY_BERT || desc->tensor_count != TURBO_BERT_EMBEDDING_TENSORS ||
            !desc->tensors)
            return refuse(err, TURBO_E_UNSUPPORTED, "family %u with %u tensors: the hailo backend needs a BERT's %d "
                          "embedding tensors",
                          desc->family, desc->tensor_count, TURBO_BERT_EMBEDDING_TENSORS);
        const turbo_backend_tensor &w = desc->tensors[TURBO_BERT_WORD_EMBEDDINGS];
        if (w.dtype != TURBO_DTYPE_F32)
            return refuse(err, TURBO_E_UNSUPPORTED, "%s: dtype %u: the hailo backend gathers F32 word rows", w.name,
                          w.dtype);
        if (desc->heads == 0 || desc->hidden % desc->heads != 0)
            return refuse(err, TURBO_E_BUNDLE_INVALID, "hidden %u is not a multiple of heads %u", desc->hidden,
                          desc->heads);
        turbo_hailo::ModelDesc d;
        d.hef = desc->artifact;
        d.hef_bytes = desc->artifact_bytes;
        d.word_table = static_cast<const float *>(w.data);
        d.vocab = desc->vocab_size;
        d.hidden = desc->hidden;
        d.heads = desc->heads;
        d.seq = desc->fixed_seq;
        std::unique_ptr<Model> m(new Model());
        m->vdevice = static_cast<Context *>(ctx)->vdevice;
        const turbo_hailo::Failure f = turbo_hailo::Model::load(*m->vdevice, d, m->hef);
        if (f) return failed(err, f);
        *out = m.release();
        return TURBO_OK;
    });
}

void model_release(void *model) { delete static_cast<Model *>(model); }

// ---- Sessions ------------------------------------------------------------------

struct Session {
    std::unique_ptr<turbo_hailo::Session> run;
    Buffer output;
    uint32_t hidden;
};

int32_t session_create(void *model, uint32_t task, uint32_t max_batch, uint32_t max_seq, uint32_t precision,
                       uint32_t *compute_dtype, void **out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Model *m = static_cast<Model *>(model);
        if (task != TURBO_TASK_EMBED)
            return refuse(err, TURBO_E_UNSUPPORTED_TASK, "task %u: the hailo backend runs embed", task);
        if (precision == TURBO_PRECISION_EXACT)
            return refuse_field(err, TURBO_E_UNSUPPORTED_OPTION, 3,
                                "precision: EXACT computes in F32, and this HEF computes in the I8 it was compiled in");
        std::unique_ptr<turbo_hailo::Session> s;
        const turbo_hailo::Failure f = turbo_hailo::Session::create(*m->hef, max_batch, max_seq, s);
        if (f) return failed(err, f);
        const uint32_t hidden = m->hef->desc().hidden;
        const uint64_t bytes = (uint64_t)max_batch * hidden * 4;
        void *p = aligned(bytes);
        if (!p)
            return refuse(err, TURBO_E_OUT_OF_MEMORY, "%llu bytes for the session's vectors",
                          (unsigned long long)bytes);
        *out = new Session{std::move(s), Buffer{p}, hidden};
        *compute_dtype = TURBO_DTYPE_I8;
        return TURBO_OK;
    });
}

void session_release(void *session) {
    Session *s = static_cast<Session *>(session);
    free(s->output.ptr);
    delete s;
}

int32_t embed_write(void *session, const turbo_backend_embed_rows *rows, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        turbo_hailo::Rows r;
        r.batch = rows->batch;
        r.seq = rows->seq;
        r.row_stride = rows->row_stride;
        r.ids = rows->ids;
        r.mask = rows->mask;
        r.types = rows->types;
        r.pooling = rows->pooling;
        r.normalize = rows->normalize;
        r.output_dim = rows->output_dim;
        const turbo_hailo::Failure f = static_cast<Session *>(session)->run->write(r);
        return f ? failed(err, f) : TURBO_OK;
    });
}

/* Stages: the lookup on the host into each frame, which HailoRT sends to
 * the device; the encoder on the device; its hidden states back, pooled
 * and normalized on the host. The vectors stay on the host. */
int32_t session_run(void *session, turbo_backend_run *out, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        Session *s = static_cast<Session *>(session);
        uint64_t h2d = 0, d2h = 0;
        const turbo_hailo::Failure f = s->run->run(static_cast<float *>(s->output.ptr), h2d, d2h);
        if (f) return failed(err, f);
        out->placement = TURBO_PLACE_HOST;
        out->output = &s->output;
        out->host = s->output.ptr;
        out->h2d_bytes = h2d;
        out->d2h_bytes = d2h;
        // The backend allocates nothing in a run: the frames, the bindings
        // and the scratch were made with the session. What HailoRT allocates
        // inside run_async is its own and is not counted here.
        out->host_allocs = 0;
        out->device_allocs = 0;
        uint32_t *st = out->stage;
        st[TURBO_EMBED_STAGE_UPLOAD] = TURBO_STAGE_DEVICE;
        st[TURBO_EMBED_STAGE_LOOKUP] = TURBO_STAGE_HOST;
        st[TURBO_EMBED_STAGE_ENCODE] = TURBO_STAGE_DEVICE;
        st[TURBO_EMBED_STAGE_POOL] = TURBO_STAGE_HOST;
        st[TURBO_EMBED_STAGE_NORMALIZE] =
            s->run->normalize() == TURBO_NORMALIZE_L2 ? TURBO_STAGE_HOST : TURBO_STAGE_UNUSED;
        st[TURBO_EMBED_STAGE_DOWNLOAD] = TURBO_STAGE_DEVICE;
        return TURBO_OK;
    });
}

} // namespace

extern "C" {

extern const turbo_backend turbo_hailo_backend;

const turbo_backend turbo_hailo_backend = {
    sizeof(turbo_backend),
    0,
    "hailo",
    // HailoRT has no version in its headers; each device reports the
    // library's as its runtime_version.
    "",
    device_count,
    device_info,
    capability,
    context_create,
    context_release,
    buffer_alloc,
    nullptr,
    buffer_release,
    buffer_export,
    model_load,
    model_release,
    session_create,
    session_release,
    embed_write,
    session_run,
    nullptr,
    TURBO_FORMAT_BIT(TURBO_FORMAT_HEF),
    0,
};

} // extern "C"
