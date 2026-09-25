/* SPDX-License-Identifier: Apache-2.0
 *
 * The Hailo backend: Hailo accelerators through HailoRT's C API, behind
 * include/turbo/turbo_backend.h. It lists the devices HailoRT finds, each
 * identified once, and says what it can run on each. One that does not
 * answer is refused by device_info with the reason, which the runtime logs
 * as it leaves the device out. It runs no task yet, so its table stops
 * at capability.
 *
 * Every function here is called from any thread. The device list is made
 * once per process, on first need, under std::call_once: HailoRT opens a
 * device to identify it, and devices do not come and go while a process
 * runs.
 */

#include <turbo/turbo_backend.h>

#include <hailo/hailort.h>

#include <cstdarg>
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
 * be NUL-terminated. */
std::string counted(const char *s, size_t len, size_t cap) {
    if (len > cap) len = cap;
    size_t n = 0;
    while (n < len && s[n]) n++;
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
    // More devices than one machine holds; HailoRT says so if not.
    std::vector<hailo_device_id_t> ids(32);
    size_t n = ids.size();
    const hailo_status s = hailo_scan_devices(nullptr, ids.data(), &n);
    if (s == HAILO_DRIVER_NOT_INSTALLED) return l;   // no driver: nothing to list
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

/* No task runs on a Hailo device through this backend yet. */
int32_t capability(uint32_t, uint32_t, uint32_t, uint32_t *status, uint32_t *dtype, uint32_t *options_honored,
                   char *reason, uint32_t reason_len, turbo_error *err) {
    return guarded(err, [&]() -> int32_t {
        *status = TURBO_CAP_UNSUPPORTED;
        *dtype = 0;
        *options_honored = 0;
        if (reason_len) copy_str(reason, reason_len, "the hailo backend lists devices and runs no task yet");
        return TURBO_OK;
    });
}

} // namespace

extern "C" {

extern const turbo_backend turbo_hailo_backend;

/* The table through capability: the backend offers no contexts, models or
 * sessions yet. */
const turbo_backend turbo_hailo_backend = {
    (uint32_t)offsetof(turbo_backend, context_create),
    0,
    "hailo",
    // HailoRT has no version in its headers; each device reports the
    // library's as its runtime_version.
    "",
    device_count,
    device_info,
    capability,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
    nullptr,
};

} // extern "C"
