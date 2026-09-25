// SPDX-License-Identifier: Apache-2.0
//
// The Metal backend: Apple GPUs through turbo_backend.h. This cut lists the
// devices; no task runs on them yet.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <string>

#include <turbo/turbo_backend.h>

#ifndef TURBO_METAL_SDK
#error "build.rs defines TURBO_METAL_SDK, the macOS SDK version this build compiled against"
#endif

namespace {

// Copy s into a fixed field, cut to fit and always terminated.
void put(char *dst, size_t len, const std::string &s) {
    if (len == 0) return;
    size_t n = std::min(s.size(), len - 1);
    memcpy(dst, s.data(), n);
    dst[n] = 0;
}

template <size_t N> void put(char (&dst)[N], const std::string &s) { put(dst, N, s); }

int32_t refuse(turbo_error *err, int32_t code, const std::string &message) {
    if (err) {
        err->code = code;
        err->field = 0;
        put(err->message, message);
    }
    return code;
}

// The devices, probed once and held for the life of the process, so an
// ordinal names the same device on every call. Metal lists only devices
// that are present; a Mac without one lists nothing.
NSArray<id<MTLDevice>> *devices() {
    static NSArray<id<MTLDevice>> *all = [] {
        @autoreleasepool {
            NSArray<id<MTLDevice>> *found = MTLCopyAllDevices();
            return found ? found : @[];
        }
    }();
    return all;
}

id<MTLDevice> device(uint32_t ordinal, turbo_error *err, int32_t *rc) {
    NSArray<id<MTLDevice>> *all = devices();
    if (ordinal >= all.count) {
        *rc = refuse(err, TURBO_E_INVALID_ARGUMENT,
                     "metal device " + std::to_string(ordinal) + ": " + std::to_string(all.count) + " listed");
        return nil;
    }
    return all[ordinal];
}

// The label benchmarks are filed under: the chip, lowercased with the
// spaces dropped ("Apple M2 Pro" is m2pro). A GPU that is not Apple's
// keeps its whole name, less trademark marks.
std::string arch(const std::string &name) {
    std::string s = name.rfind("Apple ", 0) == 0 ? name.substr(6) : name;
    for (const char *mark : {"(R)", "(TM)"})
        for (size_t at; (at = s.find(mark)) != std::string::npos;) s.erase(at, strlen(mark));
    std::string out;
    for (unsigned char c : s)
        if (std::isalnum(c)) out += (char)std::tolower(c);
    return out;
}

// Metal has no vendor field; the name leads with it ("Apple M2",
// "AMD Radeon Pro 5500M", "Intel(R) UHD Graphics 630").
std::string vendor(const std::string &name) {
    std::string first = name.substr(0, name.find(' '));
    return first.substr(0, first.find('('));
}

// What the host has free now: free and inactive pages, as the kernel
// counts them. 0 if it does not say.
uint64_t host_free() {
    vm_statistics64_data_t vm;
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    mach_port_t host = mach_host_self();
    kern_return_t kr = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &n);
    mach_port_deallocate(mach_task_self(), host);
    if (kr != KERN_SUCCESS) return 0;
    return ((uint64_t)vm.free_count + vm.inactive_count) * vm_kernel_page_size;
}

// The operating system is the driver: Metal ships with it.
std::string os_version() {
    NSOperatingSystemVersion v = [NSProcessInfo processInfo].operatingSystemVersion;
    std::string s = "macOS " + std::to_string(v.majorVersion) + "." + std::to_string(v.minorVersion);
    if (v.patchVersion) s += "." + std::to_string(v.patchVersion);
    char build[32] = {0};
    size_t len = sizeof build - 1;
    if (sysctlbyname("kern.osversion", build, &len, nullptr, 0) == 0 && build[0]) s += std::string(" (") + build + ")";
    return s;
}

const char kRuntime[] = "Metal, macOS SDK " TURBO_METAL_SDK;

int32_t device_count(uint32_t *out, turbo_error *) noexcept {
    *out = (uint32_t)devices().count;
    return TURBO_OK;
}

int32_t device_info(uint32_t ordinal, turbo_device_info *out, turbo_error *err) noexcept {
    @autoreleasepool {
        int32_t rc = TURBO_OK;
        id<MTLDevice> d = device(ordinal, err, &rc);
        if (!d) return rc;
        std::string name = d.name.UTF8String ?: "";
        bool unified = d.hasUnifiedMemory;
        // What Metal lets this device use without paging, the most a
        // model on it can hold.
        uint64_t total = d.recommendedMaxWorkingSetSize;
        uint64_t used = d.currentAllocatedSize;
        uint64_t left = total > used ? total - used : 0;
        out->kind = unified ? TURBO_DEVICE_IGPU : TURBO_DEVICE_GPU;
        out->ordinal = ordinal;
        out->unified_memory = unified ? 1 : 0;
        out->memory_total = total;
        // With one memory the host's other users count against it too.
        // Metal does not say what other processes hold on a discrete GPU,
        // so there it is unknown.
        out->memory_free = unified ? std::min(left, host_free()) : 0;
        put(out->arch, arch(name));
        put(out->name, name);
        put(out->vendor, vendor(name));
        put(out->runtime_version, kRuntime);
        put(out->driver_version, os_version());
        return TURBO_OK;
    }
}

int32_t capability(uint32_t ordinal, uint32_t, uint32_t, uint32_t *status, uint32_t *dtype,
                   uint32_t *options_honored, char *reason, uint32_t reason_len, turbo_error *err) noexcept {
    @autoreleasepool {
        int32_t rc = TURBO_OK;
        if (!device(ordinal, err, &rc)) return rc;
    }
    *status = TURBO_CAP_UNSUPPORTED;
    *dtype = 0;
    *options_honored = 0;
    put(reason, reason_len, "the metal backend has no embed kernels in this build");
    return TURBO_OK;
}

} // namespace

extern "C" const turbo_backend turbo_metal_backend = {
    sizeof(turbo_backend), 0, "metal", kRuntime, device_count, device_info, capability,
};
