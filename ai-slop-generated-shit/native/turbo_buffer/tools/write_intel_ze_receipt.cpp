// SPDX-License-Identifier: Apache-2.0
//
// Machine B LIVE receipt for turbo_buffer ZE USM HOST/SHARED/DEVICE.
// Writes testdata/receipts/turbo_buffer/intel-ze.json. No hostnames.

#include "turbo_buffer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

static std::string run_cmd(const char *cmd) {
    FILE *p = popen(cmd, "r");
    if (p == nullptr) {
        return {};
    }
    char buf[256];
    std::string out;
    while (fgets(buf, sizeof(buf), p) != nullptr) {
        out += buf;
    }
    pclose(p);
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) {
        out.pop_back();
    }
    return out;
}

int main() {
    if (turbo_buffer_backend_probe(
            TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_HOST
        ) != TURBO_BUFFER_OK ||
        turbo_buffer_backend_probe(
            TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_SHARED
        ) != TURBO_BUFFER_OK ||
        turbo_buffer_backend_probe(
            TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_DEVICE
        ) != TURBO_BUFFER_OK) {
        std::fprintf(
            stderr,
            "write_intel_ze_receipt: ZE HOST/SHARED/DEVICE not all live: %s\n",
            turbo_buffer_last_error(nullptr)
        );
        return 1;
    }

    turbo_buffer_arena *a = nullptr;
    if (turbo_buffer_arena_create(TURBO_BUFFER_DEVICE_ZE, &a) != TURBO_BUFFER_OK) {
        std::fprintf(stderr, "arena_create: %s\n", turbo_buffer_last_error(nullptr));
        return 1;
    }

    turbo_buffer_view host {};
    turbo_buffer_view shared {};
    turbo_buffer_view device {};
    if (turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &host
        ) != TURBO_BUFFER_OK ||
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_SHARED, 1, 16, 16, &shared
        ) != TURBO_BUFFER_OK ||
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_DEVICE, 1, 16, 16, &device
        ) != TURBO_BUFFER_OK) {
        std::fprintf(stderr, "rent: %s\n", turbo_buffer_last_error(a));
        turbo_buffer_arena_destroy(a);
        return 1;
    }

    turbo_buffer_placement qh = TURBO_BUFFER_PLACE_PINNED;
    turbo_buffer_placement qs = TURBO_BUFFER_PLACE_PINNED;
    turbo_buffer_placement qd = TURBO_BUFFER_PLACE_PINNED;
    if (turbo_buffer_ze_query(host.ptr, &qh) != TURBO_BUFFER_OK ||
        qh != TURBO_BUFFER_PLACE_HOST ||
        turbo_buffer_ze_query(shared.ptr, &qs) != TURBO_BUFFER_OK ||
        qs != TURBO_BUFFER_PLACE_SHARED ||
        turbo_buffer_ze_query(device.ptr, &qd) != TURBO_BUFFER_OK ||
        qd != TURBO_BUFFER_PLACE_DEVICE) {
        std::fprintf(stderr, "ze_query remapped USM types\n");
        turbo_buffer_arena_destroy(a);
        return 1;
    }

    const int32_t pattern[4] = {101, 7592, 102, 7};
    std::memcpy(turbo_buffer_view_i32(&host), pattern, sizeof(pattern));
    if (turbo_buffer_ze_memcpy(device.ptr, host.ptr, sizeof(pattern)) !=
            TURBO_BUFFER_OK ||
        turbo_buffer_ze_memcpy(host.ptr, device.ptr, sizeof(pattern)) !=
            TURBO_BUFFER_OK) {
        std::fprintf(stderr, "ze_memcpy: %s\n", turbo_buffer_last_error(nullptr));
        turbo_buffer_arena_destroy(a);
        return 1;
    }
    const bool memcpy_ok = turbo_buffer_view_i32(&host)[0] == 101 &&
                           turbo_buffer_view_i32(&host)[1] == 7592 &&
                           turbo_buffer_view_i32(&host)[2] == 102;

    (void)turbo_buffer_arena_return(a, &host);
    (void)turbo_buffer_arena_return(a, &shared);
    (void)turbo_buffer_arena_return(a, &device);
    turbo_buffer_alloc_counter_reset();
    turbo_buffer_view h2 {};
    turbo_buffer_view s2 {};
    turbo_buffer_view d2 {};
    const bool reuse =
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_HOST, 1, 16, 16, &h2
        ) == TURBO_BUFFER_OK &&
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_SHARED, 1, 16, 16, &s2
        ) == TURBO_BUFFER_OK &&
        turbo_buffer_arena_rent(
            a, TURBO_BUFFER_DTYPE_I32, TURBO_BUFFER_PLACE_DEVICE, 1, 16, 16, &d2
        ) == TURBO_BUFFER_OK &&
        turbo_buffer_alloc_counter() == 0;
    (void)turbo_buffer_arena_return(a, &h2);
    (void)turbo_buffer_arena_return(a, &s2);
    (void)turbo_buffer_arena_return(a, &d2);

    const turbo_buffer_status pinned = turbo_buffer_backend_probe(
        TURBO_BUFFER_DEVICE_ZE, TURBO_BUFFER_PLACE_PINNED
    );
    const bool no_remap = pinned == TURBO_BUFFER_ERR_NOT_IMPLEMENTED;

    turbo_buffer_arena_destroy(a);

    const bool pass = memcpy_ok && reuse && no_remap;
    const char *root_c = std::getenv("INFERSTREAM_ROOT");
    const std::string root = root_c && root_c[0] ? root_c : ".";
    const std::string out_path = root + "/testdata/receipts/turbo_buffer/intel-ze.json";
    const std::string sha = run_cmd("git rev-parse HEAD");

    std::ostringstream js;
    js << "{\n";
    js << "  \"machine\": \"Machine B\",\n";
    js << "  \"backend\": \"turbo_buffer ZE USM arena\",\n";
    js << "  \"device\": \"ZE\",\n";
    js << "  \"placements\": [\"HOST\", \"SHARED\", \"DEVICE\"],\n";
    js << "  \"compute\": {\n";
    js << "    \"usm_host\": true,\n";
    js << "    \"usm_shared\": true,\n";
    js << "    \"usm_device\": true,\n";
    js << "    \"silent_host_remap\": false,\n";
    js << "    \"device_memcpy_roundtrip\": " << (memcpy_ok ? "true" : "false")
       << ",\n";
    js << "    \"reuse_allocs\": " << (reuse ? 0 : 1) << ",\n";
    js << "    \"pinned_on_ze\": \"NOT_IMPLEMENTED\"\n";
    js << "  },\n";
    js << "  \"pass\": " << (pass ? "true" : "false") << ",\n";
    js << "  \"git_sha\": \"" << sha << "\",\n";
    js << "  \"command\": \"make turbo-buffer-intel-receipt\",\n";
    js << "  \"note\": \"Machine B LIVE: zeMemAllocHost/Shared/Device via "
          "turbo_buffer rent/return. SHARED/DEVICE query as those types, not "
          "HOST. DEVICE proven with Level Zero memcpy, not a host stand-in. "
          "PINNED on ZE is NOT_IMPLEMENTED.\"\n";
    js << "}\n";

    std::ofstream out(out_path);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", out_path.c_str());
        return 1;
    }
    out << js.str();
    std::fprintf(
        stderr,
        "wrote %s pass=%s memcpy=%s reuse=%s\n",
        out_path.c_str(),
        pass ? "true" : "false",
        memcpy_ok ? "ok" : "fail",
        reuse ? "0" : "nonzero"
    );
    return pass ? 0 : 2;
}
