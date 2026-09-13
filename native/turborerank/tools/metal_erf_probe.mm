// SPDX-License-Identifier: Apache-2.0
//
// Machine C proof that MSL still has no erf(), and that the Hart GELU
// path on Metal matches host libm closer than A&S 7.1.26.
// No CPU GELU fallback — the grid runs as a compute kernel.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "erf_approx.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

const char *kTryNativeErf = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] = erf(x[i]);
}
)METAL";

const char *kTryPreciseErf = R"METAL(
#include <metal_stdlib>
using namespace metal;
kernel void k(device float *x [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    x[i] = precise::erf(x[i]);
}
)METAL";

const char *kGridSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

inline float erf_as(float x) {
    const float ax = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y =
        1.0f -
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
          0.284496736f) *
             t +
         0.254829592f) *
            t * exp(-x * x);
    return copysign(y, x);
}

inline float erf_hart(float x) {
    const float ax = fabs(x);
    if (!(ax == ax)) {
        return x;
    }
    if (ax >= 4.0f) {
        return copysign(1.0f, x);
    }
    if (ax < 0.84375f) {
        const float z = x * x;
        const float p =
            fma(fma(-1.86261395e-03f, z, -3.36030394e-01f), z, 1.28379166e-01f);
        const float q = fma(
            fma(fma(-1.98859372e-03f, z, 2.16070414e-02f), z, 3.12324315e-01f),
            z,
            1.0f
        );
        return fma(x, p / q, x);
    }
    if (ax < 1.25f) {
        const float s = ax - 1.0f;
        const float P = fma(
            fma(fma(8.67677554e-02f, s, -2.09395722e-01f), s, 4.15109307e-01f),
            s,
            3.65041046e-06f
        );
        const float Q = fma(
            fma(fma(3.92478965e-02f, s, 3.71248513e-01f), s, 4.95560974e-01f),
            s,
            1.0f
        );
        return copysign(8.42697144e-01f + P / Q, x);
    }
    const float inv2 = 1.0f / (ax * ax);
    float R;
    float S;
    if (ax < 2.85715f) {
        R = fma(
            fma(fma(-6.91554189e-01f, inv2, -1.66828310f), inv2, -5.43658376e-01f),
            inv2,
            -9.88156721e-03f
        );
        S = fma(
            fma(fma(5.53855181e-01f, inv2, 4.10799170f), inv2, 4.48581553f),
            inv2,
            1.0f
        );
    } else {
        R = fma(fma(-1.84115684f, inv2, -5.48049808e-01f), inv2, -9.86496918e-03f);
        S = fma(
            fma(fma(-7.61900663e-01f, inv2, 3.04982710f), inv2, 4.87132740f),
            inv2,
            1.0f
        );
    }
    const float z = as_type<float>(as_type<uint>(ax) & 0xffffe000u);
    const float r =
        exp(-z * z - 0.5625f) * exp((z - ax) * (z + ax) + R / S);
    return copysign(1.0f - r / ax, x);
}

kernel void as_kernel(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    out[i] = erf_as(in[i]);
}

kernel void hart_kernel(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    out[i] = erf_hart(in[i]);
}

kernel void gelu_hart_kernel(
    device const float *in [[buffer(0)]],
    device float *out [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    const float v = in[i];
    out[i] = 0.5f * v * (1.0f + erf_hart(v * 0.7071067811865476f));
}
)METAL";

id<MTLLibrary> compile(
    id<MTLDevice> dev, const char *src, NSError **err
) {
    MTLCompileOptions *opts = [MTLCompileOptions new];
    if (@available(macOS 15.0, *)) {
        opts.mathMode = MTLMathModeSafe;
    } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        opts.fastMathEnabled = NO;
#pragma clang diagnostic pop
    }
    return [dev newLibraryWithSource:[NSString stringWithUTF8String:src]
                             options:opts
                               error:err];
}

bool expect_erf_missing(id<MTLDevice> dev, const char *src, const char *label) {
    NSError *err = nil;
    id<MTLLibrary> lib = compile(dev, src, &err);
    if (lib != nil) {
        std::fprintf(
            stderr,
            "FAIL %s: MSL compiled erf(); switch GELU to the native builtin\n",
            label
        );
        return false;
    }
    const char *msg =
        err != nil ? err.localizedDescription.UTF8String : "(no NSError)";
    const bool mentions =
        std::strstr(msg, "erf") != nullptr ||
        std::strstr(msg, "undeclared") != nullptr ||
        std::strstr(msg, "no matching") != nullptr ||
        std::strstr(msg, "use of undeclared") != nullptr;
    std::fprintf(stderr, "MSL %s rejected (%s)\n", label, msg);
    if (!mentions) {
        std::fprintf(stderr, "FAIL %s: compile failed but not an erf miss\n", label);
        return false;
    }
    return true;
}

bool run_kernel(
    id<MTLDevice> dev,
    id<MTLCommandQueue> q,
    id<MTLComputePipelineState> pso,
    const float *in,
    float *out,
    NSUInteger n
) {
    const NSUInteger bytes = n * sizeof(float);
    id<MTLBuffer> bin = [dev newBufferWithBytes:in
                                         length:bytes
                                        options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout = [dev newBufferWithLength:bytes
                                          options:MTLResourceStorageModeShared];
    if (bin == nil || bout == nil) {
        std::fprintf(stderr, "FAIL Metal SHARED buffer alloc\n");
        return false;
    }
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    if (cmd == nil || enc == nil) {
        std::fprintf(stderr, "FAIL Metal encoder; refusing CPU fallback\n");
        return false;
    }
    [enc setComputePipelineState:pso];
    [enc setBuffer:bin offset:0 atIndex:0];
    [enc setBuffer:bout offset:0 atIndex:1];
    const NSUInteger tw = std::min(n, pso.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(n, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tw, 1, 1)];
    [enc endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.error != nil) {
        std::fprintf(
            stderr,
            "FAIL kernel: %s; refusing CPU fallback\n",
            cmd.error.localizedDescription.UTF8String
        );
        return false;
    }
    std::memcpy(out, bout.contents, bytes);
    return true;
}

} // namespace

int main() {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (dev == nil) {
        std::fprintf(
            stderr,
            "metal_erf_probe: no MTL device; refusing CPU fallback\n"
        );
        return 1;
    }
    if (!dev.hasUnifiedMemory) {
        std::fprintf(stderr, "metal_erf_probe: no unified memory\n");
        return 1;
    }
    std::fprintf(
        stderr,
        "metal_erf_probe: device=%s metal_stdlib has no erf (Xcode Metal)\n",
        dev.name.UTF8String
    );

    if (!expect_erf_missing(dev, kTryNativeErf, "erf()")) {
        return 2;
    }
    if (!expect_erf_missing(dev, kTryPreciseErf, "precise::erf()")) {
        return 2;
    }

    NSError *err = nil;
    id<MTLLibrary> lib = compile(dev, kGridSrc, &err);
    if (lib == nil) {
        std::fprintf(
            stderr,
            "FAIL Hart/A&S grid compile: %s\n",
            err != nil ? err.localizedDescription.UTF8String : "?"
        );
        return 1;
    }
    auto pipe = [&](const char *name) -> id<MTLComputePipelineState> {
        id<MTLFunction> fn =
            [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
        NSError *pe = nil;
        return fn == nil ? nil : [dev newComputePipelineStateWithFunction:fn error:&pe];
    };
    id<MTLComputePipelineState> as_pso = pipe("as_kernel");
    id<MTLComputePipelineState> hart_pso = pipe("hart_kernel");
    id<MTLComputePipelineState> gelu_pso = pipe("gelu_hart_kernel");
    if (as_pso == nil || hart_pso == nil || gelu_pso == nil) {
        std::fprintf(stderr, "FAIL missing grid pipeline\n");
        return 1;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    if (q == nil) {
        std::fprintf(stderr, "FAIL command queue; refusing CPU fallback\n");
        return 1;
    }

    const int n = 16001;
    std::vector<float> in(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        in[static_cast<size_t>(i)] = static_cast<float>(i - 8000) * 0.001f;
    }
    std::vector<float> as_out(in.size()), hart_out(in.size()), gelu_out(in.size());
    if (!run_kernel(dev, q, as_pso, in.data(), as_out.data(), in.size()) ||
        !run_kernel(dev, q, hart_pso, in.data(), hart_out.data(), in.size()) ||
        !run_kernel(dev, q, gelu_pso, in.data(), gelu_out.data(), in.size())) {
        return 1;
    }

    float max_as = 0, max_hart = 0, max_twin = 0, max_gelu = 0;
    for (int i = 0; i < n; ++i) {
        const float x = in[static_cast<size_t>(i)];
        const float gold = std::erff(x);
        max_as = std::fmax(max_as, std::fabs(as_out[static_cast<size_t>(i)] - gold));
        max_hart =
            std::fmax(max_hart, std::fabs(hart_out[static_cast<size_t>(i)] - gold));
        max_twin = std::fmax(
            max_twin,
            std::fabs(
                hart_out[static_cast<size_t>(i)] - turborerank::impl::erf_hart(x)
            )
        );
        const float gelu_gold =
            0.5f * x * (1.0f + std::erff(x * 0.7071067811865476f));
        max_gelu =
            std::fmax(max_gelu, std::fabs(gelu_out[static_cast<size_t>(i)] - gelu_gold));
    }

    std::fprintf(
        stderr,
        "Metal grid n=%d vs libm erff: max|A&S|=%.6e max|Hart|=%.6e "
        "max|MetalHart-hostHart|=%.6e max|GELU Hart|=%.6e\n",
        n,
        max_as,
        max_hart,
        max_twin,
        max_gelu
    );

    if (!(max_hart < max_as)) {
        std::fprintf(stderr, "FAIL Hart did not beat A&S on Metal vs libm\n");
        return 2;
    }
    if (max_hart > 2e-7f) {
        std::fprintf(stderr, "FAIL Metal Hart drifted from libm erff\n");
        return 2;
    }
    if (max_twin > 2e-7f) {
        std::fprintf(stderr, "FAIL Metal Hart != host twin\n");
        return 2;
    }
    if (max_gelu > 5e-7f) {
        std::fprintf(stderr, "FAIL Metal GELU Hart drifted from libm\n");
        return 2;
    }
    std::fprintf(stderr, "metal_erf_probe: PASS (MSL needs software erf; Hart wins)\n");
    return 0;
}
