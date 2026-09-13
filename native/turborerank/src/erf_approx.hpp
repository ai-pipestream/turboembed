// SPDX-License-Identifier: Apache-2.0
//
// Software erf for GELU on Metal. MSL metal_math (Xcode 26 / Metal 32023)
// has exp, fma, fabs, copysign — and no erf / erfc / tgamma.
// Same Horner and domains as the Metal kernel in metal_api.mm; keep in sync.
//
// Near / mid rationals are the Hart–Cheney single-precision tables
// (Computer Approximations, 1968). The tail is the complementary
// exp(-x^2) form so we never call a missing erfc().

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace turborerank {
namespace impl {

inline float erf_as(float x) {
    const float ax = std::fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * ax);
    const float y =
        1.0f -
        (((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t -
          0.284496736f) *
             t +
         0.254829592f) *
            t * std::exp(-x * x);
    return std::copysign(y, x);
}

inline uint32_t f32_bits(float x) {
    uint32_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
    return u;
}

inline float bits_f32(uint32_t u) {
    float x = 0;
    std::memcpy(&x, &u, sizeof(x));
    return x;
}

inline float erf_hart(float x) {
    const float ax = std::fabs(x);
    if (!(ax == ax)) {
        return x;
    }
    if (ax >= 4.0f) {
        return std::copysign(1.0f, x);
    }
    if (ax < 0.84375f) {
        const float z = x * x;
        const float p = std::fma(std::fma(-1.86261395e-03f, z, -3.36030394e-01f), z,
                                 1.28379166e-01f);
        const float q = std::fma(
            std::fma(std::fma(-1.98859372e-03f, z, 2.16070414e-02f), z,
                     3.12324315e-01f),
            z,
            1.0f
        );
        return std::fma(x, p / q, x);
    }
    if (ax < 1.25f) {
        const float s = ax - 1.0f;
        const float P = std::fma(
            std::fma(std::fma(8.67677554e-02f, s, -2.09395722e-01f), s,
                     4.15109307e-01f),
            s,
            3.65041046e-06f
        );
        const float Q = std::fma(
            std::fma(std::fma(3.92478965e-02f, s, 3.71248513e-01f), s,
                     4.95560974e-01f),
            s,
            1.0f
        );
        return std::copysign(8.42697144e-01f + P / Q, x);
    }
    // 1.25 <= |x| < 4: erfc via exp(-x^2) * rational(1/x^2) / x
    const float s = 1.0f / (ax * ax);
    float R;
    float S;
    if (ax < 2.85715f) {
        R = std::fma(
            std::fma(std::fma(-6.91554189e-01f, s, -1.66828310f), s,
                     -5.43658376e-01f),
            s,
            -9.88156721e-03f
        );
        S = std::fma(
            std::fma(std::fma(5.53855181e-01f, s, 4.10799170f), s, 4.48581553f),
            s,
            1.0f
        );
    } else {
        R = std::fma(std::fma(-1.84115684f, s, -5.48049808e-01f), s,
                     -9.86496918e-03f);
        S = std::fma(
            std::fma(std::fma(-7.61900663e-01f, s, 3.04982710f), s, 4.87132740f),
            s,
            1.0f
        );
    }
    const float z = bits_f32(f32_bits(ax) & 0xffffe000u);
    const float r = std::exp(-z * z - 0.5625f) *
                    std::exp((z - ax) * (z + ax) + R / S);
    return std::copysign(1.0f - r / ax, x);
}

inline float gelu_erf_as(float x) {
    return 0.5f * x * (1.0f + erf_as(x * 0.7071067811865476f));
}

inline float gelu_erf_hart(float x) {
    return 0.5f * x * (1.0f + erf_hart(x * 0.7071067811865476f));
}

} // namespace impl
} // namespace turborerank
