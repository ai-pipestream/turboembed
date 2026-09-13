// SPDX-License-Identifier: Apache-2.0
// Host A&S vs Hart vs libm erff. No Metal.

#include "erf_approx.hpp"

#include <cmath>
#include <cstdio>

int main() {
    float max_as = 0, max_hart = 0;
    float max_gelu_as = 0, max_gelu_hart = 0;
    double sum_as = 0, sum_hart = 0;
    int n = 0;
    for (int i = -8000; i <= 8000; ++i) {
        const float x = static_cast<float>(i) * 0.001f;
        const float gold = std::erff(x);
        const float a = turborerank::impl::erf_as(x);
        const float h = turborerank::impl::erf_hart(x);
        max_as = std::fmax(max_as, std::fabs(a - gold));
        max_hart = std::fmax(max_hart, std::fabs(h - gold));
        sum_as += std::fabs(static_cast<double>(a) - gold);
        sum_hart += std::fabs(static_cast<double>(h) - gold);
        const float g = std::erff(x * 0.7071067811865476f);
        const float gelu_gold = 0.5f * x * (1.0f + g);
        max_gelu_as = std::fmax(
            max_gelu_as, std::fabs(turborerank::impl::gelu_erf_as(x) - gelu_gold)
        );
        max_gelu_hart = std::fmax(
            max_gelu_hart,
            std::fabs(turborerank::impl::gelu_erf_hart(x) - gelu_gold)
        );
        ++n;
    }
    std::printf(
        "grid n=%d  max|A&S-erff|=%.6e  max|Hart-erff|=%.6e\n"
        "         mean|A&S|=%.6e  mean|Hart|=%.6e\n"
        "GELU     max|A&S|=%.6e  max|Hart|=%.6e\n",
        n,
        max_as,
        max_hart,
        sum_as / n,
        sum_hart / n,
        max_gelu_as,
        max_gelu_hart
    );
    return max_hart < max_as ? 0 : 2;
}
