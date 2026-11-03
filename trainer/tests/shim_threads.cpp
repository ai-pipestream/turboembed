// SPDX-License-Identifier: Apache-2.0
#include "../cuda/shim.h"
#include "../cuda/train.h"
#include <array>
#include <atomic>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    constexpr int workers = 4, count = 8192, rounds = 50;
    std::array<std::array<const dim3 *, 4>, workers> addresses{};
    std::atomic<int> ready{0}, wrong{0};
    std::atomic<bool> go{false}, safe{false};
    std::vector<std::thread> jobs;
    for (int t = 0; t < workers; ++t) jobs.emplace_back([&, t] {
        addresses[t] = {&gridDim, &blockDim, &blockIdx, &threadIdx};
        ready.fetch_add(1);
        while (!go.load()) std::this_thread::yield();
        // Refuse shared launch state before running kernels, making the
        // regression deterministic without deliberately causing a race.
        if (!safe.load()) return;
        std::vector<float> x(count, float(t + 1)), y(count, 0.0f);
        for (int i = 0; i < rounds; ++i)
            if (tt_axpy(y.data(), x.data(), count, 1.0f) != 0) wrong.fetch_add(1);
        for (float v : y) if (v != float((t + 1) * rounds)) wrong.fetch_add(1);
    });
    while (ready.load() != workers) std::this_thread::yield();
    bool distinct = true;
    for (int t = 0; t < workers; ++t)
        for (int u = t + 1; u < workers; ++u)
            for (int i = 0; i < 4; ++i)
                if (addresses[t][i] == addresses[u][i]) distinct = false;
    safe.store(distinct);
    go.store(true);
    for (auto &job : jobs) job.join();
    if (!distinct) std::fputs("launch state is shared across host threads\n", stderr);
    if (wrong.load()) std::fprintf(stderr, "incorrect kernel results: %d\n", wrong.load());
    return distinct && wrong.load() == 0 ? 0 : 1;
}
