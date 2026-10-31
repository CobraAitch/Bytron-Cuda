/**
 * @file bench_cpu.cpp
 * @brief Micro-benchmarks for the CPU-side hot paths (letterbox + postprocess).
 *
 * These are the only per-frame stages that run on the CPU; everything else is
 * on the GPU and measured via CUDA events at runtime. Kept dependency-free
 * (std::chrono only) so it builds and runs without a GPU.
 */

#include "preprocess/letterbox.hpp"
#include "postprocess/postprocessor.hpp"
#include "core/frame_context.hpp"

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

using namespace bytron;
using clk = std::chrono::steady_clock;

namespace {
    template <typename F>
    double bench_ns(const char* name, std::size_t iters, F&& fn) {
        // Warm up.
        for (std::size_t i = 0; i < iters / 10 + 1; ++i) fn();

        auto t0 = clk::now();
        for (std::size_t i = 0; i < iters; ++i) fn();
        auto t1 = clk::now();

        double ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / iters;
        std::printf("  %-28s %8.1f ns/op   %10.2f M ops/s\n",
                    name, ns, 1e3 / ns);
        return ns;
    }
}

int main() {
    std::printf("Bytron CPU micro-benchmarks\n---------------------------\n");

    // --- Letterbox ---
    {
        // Vary source sizes so the result can't be constant-folded away.
        std::mt19937 rng(1);
        std::uniform_int_distribution<int> dw(640, 3840), dh(480, 2160);
        std::vector<std::pair<int, int>> sizes(4096);
        for (auto& s : sizes) s = { dw(rng), dh(rng) };

        std::size_t idx = 0;
        volatile int sink = 0;
        bench_ns("compute_letterbox", 5'000'000, [&] {
            auto& s = sizes[idx++ & 4095];
            auto p = compute_letterbox(s.first, s.second, 640, 640);
            sink += p.pad_x + p.pad_y + p.new_w;
        });
    }

    // --- Postprocess ---
    {
        constexpr int max_dets = 300, stride = 6;
        Postprocessor pp(0.25f, max_dets, stride);

        // Realistic output: ~40 detections above threshold out of 300 rows.
        std::mt19937 rng(2);
        std::uniform_real_distribution<float> coord(0.0f, 640.0f);
        std::uniform_real_distribution<float> conf(0.0f, 1.0f);
        std::vector<float> out(max_dets * stride);
        for (int i = 0; i < max_dets; ++i) {
            float* d = out.data() + i * stride;
            d[0] = coord(rng); d[1] = coord(rng);
            d[2] = d[0] + 40.0f; d[3] = d[1] + 40.0f;
            d[4] = (i < 40) ? 0.9f : conf(rng) * 0.24f;  // ~40 kept
            d[5] = static_cast<float>(i % 80);
        }

        FrameContext ctx;
        ctx.src_width = 1920; ctx.src_height = 1080;
        ctx.scale = 640.0f / 1920.0f; ctx.pad_x = 0.0f; ctx.pad_y = 140.0f;

        bench_ns("Postprocessor::process(300)", 1'000'000, [&] {
            pp.process(out, ctx);
        });
    }

    std::printf("done.\n");
    return 0;
}
