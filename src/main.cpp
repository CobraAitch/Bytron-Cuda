/**
 * @file main.cpp
 * @brief Application entry point: CLI parsing, sink selection, and pipeline execution.
 */

#include "core/pipeline.hpp"
#include "output/overlay_sink.hpp"
#include "output/display_sink.hpp"
#include "output/callback_sink.hpp"
#include "util/config.hpp"
#include "util/logger.hpp"

#include <spdlog/spdlog.h>
#include <iostream>
#include <memory>

#include <windows.h>

static std::unique_ptr<bytron::Pipeline> g_pipeline;

/**
 * @brief Windows console control handler for graceful shutdown.
 *
 * Unlike std::signal, SetConsoleCtrlHandler callbacks are invoked on a
 * dedicated OS thread, making it safe to call Pipeline::stop() which
 * touches atomics and condition variables.
 */
static BOOL WINAPI console_handler(DWORD event) {
    if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT || event == CTRL_CLOSE_EVENT) {
        if (g_pipeline) g_pipeline->stop();
        return TRUE;
    }
    return FALSE;
}

static void print_usage() {
    std::cout << R"(
Bytron - Pipelined CUDA + TensorRT YOLO26 Inference Engine

Usage:
  bytron --onnx <model.onnx> [options]

Captures the desktop via DXGI Desktop Duplication (primary monitor by default).

Options:
  -m, --onnx <path>       Path to YOLO26 ONNX model
  -g, --gpu <index>       CUDA device index (default: 0)
  -d, --monitor <index>   Monitor index for desktop capture (default: 0)
  -p, --precision <fp16|int8>  Inference precision (default: fp16)
  -c, --config <path>     JSON config file
  --conf <float>          Confidence threshold (default: 0.25)
  --slots <n>             Pipeline slots / frames in flight (default: 3)
  --capture <wgc|dxgi>    Capture backend (default: wgc = Windows.Graphics.Capture)
  --calibration-dir <dir> Directory of images for INT8 calibration
  --calibration-batch <n> Batch size for INT8 calibration (default: 8)
  --no-display            Disable display window
  --no-overlay            Use full-frame display instead of transparent overlay
  --no-cuda-graphs        Disable CUDA graph optimization
)";
}

int main(int argc, char** argv) {
    bytron::log::init();

    if (argc < 2) {
        print_usage();
        return 1;
    }

    try {
        bytron::Config cfg;
        cfg.apply_cli(argc, argv);

        if (cfg.onnx_path.empty()) {
            spdlog::error("No ONNX model specified. Use --onnx <model.onnx>");
            return 1;
        }

        spdlog::info("Bytron starting...");
        spdlog::info("GPU:       {}", cfg.gpu_device);
        spdlog::info("Monitor:   {}", cfg.monitor_index);
        spdlog::info("Model:     {}", cfg.onnx_path.string());
        spdlog::info("Precision: {}", cfg.precision == bytron::Precision::FP16 ? "FP16" : "INT8");
        spdlog::info("Slots:     {}", cfg.num_pipeline_slots);

        // Raise the process class so capture/inference threads (which run at
        // THREAD_PRIORITY_HIGHEST) outrank background load. Threads block on
        // events/CVs between frames, so this does not starve the system.
        if (!SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
            spdlog::warn("Failed to raise process priority class");

        g_pipeline = std::make_unique<bytron::Pipeline>(cfg);

        SetConsoleCtrlHandler(console_handler, TRUE);

        // Sink priority: overlay (transparent, capture-excluded) > full-frame
        // display (Vulkan CUDA interop) > headless callback (metadata only).
        if (cfg.display && cfg.overlay) {
            bytron::OverlaySink overlay_sink(g_pipeline->capture_width(),
                                              g_pipeline->capture_height());
            g_pipeline->run(overlay_sink);
        } else if (cfg.display) {
            bytron::DisplaySink display_sink("Bytron",
                                              g_pipeline->capture_width(),
                                              g_pipeline->capture_height());
            g_pipeline->run(display_sink);
        } else {
            bytron::CallbackSink callback_sink([](const bytron::FrameContext&) {});
            g_pipeline->run(callback_sink);
        }

        auto m = g_pipeline->metrics();
        if (m.eager_frames > 0) {
            spdlog::info("Final: {} frames, {:.2f} ms avg [pre {:.2f} | trt {:.2f} | d2h {:.2f}]",
                         m.total_frames, m.avg_latency_ms,
                         m.last_pre_ms, m.last_trt_ms, m.last_d2h_ms);
        } else {
            spdlog::info("Final: {} frames processed, {:.2f} ms avg latency",
                         m.total_frames, m.avg_latency_ms);
        }

    } catch (const std::exception& e) {
        spdlog::error("Fatal: {}", e.what());
        return 1;
    }

    return 0;
}
