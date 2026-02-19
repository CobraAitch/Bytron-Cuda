#include "core/pipeline.hpp"
#include "output/display_sink.hpp"
#include "output/callback_sink.hpp"
#include "util/config.hpp"
#include "util/logger.hpp"

#include <spdlog/spdlog.h>
#include <iostream>
#include <csignal>
#include <memory>

static std::unique_ptr<bytron::Pipeline> g_pipeline;

static void signal_handler(int /*sig*/) {
    spdlog::info("Interrupt received, stopping pipeline...");
    if (g_pipeline) g_pipeline->stop();
}

static void print_usage() {
    std::cout << R"(
Bytron - Pipelined CUDA + TensorRT YOLO26 Inference Engine

Usage:
  bytron --video <source> --onnx <model.onnx> [options]

Options:
  -v, --video <uri>       Video source (file path, RTSP URL, or camera index)
  -m, --onnx <path>       Path to YOLO26 ONNX model
  -p, --precision <fp16|int8>  Inference precision (default: fp16)
  -c, --config <path>     JSON config file
  --conf <float>          Confidence threshold (default: 0.25)
  --slots <n>             Pipeline slots / frames in flight (default: 3)
  --calibration-dir <dir> Directory of images for INT8 calibration
  --calibration-batch <n> Batch size for INT8 calibration (default: 8)
  --no-display            Disable display window
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

        if (cfg.video_uri.empty()) {
            spdlog::error("No video source specified. Use --video <source>");
            return 1;
        }
        if (cfg.onnx_path.empty()) {
            spdlog::error("No ONNX model specified. Use --onnx <model.onnx>");
            return 1;
        }

        spdlog::info("Bytron starting...");
        spdlog::info("Video:     {}", cfg.video_uri);
        spdlog::info("Model:     {}", cfg.onnx_path.string());
        spdlog::info("Precision: {}", cfg.precision == bytron::Precision::FP16 ? "FP16" : "INT8");
        spdlog::info("Slots:     {}", cfg.num_pipeline_slots);

        g_pipeline = std::make_unique<bytron::Pipeline>(cfg);

        std::signal(SIGINT, signal_handler);
        std::signal(SIGTERM, signal_handler);

        if (cfg.display) {
            bytron::DisplaySink display_sink("Bytron");
            g_pipeline->run(display_sink);
        } else {
            bytron::CallbackSink callback_sink([](const bytron::FrameContext&) {});
            g_pipeline->run(callback_sink);
        }

        auto m = g_pipeline->metrics();
        spdlog::info("Final: {} frames processed, {:.2f} ms avg latency",
                     m.total_frames, m.avg_latency_ms);

    } catch (const std::exception& e) {
        spdlog::error("Fatal: {}", e.what());
        return 1;
    }

    return 0;
}
