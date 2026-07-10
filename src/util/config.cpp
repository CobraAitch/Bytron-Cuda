/**
 * @file config.cpp
 * @brief JSON parsing and CLI argument processing for Config.
 */

#include "util/config.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <format>
#include <stdexcept>
#include <string_view>

namespace bytron {

    std::filesystem::path Config::engine_cache_path() const {
        auto stem = onnx_path.stem().string();
        std::string precision_str = (precision == Precision::FP16) ? "fp16" : "int8";
        auto filename = stem + "_" + precision_str + ".engine";
        return engine_cache_dir / filename;
    }

    Config Config::from_file(const std::filesystem::path& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open config file: " + path.string());
        }

        auto j = nlohmann::json::parse(file);
        Config cfg;

        if (j.contains("gpu_device"))              cfg.gpu_device             = j["gpu_device"];
        if (j.contains("monitor_index"))          cfg.monitor_index          = j["monitor_index"];
        if (j.contains("onnx_path"))              cfg.onnx_path              = j["onnx_path"].get<std::string>();
        if (j.contains("engine_cache_dir"))       cfg.engine_cache_dir       = j["engine_cache_dir"].get<std::string>();
        if (j.contains("precision")) {
            auto p = j["precision"].get<std::string>();
            cfg.precision = (p == "int8" || p == "INT8") ? Precision::INT8 : Precision::FP16;
        }
        if (j.contains("model_width"))            cfg.model_width            = j["model_width"];
        if (j.contains("model_height"))           cfg.model_height           = j["model_height"];
        if (j.contains("confidence_threshold"))   cfg.confidence_threshold   = j["confidence_threshold"];
        if (j.contains("max_detections"))         cfg.max_detections         = j["max_detections"];
        if (j.contains("num_pipeline_slots"))     cfg.num_pipeline_slots     = j["num_pipeline_slots"];
        if (j.contains("use_cuda_graphs"))        cfg.use_cuda_graphs        = j["use_cuda_graphs"];
        if (j.contains("calibration_images_dir")) cfg.calibration_images_dir = j["calibration_images_dir"].get<std::string>();
        if (j.contains("calibration_batch_size")) cfg.calibration_batch_size = j["calibration_batch_size"];
        if (j.contains("max_workspace_bytes"))     cfg.max_workspace_bytes    = j["max_workspace_bytes"].get<std::size_t>();
        if (j.contains("max_aux_streams"))        cfg.max_aux_streams        = j["max_aux_streams"];
        if (j.contains("builder_opt_level"))      cfg.builder_opt_level      = j["builder_opt_level"];
        if (j.contains("l2_cache_bytes"))         cfg.l2_cache_bytes         = j["l2_cache_bytes"].get<std::size_t>();
        if (j.contains("display"))                cfg.display                = j["display"];
        if (j.contains("overlay"))                cfg.overlay                = j["overlay"];
        if (j.contains("capture_backend")) {
            auto b = j["capture_backend"].get<std::string>();
            cfg.capture_backend = (b == "dxgi" || b == "duplication" || b == "desktop_duplication")
                                ? CaptureBackend::DesktopDuplication
                                : CaptureBackend::GraphicsCapture;
        }

        if (j.contains("detection_schema")) {
            auto& s = j["detection_schema"];
            if (s.contains("x1_idx"))         cfg.detection_schema.x1_idx         = s["x1_idx"];
            if (s.contains("y1_idx"))         cfg.detection_schema.y1_idx         = s["y1_idx"];
            if (s.contains("x2_idx"))         cfg.detection_schema.x2_idx         = s["x2_idx"];
            if (s.contains("y2_idx"))         cfg.detection_schema.y2_idx         = s["y2_idx"];
            if (s.contains("confidence_idx")) cfg.detection_schema.confidence_idx = s["confidence_idx"];
            if (s.contains("class_id_idx"))   cfg.detection_schema.class_id_idx   = s["class_id_idx"];
        }

        cfg.validate();
        return cfg;
    }

    void Config::validate() const {
        if (num_pipeline_slots < 1 || num_pipeline_slots > 8)
            throw std::runtime_error(std::format("num_pipeline_slots {} out of range [1, 8]", num_pipeline_slots));
        if (confidence_threshold < 0.0f || confidence_threshold > 1.0f)
            throw std::runtime_error(std::format("confidence_threshold {} out of range [0, 1]", confidence_threshold));
        if (max_detections <= 0)
            throw std::runtime_error(std::format("max_detections must be > 0, got {}", max_detections));
        if (max_frame_width <= 0 || max_frame_height <= 0)
            throw std::runtime_error("max_frame_width and max_frame_height must be > 0");
        if (calibration_batch_size <= 0)
            throw std::runtime_error(std::format("calibration_batch_size must be > 0, got {}", calibration_batch_size));
        if (builder_opt_level < 0 || builder_opt_level > 5)
            throw std::runtime_error(std::format("builder_opt_level {} out of range [0, 5]", builder_opt_level));
    }

    void Config::apply_cli(int argc, char** argv) {
        // Pass 1: find --config early so pass 2 CLI flags can override JSON values.
        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];
            if (arg == "--config" || arg == "-c") {
                if (i + 1 < argc) {
                    *this = Config::from_file(argv[++i]);
                    break;
                }
            }
        }

        for (int i = 1; i < argc; ++i) {
            std::string_view arg = argv[i];

            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error(std::format("Missing value for {}", arg));
                return argv[++i];
            };

            if      (arg == "--gpu"     || arg == "-g") gpu_device             = std::stoi(next());
            else if (arg == "--monitor"  || arg == "-d") monitor_index          = std::stoi(next());
            else if (arg == "--onnx"     || arg == "-m") onnx_path              = next();
            else if (arg == "--precision" || arg == "-p") {
                auto p = next();
                precision = (p == "int8" || p == "INT8") ? Precision::INT8 : Precision::FP16;
            }
            else if (arg == "--conf")                    confidence_threshold   = std::stof(next());
            else if (arg == "--slots")                   num_pipeline_slots     = std::stoi(next());
            else if (arg == "--capture") {
                auto b = next();
                capture_backend = (b == "dxgi" || b == "duplication" || b == "desktop_duplication")
                                ? CaptureBackend::DesktopDuplication
                                : CaptureBackend::GraphicsCapture;
            }
            else if (arg == "--no-display")              display                = false;
            else if (arg == "--no-overlay")              overlay                = false;
            else if (arg == "--no-cuda-graphs")          use_cuda_graphs        = false;
            else if (arg == "--calibration-dir")         calibration_images_dir = next();
            else if (arg == "--calibration-batch")       calibration_batch_size = std::stoi(next());
            else if (arg == "--config" || arg == "-c")   { next(); }
            else {
                spdlog::warn("Unknown argument: {}", arg);
            }
        }
    }
}
