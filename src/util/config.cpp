#include "util/config.hpp"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <fstream>
#include <stdexcept>

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

        if (j.contains("video_uri"))              cfg.video_uri = j["video_uri"];
        if (j.contains("onnx_path"))              cfg.onnx_path = j["onnx_path"].get<std::string>();
        if (j.contains("engine_cache_dir"))       cfg.engine_cache_dir = j["engine_cache_dir"].get<std::string>();
        if (j.contains("precision")) {
            auto p = j["precision"].get<std::string>();
            cfg.precision = (p == "int8" || p == "INT8") ? Precision::INT8 : Precision::FP16;
        }
        if (j.contains("model_width"))            cfg.model_width = j["model_width"];
        if (j.contains("model_height"))           cfg.model_height = j["model_height"];
        if (j.contains("confidence_threshold"))   cfg.confidence_threshold = j["confidence_threshold"];
        if (j.contains("max_detections"))         cfg.max_detections = j["max_detections"];
        if (j.contains("num_pipeline_slots"))     cfg.num_pipeline_slots = j["num_pipeline_slots"];
        if (j.contains("use_cuda_graphs"))        cfg.use_cuda_graphs = j["use_cuda_graphs"];
        if (j.contains("calibration_images_dir")) cfg.calibration_images_dir = j["calibration_images_dir"].get<std::string>();
        if (j.contains("calibration_batch_size")) cfg.calibration_batch_size = j["calibration_batch_size"];
        if (j.contains("display"))                cfg.display = j["display"];

        return cfg;
    }

    void Config::apply_cli(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];

            auto next = [&]() -> std::string {
                if (i + 1 >= argc) throw std::runtime_error("Missing value for " + arg);
                return argv[++i];
            };

            if (arg == "--video" || arg == "-v")           video_uri = next();
            else if (arg == "--onnx" || arg == "-m")       onnx_path = next();
            else if (arg == "--precision" || arg == "-p") {
                auto p = next();
                precision = (p == "int8" || p == "INT8") ? Precision::INT8 : Precision::FP16;
            }
            else if (arg == "--conf")                      confidence_threshold = std::stof(next());
            else if (arg == "--slots")                     num_pipeline_slots = std::stoi(next());
            else if (arg == "--no-display")                display = false;
            else if (arg == "--no-cuda-graphs")            use_cuda_graphs = false;
            else if (arg == "--calibration-dir")           calibration_images_dir = next();
            else if (arg == "--calibration-batch")         calibration_batch_size = std::stoi(next());
            else if (arg == "--config" || arg == "-c") {
                auto file_cfg = Config::from_file(next());
                if (video_uri.empty()) video_uri = file_cfg.video_uri;
                if (onnx_path.empty()) onnx_path = file_cfg.onnx_path;
            }
            else {
                spdlog::warn("Unknown argument: {}", arg);
            }
        }
    }
}
