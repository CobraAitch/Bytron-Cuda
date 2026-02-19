#pragma once

#include "bytron/types.hpp"
#include <filesystem>
#include <string>

namespace bytron {

    /// <summary>
    /// Runtime configuration for the Bytron pipeline.
    /// Can be loaded from a JSON file and/or overridden by CLI arguments.
    /// </summary>
    struct Config {
        // Video source
        std::string video_uri;

        // Model
        std::filesystem::path onnx_path;
        std::filesystem::path engine_cache_dir = "models";
        Precision precision = Precision::FP16;
        int model_width = 640;
        int model_height = 640;

        // Detection
        float confidence_threshold = 0.25f;
        int   max_detections = 300;

        // Pipeline
        int  num_pipeline_slots = 3;
        bool use_cuda_graphs = true;
        int  max_frame_width = 3840;
        int  max_frame_height = 2160;

        // INT8 calibration
        std::filesystem::path calibration_images_dir;
        int calibration_batch_size = 8;

        // Output
        bool display = true;

        [[nodiscard]] std::filesystem::path engine_cache_path() const;

        static Config from_file(const std::filesystem::path& path);
        void apply_cli(int argc, char** argv);
    };
}
