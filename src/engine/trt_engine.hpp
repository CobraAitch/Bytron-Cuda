#pragma once

#include <NvInfer.h>
#include <filesystem>
#include <string>
#include <memory>
#include "bytron/types.hpp"
#include "core/pipeline_slot.hpp"

namespace bytron {

    /// <summary>
    /// Manages TensorRT engine lifecycle: build from ONNX, cache to disk,
    /// load from cache, and create per-slot execution contexts.
    /// </summary>
    class TrtEngine {
        public:
            struct BuildConfig {
                std::filesystem::path onnx_path;
                std::filesystem::path engine_cache_path;
                Precision             precision = Precision::FP16;
                int                   max_batch_size = 1;
                int                   opt_batch_size = 1;
                std::size_t           max_workspace_bytes = 1ULL << 28; // 256 MB
                int                   max_aux_streams = 0;
                std::filesystem::path calibration_data_dir;
                int                   calibration_batch_size = 8;
            };

            explicit TrtEngine(const BuildConfig& config);

            /// <summary>
            /// Create an independent execution context (one per pipeline slot).
            /// </summary>
            [[nodiscard]] TrtUniquePtr<nvinfer1::IExecutionContext> create_context();

            [[nodiscard]] const char* input_name() const;
            [[nodiscard]] const char* output_name() const;
            [[nodiscard]] nvinfer1::Dims input_dims() const;
            [[nodiscard]] nvinfer1::Dims output_dims() const;
            [[nodiscard]] int model_width() const;
            [[nodiscard]] int model_height() const;
            [[nodiscard]] int max_detections() const;

        private:
            void build_from_onnx(const BuildConfig& config);
            void load_from_cache(const std::filesystem::path& path);
            void serialize_to_file(const std::filesystem::path& path);

            TrtUniquePtr<nvinfer1::IRuntime>    runtime_;
            TrtUniquePtr<nvinfer1::ICudaEngine> engine_;

            std::string input_name_;
            std::string output_name_;
    };
}
