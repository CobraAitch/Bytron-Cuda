/**
 * @file config.hpp
 * @brief Runtime configuration for the Bytron inference pipeline.
 */

#pragma once

#include "bytron/types.hpp"
#include <filesystem>
#include <string>

namespace bytron {

    /**
     * @brief Complete runtime configuration for pipeline, engine, and display.
     *
     * Populated from JSON file and/or CLI arguments. All fields have sensible
     * defaults; only onnx_path is required for operation.
     *
     * @code
     * // JSON config example:
     * // {
     * //   "onnx_path": "models/yolov26s.onnx",
     * //   "precision": "fp16",
     * //   "monitor_index": 0,
     * //   "confidence_threshold": 0.3,
     * //   "num_pipeline_slots": 3,
     * //   "use_cuda_graphs": true,
     * //   "display": true,
     * //   "overlay": true
     * // }
     * @endcode
     *
     * @note Thread safety: Not thread-safe. Construct and configure before
     *       passing to Pipeline.
     */
    struct Config {
        int gpu_device    = 0;  ///< CUDA device index (0 = first GPU)
        int monitor_index = 0;  ///< DXGI output index for desktop capture (0 = primary)

        std::filesystem::path onnx_path;                       ///< Path to YOLO26 ONNX model (required)
        std::filesystem::path engine_cache_dir = "models";     ///< Directory for cached .engine files
        Precision precision    = Precision::FP16;              ///< Inference precision (FP16 or INT8)
        int       model_width  = 640;                          ///< Model input width in pixels
        int       model_height = 640;                          ///< Model input height in pixels

        float confidence_threshold = 0.25f;  ///< Minimum detection confidence [0.0, 1.0]
        int   max_detections       = 300;    ///< Maximum detections per frame (matches YOLO26 output dim)

        int         num_pipeline_slots  = 3;             ///< Concurrent inference slots [2, MAX_SLOTS]
        bool        use_cuda_graphs    = true;           ///< Enable CUDA graph capture for reduced launch overhead
        int         max_frame_width    = 3840;           ///< Maximum capture frame width for buffer pre-allocation
        int         max_frame_height   = 2160;           ///< Maximum capture frame height for buffer pre-allocation
        std::size_t max_workspace_bytes = 1ULL << 30;    ///< TRT workspace limit (default 1 GB)
        int         max_aux_streams    = 4;              ///< TRT auxiliary streams for layer parallelism

        std::filesystem::path calibration_images_dir;  ///< Image directory for INT8 calibration
        int calibration_batch_size = 8;                ///< Images per INT8 calibration batch

        bool display = true;   ///< Enable visual display output
        bool overlay = true;   ///< Use transparent overlay (true) vs full-frame Vulkan window (false)

        DetectionSchema detection_schema;  ///< Output tensor field mapping (default: YOLOv26 layout)

        /**
         * @brief Compute the engine cache file path from onnx_path and precision.
         * @return Path like "models/yolov26s_fp16.engine".
         */
        [[nodiscard]] std::filesystem::path engine_cache_path() const;

        /**
         * @brief Load configuration from a JSON file.
         *
         * Missing keys retain their default values.
         *
         * @throws std::runtime_error If the file cannot be opened or parsed.
         */
        [[nodiscard]] static Config from_file(const std::filesystem::path& path);

        /**
         * @brief Apply command-line arguments over current configuration.
         *
         * Two-pass design: first pass loads --config file (replacing all values),
         * second pass applies individual flags (overriding file values).
         *
         * @throws std::runtime_error On missing argument values.
         */
        void apply_cli(int argc, char** argv);

        /** @brief Validate field ranges. Throws std::runtime_error on invalid values. */
        void validate() const;
    };
}
