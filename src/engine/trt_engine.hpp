/**
 * @file trt_engine.hpp
 * @brief TensorRT engine builder, cache manager, and inference context factory.
 */

#pragma once

#include <NvInfer.h>
#include <filesystem>
#include <memory>
#include <string>
#include "bytron/types.hpp"
#include "core/pipeline_slot.hpp"

namespace bytron {

    /**
     * @brief Manages TensorRT engine lifecycle: build from ONNX, cache, and create contexts.
     *
     * On construction, loads a cached .engine file if it exists, otherwise builds
     * from ONNX and serializes. A .timing file accelerates subsequent builds.
     * If cache deserialization fails (stale from TRT version change), delete
     * the .engine file and rebuild.
     *
     * @note Thread safety: create_context() and const accessors are safe after
     *       construction. The engine itself is immutable post-construction.
     */
    class TrtEngine {
    public:
        /** @brief Configuration for building a TensorRT engine from ONNX. */
        struct BuildConfig {
            std::filesystem::path onnx_path;                       ///< Source ONNX model path
            std::filesystem::path engine_cache_path;                ///< Where to save/load serialized engine
            Precision             precision              = Precision::FP16; ///< FP16 or INT8
            int                   max_batch_size         = 1;       ///< Max batch for optimization profile
            int                   opt_batch_size         = 1;       ///< Optimal batch for optimization profile
            std::size_t           max_workspace_bytes    = 1ULL << 28; ///< TRT workspace limit (default 256 MB)
            int                   max_aux_streams        = 0;       ///< TRT auxiliary streams for layer parallelism
            std::filesystem::path calibration_data_dir;             ///< Image directory for INT8 calibration
            int                   calibration_batch_size = 8;       ///< Images per INT8 calibration batch
        };

        /**
         * @brief Build or load a TensorRT engine.
         * @throws std::runtime_error On ONNX parse failure, build failure,
         *         or stale cache deserialization failure.
         */
        explicit TrtEngine(const BuildConfig& config);

        /**
         * @brief Create a new TRT execution context for a pipeline slot.
         *
         * Each slot needs its own context for concurrent enqueue.
         *
         * @throws std::runtime_error If context creation fails.
         */
        [[nodiscard]] TrtUniquePtr<nvinfer1::IExecutionContext> create_context();

        [[nodiscard]] const char*    input_name()     const noexcept;  ///< First input tensor name
        [[nodiscard]] const char*    output_name()    const noexcept;  ///< First output tensor name
        [[nodiscard]] nvinfer1::Dims input_dims()     const;           ///< Input shape [N,C,H,W]
        [[nodiscard]] nvinfer1::Dims output_dims()    const;           ///< Output shape [N,300,6]
        [[nodiscard]] int            model_width()    const;           ///< input_dims().d[3]
        [[nodiscard]] int            model_height()   const;           ///< input_dims().d[2]
        [[nodiscard]] int            max_detections() const;           ///< output_dims().d[1]

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
