/**
 * @file calibrator.hpp
 * @brief INT8 entropy calibrator for TensorRT using stb_image.
 */

#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <string>
#include <vector>

namespace bytron {

    /**
     * @brief Feeds calibration images to TensorRT's INT8 quantization pass.
     *
     * Loads JPEG/PNG/BMP images from a directory, preprocesses each through
     * the fused BGRA kernel (with RGBA-to-BGRA swap on CPU), and returns
     * batches of CHW float tensors. Cache is saved to skip re-calibration.
     *
     * @pre Image directory must contain at least one supported image.
     *
     * @note Thread safety: Called by TRT's builder thread during engine build.
     */
    class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
    public:
        Int8EntropyCalibrator(const std::filesystem::path& image_dir,
                              int batch_size, int input_w, int input_h,
                              const std::string& cache_file = "calibration.cache");
        ~Int8EntropyCalibrator() override;

        [[nodiscard]] int32_t      getBatchSize() const noexcept override;
        bool                       getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override;
        [[nodiscard]] const void*  readCalibrationCache(std::size_t& length) noexcept override;
        void                       writeCalibrationCache(const void* cache, std::size_t length) noexcept override;

    private:
        int         batch_size_;
        int         input_w_;
        int         input_h_;
        std::string cache_file_;

        std::vector<std::filesystem::path> image_paths_;
        int current_batch_ = 0;

        void*        device_input_       = nullptr;
        void*        device_raw_         = nullptr;
        cudaStream_t stream_             = nullptr;
        void*        pinned_staging_     = nullptr;
        std::size_t  pinned_staging_size_ = 0;

        std::vector<char> calibration_cache_;
    };

}
