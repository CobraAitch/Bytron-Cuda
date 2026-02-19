#pragma once

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <vector>
#include <string>

namespace bytron {

    /// <summary>
    /// INT8 entropy calibrator for TensorRT. Loads calibration images from disk,
    /// preprocesses them on the GPU with the fused kernel, and feeds batches
    /// to TensorRT for INT8 range estimation.
    /// </summary>
    class Int8EntropyCalibrator : public nvinfer1::IInt8EntropyCalibrator2 {
        public:
            Int8EntropyCalibrator(
                const std::filesystem::path& image_dir,
                int batch_size,
                int input_w, int input_h,
                const std::string& cache_file = "calibration.cache"
            );

            ~Int8EntropyCalibrator() override;

            [[nodiscard]] int32_t getBatchSize() const noexcept override;
            bool getBatch(void* bindings[], const char* names[], int32_t nbBindings) noexcept override;
            [[nodiscard]] const void* readCalibrationCache(std::size_t& length) noexcept override;
            void writeCalibrationCache(const void* cache, std::size_t length) noexcept override;

        private:
            int batch_size_;
            int input_w_;
            int input_h_;
            std::string cache_file_;
            std::vector<std::filesystem::path> image_paths_;
            int current_batch_ = 0;

            void*        device_input_ = nullptr;   // [B, 3, H, W] preprocessed
            void*        device_raw_ = nullptr;      // single raw image upload
            cudaStream_t stream_ = nullptr;

            void*       pinned_staging_ = nullptr;
            std::size_t pinned_staging_size_ = 0;

            std::vector<char> calibration_cache_;
    };
}
