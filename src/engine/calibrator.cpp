#include "engine/calibrator.hpp"
#include "preprocess/preprocess_kernels.cuh"
#include "util/cuda_check.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <fstream>
#include <algorithm>
#include <cstring>

namespace bytron {

    Int8EntropyCalibrator::Int8EntropyCalibrator(
        const std::filesystem::path& image_dir,
        int batch_size,
        int input_w, int input_h,
        const std::string& cache_file
    )
        : batch_size_(batch_size)
        , input_w_(input_w)
        , input_h_(input_h)
        , cache_file_(cache_file)
    {
        // Collect supported image files
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp") {
                image_paths_.push_back(entry.path());
            }
        }

        std::sort(image_paths_.begin(), image_paths_.end());

        if (image_paths_.empty()) {
            spdlog::warn("No calibration images found in {}", image_dir.string());
        } else {
            spdlog::info("INT8 calibration: {} images from {}", image_paths_.size(), image_dir.string());
        }

        std::size_t input_bytes = static_cast<std::size_t>(batch_size) * 3 * input_h * input_w * sizeof(float);
        CUDA_CHECK(cudaMalloc(&device_input_, input_bytes));

        // 4K is the largest source resolution we expect during calibration
        constexpr std::size_t max_raw_size = 3840ULL * 2160 * 3;
        CUDA_CHECK(cudaMalloc(&device_raw_, max_raw_size));

        pinned_staging_size_ = max_raw_size;
        CUDA_CHECK(cudaMallocHost(&pinned_staging_, pinned_staging_size_));

        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    Int8EntropyCalibrator::~Int8EntropyCalibrator() {
        if (device_input_) cudaFree(device_input_);
        if (device_raw_) cudaFree(device_raw_);
        if (pinned_staging_) cudaFreeHost(pinned_staging_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    int32_t Int8EntropyCalibrator::getBatchSize() const noexcept {
        return batch_size_;
    }

    bool Int8EntropyCalibrator::getBatch(
        void* bindings[], const char* /*names*/[], int32_t /*nbBindings*/
    ) noexcept {
        try {
            int start_idx = current_batch_ * batch_size_;
            if (start_idx >= static_cast<int>(image_paths_.size())) {
                return false; // all batches consumed
            }

            int end_idx = std::min(start_idx + batch_size_, static_cast<int>(image_paths_.size()));
            int actual_batch = end_idx - start_idx;

            std::size_t single_input_bytes = static_cast<std::size_t>(3) * input_h_ * input_w_ * sizeof(float);

            for (int i = 0; i < actual_batch; ++i) {
                const auto& path = image_paths_[start_idx + i];

                cv::Mat img = cv::imread(path.string(), cv::IMREAD_COLOR);
                if (img.empty()) {
                    spdlog::warn("Failed to load calibration image: {}", path.string());
                    CUDA_CHECK(cudaMemsetAsync(
                        static_cast<char*>(device_input_) + i * single_input_bytes,
                        0, single_input_bytes, stream_));
                    continue;
                }

                int src_w = img.cols;
                int src_h = img.rows;
                std::size_t raw_bytes = static_cast<std::size_t>(src_w) * src_h * 3;

                // Pinned staging -> GPU upload -> fused preprocess
                std::memcpy(pinned_staging_, img.data, raw_bytes);
                CUDA_CHECK(cudaMemcpyAsync(device_raw_, pinned_staging_, raw_bytes,
                                            cudaMemcpyHostToDevice, stream_));

                float* dst_ptr = reinterpret_cast<float*>(
                    static_cast<char*>(device_input_) + i * single_input_bytes);

                launch_fused_preprocess(
                    static_cast<const uint8_t*>(device_raw_),
                    dst_ptr,
                    src_w, src_h,
                    src_w * 3,
                    input_w_, input_h_,
                    stream_
                );
            }

            // Zero-fill any remaining slots in a partial last batch
            for (int i = actual_batch; i < batch_size_; ++i) {
                CUDA_CHECK(cudaMemsetAsync(
                    static_cast<char*>(device_input_) + i * single_input_bytes,
                    0, single_input_bytes, stream_));
            }

            CUDA_CHECK(cudaStreamSynchronize(stream_));

            bindings[0] = device_input_;
            ++current_batch_;

            int total_batches = (static_cast<int>(image_paths_.size()) + batch_size_ - 1) / batch_size_;
            spdlog::info("Calibration batch {}/{} ({} images)",
                         current_batch_, total_batches, actual_batch);
            return true;

        } catch (const std::exception& e) {
            spdlog::error("Calibration getBatch failed: {}", e.what());
            return false;
        }
    }

    const void* Int8EntropyCalibrator::readCalibrationCache(std::size_t& length) noexcept {
        std::ifstream file(cache_file_, std::ios::binary);
        if (!file.is_open()) {
            length = 0;
            return nullptr;
        }

        calibration_cache_ = std::vector<char>(
            std::istreambuf_iterator<char>(file), {});
        length = calibration_cache_.size();
        spdlog::info("Read calibration cache ({} bytes): {}", length, cache_file_);
        return calibration_cache_.data();
    }

    void Int8EntropyCalibrator::writeCalibrationCache(
        const void* cache, std::size_t length
    ) noexcept {
        std::ofstream file(cache_file_, std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char*>(cache), length);
            spdlog::info("Wrote calibration cache ({} bytes): {}", length, cache_file_);
        }
    }
}
