/**
 * @file calibrator.cpp
 * @brief INT8 calibration image loading, preprocessing, and cache I/O.
 */

// TRT 10.x deprecates IInt8EntropyCalibrator2 (implicit-quantization APIs);
// this project still supports INT8 PTQ, so silence the deprecation locally.
#pragma warning(push)
#pragma warning(disable: 4996)

#include "engine/calibrator.hpp"
#include "preprocess/preprocess_kernels.cuh"
#include "util/cuda_check.hpp"

#include <stb/stb_image.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <random>

namespace bytron {

    Int8EntropyCalibrator::Int8EntropyCalibrator(
        const std::filesystem::path& image_dir,
        int batch_size, int input_w, int input_h,
        const std::string& cache_file
    )
        : batch_size_(batch_size)
        , input_w_(input_w)
        , input_h_(input_h)
        , cache_file_(cache_file)
    {
        for (const auto& entry : std::filesystem::directory_iterator(image_dir)) {
            auto ext = entry.path().extension().string();
            std::ranges::transform(ext, ext.begin(), ::tolower);
            if (ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".bmp")
                image_paths_.push_back(entry.path());
        }

        // Shuffle for better calibration statistics (fixed seed for reproducibility).
        std::mt19937 rng(42);
        std::ranges::shuffle(image_paths_, rng);

        if (image_paths_.empty())
            spdlog::warn("No calibration images found in {}", image_dir.string());
        else
            spdlog::info("INT8 calibration: {} images from {}", image_paths_.size(), image_dir.string());

        auto input_bytes = std::size_t(batch_size) * 3 * input_h * input_w * sizeof(float);
        CUDA_CHECK(cudaMalloc(&device_input_, input_bytes));

        constexpr std::size_t max_raw_size = 3840ULL * 2160 * 4;
        pinned_staging_size_ = max_raw_size;

        for (int b = 0; b < NUM_BUFS; ++b) {
            CUDA_CHECK(cudaMalloc(&device_raw_[b], max_raw_size));
            CUDA_CHECK(cudaMallocHost(&pinned_staging_[b], pinned_staging_size_));
            CUDA_CHECK(cudaStreamCreate(&stream_[b]));
        }
    }

    Int8EntropyCalibrator::~Int8EntropyCalibrator() {
        if (device_input_) cudaFree(device_input_);
        for (int b = 0; b < NUM_BUFS; ++b) {
            if (device_raw_[b])     cudaFree(device_raw_[b]);
            if (pinned_staging_[b]) cudaFreeHost(pinned_staging_[b]);
            if (stream_[b])         cudaStreamDestroy(stream_[b]);
        }
    }

    int32_t Int8EntropyCalibrator::getBatchSize() const noexcept { return batch_size_; }

    bool Int8EntropyCalibrator::getBatch(
        void* bindings[], const char* /*names*/[], int32_t /*nbBindings*/
    ) noexcept {
        try {
            int start_idx = current_batch_ * batch_size_;
            if (start_idx >= static_cast<int>(image_paths_.size()))
                return false;

            int end_idx      = std::min(start_idx + batch_size_, static_cast<int>(image_paths_.size()));
            int actual_batch = end_idx - start_idx;
            auto single_input_bytes = std::size_t(3) * input_h_ * input_w_ * sizeof(float);
            int failed_in_batch = 0;

            for (int i = 0; i < actual_batch; ++i) {
                int buf = i % NUM_BUFS;

                // Ensure previous use of this buffer's stream is complete before
                // overwriting pinned staging memory.
                if (i >= NUM_BUFS)
                    CUDA_CHECK(cudaStreamSynchronize(stream_[buf]));

                const auto& path = image_paths_[start_idx + i];

                int src_w = 0, src_h = 0, channels = 0;
                auto* rgba_data = stbi_load(path.string().c_str(), &src_w, &src_h, &channels, 4);
                if (!rgba_data) {
                    spdlog::warn("Failed to load calibration image: {}", path.string());
                    ++failed_in_batch;
                    CUDA_CHECK(cudaMemsetAsync(
                        static_cast<char*>(device_input_) + i * single_input_bytes,
                        0, single_input_bytes, stream_[buf]));
                    continue;
                }

                auto raw_bytes = std::size_t(src_w) * src_h * 4;

                if (raw_bytes > pinned_staging_size_) {
                    spdlog::warn("Calibration image too large ({}x{}, {} bytes > {} limit), skipping: {}",
                                 src_w, src_h, raw_bytes, pinned_staging_size_, path.string());
                    stbi_image_free(rgba_data);
                    ++failed_in_batch;
                    CUDA_CHECK(cudaMemsetAsync(
                        static_cast<char*>(device_input_) + i * single_input_bytes,
                        0, single_input_bytes, stream_[buf]));
                    continue;
                }

                auto* pixels = static_cast<unsigned char*>(pinned_staging_[buf]);
                std::memcpy(pixels, rgba_data, raw_bytes);
                stbi_image_free(rgba_data);

                // stb_image loads as RGBA; swap R<->B to produce BGRA for the preprocess kernel.
                for (std::size_t p = 0; p < raw_bytes; p += 4)
                    std::swap(pixels[p], pixels[p + 2]);

                CUDA_CHECK(cudaMemcpyAsync(device_raw_[buf], pinned_staging_[buf], raw_bytes,
                                            cudaMemcpyHostToDevice, stream_[buf]));

                auto* dst_ptr = reinterpret_cast<float*>(
                    static_cast<char*>(device_input_) + i * single_input_bytes);

                launch_fused_preprocess_bgra(
                    static_cast<const std::uint8_t*>(device_raw_[buf]), dst_ptr,
                    src_w, src_h, src_w * 4, input_w_, input_h_,
                    compute_letterbox(src_w, src_h, input_w_, input_h_), stream_[buf]);
            }

            for (int i = actual_batch; i < batch_size_; ++i) {
                CUDA_CHECK(cudaMemsetAsync(
                    static_cast<char*>(device_input_) + i * single_input_bytes,
                    0, single_input_bytes, stream_[0]));
            }

            if (failed_in_batch == actual_batch) {
                spdlog::error("All {} images in calibration batch failed to load", actual_batch);
                return false;
            }
            if (failed_in_batch > 0)
                spdlog::warn("{}/{} calibration images failed in batch", failed_in_batch, actual_batch);

            // Synchronize both streams before returning the batch.
            for (int b = 0; b < NUM_BUFS; ++b)
                CUDA_CHECK(cudaStreamSynchronize(stream_[b]));

            bindings[0] = device_input_;
            ++current_batch_;

            int total_batches = (static_cast<int>(image_paths_.size()) + batch_size_ - 1) / batch_size_;
            spdlog::info("Calibration batch {}/{} ({} images, {} failed)",
                         current_batch_, total_batches, actual_batch, failed_in_batch);
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
        calibration_cache_ = std::vector<char>(std::istreambuf_iterator<char>(file), {});
        length = calibration_cache_.size();
        spdlog::info("Read calibration cache ({} bytes): {}", length, cache_file_);
        return calibration_cache_.data();
    }

    void Int8EntropyCalibrator::writeCalibrationCache(const void* cache, std::size_t length) noexcept {
        std::ofstream file(cache_file_, std::ios::binary);
        if (file.is_open()) {
            file.write(static_cast<const char*>(cache), length);
            spdlog::info("Wrote calibration cache ({} bytes): {}", length, cache_file_);
        }
    }

}

#pragma warning(pop)
