#pragma once

#include <cuda_runtime.h>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /// <summary>
    /// RAII wrapper for a non-blocking CUDA stream.
    /// </summary>
    class CudaStream {
        public:
            CudaStream() {
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            }

            ~CudaStream() {
                if (stream_) cudaStreamDestroy(stream_);
            }

            CudaStream(CudaStream&& other) noexcept
                : stream_(std::exchange(other.stream_, nullptr)) {}

            CudaStream& operator=(CudaStream&& other) noexcept {
                if (this != &other) {
                    if (stream_) cudaStreamDestroy(stream_);
                    stream_ = std::exchange(other.stream_, nullptr);
                }
                return *this;
            }

            CudaStream(const CudaStream&) = delete;
            CudaStream& operator=(const CudaStream&) = delete;

            [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }
            operator cudaStream_t() const noexcept { return stream_; }

            void synchronize() const { CUDA_CHECK(cudaStreamSynchronize(stream_)); }

        private:
            cudaStream_t stream_ = nullptr;
    };
}
