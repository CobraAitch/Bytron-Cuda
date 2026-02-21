/**
 * @file cuda_stream.hpp
 * @brief RAII wrapper for a CUDA stream.
 */

#pragma once

#include <cuda_runtime.h>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /**
     * @brief Move-only RAII CUDA stream, created with cudaStreamNonBlocking.
     *
     * Non-blocking mode prevents implicit synchronization with the default
     * stream, which is critical for pipeline parallelism across multiple slots.
     *
     * @note Thread safety: The CUDA stream handle can be used from any CPU
     *       thread; CUDA serializes operations on the same stream.
     */
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
