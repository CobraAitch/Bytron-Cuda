/**
 * @file cuda_stream.hpp
 * @brief RAII wrapper for a CUDA stream with optional priority scheduling.
 */

#pragma once

#include <cuda_runtime.h>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /**
     * @brief CUDA stream priority levels for GPU work scheduling.
     *
     * Maps to the device's priority range via cudaDeviceGetStreamPriorityRange.
     * High-priority streams preempt lower-priority work when resources are
     * contended, reducing latency on the critical inference path.
     */
    enum class StreamPriority { High, Normal, Low };

    /**
     * @brief Move-only RAII CUDA stream, created with cudaStreamNonBlocking.
     *
     * Non-blocking mode prevents implicit synchronization with the default
     * stream, which is critical for pipeline parallelism across multiple slots.
     * An optional StreamPriority allows inference streams to preempt display
     * work on the same GPU.
     *
     * @note Thread safety: The CUDA stream handle can be used from any CPU
     *       thread; CUDA serializes operations on the same stream.
     */
    class CudaStream {
    public:
        explicit CudaStream(StreamPriority priority = StreamPriority::Normal) {
            if (priority == StreamPriority::Normal) {
                CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
            } else {
                int lo = 0, hi = 0;  // lo = lowest priority (largest value), hi = highest (smallest)
                CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lo, &hi));
                int p = (priority == StreamPriority::High) ? hi : lo;
                CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, p));
            }
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
