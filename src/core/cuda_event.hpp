#pragma once

#include <cuda_runtime.h>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /// <summary>
    /// RAII wrapper for a CUDA event. Optionally supports timing queries.
    /// </summary>
    class CudaEvent {
        public:
            explicit CudaEvent(bool enable_timing = false) {
                unsigned int flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
                CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
            }

            ~CudaEvent() {
                if (event_) cudaEventDestroy(event_);
            }

            CudaEvent(CudaEvent&& other) noexcept
                : event_(std::exchange(other.event_, nullptr)) {}

            CudaEvent& operator=(CudaEvent&& other) noexcept {
                if (this != &other) {
                    if (event_) cudaEventDestroy(event_);
                    event_ = std::exchange(other.event_, nullptr);
                }
                return *this;
            }

            CudaEvent(const CudaEvent&) = delete;
            CudaEvent& operator=(const CudaEvent&) = delete;

            void record(cudaStream_t stream) const {
                CUDA_CHECK(cudaEventRecord(event_, stream));
            }

            void synchronize() const {
                CUDA_CHECK(cudaEventSynchronize(event_));
            }

            [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }
            operator cudaEvent_t() const noexcept { return event_; }

            /// <summary>
            /// Returns elapsed milliseconds between two recorded timing events.
            /// </summary>
            [[nodiscard]] static float elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
                float ms = 0.0f;
                CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
                return ms;
            }

        private:
            cudaEvent_t event_ = nullptr;
    };
}
