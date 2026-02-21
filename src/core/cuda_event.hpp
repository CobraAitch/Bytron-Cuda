/**
 * @file cuda_event.hpp
 * @brief RAII wrapper for a CUDA event with optional timing support.
 */

#pragma once

#include <cuda_runtime.h>
#include <utility>
#include "util/cuda_check.hpp"

namespace bytron {

    /**
     * @brief Move-only RAII CUDA event.
     *
     * Events created without timing (default) have lower overhead and are
     * suitable for pure synchronization. Enable timing only for events
     * used with elapsed_ms().
     *
     * @note Thread safety: CUDA events are thread-safe at the driver level.
     */
    class CudaEvent {
    public:
        /**
         * @param [in] enable_timing If true, creates with cudaEventDefault (timing-capable).
         *        If false, creates with cudaEventDisableTiming for lower overhead.
         */
        explicit CudaEvent(bool enable_timing = false) {
            unsigned flags = enable_timing ? cudaEventDefault : cudaEventDisableTiming;
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

        void record(cudaStream_t stream) const { CUDA_CHECK(cudaEventRecord(event_, stream)); }
        void synchronize() const               { CUDA_CHECK(cudaEventSynchronize(event_)); }

        [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }
        operator cudaEvent_t() const noexcept { return event_; }

        /**
         * @brief Compute elapsed GPU time between two recorded events.
         *
         * @pre Both events must have been created with timing enabled and
         *      recorded on the same (or synchronized) stream(s).
         * @throws CudaError if events were not recorded or timing is disabled.
         */
        [[nodiscard]] static float elapsed_ms(const CudaEvent& start, const CudaEvent& end) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, end.event_));
            return ms;
        }

    private:
        cudaEvent_t event_ = nullptr;
    };

}
