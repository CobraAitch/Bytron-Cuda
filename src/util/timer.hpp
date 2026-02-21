/**
 * @file timer.hpp
 * @brief Wall-clock timer for CPU-side latency measurement.
 */

#pragma once

#include <chrono>

namespace bytron {

    /**
     * @brief High-resolution wall-clock timer.
     *
     * Starts counting on construction. Not suitable for GPU timing;
     * use CudaEvent::elapsed_ms() for kernel timing.
     *
     * @note Thread safety: Safe for concurrent reads; reset() is not thread-safe.
     */
    class Timer {
    public:
        Timer() noexcept : start_(std::chrono::high_resolution_clock::now()) {}

        /** @brief Restart the timer from now. */
        void reset() noexcept { start_ = std::chrono::high_resolution_clock::now(); }

        /** @brief Elapsed time since construction or last reset(), in milliseconds. */
        [[nodiscard]] double elapsed_ms() const noexcept {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::milli>(now - start_).count();
        }

        /** @brief Elapsed time since construction or last reset(), in microseconds. */
        [[nodiscard]] double elapsed_us() const noexcept {
            auto now = std::chrono::high_resolution_clock::now();
            return std::chrono::duration<double, std::micro>(now - start_).count();
        }

    private:
        std::chrono::high_resolution_clock::time_point start_;
    };

}
