#pragma once

#include <chrono>

namespace bytron {

    /// <summary>
    /// Simple wall-clock timer using high_resolution_clock.
    /// </summary>
    class Timer {
        public:
            Timer() : start_(std::chrono::high_resolution_clock::now()) {}

            void reset() { start_ = std::chrono::high_resolution_clock::now(); }

            [[nodiscard]] double elapsed_ms() const {
                auto now = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::milli>(now - start_).count();
            }

            [[nodiscard]] double elapsed_us() const {
                auto now = std::chrono::high_resolution_clock::now();
                return std::chrono::duration<double, std::micro>(now - start_).count();
            }

        private:
            std::chrono::high_resolution_clock::time_point start_;
    };
}
