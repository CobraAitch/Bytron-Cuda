#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <format>

namespace bytron {

    /// <summary>
    /// Exception type thrown by CUDA_CHECK on failure.
    /// Captures error code, call expression, and source location.
    /// </summary>
    class CudaError : public std::runtime_error {
        public:
            CudaError(cudaError_t err, const char* call, const char* file, int line)
                : std::runtime_error(format_message(err, call, file, line))
                , error_code_(err) {}

            [[nodiscard]] cudaError_t code() const noexcept { return error_code_; }

        private:
            cudaError_t error_code_;

            static std::string format_message(cudaError_t err, const char* call,
                                               const char* file, int line) {
                return std::format("CUDA error {} ({}) at {}:{}: {}",
                                   static_cast<int>(err), cudaGetErrorString(err),
                                   file, line, call);
            }
    };
}

#define CUDA_CHECK(call) do {                                     \
    cudaError_t err_ = (call);                                    \
    if (err_ != cudaSuccess) {                                    \
        throw ::bytron::CudaError(err_, #call, __FILE__, __LINE__); \
    }                                                             \
} while (0)
