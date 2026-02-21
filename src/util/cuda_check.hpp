/**
 * @file cuda_check.hpp
 * @brief CUDA error checking macro and exception type.
 */

#pragma once

#include <cuda_runtime.h>
#include <format>
#include <stdexcept>
#include <string>

namespace bytron {

    /**
     * @brief Exception thrown on CUDA runtime errors.
     *
     * Captures the cudaError_t code, stringified call expression,
     * and source location for diagnostic messages.
     */
    class CudaError : public std::runtime_error {
    public:
        CudaError(cudaError_t err, const char* call, const char* file, int line)
            : std::runtime_error(std::format("CUDA error {} ({}) at {}:{}: {}",
                  static_cast<int>(err), cudaGetErrorString(err), file, line, call))
            , error_code_(err) {}

        [[nodiscard]] cudaError_t code() const noexcept { return error_code_; }

    private:
        cudaError_t error_code_;
    };

}

/**
 * @brief Check a CUDA runtime call and throw CudaError on failure.
 *
 * @code
 * CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream));
 * @endcode
 */
#define CUDA_CHECK(call) do {                                             \
    cudaError_t err_ = (call);                                            \
    if (err_ != cudaSuccess)                                              \
        throw ::bytron::CudaError(err_, #call, __FILE__, __LINE__);       \
} while (0)
