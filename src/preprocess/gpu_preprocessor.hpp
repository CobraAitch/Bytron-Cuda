/**
 * @file gpu_preprocessor.hpp
 * @brief Thin wrapper around the fused CUDA preprocessing kernel.
 */

#pragma once

#include "core/cuda_buffer.hpp"
#include "core/frame_context.hpp"
#include <cuda_runtime.h>

namespace bytron {

    /**
     * @brief Orchestrates BGRA-to-CHW preprocessing and populates letterbox metadata.
     *
     * Computes letterbox parameters, stores them in FrameContext for later
     * coordinate rescaling, then launches the fused kernel.
     *
     * @note Thread safety: Stateless after construction; safe with stream serialization.
     */
    class GpuPreprocessor {
    public:
        GpuPreprocessor(int model_w, int model_h);

        void preprocess_bgra(const CudaDeviceBuffer& src, int src_w, int src_h,
                             int src_pitch, CudaDeviceBuffer& dst, FrameContext& ctx,
                             cudaStream_t stream);

        [[nodiscard]] int model_width()  const noexcept { return model_w_; }
        [[nodiscard]] int model_height() const noexcept { return model_h_; }

    private:
        int model_w_;
        int model_h_;
    };

}
