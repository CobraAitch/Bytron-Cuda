/**
 * @file gpu_preprocessor.cpp
 * @brief GpuPreprocessor implementation -- letterbox computation and kernel launch.
 */

#include "preprocess/gpu_preprocessor.hpp"
#include "preprocess/preprocess_kernels.cuh"

#include <cstdint>

namespace bytron {

    GpuPreprocessor::GpuPreprocessor(int model_w, int model_h)
        : model_w_(model_w), model_h_(model_h) {}

    void GpuPreprocessor::preprocess_bgra(
        const CudaDeviceBuffer& src, int src_w, int src_h,
        int src_pitch, CudaDeviceBuffer& dst, FrameContext& ctx,
        cudaStream_t stream
    ) {
        auto params = compute_letterbox(src_w, src_h, model_w_, model_h_);

        ctx.src_width  = src_w;
        ctx.src_height = src_h;
        ctx.src_pitch  = src_pitch;
        ctx.scale = params.scale;
        ctx.pad_x = static_cast<float>(params.pad_x);
        ctx.pad_y = static_cast<float>(params.pad_y);

        launch_fused_preprocess_bgra(
            src.as<const std::uint8_t>(), dst.as<float>(),
            src_w, src_h, src_pitch, model_w_, model_h_, params, stream);
    }

}
