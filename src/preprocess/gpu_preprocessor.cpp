#include "preprocess/gpu_preprocessor.hpp"
#include "preprocess/preprocess_kernels.cuh"

#include <cstdint>

namespace bytron {

    GpuPreprocessor::GpuPreprocessor(int model_w, int model_h)
        : model_w_(model_w), model_h_(model_h) {}

    void GpuPreprocessor::preprocess(
        const CudaDeviceBuffer& src, int src_w, int src_h,
        CudaDeviceBuffer& dst, FrameContext& ctx,
        cudaStream_t stream
    ) {
        auto params = compute_letterbox(src_w, src_h, model_w_, model_h_);

        ctx.src_width = src_w;
        ctx.src_height = src_h;
        ctx.scale = params.scale;
        ctx.pad_x = static_cast<float>(params.pad_x);
        ctx.pad_y = static_cast<float>(params.pad_y);

        launch_fused_preprocess(
            static_cast<const std::uint8_t*>(src.ptr()),
            static_cast<float*>(dst.ptr()),
            src_w, src_h,
            src_w * 3, // pitch = width * 3 (packed BGR)
            model_w_, model_h_,
            stream
        );
    }
}
