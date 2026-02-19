#pragma once

#include "core/cuda_buffer.hpp"
#include "core/frame_context.hpp"
#include <cuda_runtime.h>

namespace bytron {

    /// <summary>
    /// Launches the fused GPU preprocess kernel (resize + letterbox + BGR2RGB + normalize + HWC2CHW).
    /// Populates FrameContext with letterbox params so postprocess can rescale coords back.
    /// </summary>
    class GpuPreprocessor {
        public:
            GpuPreprocessor(int model_w, int model_h);

            void preprocess(const CudaDeviceBuffer& src, int src_w, int src_h,
                            CudaDeviceBuffer& dst, FrameContext& ctx,
                            cudaStream_t stream);

            [[nodiscard]] int model_width() const noexcept { return model_w_; }
            [[nodiscard]] int model_height() const noexcept { return model_h_; }

        private:
            int model_w_;
            int model_h_;
    };
}
